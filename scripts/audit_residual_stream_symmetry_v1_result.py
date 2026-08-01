from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import param_groups  # noqa: E402
from src.model import VisionTransformer  # noqa: E402
from src.norm_control import control_units, frobenius_norm  # noqa: E402
from src.policy_overlay_residual_stream_symmetry import (  # noqa: E402
    POLICY,
    install,
)


BRANCHES = ("constant", "linear_up", "linear_down", "cyclic")
CONTROLLED_BRANCHES = BRANCHES[1:]
EXPECTED_PREFIX_SHA256 = (
    "90b221bb6b560f30121e09ce89d019b9bcba91884fb94bb1a30ca9dbac25a0fd"
)
EXPECTED_CONTROL_START = 2_500
EXPECTED_TOTAL_STEPS = 20_000
EXPECTED_TRAIN_SIZE = 100_000
EXPECTED_CONTROLLED_TENSORS = 52
EXPECTED_UNCONTROLLED_TENSORS = 100
EXPECTED_OPTIMIZER_STATES = 152

METRICS_HEADER = (
    "global_state_step",
    "branch",
    "train_loss_raw",
    "train_loss_ema",
    "train_top1",
    "base_lr",
    "controlled_lr",
    "schedule_ratio",
    "controlled_frobenius_norm",
    "target_norm_relative_error_mean",
    "target_norm_relative_error_max",
    "tensorwise_elr_mean",
    "tensorwise_elr_median",
    "tensorwise_elr_min",
    "tensorwise_elr_max",
    "elr_relative_error_mean",
    "elr_relative_error_max",
    "gradient_finite",
    "angular_step_mean",
    "angular_step_median",
)
FINITE_METRIC_FIELDS = (
    "train_loss_raw",
    "train_loss_ema",
    "train_top1",
    "base_lr",
    "controlled_lr",
    "schedule_ratio",
    "controlled_frobenius_norm",
    "target_norm_relative_error_mean",
    "target_norm_relative_error_max",
    "tensorwise_elr_mean",
    "tensorwise_elr_median",
    "tensorwise_elr_min",
    "tensorwise_elr_max",
    "elr_relative_error_mean",
    "elr_relative_error_max",
)
SUMMARY_FIELDS = (
    "branch",
    "loss_raw_mae_vs_constant",
    "loss_ema_mae_vs_constant",
    "loss_ema_max_abs_vs_constant",
    "probe_loss_mae_vs_constant",
    "nce",
    "mean_elr_relative_error",
    "max_elr_relative_error",
    "mean_target_norm_relative_error",
    "max_target_norm_relative_error",
    "mean_angular_step",
    "angular_step_mae_vs_constant",
    "final_train_loss_ema",
    "final_probe_loss",
    "final_val_loss",
    "final_val_top1",
)


class AuditError(ValueError):
    """The result artifacts violate an independent audit invariant."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml_mapping(path: Path) -> dict:
    _require(path.is_file(), f"missing YAML: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected YAML mapping: {path}")
    return value


def _load_json_mapping(path: Path) -> dict:
    _require(path.is_file(), f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON mapping: {path}")
    return value


def _finite_float(value: Any, label: str, *, nonnegative: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{label} is not numeric: {value!r}") from exc
    _require(math.isfinite(result), f"{label} is non-finite: {result!r}")
    if nonnegative:
        _require(result >= 0.0, f"{label} is negative: {result!r}")
    return result


def _close(
    actual: float,
    expected: float,
    label: str,
    *,
    rel_tol: float = 2e-11,
    abs_tol: float = 2e-12,
) -> None:
    _require(
        math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol),
        f"{label} mismatch: actual={actual:.17g}, expected={expected:.17g}",
    )


def _base_lr_at(step: int, cfg: Mapping[str, Any]) -> float:
    peak = float(cfg["peak_lr"])
    final = float(cfg["final_lr"])
    warmup = int(cfg["warmup_steps"])
    decay_start = int(cfg["decay_start_step"])
    total = int(cfg["total_steps"])
    _require(0 <= step < total, f"base-LR step outside range: {step}")
    if step < warmup:
        return peak * (step + 1) / warmup
    if step < decay_start:
        return peak
    return final + (peak - final) * (total - step) / (total - decay_start)


def _schedule_ratio(
    branch: str,
    step: int,
    optimizer_cfg: Mapping[str, Any],
    control_cfg: Mapping[str, Any],
) -> float:
    start = int(optimizer_cfg["control_start_step"])
    total = int(optimizer_cfg["total_steps"])
    _require(start <= step <= total, f"schedule step outside range: {step}")
    progress = (step - start) / (total - start)
    if branch == "constant":
        return 1.0
    if branch == "linear_up":
        return 1.0 + progress * (
            float(control_cfg["linear_up_final"]) - 1.0
        )
    if branch == "linear_down":
        return 1.0 + progress * (
            float(control_cfg["linear_down_final"]) - 1.0
        )
    if branch == "cyclic":
        return 1.0 + float(control_cfg["cyclic_amplitude"]) * math.sin(
            2.0
            * math.pi
            * (step - start)
            / int(control_cfg["cyclic_period_steps"])
        )
    raise AuditError(f"unexpected branch: {branch}")


def validate_config(output: Path, expected_config: Path) -> dict:
    resolved = _load_yaml_mapping(output / "resolved_config.yaml")
    expected = _load_yaml_mapping(expected_config)
    _require(
        resolved == expected,
        "resolved_config.yaml differs from the approved residual-v1 config",
    )
    _require(
        resolved["experiment"]["output_dir"]
        == "outputs/residual_stream_symmetry_v1",
        "unexpected output_dir",
    )
    _require(
        resolved["control"]["policy"] == POLICY,
        f"control.policy must be {POLICY}",
    )
    _require(
        resolved["control"]["schedules"] == list(BRANCHES),
        "the four schedule names/order changed",
    )
    _require(
        resolved["control"]["split_fused_qkv"] is True,
        "split_fused_qkv must be exactly true",
    )
    _require(
        resolved["control"]["modify_optimizer_moments"] is False,
        "residual v1 must leave optimizer moments unmodified",
    )
    _require(
        int(resolved["optimizer"]["control_start_step"])
        == EXPECTED_CONTROL_START,
        f"control_start_step must be {EXPECTED_CONTROL_START}",
    )
    _require(
        int(resolved["optimizer"]["total_steps"]) == EXPECTED_TOTAL_STEPS,
        f"total_steps must be {EXPECTED_TOTAL_STEPS}",
    )
    _require(
        int(resolved["logging"]["train_log_interval"]) == 1,
        "train_log_interval must be 1 for a full-window audit",
    )
    _close(
        float(resolved["logging"]["ema_beta"]),
        0.99,
        "EMA beta",
        rel_tol=0.0,
        abs_tol=0.0,
    )
    _require(
        int(resolved["model"]["depth"]) == 12,
        "residual-v1 audit expects 12 transformer blocks",
    )
    _close(
        float(resolved["model"]["norm_eps"]),
        1e-6,
        "LayerNorm epsilon",
        rel_tol=0.0,
        abs_tol=0.0,
    )
    _close(
        float(resolved["optimizer"]["eps"]),
        1e-8,
        "Adam epsilon",
        rel_tol=0.0,
        abs_tol=0.0,
    )
    _close(
        float(resolved["optimizer"]["weight_decay"]),
        0.0,
        "weight decay",
        rel_tol=0.0,
        abs_tol=0.0,
    )
    return resolved


def expected_controlled_names(depth: int) -> set[str]:
    names = {
        "patch_embed.proj.weight",
        "patch_embed.proj.bias",
        "cls_token",
        "pos_embed",
    }
    for index in range(depth):
        names.update({
            f"blocks.{index}.attn.proj.weight",
            f"blocks.{index}.attn.proj.bias",
            f"blocks.{index}.mlp.fc2.weight",
            f"blocks.{index}.mlp.fc2.bias",
        })
    return names


def validate_policy_scope(cfg: dict) -> tuple[VisionTransformer, dict, dict, dict]:
    install()
    model = VisionTransformer(cfg["model"])
    controlled, uncontrolled = param_groups.classify_parameters(model, POLICY)
    units = control_units(controlled, split_fused_qkv=True)
    expected = expected_controlled_names(int(cfg["model"]["depth"]))
    trainable = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    _require(set(controlled) == expected, "controlled policy scope is not exact")
    _require(
        len(controlled) == EXPECTED_CONTROLLED_TENSORS,
        f"expected {EXPECTED_CONTROLLED_TENSORS} controlled tensors",
    )
    _require(
        len(uncontrolled) == EXPECTED_UNCONTROLLED_TENSORS,
        f"expected {EXPECTED_UNCONTROLLED_TENSORS} uncontrolled tensors",
    )
    _require(
        set(controlled) | set(uncontrolled) == trainable
        and not (set(controlled) & set(uncontrolled)),
        "controlled/uncontrolled sets are not a trainable partition",
    )
    _require(set(units) == expected, "control-unit scope differs from policy scope")
    forbidden = {
        name
        for name in controlled
        if (
            ".attn.qkv." in name
            or ".mlp.fc1." in name
            or ".norm" in name
            or name.startswith("norm.")
            or name.startswith("head.")
        )
    }
    _require(not forbidden, f"forbidden tensors are controlled: {sorted(forbidden)}")
    unit_numels = {name: unit.numel() for name, unit in units.items()}
    return model, controlled, uncontrolled, unit_numels


def _all_tensors_finite(value: Any, label: str) -> int:
    count = 0
    if isinstance(value, torch.Tensor):
        _require(
            bool(torch.isfinite(value).all().item()),
            f"{label} contains a non-finite tensor",
        )
        return 1
    if isinstance(value, Mapping):
        for key, child in value.items():
            count += _all_tensors_finite(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            count += _all_tensors_finite(child, f"{label}[{index}]")
    return count


def _validate_sampler_state(
    checkpoint: Mapping[str, Any],
    expected_steps: int,
    cfg: dict,
    train_size: int,
    label: str,
) -> dict:
    sampler = checkpoint.get("sampler_state")
    _require(isinstance(sampler, dict), f"{label}: sampler_state missing")
    epoch = sampler.get("epoch")
    offset = sampler.get("batch_offset")
    _require(
        isinstance(epoch, int) and epoch >= 0,
        f"{label}: invalid sampler epoch",
    )
    _require(
        isinstance(offset, int) and offset >= 0,
        f"{label}: invalid sampler batch_offset",
    )
    micro = int(cfg["data"]["micro_batch_size"])
    global_batch = int(cfg["data"]["global_batch_size"])
    _require(global_batch % micro == 0, "invalid gradient accumulation")
    accumulation = global_batch // micro
    drop_last = bool(cfg["data"]["drop_last"])
    batches_per_epoch = (
        train_size // micro
        if drop_last
        else (train_size + micro - 1) // micro
    )
    expected_epoch, expected_offset = divmod(
        expected_steps * accumulation, batches_per_epoch
    )
    _require(
        (epoch, offset) == (expected_epoch, expected_offset),
        f"{label}: sampler position {(epoch, offset)} != "
        f"expected {(expected_epoch, expected_offset)}",
    )
    _require(
        checkpoint.get("epoch") == epoch
        and checkpoint.get("batch_offset") == offset,
        f"{label}: duplicated sampler fields disagree",
    )
    return {
        "epoch": epoch,
        "batch_offset": offset,
        "batches_per_epoch": batches_per_epoch,
        "accumulation": accumulation,
    }


def _validate_prefix_scientific_config(source: dict, target: dict) -> None:
    _require(source["model"] == target["model"], "prefix model config changed")
    _require(source["optimizer"] == target["optimizer"], "prefix optimizer config changed")
    for key in ("seed", "data_seed", "deterministic"):
        _require(
            source["experiment"][key] == target["experiment"][key],
            f"prefix experiment.{key} differs",
        )
    for key in (
        "global_batch_size",
        "micro_batch_size",
        "drop_last",
    ):
        _require(
            source["data"][key] == target["data"][key],
            f"prefix data.{key} differs",
        )
    for key in (
        "projection_eps",
        "schedules",
        "cyclic_period_steps",
        "cyclic_amplitude",
        "linear_up_final",
        "linear_down_final",
        "modify_optimizer_moments",
    ):
        _require(
            source["control"][key] == target["control"][key],
            f"prefix control.{key} differs",
        )
    _require(
        source["control"]["policy"] == "hidden_matrices",
        "shared prefix source policy is not hidden_matrices",
    )
    _require(
        source["logging"]["ema_beta"] == target["logging"]["ema_beta"],
        "prefix EMA beta differs",
    )


def validate_prefix(
    output: Path,
    cfg: dict,
    model: VisionTransformer,
    controlled: dict,
    train_size: int,
) -> tuple[dict[str, float], float, dict]:
    path = output / "prefix" / "checkpoint_step_002500.pt"
    _require(path.is_file() and path.stat().st_size > 0, "shared prefix is missing")
    actual_sha = sha256(path)
    _require(
        actual_sha == EXPECTED_PREFIX_SHA256,
        f"shared prefix SHA mismatch: {actual_sha}",
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, dict), "shared prefix is not a checkpoint mapping")
    _require(
        checkpoint.get("global_state_step") == EXPECTED_CONTROL_START,
        "shared prefix checkpoint step is not 2500",
    )
    source_cfg = checkpoint.get("config")
    _require(isinstance(source_cfg, dict), "shared prefix config missing")
    _validate_prefix_scientific_config(source_cfg, cfg)
    model_state = checkpoint.get("model")
    _require(isinstance(model_state, dict), "shared prefix model state missing")
    model.load_state_dict(model_state, strict=True)
    _all_tensors_finite(model_state, "prefix.model")
    _all_tensors_finite(checkpoint.get("optimizer"), "prefix.optimizer")
    prefix_ema = _finite_float(
        checkpoint.get("train_loss_ema"), "prefix train_loss_ema", nonnegative=True
    )
    sampler = _validate_sampler_state(
        checkpoint,
        EXPECTED_CONTROL_START,
        cfg,
        train_size,
        "prefix",
    )
    source_names = checkpoint.get("controlled_names")
    source_references = checkpoint.get("reference_norms")
    _require(isinstance(source_names, list), "prefix controlled_names missing")
    _require(isinstance(source_references, dict), "prefix reference_norms missing")
    _require(
        len(source_names) == len(set(source_names)),
        "prefix controlled_names contains duplicates",
    )
    _require(
        set(source_names) == set(source_references),
        "prefix controlled_names/reference_norm keys differ",
    )

    units = control_units(controlled, split_fused_qkv=True)
    references = {name: frobenius_norm(unit) for name, unit in units.items()}
    _require(
        set(references) == expected_controlled_names(int(cfg["model"]["depth"])),
        "prefix-derived residual reference scope differs",
    )
    for name, value in references.items():
        _require(
            math.isfinite(value) and value > float(cfg["control"]["projection_eps"]),
            f"invalid prefix-derived reference norm for {name}: {value!r}",
        )
        if name in source_references:
            _close(
                value,
                _finite_float(source_references[name], f"prefix reference {name}"),
                f"stored prefix reference {name}",
                rel_tol=2e-7,
                abs_tol=1e-9,
            )

    preflight = _load_json_mapping(output / "residual_stream_preflight.json")
    _require(
        preflight.get("checkpoint_sha256") == EXPECTED_PREFIX_SHA256
        and preflight.get("expected_checkpoint_sha256") == EXPECTED_PREFIX_SHA256,
        "preflight does not bind to the approved shared prefix",
    )
    _require(
        preflight.get("source_policy") == "hidden_matrices"
        and preflight.get("target_policy") == POLICY,
        "preflight policy transition differs",
    )
    for key, expected in (
        ("controlled_tensors", EXPECTED_CONTROLLED_TENSORS),
        ("control_units", EXPECTED_CONTROLLED_TENSORS),
        ("uncontrolled_tensors", EXPECTED_UNCONTROLLED_TENSORS),
        ("optimizer_state_entries", EXPECTED_OPTIMIZER_STATES),
    ):
        _require(preflight.get(key) == expected, f"preflight {key} differs")
    _require(
        preflight.get("optimizer_state_values_bitwise_equal") is True,
        "preflight did not prove lossless optimizer-state migration",
    )
    _require(
        set(preflight.get("added_from_hidden", []))
        == (
            set(references)
            - {
                name
                for name in source_names
                if name in set(references)
            }
        ),
        "preflight added-from-hidden scope differs from prefix evidence",
    )
    _require(
        len(preflight.get("added_from_hidden", [])) == 28,
        "preflight must report 28 added tensors",
    )
    _require(
        len(preflight.get("removed_from_hidden", [])) == 24,
        "preflight must report 24 removed tensors",
    )
    _close(
        float(preflight["reference_norm_min"]),
        min(references.values()),
        "preflight reference_norm_min",
        rel_tol=2e-7,
        abs_tol=1e-9,
    )
    _close(
        float(preflight["reference_norm_max"]),
        max(references.values()),
        "preflight reference_norm_max",
        rel_tol=2e-7,
        abs_tol=1e-9,
    )
    return references, prefix_ema, {
        "path": str(path.resolve()),
        "sha256": actual_sha,
        "global_state_step": EXPECTED_CONTROL_START,
        "train_loss_ema": prefix_ema,
        "sampler_state": sampler,
        "source_policy": source_cfg["control"]["policy"],
        "preflight_verified": True,
    }


def _parse_canonical_step(raw: str, path: Path, row_number: int) -> int:
    stripped = raw.strip()
    try:
        step = int(stripped)
    except ValueError as exc:
        raise AuditError(
            f"{path}: row {row_number} has invalid step {raw!r}"
        ) from exc
    _require(
        str(step) == stripped and step >= 0,
        f"{path}: row {row_number} has noncanonical step {raw!r}",
    )
    return step


def audit_metrics_file(
    path: Path,
    cfg: dict,
    branch: str,
    prefix_ema: float,
    reference_norms: Mapping[str, float],
    unit_numels: Mapping[str, int],
    validity_threshold: float,
) -> tuple[list[dict[str, float]], dict]:
    _require(path.is_file(), f"{branch}: metrics.csv missing")
    optimizer_cfg = cfg["optimizer"]
    control_cfg = cfg["control"]
    logging_cfg = cfg["logging"]
    start = int(optimizer_cfg["control_start_step"])
    total = int(optimizer_cfg["total_steps"])
    beta = float(logging_cfg["ema_beta"])
    expected_steps = list(range(start + 1, total + 1))
    _require(set(reference_norms) == set(unit_numels), "reference/unit scope differs")
    combined_reference_norm = math.sqrt(
        sum(float(value) ** 2 for value in reference_norms.values())
    )
    rows: list[dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require(
            tuple(reader.fieldnames or ()) == METRICS_HEADER,
            f"{branch}: metrics header mismatch",
        )
        raw_rows = list(reader)
    _require(
        len(raw_rows) == len(expected_steps),
        f"{branch}: expected {len(expected_steps)} metrics rows, "
        f"got {len(raw_rows)}",
    )
    previous_ema = prefix_ema
    seen: set[int] = set()
    interval = int(logging_cfg["angular_interval"])
    for offset, (raw_row, expected_step) in enumerate(
        zip(raw_rows, expected_steps),
        start=2,
    ):
        _require(
            None not in raw_row and all(value is not None for value in raw_row.values()),
            f"{branch}: malformed metrics row {offset}",
        )
        step = _parse_canonical_step(
            raw_row["global_state_step"], path, offset
        )
        _require(
            step == expected_step,
            f"{branch}: row {offset} step {step} != expected {expected_step}",
        )
        _require(step not in seen, f"{branch}: duplicate metrics step {step}")
        seen.add(step)
        _require(
            raw_row["branch"] == branch,
            f"{branch}: row {offset} claims branch {raw_row['branch']!r}",
        )
        row = {
            field: _finite_float(
                raw_row[field],
                f"{branch} step {step} {field}",
                nonnegative=field
                in {
                    "train_loss_raw",
                    "train_loss_ema",
                    "base_lr",
                    "controlled_lr",
                    "controlled_frobenius_norm",
                    "target_norm_relative_error_mean",
                    "target_norm_relative_error_max",
                    "tensorwise_elr_mean",
                    "tensorwise_elr_median",
                    "tensorwise_elr_min",
                    "tensorwise_elr_max",
                    "elr_relative_error_mean",
                    "elr_relative_error_max",
                },
            )
            for field in FINITE_METRIC_FIELDS
        }
        row["global_state_step"] = float(step)
        _require(
            0.0 <= row["train_top1"] <= 1.0,
            f"{branch} step {step}: top1 outside [0, 1]",
        )
        _require(
            raw_row["gradient_finite"] == "True",
            f"{branch} step {step}: gradient_finite is not exactly True",
        )
        angular_mean = float(raw_row["angular_step_mean"])
        angular_median = float(raw_row["angular_step_median"])
        angular_expected = (step - 1) % interval == 0
        if angular_expected:
            _require(
                math.isfinite(angular_mean)
                and math.isfinite(angular_median)
                and angular_mean >= 0.0
                and angular_median >= 0.0,
                f"{branch} step {step}: angular metrics are not finite",
            )
        else:
            _require(
                math.isnan(angular_mean) and math.isnan(angular_median),
                f"{branch} step {step}: unexpected angular values",
            )
        row["angular_step_mean"] = angular_mean
        row["angular_step_median"] = angular_median

        optimizer_step = step - 1
        expected_base_lr = _base_lr_at(optimizer_step, optimizer_cfg)
        expected_ratio = _schedule_ratio(
            branch, optimizer_step, optimizer_cfg, control_cfg
        )
        expected_post_ratio = _schedule_ratio(
            branch, step, optimizer_cfg, control_cfg
        )
        _close(row["base_lr"], expected_base_lr, f"{branch} step {step} base_lr")
        _close(
            row["schedule_ratio"],
            expected_ratio,
            f"{branch} step {step} schedule_ratio",
        )
        _close(
            row["controlled_lr"],
            expected_base_lr * expected_ratio,
            f"{branch} step {step} controlled_lr",
        )
        expected_combined_norm = expected_post_ratio * combined_reference_norm
        _close(
            row["controlled_frobenius_norm"],
            expected_combined_norm,
            f"{branch} step {step} combined controlled norm",
            rel_tol=max(3e-6, validity_threshold * 2.0),
            abs_tol=2e-7,
        )

        expected_ema = beta * previous_ema + (1.0 - beta) * row["train_loss_raw"]
        _close(
            row["train_loss_ema"],
            expected_ema,
            f"{branch} step {step} EMA recurrence",
            rel_tol=3e-12,
            abs_tol=3e-12,
        )
        previous_ema = row["train_loss_ema"]

        _require(
            row["target_norm_relative_error_mean"]
            <= row["target_norm_relative_error_max"],
            f"{branch} step {step}: norm mean exceeds max",
        )
        _require(
            row["elr_relative_error_mean"]
            <= row["elr_relative_error_max"],
            f"{branch} step {step}: ELR mean exceeds max",
        )
        _require(
            row["target_norm_relative_error_max"] <= validity_threshold,
            f"{branch} step {step}: norm error exceeds threshold",
        )
        _require(
            row["elr_relative_error_max"] <= validity_threshold,
            f"{branch} step {step}: ELR error exceeds threshold",
        )
        _require(
            row["tensorwise_elr_min"]
            <= row["tensorwise_elr_median"]
            <= row["tensorwise_elr_max"],
            f"{branch} step {step}: tensorwise ELR order is invalid",
        )
        targets = [
            expected_base_lr
            / (float(reference_norms[name]) / math.sqrt(unit_numels[name]))
            for name in reference_norms
        ]
        expected_elr_stats = {
            "tensorwise_elr_mean": statistics.mean(targets),
            "tensorwise_elr_median": statistics.median(targets),
            "tensorwise_elr_min": min(targets),
            "tensorwise_elr_max": max(targets),
        }
        for field, expected_value in expected_elr_stats.items():
            _close(
                row[field],
                expected_value,
                f"{branch} step {step} {field}",
                rel_tol=max(3e-6, validity_threshold * 2.0),
                abs_tol=2e-12,
            )
        rows.append(row)
    _require(
        seen == set(expected_steps),
        f"{branch}: metrics step set is not the exact controlled window",
    )
    return rows, {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "rows": len(rows),
        "first_step": expected_steps[0],
        "last_step": expected_steps[-1],
        "steps_unique_and_complete": True,
        "ema_recurrence_verified_from_prefix": True,
        "schedule_and_lr_verified": True,
        "all_core_metrics_finite": True,
        "all_gradients_reported_finite": True,
        "max_target_norm_relative_error": max(
            row["target_norm_relative_error_max"] for row in rows
        ),
        "max_elr_relative_error": max(
            row["elr_relative_error_max"] for row in rows
        ),
        "final_train_loss_ema": rows[-1]["train_loss_ema"],
    }


def _recursive_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(
            _recursive_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _recursive_equal(a, b) for a, b in zip(left, right)
        )
    try:
        import numpy as np

        if isinstance(left, np.ndarray):
            return bool(np.array_equal(left, right))
    except ImportError:
        pass
    return bool(left == right)


def _extract_rng_state(checkpoint: Mapping[str, Any]) -> dict:
    keys = (
        "python_rng",
        "numpy_rng",
        "torch_cpu_rng",
        "torch_cuda_rng_all",
    )
    _require(all(key in checkpoint for key in keys), "checkpoint RNG state missing")
    return {key: copy.deepcopy(checkpoint[key]) for key in keys}


def _validate_optimizer_layout(
    optimizer_state: Any,
    branch: str,
    controlled_count: int,
    uncontrolled_count: int,
) -> None:
    _require(isinstance(optimizer_state, dict), f"{branch}: optimizer state missing")
    state = optimizer_state.get("state")
    groups = optimizer_state.get("param_groups")
    _require(isinstance(state, dict), f"{branch}: optimizer.state missing")
    _require(
        len(state) == EXPECTED_OPTIMIZER_STATES,
        f"{branch}: optimizer has {len(state)} states, "
        f"expected {EXPECTED_OPTIMIZER_STATES}",
    )
    _require(
        isinstance(groups, list) and len(groups) == 2,
        f"{branch}: optimizer must have two parameter groups",
    )
    by_name = {group.get("name"): group for group in groups}
    _require(
        set(by_name) == {"controlled", "uncontrolled"},
        f"{branch}: optimizer group names differ",
    )
    _require(
        len(by_name["controlled"].get("params", [])) == controlled_count,
        f"{branch}: optimizer controlled group size differs",
    )
    _require(
        len(by_name["uncontrolled"].get("params", [])) == uncontrolled_count,
        f"{branch}: optimizer uncontrolled group size differs",
    )
    parameter_ids = [
        parameter_id
        for group in groups
        for parameter_id in group.get("params", [])
    ]
    _require(
        len(parameter_ids) == len(set(parameter_ids)),
        f"{branch}: optimizer parameter IDs overlap",
    )
    _require(
        set(parameter_ids) == set(state),
        f"{branch}: optimizer state/group parameter IDs differ",
    )
    _all_tensors_finite(optimizer_state, f"{branch}.optimizer")


def validate_final_checkpoint(
    path: Path,
    cfg: dict,
    branch: str,
    model: VisionTransformer,
    controlled: dict,
    uncontrolled: dict,
    reference_norms: Mapping[str, float],
    final_metric_ema: float,
    validity_threshold: float,
    train_size: int,
) -> tuple[dict, dict]:
    _require(path.is_file() and path.stat().st_size > 0, f"{branch}: final checkpoint missing")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, dict), f"{branch}: final checkpoint is not a mapping")
    _require(
        checkpoint.get("global_state_step") == EXPECTED_TOTAL_STEPS,
        f"{branch}: final checkpoint step differs",
    )
    _require(
        checkpoint.get("config") == cfg,
        f"{branch}: checkpoint config differs from resolved config",
    )
    names = checkpoint.get("controlled_names")
    _require(
        isinstance(names, list)
        and len(names) == EXPECTED_CONTROLLED_TENSORS
        and len(set(names)) == EXPECTED_CONTROLLED_TENSORS
        and set(names) == set(controlled),
        f"{branch}: final controlled scope differs",
    )
    stored_references = checkpoint.get("reference_norms")
    _require(isinstance(stored_references, dict), f"{branch}: reference norms missing")
    _require(
        set(stored_references) == set(reference_norms),
        f"{branch}: reference norm scope differs",
    )
    for name, expected in reference_norms.items():
        actual = _finite_float(
            stored_references[name],
            f"{branch} final reference {name}",
            nonnegative=True,
        )
        _close(
            actual,
            float(expected),
            f"{branch} final reference {name}",
            rel_tol=2e-7,
            abs_tol=1e-9,
        )
    checkpoint_ema = _finite_float(
        checkpoint.get("train_loss_ema"),
        f"{branch}: final checkpoint EMA",
        nonnegative=True,
    )
    _close(
        checkpoint_ema,
        final_metric_ema,
        f"{branch}: checkpoint/metrics final EMA",
    )
    sampler = _validate_sampler_state(
        checkpoint,
        EXPECTED_TOTAL_STEPS,
        cfg,
        train_size,
        branch,
    )
    _validate_optimizer_layout(
        checkpoint.get("optimizer"),
        branch,
        len(controlled),
        len(uncontrolled),
    )
    model_state = checkpoint.get("model")
    _require(isinstance(model_state, dict), f"{branch}: model state missing")
    _all_tensors_finite(model_state, f"{branch}.model")
    model.load_state_dict(model_state, strict=True)
    branch_controlled, branch_uncontrolled = param_groups.classify_parameters(
        model, POLICY
    )
    _require(
        set(branch_controlled) == set(controlled)
        and set(branch_uncontrolled) == set(uncontrolled),
        f"{branch}: reconstructed policy scope differs",
    )
    units = control_units(branch_controlled, split_fused_qkv=True)
    final_ratio = _schedule_ratio(
        branch,
        EXPECTED_TOTAL_STEPS,
        cfg["optimizer"],
        cfg["control"],
    )
    final_norm_errors = {}
    for name, unit in units.items():
        expected = final_ratio * float(reference_norms[name])
        actual = frobenius_norm(unit)
        relative = abs(actual - expected) / max(abs(expected), 1e-12)
        _require(
            math.isfinite(relative) and relative <= validity_threshold,
            f"{branch}: final checkpoint norm error for {name} is {relative}",
        )
        final_norm_errors[name] = relative
    rng = _extract_rng_state(checkpoint)
    report = {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "global_state_step": checkpoint["global_state_step"],
        "config_exact": True,
        "controlled_scope_exact": True,
        "model_and_optimizer_finite": True,
        "optimizer_state_entries": len(checkpoint["optimizer"]["state"]),
        "sampler_state": sampler,
        "train_loss_ema": checkpoint_ema,
        "final_schedule_ratio": final_ratio,
        "max_recomputed_final_norm_relative_error": max(
            final_norm_errors.values()
        ),
    }
    del checkpoint
    gc.collect()
    return report, rng


def _summary_recomputed(
    branch_rows: Mapping[str, list[dict[str, float]]],
) -> dict[str, dict[str, float]]:
    constant = branch_rows["constant"]
    result = {}
    for branch in BRANCHES:
        rows = branch_rows[branch]
        _require(
            len(rows) == len(constant),
            f"{branch}: row count differs from constant",
        )
        _require(
            [row["global_state_step"] for row in rows]
            == [row["global_state_step"] for row in constant],
            f"{branch}: EMA alignment step vector differs from constant",
        )
        raw_residual = [
            abs(row["train_loss_raw"] - baseline["train_loss_raw"])
            for row, baseline in zip(rows, constant)
        ]
        ema_residual = [
            abs(row["train_loss_ema"] - baseline["train_loss_ema"])
            for row, baseline in zip(rows, constant)
        ]
        denominator = sum(abs(row["train_loss_ema"]) for row in constant) + 1e-12
        result[branch] = {
            "loss_raw_mae_vs_constant": statistics.mean(raw_residual),
            "loss_ema_mae_vs_constant": statistics.mean(ema_residual),
            "loss_ema_max_abs_vs_constant": max(ema_residual),
            "nce": sum(ema_residual) / denominator,
            "mean_elr_relative_error": statistics.mean(
                row["elr_relative_error_mean"] for row in rows
            ),
            "max_elr_relative_error": max(
                row["elr_relative_error_max"] for row in rows
            ),
            "mean_target_norm_relative_error": statistics.mean(
                row["target_norm_relative_error_mean"] for row in rows
            ),
            "max_target_norm_relative_error": max(
                row["target_norm_relative_error_max"] for row in rows
            ),
            "mean_angular_step": statistics.mean(
                row["angular_step_mean"]
                for row in rows
                if math.isfinite(row["angular_step_mean"])
            ),
            "final_train_loss_ema": rows[-1]["train_loss_ema"],
        }
    return result


def validate_collapse_summary(
    path: Path,
    recomputed: Mapping[str, Mapping[str, float]],
) -> dict:
    _require(path.is_file(), "analysis/collapse_metrics.csv missing")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require(
            tuple(reader.fieldnames or ()) == SUMMARY_FIELDS,
            "collapse_metrics.csv header differs",
        )
        rows = list(reader)
    _require(len(rows) == len(BRANCHES), "collapse summary must have four rows")
    by_branch = {}
    for row_number, row in enumerate(rows, start=2):
        branch = row["branch"]
        _require(branch in BRANCHES, f"summary row {row_number}: unexpected branch")
        _require(branch not in by_branch, f"summary duplicate branch: {branch}")
        by_branch[branch] = row
    _require(set(by_branch) == set(BRANCHES), "summary branch set differs")
    compare_fields = (
        "loss_raw_mae_vs_constant",
        "loss_ema_mae_vs_constant",
        "loss_ema_max_abs_vs_constant",
        "nce",
        "mean_elr_relative_error",
        "max_elr_relative_error",
        "mean_target_norm_relative_error",
        "max_target_norm_relative_error",
        "mean_angular_step",
        "final_train_loss_ema",
    )
    for branch in BRANCHES:
        for field in compare_fields:
            actual = _finite_float(
                by_branch[branch][field],
                f"summary {branch} {field}",
                nonnegative=field
                not in {"final_train_loss_ema"},
            )
            _close(
                actual,
                float(recomputed[branch][field]),
                f"summary {branch} {field}",
                rel_tol=2e-9,
                abs_tol=2e-11,
            )
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "branches_unique_and_complete": True,
        "values_recomputed_from_raw_metrics": True,
    }


def validate_manifest(output: Path, required_paths: list[Path]) -> dict:
    manifest = _load_json_mapping(output / "run_manifest.json")
    stages = manifest.get("stages")
    _require(isinstance(stages, dict), "run manifest stages missing")
    for stage in ("tests", *BRANCHES, "analysis"):
        _require(
            stages.get(stage) == "complete",
            f"run manifest stage {stage!r} is not complete",
        )
    files = manifest.get("files")
    _require(isinstance(files, dict), "run manifest file hashes missing")
    normalized = {
        str(key).replace("\\", "/"): value for key, value in files.items()
    }
    verified = {}
    for path in required_paths:
        relative = path.relative_to(output).as_posix()
        _require(relative in normalized, f"manifest hash missing for {relative}")
        actual = sha256(path)
        _require(
            normalized[relative] == actual,
            f"manifest hash mismatch for {relative}",
        )
        verified[relative] = actual
    return {
        "stages_complete": True,
        "required_file_hashes_verified": verified,
    }


def validate_existing_gate(
    output: Path,
    recomputed: Mapping[str, Mapping[str, float]],
    threshold: float,
    validity_threshold: float,
    independent_passed: bool,
) -> dict:
    gate = _load_json_mapping(output / "analysis" / "iteration_gate.json")
    _close(float(gate.get("threshold")), threshold, "existing gate threshold")
    _close(
        float(gate.get("validity_threshold")),
        validity_threshold,
        "existing gate validity threshold",
    )
    _require(
        gate.get("total_steps") == EXPECTED_TOTAL_STEPS,
        "existing gate total_steps differs",
    )
    checkpoint_flags = gate.get("final_checkpoints")
    _require(
        isinstance(checkpoint_flags, dict)
        and set(checkpoint_flags) == set(BRANCHES)
        and all(checkpoint_flags.values()),
        "existing gate did not accept all four final checkpoints",
    )
    gate_mae = gate.get("branch_ema_mae")
    _require(isinstance(gate_mae, dict), "existing gate branch MAE missing")
    for branch in CONTROLLED_BRANCHES:
        _close(
            float(gate_mae[branch]),
            float(recomputed[branch]["loss_ema_mae_vs_constant"]),
            f"existing gate MAE {branch}",
            rel_tol=2e-9,
            abs_tol=2e-11,
        )
    _require(
        gate.get("passed") is independent_passed,
        "existing gate pass/fail differs from independent recomputation",
    )
    return {
        "path": str((output / "analysis" / "iteration_gate.json").resolve()),
        "sha256": sha256(output / "analysis" / "iteration_gate.json"),
        "agrees_with_independent_recomputation": True,
        "reported_passed": gate["passed"],
    }


def code_fingerprints() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "run_pipeline.py",
        ROOT / "run_pipeline_residual_stream_symmetry.py",
        ROOT / "src" / "train_branch.py",
        ROOT / "src" / "analyze.py",
        ROOT / "src" / "norm_control.py",
        ROOT / "src" / "policy_overlay_residual_stream_symmetry.py",
        ROOT / "scripts" / "iteration_gate.py",
    )
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
        for path in paths
    }


def audit_result(
    output: Path,
    expected_config: Path,
    *,
    threshold: float = 0.03,
    validity_threshold: float = 1e-5,
    train_size: int = EXPECTED_TRAIN_SIZE,
) -> dict:
    output = output.resolve()
    expected_config = expected_config.resolve()
    threshold = _finite_float(threshold, "collapse threshold", nonnegative=True)
    validity_threshold = _finite_float(
        validity_threshold, "validity threshold", nonnegative=True
    )
    _require(train_size == EXPECTED_TRAIN_SIZE, "unexpected Tiny-ImageNet train size")

    cfg = validate_config(output, expected_config)
    model, controlled, uncontrolled, unit_numels = validate_policy_scope(cfg)
    references, prefix_ema, prefix_report = validate_prefix(
        output, cfg, model, controlled, train_size
    )

    branch_rows = {}
    metrics_reports = {}
    required_manifest_paths = [
        output / "resolved_config.yaml",
        output / "residual_stream_preflight.json",
        output / "prefix" / "checkpoint_step_002500.pt",
    ]
    for branch in BRANCHES:
        metrics_path = output / "branches" / branch / "metrics.csv"
        rows, report = audit_metrics_file(
            metrics_path,
            cfg,
            branch,
            prefix_ema,
            references,
            unit_numels,
            validity_threshold,
        )
        branch_rows[branch] = rows
        metrics_reports[branch] = report
        required_manifest_paths.append(metrics_path)

    first_step_fields = ("train_loss_raw", "train_loss_ema", "train_top1")
    for field in first_step_fields:
        values = [branch_rows[branch][0][field] for branch in BRANCHES]
        _require(
            all(value == values[0] for value in values[1:]),
            f"four branches do not share an identical first-step {field}",
        )

    checkpoint_reports = {}
    rng_states = {}
    final_sampler_states = {}
    for branch in BRANCHES:
        checkpoint_path = (
            output
            / "branches"
            / branch
            / "checkpoint_step_020000.pt"
        )
        checkpoint_report, rng_state = validate_final_checkpoint(
            checkpoint_path,
            cfg,
            branch,
            model,
            controlled,
            uncontrolled,
            references,
            branch_rows[branch][-1]["train_loss_ema"],
            validity_threshold,
            train_size,
        )
        checkpoint_reports[branch] = checkpoint_report
        rng_states[branch] = rng_state
        final_sampler_states[branch] = checkpoint_report["sampler_state"]
        required_manifest_paths.append(checkpoint_path)
    baseline_rng = rng_states["constant"]
    for branch in CONTROLLED_BRANCHES:
        _require(
            _recursive_equal(rng_states[branch], baseline_rng),
            f"{branch}: final RNG state differs from constant",
        )
        _require(
            final_sampler_states[branch] == final_sampler_states["constant"],
            f"{branch}: final sampler state differs from constant",
        )

    recomputed = _summary_recomputed(branch_rows)
    collapse_summary_path = output / "analysis" / "collapse_metrics.csv"
    summary_report = validate_collapse_summary(
        collapse_summary_path, recomputed
    )
    required_manifest_paths.append(collapse_summary_path)
    manifest_report = validate_manifest(output, required_manifest_paths)

    collapse_passed = all(
        recomputed[branch]["loss_ema_mae_vs_constant"] <= threshold
        for branch in CONTROLLED_BRANCHES
    )
    validity_passed = all(
        recomputed[branch]["max_elr_relative_error"] <= validity_threshold
        and recomputed[branch]["max_target_norm_relative_error"]
        <= validity_threshold
        for branch in BRANCHES
    )
    independent_passed = collapse_passed and validity_passed
    existing_gate_report = validate_existing_gate(
        output,
        recomputed,
        threshold,
        validity_threshold,
        independent_passed,
    )
    result = {
        "schema_version": 1,
        "audit_kind": "read_only_independent_residual_stream_symmetry_v1",
        "output": str(output),
        "expected_config": str(expected_config),
        "expected_config_sha256": sha256(expected_config),
        "window": {
            "first_step": EXPECTED_CONTROL_START + 1,
            "last_step": EXPECTED_TOTAL_STEPS,
            "inclusive_rows_per_branch": (
                EXPECTED_TOTAL_STEPS - EXPECTED_CONTROL_START
            ),
            "alignment": "exact global_state_step equality; no intersection/dropna",
        },
        "policy_scope": {
            "policy": POLICY,
            "controlled_tensors": len(controlled),
            "uncontrolled_tensors": len(uncontrolled),
            "control_units": len(unit_numels),
            "controlled_names": sorted(controlled),
            "exact": True,
        },
        "prefix": prefix_report,
        "lineage": {
            "approved_prefix_sha256": EXPECTED_PREFIX_SHA256,
            "prefix_hash_and_preflight_verified": True,
            "all_branch_first_step_loss_ema_top1_identical": True,
            "all_branch_reference_norms_derived_from_prefix": True,
            "all_branch_final_sampler_states_identical_and_expected": True,
            "all_branch_final_rng_states_identical": True,
            "note": (
                "These are consequential lineage invariants. The checkpoint "
                "schema has no cryptographic parent-prefix field."
            ),
        },
        "raw_metrics": metrics_reports,
        "final_checkpoints": checkpoint_reports,
        "analysis_summary": summary_report,
        "manifest": manifest_report,
        "existing_gate": existing_gate_report,
        "ema": {
            "beta": float(cfg["logging"]["ema_beta"]),
            "initial_value": prefix_ema,
            "definition": "ema_t=beta*ema_(t-1)+(1-beta)*train_loss_raw_t",
            "recurrence_verified_for_every_branch_and_every_row": True,
        },
        "branch_ema_mae_full_window": {
            branch: recomputed[branch]["loss_ema_mae_vs_constant"]
            for branch in CONTROLLED_BRANCHES
        },
        "branch_max_elr_relative_error": {
            branch: recomputed[branch]["max_elr_relative_error"]
            for branch in BRANCHES
        },
        "branch_max_target_norm_relative_error": {
            branch: recomputed[branch]["max_target_norm_relative_error"]
            for branch in BRANCHES
        },
        "threshold": threshold,
        "validity_threshold": validity_threshold,
        "collapse_passed": collapse_passed,
        "validity_passed": validity_passed,
        "passed": independent_passed,
        "failing_branches": [
            branch
            for branch in CONTROLLED_BRANCHES
            if recomputed[branch]["loss_ema_mae_vs_constant"] > threshold
        ],
        "code_sha256_at_audit_time": code_fingerprints(),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only independent final-result audit for "
            "residual_stream_symmetry_v1."
        )
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-config", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.03)
    parser.add_argument("--validity-threshold", type=float, default=1e-5)
    parser.add_argument("--train-size", type=int, default=EXPECTED_TRAIN_SIZE)
    args = parser.parse_args()
    try:
        result = audit_result(
            args.output,
            args.expected_config,
            threshold=args.threshold,
            validity_threshold=args.validity_threshold,
            train_size=args.train_size,
        )
    except Exception as exc:
        print(json.dumps({
            "audit_kind": (
                "read_only_independent_residual_stream_symmetry_v1"
            ),
            "status": "invalid",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, indent=2))
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from scripts import audit_residual_stream_symmetry_v1_result as base


FLOAT32_UNIT_ROUNDOFF = torch.finfo(torch.float32).eps / 2.0
FIX_DESCRIPTION = (
    "Device-independent reference audit: compare serialized float32 "
    "reference norms with the checkpoint tensor's CPU-float64 norm, while "
    "requiring exact source-reference inheritance and an identical runtime "
    "reference map in all four branches."
)


def _norm64(tensor: torch.Tensor) -> float:
    """Euclidean norm of stored float32 values with float64 accumulation."""
    return float(
        torch.linalg.vector_norm(
            tensor.detach().to(device="cpu", dtype=torch.float64)
        ).item()
    )


def _float32_norm_roundoff_bound(numel: int) -> float:
    """Conservative relative error for a hierarchical float32 2-norm.

    For nonnegative squared terms the summation condition number is one.
    A balanced reduction has ceil(log2(n)) addition levels.  The operation
    budget below allocates two such levels plus eight operations for
    squaring, multi-stage block reduction, square root, scalar conversion,
    and projection scaling.  gamma_k = k*u/(1-k*u) is the standard
    floating-point forward-error bound.
    """
    if not isinstance(numel, int) or numel <= 0:
        raise base.AuditError(f"invalid tensor size for norm bound: {numel!r}")
    depth = math.ceil(math.log2(max(2, numel)))
    operation_budget = 2 * depth + 8
    product = operation_budget * FLOAT32_UNIT_ROUNDOFF
    if product >= 1.0:
        raise base.AuditError("float32 norm error bound is not finite")
    return product / (1.0 - product)


def _reference_error(
    stored: float,
    mathematical: float,
    numel: int,
    label: str,
) -> dict:
    stored = base._finite_float(stored, f"{label} stored", nonnegative=True)
    mathematical = base._finite_float(
        mathematical, f"{label} float64 norm", nonnegative=True
    )
    base._require(mathematical > 0.0, f"{label}: zero float64 norm")
    relative = abs(stored - mathematical) / mathematical
    bound = _float32_norm_roundoff_bound(numel)
    base._require(
        relative <= bound,
        f"{label}: stored float32 reference differs from the float64 norm; "
        f"relative_error={relative:.17g}, bound={bound:.17g}, "
        f"stored={stored:.17g}, float64={mathematical:.17g}",
    )
    return {
        "stored": stored,
        "float64_norm": mathematical,
        "relative_error": relative,
        "float32_roundoff_bound": bound,
        "within_bound": True,
    }


def _runtime_reference_maps(
    output: Path,
    cfg: dict,
    mathematical_references: Mapping[str, float],
    source_references: Mapping[str, Any],
    unit_numels: Mapping[str, int],
) -> tuple[dict[str, float], dict]:
    maps: dict[str, dict[str, float]] = {}
    checkpoint_hashes = {}
    for branch in base.BRANCHES:
        path = (
            output
            / "branches"
            / branch
            / f"checkpoint_step_{base.EXPECTED_TOTAL_STEPS:06d}.pt"
        )
        base._require(
            path.is_file() and path.stat().st_size > 0,
            f"{branch}: final checkpoint missing while auditing references",
        )
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        base._require(
            isinstance(checkpoint, dict)
            and checkpoint.get("global_state_step")
            == base.EXPECTED_TOTAL_STEPS,
            f"{branch}: invalid final checkpoint for runtime references",
        )
        base._require(
            checkpoint.get("config") == cfg,
            f"{branch}: config differs while auditing runtime references",
        )
        raw = checkpoint.get("reference_norms")
        base._require(
            isinstance(raw, dict)
            and set(raw) == set(mathematical_references),
            f"{branch}: runtime reference scope differs",
        )
        maps[branch] = {
            name: base._finite_float(
                raw[name],
                f"{branch} runtime reference {name}",
                nonnegative=True,
            )
            for name in mathematical_references
        }
        checkpoint_hashes[branch] = base.sha256(path)
        del checkpoint
        gc.collect()

    canonical = maps["constant"]
    for branch in base.CONTROLLED_BRANCHES:
        base._require(
            maps[branch] == canonical,
            f"{branch}: runtime reference map is not exactly identical "
            "to constant",
        )

    inherited = {}
    added = {}
    for name, runtime_value in canonical.items():
        if name in source_references:
            source_value = float(source_references[name])
            base._require(
                runtime_value == source_value,
                f"{name}: runtime reference did not exactly inherit the "
                "serialized prefix reference",
            )
            inherited[name] = {
                "runtime": runtime_value,
                "prefix_stored": source_value,
                "bitwise_float_value_equal": True,
            }
        else:
            added[name] = _reference_error(
                runtime_value,
                mathematical_references[name],
                unit_numels[name],
                f"runtime-added {name}",
            )
    return canonical, {
        "all_four_runtime_maps_exactly_identical": True,
        "inherited_reference_count": len(inherited),
        "added_reference_count": len(added),
        "inherited_references": inherited,
        "added_references": added,
        "checkpoint_sha256": checkpoint_hashes,
    }


def validate_prefix_fixed(
    output: Path,
    cfg: dict,
    model: base.VisionTransformer,
    controlled: dict,
    train_size: int,
) -> tuple[dict[str, float], float, dict]:
    path = output / "prefix" / "checkpoint_step_002500.pt"
    base._require(
        path.is_file() and path.stat().st_size > 0,
        "shared prefix is missing",
    )
    actual_sha = base.sha256(path)
    base._require(
        actual_sha == base.EXPECTED_PREFIX_SHA256,
        f"shared prefix SHA mismatch: {actual_sha}",
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    base._require(
        isinstance(checkpoint, dict),
        "shared prefix is not a checkpoint mapping",
    )
    base._require(
        checkpoint.get("global_state_step") == base.EXPECTED_CONTROL_START,
        "shared prefix checkpoint step is not 2500",
    )
    source_cfg = checkpoint.get("config")
    base._require(isinstance(source_cfg, dict), "shared prefix config missing")
    base._validate_prefix_scientific_config(source_cfg, cfg)
    model_state = checkpoint.get("model")
    base._require(isinstance(model_state, dict), "shared prefix model state missing")
    model.load_state_dict(model_state, strict=True)
    base._all_tensors_finite(model_state, "prefix.model")
    base._all_tensors_finite(checkpoint.get("optimizer"), "prefix.optimizer")
    prefix_ema = base._finite_float(
        checkpoint.get("train_loss_ema"),
        "prefix train_loss_ema",
        nonnegative=True,
    )
    sampler = base._validate_sampler_state(
        checkpoint,
        base.EXPECTED_CONTROL_START,
        cfg,
        train_size,
        "prefix",
    )

    source_names = checkpoint.get("controlled_names")
    source_references = checkpoint.get("reference_norms")
    base._require(isinstance(source_names, list), "prefix controlled_names missing")
    base._require(
        isinstance(source_references, dict),
        "prefix reference_norms missing",
    )
    base._require(
        len(source_names) == len(set(source_names)),
        "prefix controlled_names contains duplicates",
    )
    base._require(
        set(source_names) == set(source_references),
        "prefix controlled_names/reference_norm keys differ",
    )

    source_controlled, _ = base.param_groups.classify_parameters(
        model, "hidden_matrices"
    )
    source_units = base.control_units(
        source_controlled,
        split_fused_qkv=bool(
            source_cfg["control"].get("split_fused_qkv", False)
        ),
    )
    base._require(
        set(source_units) == set(source_references),
        "serialized source-reference scope differs from hidden_matrices",
    )
    source_reference_audit = {
        name: _reference_error(
            float(source_references[name]),
            _norm64(unit),
            unit.numel(),
            f"prefix-source {name}",
        )
        for name, unit in source_units.items()
    }

    target_units = base.control_units(controlled, split_fused_qkv=True)
    mathematical_references = {
        name: _norm64(unit) for name, unit in target_units.items()
    }
    unit_numels = {name: unit.numel() for name, unit in target_units.items()}
    base._require(
        set(mathematical_references)
        == base.expected_controlled_names(int(cfg["model"]["depth"])),
        "prefix-derived residual reference scope differs",
    )
    for name, value in mathematical_references.items():
        base._require(
            math.isfinite(value)
            and value > float(cfg["control"]["projection_eps"]),
            f"invalid float64 prefix norm for {name}: {value!r}",
        )

    runtime_references, runtime_report = _runtime_reference_maps(
        output,
        cfg,
        mathematical_references,
        source_references,
        unit_numels,
    )

    preflight = base._load_json_mapping(
        output / "residual_stream_preflight.json"
    )
    base._require(
        preflight.get("checkpoint_sha256") == base.EXPECTED_PREFIX_SHA256
        and preflight.get("expected_checkpoint_sha256")
        == base.EXPECTED_PREFIX_SHA256,
        "preflight does not bind to the approved shared prefix",
    )
    base._require(
        preflight.get("source_policy") == "hidden_matrices"
        and preflight.get("target_policy") == base.POLICY,
        "preflight policy transition differs",
    )
    for key, expected in (
        ("controlled_tensors", base.EXPECTED_CONTROLLED_TENSORS),
        ("control_units", base.EXPECTED_CONTROLLED_TENSORS),
        ("uncontrolled_tensors", base.EXPECTED_UNCONTROLLED_TENSORS),
        ("optimizer_state_entries", base.EXPECTED_OPTIMIZER_STATES),
    ):
        base._require(preflight.get(key) == expected, f"preflight {key} differs")
    base._require(
        preflight.get("optimizer_state_values_bitwise_equal") is True,
        "preflight did not prove lossless optimizer-state migration",
    )
    expected_added = set(target_units) - (
        set(target_units) & set(source_references)
    )
    base._require(
        set(preflight.get("added_from_hidden", [])) == expected_added,
        "preflight added-from-hidden scope differs",
    )
    base._require(
        len(preflight.get("added_from_hidden", [])) == 28,
        "preflight must report 28 added tensors",
    )
    base._require(
        len(preflight.get("removed_from_hidden", [])) == 24,
        "preflight must report 24 removed tensors",
    )

    # The preflight itself runs on CPU.  It retains serialized GPU32 values
    # for inherited names and computes only newly added names with CPU32.
    cpu_preflight_references = {
        name: (
            float(source_references[name])
            if name in source_references
            else base.frobenius_norm(unit)
        )
        for name, unit in target_units.items()
    }
    base._close(
        float(preflight["reference_norm_min"]),
        min(cpu_preflight_references.values()),
        "preflight mixed-device reference_norm_min",
        rel_tol=2e-7,
        abs_tol=1e-9,
    )
    base._close(
        float(preflight["reference_norm_max"]),
        max(cpu_preflight_references.values()),
        "preflight mixed-device reference_norm_max",
        rel_tol=2e-7,
        abs_tol=1e-9,
    )

    prefix_report = {
        "path": str(path.resolve()),
        "sha256": actual_sha,
        "global_state_step": base.EXPECTED_CONTROL_START,
        "train_loss_ema": prefix_ema,
        "sampler_state": sampler,
        "source_policy": source_cfg["control"]["policy"],
        "preflight_verified": True,
        "reference_norm_basis": {
            "mathematical_baseline": (
                "CPU float64 norm of serialized checkpoint tensor values"
            ),
            "runtime_values": (
                "GPU float32 norm values serialized in branch checkpoints"
            ),
            "roundoff_model": (
                "gamma_k with k=2*ceil(log2(numel))+8 and u=2^-24"
            ),
        },
        "source_reference_audit": source_reference_audit,
        "runtime_reference_audit": runtime_report,
        "max_source_reference_vs_float64_relative_error": max(
            row["relative_error"] for row in source_reference_audit.values()
        ),
        "max_runtime_added_reference_vs_float64_relative_error": max(
            (
                row["relative_error"]
                for row in runtime_report["added_references"].values()
            ),
            default=0.0,
        ),
    }
    del checkpoint
    gc.collect()
    return runtime_references, prefix_ema, prefix_report


_ORIGINAL_VALIDATE_FINAL_CHECKPOINT = base.validate_final_checkpoint
_ORIGINAL_CODE_FINGERPRINTS = base.code_fingerprints


def validate_final_checkpoint_fixed(
    path: Path,
    cfg: dict,
    branch: str,
    model: base.VisionTransformer,
    controlled: dict,
    uncontrolled: dict,
    reference_norms: Mapping[str, float],
    final_metric_ema: float,
    validity_threshold: float,
    train_size: int,
) -> tuple[dict, dict]:
    maximum_roundoff = max(
        _float32_norm_roundoff_bound(parameter.numel())
        for parameter in controlled.values()
    )
    report, rng = _ORIGINAL_VALIDATE_FINAL_CHECKPOINT(
        path,
        cfg,
        branch,
        model,
        controlled,
        uncontrolled,
        reference_norms,
        final_metric_ema,
        validity_threshold + maximum_roundoff,
        train_size,
    )
    cpu32_diagnostic = report[
        "max_recomputed_final_norm_relative_error"
    ]
    branch_controlled, _ = base.param_groups.classify_parameters(
        model, base.POLICY
    )
    units = base.control_units(branch_controlled, split_fused_qkv=True)
    final_ratio = base._schedule_ratio(
        branch,
        base.EXPECTED_TOTAL_STEPS,
        cfg["optimizer"],
        cfg["control"],
    )
    details = {}
    for name, unit in units.items():
        target = final_ratio * float(reference_norms[name])
        actual = _norm64(unit)
        relative = abs(actual - target) / max(abs(target), 1e-12)
        roundoff = _float32_norm_roundoff_bound(unit.numel())
        allowed = validity_threshold + roundoff
        base._require(
            math.isfinite(relative) and relative <= allowed,
            f"{branch}: final float64 norm error for {name} is {relative}; "
            f"training validity threshold={validity_threshold}, "
            f"float32 roundoff bound={roundoff}",
        )
        details[name] = {
            "target_from_runtime_reference": target,
            "checkpoint_float64_norm": actual,
            "relative_error": relative,
            "training_validity_threshold": validity_threshold,
            "float32_projection_roundoff_bound": roundoff,
            "allowed_total": allowed,
        }
    report[
        "max_recomputed_final_norm_relative_error_cpu32_diagnostic"
    ] = cpu32_diagnostic
    report["max_recomputed_final_norm_relative_error"] = max(
        row["relative_error"] for row in details.values()
    )
    report["final_norm_validation_basis"] = (
        "checkpoint CPU-float64 mathematical norm versus target formed from "
        "the common runtime reference map; allowance is the logged training "
        "validity threshold plus the per-tensor float32 projection bound"
    )
    report["final_norm_float64_details"] = details
    return report, rng


def _fixed_code_fingerprints() -> dict[str, str]:
    values = _ORIGINAL_CODE_FINGERPRINTS()
    path = Path(__file__).resolve()
    values[str(path.relative_to(base.ROOT)).replace("\\", "/")] = base.sha256(
        path
    )
    return values


def audit_result_fixed(
    output: Path,
    expected_config: Path,
    *,
    threshold: float = 0.03,
    validity_threshold: float = 1e-5,
    train_size: int = base.EXPECTED_TRAIN_SIZE,
) -> dict:
    original_prefix = base.validate_prefix
    original_final = base.validate_final_checkpoint
    original_fingerprints = base.code_fingerprints
    try:
        base.validate_prefix = validate_prefix_fixed
        base.validate_final_checkpoint = validate_final_checkpoint_fixed
        base.code_fingerprints = _fixed_code_fingerprints
        result = base.audit_result(
            output,
            expected_config,
            threshold=threshold,
            validity_threshold=validity_threshold,
            train_size=train_size,
        )
    finally:
        base.validate_prefix = original_prefix
        base.validate_final_checkpoint = original_final
        base.code_fingerprints = original_fingerprints
    result["schema_version"] = 2
    result["audit_kind"] = (
        "read_only_independent_residual_stream_symmetry_v1_float64_fixed"
    )
    result["float32_norm_fix"] = {
        "description": FIX_DESCRIPTION,
        "unit_roundoff": FLOAT32_UNIT_ROUNDOFF,
        "blind_tolerance_widening": False,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only residual-v1 audit with device-independent float64 "
            "reference-norm validation."
        )
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-config", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.03)
    parser.add_argument("--validity-threshold", type=float, default=1e-5)
    parser.add_argument(
        "--train-size", type=int, default=base.EXPECTED_TRAIN_SIZE
    )
    args = parser.parse_args()
    try:
        result = audit_result_fixed(
            args.output,
            args.expected_config,
            threshold=args.threshold,
            validity_threshold=args.validity_threshold,
            train_size=args.train_size,
        )
    except Exception as exc:
        print(json.dumps({
            "audit_kind": (
                "read_only_independent_residual_stream_symmetry_v1_"
                "float64_fixed"
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

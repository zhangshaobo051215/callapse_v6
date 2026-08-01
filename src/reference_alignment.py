from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from .utils import append_csv, truncate_csv_after_step


METRICS_FILENAME = "reference_alignment_metrics.csv"


def alignment_settings(cfg: dict) -> dict | None:
    raw = cfg.get("control", {}).get("constant_reference_alignment")
    if not raw or not bool(raw.get("enabled", False)):
        return None
    required = {"tensor", "reference_branch"}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(
            f"constant_reference_alignment is missing keys: {missing}")
    result = dict(raw)
    result.setdefault("mode", "frobenius_norm")
    result.setdefault("tolerance", 1.0e-5)
    if result["mode"] != "frobenius_norm":
        raise ValueError(
            "V6 only supports scalar Frobenius-norm reference alignment")
    tolerance = float(result["tolerance"])
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("reference-alignment tolerance must be positive")
    result["tolerance"] = tolerance
    return result


def _norm(parameter: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(parameter.detach().float()).item())


def _optimizer_group_for(parameter: torch.Tensor, optimizer) -> dict:
    matches = [
        group for group in optimizer.param_groups
        if any(candidate is parameter for candidate in group["params"])
    ]
    if len(matches) != 1:
        raise ValueError(
            "reference-aligned tensor must occur in exactly one optimizer group")
    return matches[0]


def _load_reference_trajectory(
    path: Path,
    *,
    tensor_name: str,
    reference_branch: str,
    first_step: int,
    final_step: int,
) -> dict[int, float]:
    if not path.is_file():
        raise FileNotFoundError(
            f"constant reference trajectory is missing: {path}")
    targets: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "global_state_step",
            "branch",
            "tensor",
            "post_alignment_frobenius_norm",
        }
        fields = set(reader.fieldnames or ())
        if not required <= fields:
            raise ValueError(
                f"{path} is missing fields {sorted(required - fields)}")
        for row in reader:
            step = int(row["global_state_step"])
            if row["branch"] != reference_branch:
                raise ValueError(
                    f"reference trajectory contains branch {row['branch']!r}")
            if row["tensor"] != tensor_name:
                raise ValueError(
                    f"reference trajectory contains tensor {row['tensor']!r}")
            if step in targets:
                raise ValueError(
                    f"duplicate reference-alignment target at step {step}")
            value = float(row["post_alignment_frobenius_norm"])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"invalid reference norm at step {step}: {value!r}")
            targets[step] = value
    expected = set(range(first_step, final_step + 1))
    observed = set(targets)
    if observed != expected:
        raise ValueError(
            "constant reference trajectory is incomplete; "
            f"missing={sorted(expected - observed)[:10]}, "
            f"extra={sorted(observed - expected)[:10]}")
    return targets


@dataclass
class ConstantReferenceNormAligner:
    branch: str
    reference_branch: str
    tensor_name: str
    parameter: torch.Tensor
    optimizer_group: dict
    metrics_path: Path
    targets: dict[int, float] | None
    tolerance: float
    eps: float

    @property
    def is_reference_branch(self) -> bool:
        return self.branch == self.reference_branch

    def pre_update_norm(self) -> float:
        value = _norm(self.parameter)
        if not math.isfinite(value) or value <= self.eps:
            raise RuntimeError(
                f"invalid pre-update norm for {self.tensor_name}: {value!r}")
        return value

    def apply(
        self,
        *,
        update_start_step: int,
        base_lr: float,
        schedule_ratio: float,
        pre_update_norm: float,
    ) -> dict:
        state_step = int(update_start_step) + 1
        actual_lr = float(self.optimizer_group["lr"])
        lr_error = abs(actual_lr - base_lr) / max(abs(base_lr), 1.0e-30)
        if lr_error > self.tolerance:
            raise RuntimeError(
                f"{self.tensor_name} LR differs from constant/base LR at "
                f"step {state_step}: actual={actual_lr}, base={base_lr}")

        post_optimizer = _norm(self.parameter)
        if not math.isfinite(post_optimizer) or post_optimizer <= self.eps:
            raise RuntimeError(
                f"invalid post-optimizer norm for {self.tensor_name}: "
                f"{post_optimizer!r}")

        if self.is_reference_branch:
            target = post_optimizer
            was_projected = False
        else:
            if self.targets is None or state_step not in self.targets:
                raise RuntimeError(
                    f"missing constant target for state step {state_step}")
            target = float(self.targets[state_step])
            with torch.no_grad():
                self.parameter.mul_(target / post_optimizer)
            was_projected = True

        post_alignment = _norm(self.parameter)
        relative_error = (
            abs(post_alignment - target) / max(abs(target), self.eps)
        )
        if relative_error > self.tolerance:
            raise RuntimeError(
                f"reference alignment failed for {self.tensor_name} at "
                f"step {state_step}: error={relative_error}")

        rms_scale = math.sqrt(self.parameter.numel())
        row = {
            "update_start_state_step": int(update_start_step),
            "global_state_step": state_step,
            "branch": self.branch,
            "reference_branch": self.reference_branch,
            "tensor": self.tensor_name,
            "numel": int(self.parameter.numel()),
            "schedule_ratio": float(schedule_ratio),
            "base_lr": float(base_lr),
            "actual_lr": actual_lr,
            "pre_update_frobenius_norm": float(pre_update_norm),
            "post_optimizer_pre_alignment_frobenius_norm": post_optimizer,
            "target_reference_frobenius_norm": target,
            "post_alignment_frobenius_norm": post_alignment,
            "alignment_relative_error": relative_error,
            "lr_over_pre_frobenius_norm": actual_lr / pre_update_norm,
            "lr_over_pre_parameter_rms": (
                actual_lr / (pre_update_norm / rms_scale)
            ),
            "was_projected": was_projected,
        }
        append_csv(row, self.metrics_path)
        return row


def build_constant_reference_norm_aligner(
    cfg: dict,
    *,
    branch: str,
    model,
    optimizer,
    controlled: dict,
    output: Path,
    start_step: int,
) -> ConstantReferenceNormAligner | None:
    settings = alignment_settings(cfg)
    if settings is None:
        return None

    tensor_name = str(settings["tensor"])
    reference_branch = str(settings["reference_branch"])
    schedules = list(cfg["control"]["schedules"])
    if not schedules or schedules[0] != reference_branch:
        raise ValueError(
            "constant-reference alignment requires the reference branch to run first")
    if schedules.count(reference_branch) != 1:
        raise ValueError("reference branch must occur exactly once")

    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if tensor_name not in trainable:
        raise ValueError(f"reference-aligned tensor is not trainable: {tensor_name}")
    if tensor_name in controlled:
        raise ValueError(
            f"reference-aligned tensor must remain outside q(t) control: {tensor_name}")
    parameter = trainable[tensor_name]
    if parameter.ndim != 1:
        raise ValueError(
            f"V6 reference-aligned tensor must be one-dimensional: {tensor_name}")
    optimizer_group = _optimizer_group_for(parameter, optimizer)
    if optimizer_group.get("name") != "uncontrolled":
        raise ValueError(
            f"{tensor_name} must keep the ordinary/uncontrolled optimizer LR")

    output = Path(output)
    metrics_path = output / METRICS_FILENAME
    truncate_csv_after_step(
        metrics_path,
        int(start_step),
        dedupe_by=("branch", "tensor", "global_state_step"),
    )

    targets = None
    if branch != reference_branch:
        reference_path = output.parent / reference_branch / METRICS_FILENAME
        targets = _load_reference_trajectory(
            reference_path,
            tensor_name=tensor_name,
            reference_branch=reference_branch,
            first_step=int(cfg["optimizer"]["control_start_step"]) + 1,
            final_step=int(cfg["optimizer"]["total_steps"]),
        )

    return ConstantReferenceNormAligner(
        branch=branch,
        reference_branch=reference_branch,
        tensor_name=tensor_name,
        parameter=parameter,
        optimizer_group=optimizer_group,
        metrics_path=metrics_path,
        targets=targets,
        tolerance=float(settings["tolerance"]),
        eps=float(cfg["control"]["projection_eps"]),
    )

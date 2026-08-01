from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import torch
import yaml


BRANCHES = ("constant", "linear_up", "linear_down", "cyclic")
CONTROLLED_BRANCHES = BRANCHES[1:]


def _required_number(by_branch: pd.DataFrame, branch: str, column: str) -> float:
    if column not in by_branch.columns:
        raise ValueError(f"collapse metrics missing column: {column}")
    value = pd.to_numeric(
        pd.Series([by_branch.loc[branch, column]]), errors="coerce").iloc[0]
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"collapse metrics require a finite nonnegative {column} for "
            f"{branch}; got {value!r}")
    return value


def _total_steps(output: Path) -> int:
    config_path = output / "resolved_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"resolved config missing: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    try:
        total_steps = int(config["optimizer"]["total_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"resolved config has no valid optimizer.total_steps: "
            f"{config_path}") from exc
    if total_steps <= 0:
        raise ValueError(
            f"optimizer.total_steps must be positive, got {total_steps}")
    return total_steps


def _valid_final_checkpoint(
        path: Path, total_steps: int) -> tuple[bool, str | None]:
    if not path.is_file() or path.stat().st_size == 0:
        return False, "missing or empty"
    try:
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return False, f"unreadable: {type(exc).__name__}: {exc}"
    if not isinstance(checkpoint, dict):
        return False, f"expected mapping, got {type(checkpoint).__name__}"
    step = checkpoint.get("global_state_step")
    if step != total_steps:
        return False, f"global_state_step={step!r}, expected {total_steps}"
    return True, None


def evaluate_iteration(output: Path, threshold: float, validity_threshold: float) -> dict:
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    if not math.isfinite(validity_threshold) or validity_threshold < 0:
        raise ValueError(
            "validity_threshold must be finite and nonnegative")
    analysis = output / "analysis"
    metrics_path = analysis / "collapse_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    if "branch" not in metrics.columns:
        raise ValueError("collapse metrics missing column: branch")
    if metrics["branch"].duplicated().any():
        duplicates = sorted(
            metrics.loc[metrics["branch"].duplicated(), "branch"].unique())
        raise ValueError(
            f"collapse metrics contain duplicate branches: {duplicates}")
    by_branch = metrics.set_index("branch")
    missing = sorted(set(BRANCHES) - set(by_branch.index))
    if missing:
        raise ValueError(f"collapse metrics missing branches: {missing}")

    branch_mae = {
        branch: _required_number(
            by_branch, branch, "loss_ema_mae_vs_constant")
        for branch in CONTROLLED_BRANCHES
    }
    elr_errors = {
        branch: _required_number(
            by_branch, branch, "max_elr_relative_error")
        for branch in BRANCHES
    }
    norm_errors = {
        branch: _required_number(
            by_branch, branch, "max_target_norm_relative_error")
        for branch in BRANCHES
    }
    max_elr_error = max(elr_errors.values())
    max_norm_error = max(norm_errors.values())
    total_steps = _total_steps(output)
    checkpoint_paths = {
        branch: output / "branches" / branch
        / f"checkpoint_step_{total_steps:06d}.pt"
        for branch in BRANCHES
    }
    checkpoint_results = {
        branch: _valid_final_checkpoint(path, total_steps)
        for branch, path in checkpoint_paths.items()
    }
    checkpoints = {
        branch: valid
        for branch, (valid, _) in checkpoint_results.items()
    }
    checkpoint_errors = {
        branch: error
        for branch, (valid, error) in checkpoint_results.items()
        if not valid
    }
    validity_passed = (
        max_elr_error <= validity_threshold
        and max_norm_error <= validity_threshold
        and all(checkpoints.values())
    )
    collapse_passed = all(value <= threshold for value in branch_mae.values())
    return {
        "output": str(output),
        "threshold": threshold,
        "validity_threshold": validity_threshold,
        "total_steps": total_steps,
        "branch_ema_mae": branch_mae,
        "branch_max_elr_relative_error": elr_errors,
        "branch_max_target_norm_relative_error": norm_errors,
        "max_elr_relative_error": max_elr_error,
        "max_target_norm_relative_error": max_norm_error,
        "final_checkpoints": checkpoints,
        "checkpoint_errors": checkpoint_errors,
        "validity_passed": validity_passed,
        "collapse_passed": collapse_passed,
        "passed": validity_passed and collapse_passed,
        "failing_branches": [
            branch for branch, value in branch_mae.items() if value > threshold
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.03)
    parser.add_argument("--validity-threshold", type=float, default=1e-5)
    args = parser.parse_args()

    result = evaluate_iteration(
        args.output, args.threshold, args.validity_threshold)
    path = args.output / "analysis" / "iteration_gate.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

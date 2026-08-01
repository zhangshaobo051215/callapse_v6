from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from scripts.iteration_gate import evaluate_iteration


def _write_run(
        root: Path, maes, *, elr=2e-7, norm=3e-7, total_steps=20_000):
    analysis = root / "analysis"
    analysis.mkdir(parents=True)
    (root / "resolved_config.yaml").write_text(
        yaml.safe_dump({"optimizer": {"total_steps": total_steps}}),
        encoding="utf-8")
    rows = []
    for branch in ("constant", "linear_up", "linear_down", "cyclic"):
        rows.append({
            "branch": branch,
            "loss_ema_mae_vs_constant": 0.0 if branch == "constant" else maes[branch],
            "max_elr_relative_error": (
                elr.get(branch, 2e-7) if isinstance(elr, dict) else elr),
            "max_target_norm_relative_error": (
                norm.get(branch, 3e-7) if isinstance(norm, dict) else norm),
        })
        checkpoint = (
            root / "branches" / branch
            / f"checkpoint_step_{total_steps:06d}.pt")
        checkpoint.parent.mkdir(parents=True)
        torch.save({"global_state_step": total_steps}, checkpoint)
    pd.DataFrame(rows).to_csv(analysis / "collapse_metrics.csv", index=False)


def test_iteration_gate_requires_every_branch_below_threshold(tmp_path):
    _write_run(tmp_path, {
        "linear_up": 0.02,
        "linear_down": 0.03,
        "cyclic": 0.029,
    })
    result = evaluate_iteration(tmp_path, threshold=0.03, validity_threshold=1e-5)
    assert result["validity_passed"]
    assert result["collapse_passed"]
    assert result["passed"]


def test_iteration_gate_reports_failed_branch(tmp_path):
    _write_run(tmp_path, {
        "linear_up": 0.02,
        "linear_down": 0.031,
        "cyclic": 0.029,
    })
    result = evaluate_iteration(tmp_path, threshold=0.03, validity_threshold=1e-5)
    assert result["validity_passed"]
    assert not result["collapse_passed"]
    assert result["failing_branches"] == ["linear_down"]


@pytest.mark.parametrize(
    "bad_value", [float("nan"), float("inf"), -1.0])
def test_iteration_gate_rejects_nonfinite_or_negative_validity(
        tmp_path, bad_value):
    _write_run(
        tmp_path,
        {"linear_up": 0.02, "linear_down": 0.02, "cyclic": 0.02},
        elr={"constant": 2e-7, "linear_up": bad_value,
             "linear_down": 2e-7, "cyclic": 2e-7})
    with pytest.raises(ValueError, match="finite nonnegative"):
        evaluate_iteration(
            tmp_path, threshold=0.03, validity_threshold=1e-5)


def test_iteration_gate_rejects_corrupt_checkpoint(tmp_path):
    _write_run(
        tmp_path,
        {"linear_up": 0.02, "linear_down": 0.02, "cyclic": 0.02})
    checkpoint = (
        tmp_path / "branches" / "cyclic"
        / "checkpoint_step_020000.pt")
    checkpoint.write_bytes(b"not a torch checkpoint")
    result = evaluate_iteration(
        tmp_path, threshold=0.03, validity_threshold=1e-5)
    assert not result["validity_passed"]
    assert not result["passed"]
    assert "cyclic" in result["checkpoint_errors"]


def test_iteration_gate_uses_resolved_total_steps_and_checks_checkpoint_step(
        tmp_path):
    _write_run(
        tmp_path,
        {"linear_up": 0.02, "linear_down": 0.02, "cyclic": 0.02},
        total_steps=123)
    checkpoint = (
        tmp_path / "branches" / "linear_down"
        / "checkpoint_step_000123.pt")
    torch.save({"global_state_step": 122}, checkpoint)
    result = evaluate_iteration(
        tmp_path, threshold=0.03, validity_threshold=1e-5)
    assert result["total_steps"] == 123
    assert not result["final_checkpoints"]["linear_down"]
    assert "expected 123" in result["checkpoint_errors"]["linear_down"]

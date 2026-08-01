from __future__ import annotations

import csv
from pathlib import Path

import yaml

from src.reference_alignment_analysis import analyze_reference_alignment


def test_reference_alignment_analysis_checks_all_four_branches(tmp_path):
    cfg = {
        "optimizer": {"control_start_step": 0, "total_steps": 1},
        "control": {
            "constant_reference_alignment": {
                "enabled": True,
                "tensor": "norm.weight",
                "reference_branch": "constant",
                "mode": "frobenius_norm",
                "tolerance": 1.0e-5,
            }
        },
    }
    (tmp_path / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg), encoding="utf-8")
    fields = [
        "update_start_state_step", "global_state_step", "branch",
        "reference_branch", "tensor", "numel", "schedule_ratio",
        "base_lr", "actual_lr", "pre_update_frobenius_norm",
        "post_optimizer_pre_alignment_frobenius_norm",
        "target_reference_frobenius_norm",
        "post_alignment_frobenius_norm", "alignment_relative_error",
        "lr_over_pre_frobenius_norm", "lr_over_pre_parameter_rms",
        "was_projected",
    ]
    branches = ("constant", "linear_up", "linear_down", "cyclic")
    for branch in branches:
        path = tmp_path / "branches" / branch / "reference_alignment_metrics.csv"
        path.parent.mkdir(parents=True)
        row = {
            "update_start_state_step": 0,
            "global_state_step": 1,
            "branch": branch,
            "reference_branch": "constant",
            "tensor": "norm.weight",
            "numel": 192,
            "schedule_ratio": 1.0,
            "base_lr": 1.0e-3,
            "actual_lr": 1.0e-3,
            "pre_update_frobenius_norm": 10.0,
            "post_optimizer_pre_alignment_frobenius_norm": (
                11.0 if branch == "constant" else 9.0),
            "target_reference_frobenius_norm": 11.0,
            "post_alignment_frobenius_norm": 11.0,
            "alignment_relative_error": 0.0,
            "lr_over_pre_frobenius_norm": 1.0e-4,
            "lr_over_pre_parameter_rms": 1.0e-4 * (192 ** 0.5),
            "was_projected": branch != "constant",
        }
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)

    audit = analyze_reference_alignment(tmp_path)

    assert audit["passed"] is True
    assert audit["n_rows"] == 4
    assert audit["n_steps_per_branch"] == 1
    assert (
        tmp_path / "analysis" / "reference_alignment" /
        "final_norm_weight_alignment.png"
    ).is_file()

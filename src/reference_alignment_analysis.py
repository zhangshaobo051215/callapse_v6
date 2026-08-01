from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .reference_alignment import METRICS_FILENAME, alignment_settings


BRANCHES = ("constant", "linear_up", "linear_down", "cyclic")
COLORS = {
    "constant": "black",
    "linear_up": "tab:blue",
    "linear_down": "tab:red",
    "cyclic": "tab:green",
}


def _maximum_relative_error(actual, expected):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(expected)
    if not valid.any():
        return None
    denominator = np.maximum(np.abs(expected[valid]), 1.0e-30)
    return float(np.max(np.abs(actual[valid] - expected[valid]) / denominator))


def analyze_reference_alignment(output_dir):
    root = Path(output_dir)
    config_path = root / "resolved_config.yaml"
    if not config_path.is_file():
        return None
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = alignment_settings(cfg)
    if settings is None:
        return None

    frames, files = [], []
    for branch in BRANCHES:
        path = root / "branches" / branch / METRICS_FILENAME
        if path.is_file():
            frame = pd.read_csv(path)
            frames.append(frame)
            files.append(path)
    analysis = root / "analysis"
    plot_dir = analysis / "reference_alignment"
    plot_dir.mkdir(parents=True, exist_ok=True)
    audit_path = analysis / "reference_alignment_audit.json"
    audit_csv_path = analysis / "reference_alignment_audit.csv"

    checks = []

    def add(check, passed, value, details=""):
        checks.append({
            "check": check,
            "passed": bool(passed),
            "value": value,
            "details": details,
        })

    if not frames:
        add("alignment_files_found", False, 0)
        audit = {"passed": False, "checks": checks}
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        pd.DataFrame(checks).to_csv(audit_csv_path, index=False)
        return audit

    data = pd.concat(frames, ignore_index=True)
    required = {
        "update_start_state_step", "global_state_step", "branch",
        "reference_branch", "tensor", "numel", "schedule_ratio",
        "base_lr", "actual_lr", "pre_update_frobenius_norm",
        "post_optimizer_pre_alignment_frobenius_norm",
        "target_reference_frobenius_norm",
        "post_alignment_frobenius_norm", "alignment_relative_error",
        "lr_over_pre_frobenius_norm", "lr_over_pre_parameter_rms",
        "was_projected",
    }
    missing = sorted(required - set(data))
    add("required_columns_present", not missing, len(required) - len(missing),
        f"missing={missing}")
    if missing:
        audit = {"passed": False, "checks": checks}
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        pd.DataFrame(checks).to_csv(audit_csv_path, index=False)
        return audit

    reference_branch = str(settings["reference_branch"])
    tensor_name = str(settings["tensor"])
    tolerance = float(settings["tolerance"])
    observed_branches = set(data.branch.astype(str))
    add("all_four_branches_present", observed_branches == set(BRANCHES),
        len(observed_branches), f"present={sorted(observed_branches)}")
    add("all_files_found", len(files) == 4, len(files))
    tensors = set(data.tensor.astype(str))
    add("only_declared_tensor_is_aligned", tensors == {tensor_name},
        len(tensors), f"observed={sorted(tensors)}")
    reference_labels = set(data.reference_branch.astype(str))
    add("reference_branch_is_constant", reference_labels == {reference_branch}
        and reference_branch == "constant", len(reference_labels),
        f"observed={sorted(reference_labels)}")

    duplicates = int(data.duplicated(
        ["branch", "tensor", "global_state_step"]).sum())
    add("no_duplicate_rows", duplicates == 0, duplicates)
    first_step = int(cfg["optimizer"]["control_start_step"]) + 1
    final_step = int(cfg["optimizer"]["total_steps"])
    expected_steps = tuple(range(first_step, final_step + 1))
    grids = {
        branch: tuple(sorted(
            data.loc[data.branch == branch, "global_state_step"].astype(int).tolist()))
        for branch in BRANCHES
    }
    bad_grids = [branch for branch, grid in grids.items()
                 if grid != expected_steps]
    add("every_step_recorded_for_every_branch", not bad_grids,
        len(expected_steps), f"bad_branches={bad_grids}")

    numeric = data[[
        "base_lr", "actual_lr", "pre_update_frobenius_norm",
        "post_optimizer_pre_alignment_frobenius_norm",
        "target_reference_frobenius_norm",
        "post_alignment_frobenius_norm", "alignment_relative_error",
        "lr_over_pre_frobenius_norm", "lr_over_pre_parameter_rms",
    ]].to_numpy(float)
    finite = np.isfinite(numeric)
    add("all_metrics_are_finite", bool(finite.all()), int(finite.sum()),
        f"total={finite.size}")
    positive_columns = data[[
        "pre_update_frobenius_norm",
        "post_optimizer_pre_alignment_frobenius_norm",
        "target_reference_frobenius_norm",
        "post_alignment_frobenius_norm",
    ]].to_numpy(float)
    add("all_norms_are_positive", bool((positive_columns > 0).all()),
        int((positive_columns > 0).sum()), f"total={positive_columns.size}")

    lr_error = _maximum_relative_error(data.actual_lr, data.base_lr)
    add("aligned_tensor_uses_uncontrolled_base_lr",
        lr_error is not None and lr_error <= tolerance,
        lr_error, f"tolerance={tolerance:g}")
    post_error = _maximum_relative_error(
        data.post_alignment_frobenius_norm,
        data.target_reference_frobenius_norm,
    )
    add("post_alignment_norm_matches_constant_target",
        post_error is not None and post_error <= tolerance,
        post_error, f"tolerance={tolerance:g}")
    recorded_error = float(data.alignment_relative_error.abs().max())
    add("recorded_alignment_error_within_tolerance",
        recorded_error <= tolerance, recorded_error,
        f"tolerance={tolerance:g}")

    constant = data[data.branch == reference_branch].set_index(
        "global_state_step").sort_index()
    reference_self_error = _maximum_relative_error(
        constant.post_optimizer_pre_alignment_frobenius_norm,
        constant.post_alignment_frobenius_norm,
    )
    add("constant_branch_is_not_projected",
        reference_self_error is not None and reference_self_error <= tolerance
        and not constant.was_projected.astype(bool).any(),
        reference_self_error)

    target_errors, pre_errors, lr_cross_errors = [], [], []
    for branch in BRANCHES:
        current = data[data.branch == branch].set_index(
            "global_state_step").sort_index()
        common = current.index.intersection(constant.index)
        target_errors.append(_maximum_relative_error(
            current.loc[common, "target_reference_frobenius_norm"],
            constant.loc[common, "post_alignment_frobenius_norm"],
        ))
        pre_errors.append(_maximum_relative_error(
            current.loc[common, "pre_update_frobenius_norm"],
            constant.loc[common, "pre_update_frobenius_norm"],
        ))
        lr_cross_errors.append(_maximum_relative_error(
            current.loc[common, "actual_lr"],
            constant.loc[common, "actual_lr"],
        ))
    max_target_error = max(value for value in target_errors if value is not None)
    max_pre_error = max(value for value in pre_errors if value is not None)
    max_lr_cross_error = max(
        value for value in lr_cross_errors if value is not None)
    add("all_targets_equal_same_step_constant_norm",
        max_target_error <= tolerance, max_target_error,
        f"tolerance={tolerance:g}")
    add("pre_update_norms_align_across_branches",
        max_pre_error <= tolerance, max_pre_error,
        f"tolerance={tolerance:g}")
    add("actual_lrs_align_across_branches",
        max_lr_cross_error <= tolerance, max_lr_cross_error,
        f"tolerance={tolerance:g}")

    nonreference = data[data.branch != reference_branch]
    add("only_nonconstant_branches_are_projected",
        bool(nonreference.was_projected.astype(bool).all()),
        int(nonreference.was_projected.astype(bool).sum()),
        f"expected={len(nonreference)}")

    data.to_csv(plot_dir / "reference_alignment_metrics_all_branches.csv",
                index=False)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panels = (
        ("pre_update_frobenius_norm", "Pre-update Frobenius norm"),
        ("post_optimizer_pre_alignment_frobenius_norm",
         "Post-optimizer norm before alignment"),
        ("post_alignment_frobenius_norm", "Post-alignment Frobenius norm"),
        ("lr_over_pre_parameter_rms", "LR / pre-update parameter RMS"),
    )
    for branch in BRANCHES:
        frame = data[data.branch == branch]
        for axis, (column, title) in zip(axes.flat, panels):
            axis.plot(frame.global_state_step, frame[column],
                      color=COLORS[branch], label=branch, linewidth=1.1)
            axis.set_title(title)
            axis.set_xlabel("State step")
            axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"V6 constant-reference alignment: {tensor_name}", fontsize=14)
    fig.tight_layout()
    figure_path = plot_dir / "final_norm_weight_alignment.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    decisive = [bool(row["passed"]) for row in checks]
    audit = {
        "passed": bool(decisive and all(decisive)),
        "tolerance": tolerance,
        "tensor": tensor_name,
        "reference_branch": reference_branch,
        "n_rows": int(len(data)),
        "n_steps_per_branch": len(expected_steps),
        "max_alignment_relative_error": post_error,
        "max_pre_update_norm_relative_error_across_branches": max_pre_error,
        "max_actual_lr_relative_error_across_branches": max_lr_cross_error,
        "figure": str(figure_path.relative_to(root)).replace("\\", "/"),
        "checks": checks,
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame(checks).to_csv(audit_csv_path, index=False)
    return audit

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


BRANCHES = ("constant", "linear_up", "linear_down", "cyclic")
COLORS = {
    "constant": "black",
    "linear_up": "tab:blue",
    "linear_down": "tab:red",
    "cyclic": "tab:green",
}
AUDIT_TOLERANCE = 1e-5

# The monitor's current schema plus aliases used by earlier prototypes.
ALIASES = {
    "tensor": ("tensor", "tensor_name", "parameter_name", "parameter", "name"),
    "branch": ("branch", "schedule", "branch_name"),
    "step_pre": (
        "update_start_state_step", "state_step_pre", "update_step",
        "optimizer_step", "step",
    ),
    "step_post": (
        "global_state_step", "state_step_post", "post_step", "state_step",
    ),
    "optimizer_group": ("optimizer_group", "group", "parameter_group"),
    "is_controlled": ("is_controlled", "controlled"),
    "ndim": ("ndim", "rank"),
    "numel": ("numel", "n_elements", "size"),
    "base_lr": ("base_lr", "reference_lr"),
    "applied_lr": ("actual_lr", "applied_lr", "learning_rate", "lr"),
    "schedule_ratio": ("schedule_ratio", "q", "q_current"),
    "q_next": ("q_next", "next_schedule_ratio", "schedule_ratio_next"),
    "reference_fro_norm": (
        "reference_frobenius_norm", "reference_fro_norm",
        "reference_norm", "prefix_frobenius_norm",
    ),
    "reference_rms_norm": (
        "reference_rms_norm", "reference_rms", "prefix_rms_norm",
    ),
    "fro_norm_pre": (
        "pre_update_frobenius_norm", "fro_norm_pre",
        "pre_step_frobenius_norm", "pre_frobenius_norm",
        "norm_before_update",
    ),
    "fro_norm_post_update_pre_projection": (
        "post_optimizer_pre_projection_frobenius_norm",
        "fro_norm_post_update_pre_projection",
        "post_update_pre_projection_frobenius_norm",
        "pre_projection_fro_norm",
    ),
    "fro_norm_post_projection": (
        "post_projection_frobenius_norm", "fro_norm_post_projection",
        "post_step_frobenius_norm", "post_update_frobenius_norm",
        "post_frobenius_norm", "frobenius_norm", "tensor_norm",
    ),
    "target_post_fro_norm": (
        "target_post_frobenius_norm", "target_post_fro_norm",
        "target_frobenius_norm", "target_norm",
    ),
    "projection_relative_error": (
        "post_projection_relative_error", "projection_relative_error",
        "target_norm_relative_error",
    ),
    "norm_ratio_to_reference": (
        "post_projection_norm_ratio_to_reference",
        "post_norm_ratio_to_reference", "norm_ratio_to_reference",
        "post_reference_norm_ratio", "relative_norm",
    ),
    "lr_over_fro_norm": (
        "lr_over_frobenius_norm", "lr_over_fro_norm",
        "lr_over_frobenius", "lr_over_norm",
    ),
    "lr_over_rms_norm": (
        "lr_over_rms", "lr_over_rms_norm", "tensorwise_elr",
        "effective_lr",
    ),
    "reference_lr_over_fro_norm": (
        "reference_lr_over_frobenius_norm",
        "reference_lr_over_fro_norm", "reference_lr_over_frobenius",
    ),
    "reference_lr_over_rms_norm": (
        "reference_lr_over_rms", "reference_lr_over_rms_norm",
        "reference_tensorwise_elr",
    ),
    "lr_over_fro_ratio_to_reference": (
        "lr_over_frobenius_ratio_to_reference",
        "lr_over_fro_ratio_to_reference",
        "lr_over_frobenius_relative_to_reference",
    ),
    "lr_over_rms_ratio_to_reference": (
        "lr_over_rms_ratio_to_reference", "elr_ratio_to_reference",
    ),
}
NUMERIC = set(ALIASES) - {
    "tensor", "branch", "optimizer_group", "is_controlled",
}
METRICS = (
    (
        "fro_norm_post_projection", "step_post", "norm_absolute",
        "Frobenius norm", "target_post_fro_norm",
    ),
    (
        "norm_ratio_to_reference", "step_post", "norm_over_reference",
        "Norm / prefix norm", "target_norm_ratio_to_reference",
    ),
    (
        "lr_over_fro_norm", "step_pre", "lr_over_frobenius",
        "LR / Frobenius norm", None,
    ),
    (
        "lr_over_rms_norm", "step_pre", "lr_over_rms",
        "LR / parameter RMS", None,
    ),
)


def tensor_family(name: str) -> str:
    name = str(name).lower()
    if "patch_embed" in name or name.endswith("cls_token") or name.endswith("pos_embed"):
        return "input_embedding"
    if ".attn.qkv." in name or name.startswith("attn.qkv."):
        return "attention_qkv"
    if ".attn.proj." in name or name.startswith("attn.proj."):
        return "attention_output"
    if ".mlp.fc1." in name or name.startswith("mlp.fc1."):
        return "mlp_input"
    if ".mlp.fc2." in name or name.startswith("mlp.fc2."):
        return "mlp_output"
    if re.search(r"(?:^|\.)blocks?\.\d+\.norm[12]\.", name):
        return "block_layernorm"
    if re.search(r"(?:^|\.)norm\.(?:weight|bias)$", name):
        return "final_layernorm"
    if re.search(r"(?:^|\.)head\.(?:weight|bias)$", name):
        return "classifier_head"
    return "other"


def _natural_key(value):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def _safe_filename(value):
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return (safe or "tensor")[:140]


def _canonicalize(frame):
    frame = frame.copy()
    rename = {}
    for canonical, aliases in ALIASES.items():
        if canonical in frame:
            continue
        for alias in aliases:
            if alias in frame:
                rename[alias] = canonical
                break
    frame = frame.rename(columns=rename)
    for column in NUMERIC.intersection(frame.columns):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "is_controlled" in frame and not pd.api.types.is_bool_dtype(
        frame.is_controlled
    ):
        truthy = {"1", "true", "t", "yes", "y", "controlled"}
        frame["is_controlled"] = (
            frame.is_controlled.astype(str).str.strip().str.lower().isin(truthy)
        )
    return frame


def _merge_catalog(data, catalog):
    if catalog is None or "tensor" not in catalog:
        return data
    catalog = catalog.drop_duplicates("tensor")
    fields = (
        "tensor", "family", "optimizer_group", "is_controlled", "shape",
        "ndim", "numel", "reference_fro_norm", "reference_rms_norm",
    )
    columns = [column for column in fields if column in catalog]
    merged = data.merge(
        catalog[columns], on="tensor", how="left", suffixes=("", "_catalog")
    )
    for column in columns[1:]:
        other = f"{column}_catalog"
        if other not in merged:
            continue
        if column not in merged:
            merged[column] = merged[other]
        else:
            merged[column] = merged[column].where(
                merged[column].notna(), merged[other]
            )
        merged = merged.drop(columns=other)
    return merged


def _derive(frame):
    frame = frame.copy()
    if "step_pre" not in frame and "step_post" in frame:
        frame["step_pre"] = frame.step_post - 1
    if "step_post" not in frame and "step_pre" in frame:
        frame["step_post"] = frame.step_pre + 1
    if "fro_norm_post_projection" not in frame and "fro_norm_pre" in frame:
        frame["fro_norm_post_projection"] = frame.fro_norm_pre
    if "fro_norm_pre" not in frame and "fro_norm_post_projection" in frame:
        frame["fro_norm_pre"] = frame.fro_norm_post_projection
    if "reference_rms_norm" not in frame and {
        "reference_fro_norm", "numel",
    } <= set(frame):
        frame["reference_rms_norm"] = (
            frame.reference_fro_norm / np.sqrt(frame.numel)
        )
    if "norm_ratio_to_reference" not in frame and {
        "fro_norm_post_projection", "reference_fro_norm",
    } <= set(frame):
        frame["norm_ratio_to_reference"] = (
            frame.fro_norm_post_projection
            / frame.reference_fro_norm.replace(0, np.nan)
        )
    if "lr_over_fro_norm" not in frame and {
        "applied_lr", "fro_norm_pre",
    } <= set(frame):
        frame["lr_over_fro_norm"] = (
            frame.applied_lr / frame.fro_norm_pre.replace(0, np.nan)
        )
    if "lr_over_rms_norm" not in frame and {
        "applied_lr", "fro_norm_pre", "numel",
    } <= set(frame):
        pre_rms = frame.fro_norm_pre / np.sqrt(frame.numel)
        frame["lr_over_rms_norm"] = (
            frame.applied_lr / pre_rms.replace(0, np.nan)
        )
    if "reference_lr_over_fro_norm" not in frame and {
        "base_lr", "reference_fro_norm",
    } <= set(frame):
        frame["reference_lr_over_fro_norm"] = (
            frame.base_lr / frame.reference_fro_norm.replace(0, np.nan)
        )
    if "reference_lr_over_rms_norm" not in frame and {
        "base_lr", "reference_rms_norm",
    } <= set(frame):
        frame["reference_lr_over_rms_norm"] = (
            frame.base_lr / frame.reference_rms_norm.replace(0, np.nan)
        )
    if "lr_over_fro_ratio_to_reference" not in frame and {
        "lr_over_fro_norm", "reference_lr_over_fro_norm",
    } <= set(frame):
        frame["lr_over_fro_ratio_to_reference"] = (
            frame.lr_over_fro_norm
            / frame.reference_lr_over_fro_norm.replace(0, np.nan)
        )
    if "lr_over_rms_ratio_to_reference" not in frame and {
        "lr_over_rms_norm", "reference_lr_over_rms_norm",
    } <= set(frame):
        frame["lr_over_rms_ratio_to_reference"] = (
            frame.lr_over_rms_norm
            / frame.reference_lr_over_rms_norm.replace(0, np.nan)
        )
    if "target_post_fro_norm" not in frame and {
        "q_next", "reference_fro_norm",
    } <= set(frame):
        frame["target_post_fro_norm"] = frame.q_next * frame.reference_fro_norm
    if {
        "target_post_fro_norm", "reference_fro_norm",
    } <= set(frame):
        frame["target_norm_ratio_to_reference"] = (
            frame.target_post_fro_norm
            / frame.reference_fro_norm.replace(0, np.nan)
        )
    if "projection_relative_error" not in frame and {
        "fro_norm_post_projection", "target_post_fro_norm",
    } <= set(frame):
        frame["projection_relative_error"] = (
            (frame.fro_norm_post_projection - frame.target_post_fro_norm).abs()
            / frame.target_post_fro_norm.abs().replace(0, np.nan)
        )
    return frame


def _load_data(root, metrics_filename="tensor_metrics.csv",
               catalog_filename="tensor_catalog.csv"):
    frames, files = [], []
    for branch in BRANCHES:
        path = root / "branches" / branch / metrics_filename
        if not path.exists():
            continue
        frame = _canonicalize(pd.read_csv(path))
        if "branch" not in frame:
            frame["branch"] = branch
        else:
            frame["branch"] = frame.branch.fillna(branch).astype(str)
        frames.append(frame)
        files.append(path)
    if not frames:
        return None, None, []
    catalog_path = root / catalog_filename
    catalog = (
        _canonicalize(pd.read_csv(catalog_path))
        if catalog_path.exists() else None
    )
    data = _derive(_merge_catalog(pd.concat(frames, ignore_index=True), catalog))
    return data, catalog, files


def _build_catalog(data, catalog):
    fields = (
        "tensor", "family", "optimizer_group", "is_controlled", "shape",
        "ndim", "numel", "reference_fro_norm", "reference_rms_norm",
    )
    available = [column for column in fields if column in data]
    recorded = (
        data[available].dropna(subset=["tensor"]).drop_duplicates("tensor")
        if "tensor" in available else pd.DataFrame(columns=["tensor"])
    )
    if catalog is None or "tensor" not in catalog:
        result = recorded
    else:
        keep = [column for column in fields if column in catalog]
        result = catalog[keep].drop_duplicates("tensor").merge(
            recorded, on="tensor", how="outer", suffixes=("", "_recorded")
        )
        for column in fields[1:]:
            other = f"{column}_recorded"
            if other not in result:
                continue
            if column not in result:
                result[column] = result[other]
            else:
                result[column] = result[column].where(
                    result[column].notna(), result[other]
                )
            result = result.drop(columns=other)
    if "tensor" not in result:
        result["tensor"] = pd.Series(dtype=str)
    inferred = result.tensor.map(tensor_family)
    if "family" not in result:
        result["family"] = inferred
    else:
        result["family"] = result.family.where(result.family.notna(), inferred)
    return result.sort_values(
        "tensor", key=lambda series: series.map(_natural_key)
    ).reset_index(drop=True)


def _branch_order(data):
    present = set(data.branch.dropna().astype(str))
    return [branch for branch in BRANCHES if branch in present]


def _plot_curve(ax, data, metric, x_column, target=None, title=None):
    plotted = False
    for branch in _branch_order(data):
        group = data[data.branch == branch].sort_values(x_column)
        values = group[[x_column, metric]].dropna()
        if values.empty:
            continue
        color = COLORS[branch]
        ax.plot(
            values[x_column], values[metric], color=color,
            linewidth=1.0, alpha=.9,
        )
        plotted = True
        if target and target in group:
            target_values = group[[x_column, target]].dropna()
            if not target_values.empty:
                ax.plot(
                    target_values[x_column], target_values[target],
                    color=color, linewidth=.8, linestyle="--", alpha=.65,
                )
    if title:
        ax.set_title(title, fontsize=7.5)
    ax.grid(alpha=.22)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("State step", fontsize=7)
    return plotted


def _legend(data, include_target=False):
    handles = [
        Line2D([0], [0], color=COLORS[branch], label=branch)
        for branch in _branch_order(data)
    ]
    if include_target:
        handles.append(
            Line2D([0], [0], color="gray", linestyle="--", label="target")
        )
    return handles


def _plot_family_atlases(data, catalog, output_dir):
    generated = []
    for metric, x_column, directory, label, target in METRICS:
        if metric not in data or not data[metric].notna().any():
            continue
        metric_dir = output_dir / "family_atlas" / directory
        metric_dir.mkdir(parents=True, exist_ok=True)
        for family in sorted(catalog.family.unique(), key=_natural_key):
            tensors = catalog.loc[catalog.family == family, "tensor"].tolist()
            tensors = [name for name in tensors if (data.tensor == name).any()]
            if not tensors:
                continue
            ncols = min(4, len(tensors))
            nrows = math.ceil(len(tensors) / ncols)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(4 * ncols, 2.65 * nrows),
                squeeze=False,
            )
            axes = axes.ravel()
            for ax, tensor in zip(axes, tensors):
                tensor_data = data[data.tensor == tensor]
                row = catalog[catalog.tensor == tensor].iloc[0]
                controlled = row.get("is_controlled", np.nan)
                suffix = (
                    "" if pd.isna(controlled)
                    else (" [C]" if bool(controlled) else " [U]")
                )
                _plot_curve(
                    ax, tensor_data, metric, x_column, target,
                    f"{tensor}{suffix}",
                )
            for ax in axes[len(tensors):]:
                ax.set_visible(False)
            fig.suptitle(f"{family}: {label}", fontsize=12)
            fig.supylabel(label, fontsize=10)
            handles = _legend(data, bool(target and target in data))
            if handles:
                fig.legend(
                    handles=handles, loc="upper center", ncol=len(handles),
                    bbox_to_anchor=(.5, .995), fontsize=8,
                )
            fig.subplots_adjust(left=.065, right=.985, bottom=.055, top=.91,
                                wspace=.30, hspace=.42)
            path = metric_dir / f"{_safe_filename(family)}.png"
            fig.savefig(path, dpi=170)
            plt.close(fig)
            generated.append(path)
    return generated


def _plot_each_tensor(data, catalog, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, item in catalog.iterrows():
        tensor = item.tensor
        tensor_data = data[data.tensor == tensor]
        if tensor_data.empty:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)
        plotted = 0
        for ax, (metric, x_column, _directory, label, target) in zip(
            axes.ravel(), METRICS
        ):
            if metric not in tensor_data:
                ax.set_visible(False)
                continue
            plotted += int(_plot_curve(
                ax, tensor_data, metric, x_column, target, label
            ))
        controlled = item.get("is_controlled", np.nan)
        scope = (
            "unknown scope" if pd.isna(controlled)
            else ("controlled" if bool(controlled) else "uncontrolled")
        )
        fig.suptitle(f"{tensor} ({scope})", fontsize=11)
        handles = _legend(
            tensor_data, "target_post_fro_norm" in tensor_data
        )
        if handles:
            fig.legend(
                handles=handles, loc="upper center", ncol=len(handles),
                bbox_to_anchor=(.5, .955), fontsize=8,
            )
        fig.subplots_adjust(left=.08, right=.985, bottom=.075, top=.84,
                            wspace=.27, hspace=.34)
        path = output_dir / f"{index:03d}_{_safe_filename(tensor)}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        rows.append({
            "plot_index": int(index),
            "tensor": tensor,
            "family": item.get("family", tensor_family(tensor)),
            "is_controlled": controlled,
            "plotted_metrics": plotted,
            "relative_path": str(
                path.relative_to(output_dir.parent.parent)
            ).replace("\\", "/"),
        })
    return pd.DataFrame(rows)


def _summary(data, catalog):
    rows = []
    metadata = catalog.set_index("tensor")
    for (branch, tensor), group in data.groupby(
        ["branch", "tensor"], sort=False
    ):
        group = group.sort_values("step_post")
        row = {
            "branch": branch,
            "tensor": tensor,
            "n_samples": len(group),
            "first_state_step": group.step_post.iloc[0],
            "last_state_step": group.step_post.iloc[-1],
        }
        if tensor in metadata.index:
            item = metadata.loc[tensor]
            for column in (
                "family", "optimizer_group", "is_controlled", "shape",
                "ndim", "numel", "reference_fro_norm", "reference_rms_norm",
            ):
                if column in item:
                    row[column] = item[column]
        for column in (
            "fro_norm_post_projection", "norm_ratio_to_reference",
            "applied_lr", "lr_over_fro_norm", "lr_over_rms_norm",
        ):
            if column not in group:
                continue
            values = group[column].dropna()
            if values.empty:
                continue
            for label, value in (
                ("initial", values.iloc[0]), ("final", values.iloc[-1]),
                ("min", values.min()), ("max", values.max()),
                ("mean", values.mean()),
            ):
                row[f"{label}_{column}"] = value
        for column in (
            "projection_relative_error",
            "lr_over_fro_ratio_to_reference",
            "lr_over_rms_ratio_to_reference",
        ):
            if column in group and group[column].notna().any():
                values = group[column].dropna()
                if column.endswith("ratio_to_reference"):
                    values = (values - 1).abs()
                else:
                    values = values.abs()
                row[f"max_abs_{column}"] = values.max()
        rows.append(row)
    return pd.DataFrame(rows)


def _family_summary(summary):
    required = {
        "branch", "family", "is_controlled", "tensor", "n_samples",
        "final_norm_ratio_to_reference", "final_lr_over_fro_norm",
        "final_lr_over_rms_norm",
    }
    if summary.empty or not required <= set(summary):
        return pd.DataFrame()
    return summary.groupby(
        ["branch", "family", "is_controlled"],
        dropna=False, as_index=False,
    ).agg(
        n_tensors=("tensor", "nunique"),
        n_samples=("n_samples", "sum"),
        mean_final_norm_ratio=("final_norm_ratio_to_reference", "mean"),
        max_final_norm_ratio=("final_norm_ratio_to_reference", "max"),
        mean_final_lr_over_frobenius=("final_lr_over_fro_norm", "mean"),
        mean_final_lr_over_rms=("final_lr_over_rms_norm", "mean"),
    )


def _audit(data, catalog, files, plotted, require_both_scopes=True):
    rows = []

    def add(check, passed, value, details=""):
        rows.append({
            "check": check, "passed": passed,
            "value": value, "details": details,
        })

    def maximum_relative_error(actual, expected):
        actual = np.asarray(actual, dtype=float)
        expected = np.asarray(expected, dtype=float)
        valid = np.isfinite(actual) & np.isfinite(expected)
        if not valid.any():
            return None
        denominator = np.maximum(np.abs(expected[valid]), 1e-30)
        return float(np.max(
            np.abs(actual[valid] - expected[valid]) / denominator
        ))

    branches = set(data.branch.dropna().astype(str))
    add(
        "all_four_branches_present", branches == set(BRANCHES), len(branches),
        f"present={sorted(branches)}; missing={sorted(set(BRANCHES) - branches)}",
    )
    add("monitor_files_found", len(files) == 4, len(files))
    required = {
        "branch", "tensor", "step_pre", "step_post", "is_controlled",
        "numel", "applied_lr", "reference_fro_norm", "fro_norm_pre",
        "fro_norm_post_projection", "norm_ratio_to_reference", "q_next",
        "lr_over_fro_norm", "lr_over_rms_norm",
    }
    missing = sorted(required - set(data))
    add("required_columns_present", not missing, len(required) - len(missing),
        f"missing={missing}")

    tensor_sets = {
        branch: set(group.tensor.astype(str))
        for branch, group in data.groupby("branch")
    }
    identical = (
        len(tensor_sets) == 4
        and len({frozenset(values) for values in tensor_sets.values()}) == 1
    )
    recorded = set(data.tensor.astype(str))
    catalog_set = set(catalog.tensor.astype(str))
    add(
        "tensor_sets_identical_across_branches", identical, len(recorded),
        "; ".join(f"{key}={len(value)}" for key, value in tensor_sets.items()),
    )
    add(
        "catalog_matches_recorded_tensors", catalog_set == recorded,
        len(catalog_set),
        f"missing_from_records={sorted(catalog_set - recorded)}; "
        f"missing_from_catalog={sorted(recorded - catalog_set)}",
    )
    add(
        "every_recorded_tensor_has_plot", plotted == recorded, len(plotted),
        f"missing_plots={sorted(recorded - plotted)}",
    )
    duplicates = int(data.duplicated(["branch", "tensor", "step_pre"]).sum())
    add("no_duplicate_samples", duplicates == 0, duplicates)
    grids = data.groupby(["branch", "tensor"]).step_pre.apply(
        lambda values: tuple(sorted(values.dropna().astype(int).unique()))
    )
    matching_grid = grids.groupby(level="tensor").apply(
        lambda values: len(values) == 4 and len(set(values)) == 1
    )
    mismatch_count = int((~matching_grid).sum())
    add("sample_step_grids_match_across_branches", mismatch_count == 0,
        mismatch_count)

    expected_per_snapshot = len(catalog_set)
    snapshot_counts = data.groupby(["branch", "step_pre"]).tensor.nunique()
    incomplete_snapshots = int(
        (snapshot_counts != expected_per_snapshot).sum()
    )
    add(
        "every_snapshot_contains_every_tensor",
        incomplete_snapshots == 0 and expected_per_snapshot > 0,
        incomplete_snapshots,
        f"expected_tensors_per_snapshot={expected_per_snapshot}",
    )
    step_errors = (
        data.step_post.to_numpy(float)
        - data.step_pre.to_numpy(float)
        - 1.0
    )
    maximum_step_error = float(np.nanmax(np.abs(step_errors)))
    add(
        "pre_and_post_steps_are_aligned",
        maximum_step_error == 0.0,
        maximum_step_error,
    )

    expected_lr = np.where(
        data.is_controlled.to_numpy(bool),
        data.base_lr.to_numpy(float) * data.schedule_ratio.to_numpy(float),
        data.base_lr.to_numpy(float),
    )
    lr_formula_error = maximum_relative_error(data.applied_lr, expected_lr)
    add(
        "actual_optimizer_group_lr_matches_policy",
        lr_formula_error is not None and lr_formula_error <= AUDIT_TOLERANCE,
        lr_formula_error,
        "controlled=base_lr*q_now; uncontrolled=base_lr",
    )
    expected_lr_over_fro = (
        data.applied_lr.to_numpy(float)
        / data.fro_norm_pre.to_numpy(float)
    )
    fro_formula_error = maximum_relative_error(
        data.lr_over_fro_norm, expected_lr_over_fro
    )
    add(
        "lr_over_frobenius_uses_pre_update_norm",
        fro_formula_error is not None
        and fro_formula_error <= AUDIT_TOLERANCE,
        fro_formula_error,
    )
    expected_lr_over_rms = (
        expected_lr_over_fro * np.sqrt(data.numel.to_numpy(float))
    )
    rms_formula_error = maximum_relative_error(
        data.lr_over_rms_norm, expected_lr_over_rms
    )
    add(
        "lr_over_rms_uses_pre_update_rms",
        rms_formula_error is not None
        and rms_formula_error <= AUDIT_TOLERANCE,
        rms_formula_error,
    )
    if ("fro_norm_post_update_pre_projection" in data
            and (data.is_controlled == False).any()):  # noqa: E712
        free = data[data.is_controlled == False]  # noqa: E712
        free_projection_error = maximum_relative_error(
            free.fro_norm_post_projection,
            free.fro_norm_post_update_pre_projection,
        )
        add(
            "uncontrolled_tensors_are_not_projected",
            free_projection_error is not None
            and free_projection_error <= AUDIT_TOLERANCE,
            free_projection_error,
        )

    norm_columns = [
        column for column in (
            "reference_fro_norm", "reference_rms_norm", "fro_norm_pre",
            "fro_norm_post_projection", "norm_ratio_to_reference",
        ) if column in data
    ]
    norms = data[norm_columns].to_numpy(float)
    norm_good = np.isfinite(norms) & (norms > 0)
    add(
        "norms_are_finite_and_positive", bool(norm_good.all()),
        int(norm_good.sum()), f"total_values={norm_good.size}",
    )
    lr_columns = [
        column for column in (
            "applied_lr", "lr_over_fro_norm", "lr_over_rms_norm",
        ) if column in data
    ]
    lrs = data[lr_columns].to_numpy(float)
    lr_good = np.isfinite(lrs) & (lrs >= 0)
    add(
        "lr_metrics_are_finite_and_nonnegative", bool(lr_good.all()),
        int(lr_good.sum()), f"total_values={lr_good.size}",
    )

    controlled = data[data.is_controlled == True]  # noqa: E712
    uncontrolled = data[data.is_controlled == False]  # noqa: E712
    n_controlled = int(controlled.tensor.nunique())
    n_uncontrolled = int(uncontrolled.tensor.nunique())
    if require_both_scopes:
        scope_check = "both_control_scopes_recorded"
        scope_passed = n_controlled > 0 and n_uncontrolled > 0
    else:
        scope_check = "all_control_units_recorded"
        scope_passed = n_controlled > 0 and n_uncontrolled == 0
    add(
        scope_check,
        scope_passed,
        n_controlled + n_uncontrolled,
        f"controlled={n_controlled}; uncontrolled={n_uncontrolled}",
    )
    expected_targets = (
        controlled.reference_fro_norm.to_numpy(float)
        * controlled.q_next.to_numpy(float)
    )
    target_formula_error = maximum_relative_error(
        controlled.target_post_fro_norm, expected_targets
    )
    add(
        "controlled_target_uses_q_next",
        target_formula_error is not None
        and target_formula_error <= AUDIT_TOLERANCE,
        target_formula_error,
    )
    post_target_error = maximum_relative_error(
        controlled.fro_norm_post_projection, expected_targets
    )
    add(
        "controlled_post_projection_norm_matches_q_next_target",
        post_target_error is not None
        and post_target_error <= AUDIT_TOLERANCE,
        post_target_error,
        f"tolerance={AUDIT_TOLERANCE:g}",
    )
    if "projection_relative_error" in controlled:
        values = controlled.projection_relative_error.dropna().abs()
        maximum = float(values.max()) if not values.empty else None
        add(
            "controlled_projection_error_within_tolerance",
            maximum is not None and maximum <= AUDIT_TOLERANCE,
            maximum, f"tolerance={AUDIT_TOLERANCE:g}",
        )
    else:
        add(
            "controlled_projection_error_within_tolerance", None, None,
            "controlled target metric unavailable",
        )
    for metric, check in (
        ("lr_over_fro_ratio_to_reference",
         "controlled_lr_over_frobenius_matches_reference"),
        ("lr_over_rms_ratio_to_reference",
         "controlled_lr_over_rms_matches_reference"),
    ):
        if metric in controlled:
            values = controlled[metric].dropna()
            maximum = float((values - 1).abs().max()) if not values.empty else None
            add(
                check,
                maximum is not None and maximum <= AUDIT_TOLERANCE,
                maximum, f"maximum |ratio-1|; tolerance={AUDIT_TOLERANCE:g}",
            )
        else:
            add(check, None, None, f"{metric} unavailable")
    return rows


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def analyze_tensor_monitoring(
    output_dir, *,
    metrics_filename="tensor_metrics.csv",
    catalog_filename="tensor_catalog.csv",
    analysis_directory="tensor_monitoring",
    audit_stem="tensor_monitoring_audit",
    require_both_scopes=True,
):
    """Create all-tensor plots/audits, or return None for legacy runs."""
    root = Path(output_dir)
    data, input_catalog, files = _load_data(
        root, metrics_filename, catalog_filename)
    if data is None:
        return None
    analysis = root / "analysis"
    monitoring = analysis / analysis_directory
    monitoring.mkdir(parents=True, exist_ok=True)
    audit_json_path = analysis / f"{audit_stem}.json"
    audit_csv_path = analysis / f"{audit_stem}.csv"

    if "tensor" not in data:
        audit = {
            "passed": False,
            "checks": [{
                "check": "tensor_names_present", "passed": False,
                "value": 0, "details": "No recognized tensor column",
            }],
        }
        audit_json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        pd.DataFrame(audit["checks"]).to_csv(audit_csv_path, index=False)
        return audit
    data["tensor"] = data.tensor.astype(str)
    data["branch"] = data.branch.astype(str)
    catalog = _build_catalog(data, input_catalog)
    catalog.to_csv(monitoring / "tensor_catalog.csv", index=False)

    atlas_paths = _plot_family_atlases(data, catalog, monitoring)
    manifest = _plot_each_tensor(data, catalog, monitoring / "per_tensor")
    manifest.to_csv(monitoring / "plot_manifest.csv", index=False)
    plotted = set(manifest.tensor) if "tensor" in manifest else set()
    summary = _summary(data, catalog)
    summary.to_csv(monitoring / "tensor_monitor_summary.csv", index=False)
    _family_summary(summary).to_csv(
        monitoring / "tensor_family_summary.csv", index=False
    )

    checks = _audit(
        data, catalog, files, plotted, require_both_scopes)
    pd.DataFrame(checks).to_csv(audit_csv_path, index=False)
    decisive = [
        bool(row["passed"]) for row in checks if row["passed"] is not None
    ]
    audit = _json_ready({
        "passed": bool(decisive and all(decisive)),
        "tolerance": AUDIT_TOLERANCE,
        "n_rows": int(len(data)),
        "n_tensors": int(data.tensor.nunique()),
        "n_controlled_tensors": int(
            data.loc[data.is_controlled == True, "tensor"].nunique()  # noqa: E712
        ),
        "n_uncontrolled_tensors": int(
            data.loc[data.is_controlled == False, "tensor"].nunique()  # noqa: E712
        ),
        "branches": sorted(data.branch.unique().tolist(), key=_natural_key),
        "checks": checks,
        "generated_family_atlases": [
            str(path.relative_to(root)).replace("\\", "/")
            for path in atlas_paths
        ],
        "per_tensor_plot_manifest":
            f"analysis/{analysis_directory}/plot_manifest.csv",
    })
    serialized = json.dumps(
        audit, indent=2, ensure_ascii=False, allow_nan=False
    )
    audit_json_path.write_text(serialized, encoding="utf-8")
    (monitoring / f"{audit_stem}.json").write_text(
        serialized, encoding="utf-8"
    )
    return audit

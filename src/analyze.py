from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .tensor_monitoring_analysis import analyze_tensor_monitoring


COLORS = {"constant": "black", "linear_up": "tab:blue",
          "linear_down": "tab:red", "cyclic": "tab:green"}


def analyze_run(output_dir):
    root = Path(output_dir)
    frames = []
    for branch in COLORS:
        path = root / "branches" / branch / "metrics.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("no branch metrics found")
    data = pd.concat(frames, ignore_index=True)
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    const = data[data.branch == "constant"].set_index("global_state_step")
    summaries = []
    evaluations = {}
    for branch in COLORS:
        path = root / "branches" / branch / "evaluation_metrics.csv"
        if path.exists():
            evaluations[branch] = pd.read_csv(path)
    for branch, group in data.groupby("branch"):
        g = group.set_index("global_state_step")
        common = g.index.intersection(const.index)
        raw = (g.loc[common, "train_loss_raw"] - const.loc[common, "train_loss_raw"]).abs()
        ema = (g.loc[common, "train_loss_ema"] - const.loc[common, "train_loss_ema"]).abs()
        probe = evaluations.get(branch, pd.DataFrame())
        const_eval = evaluations.get("constant", pd.DataFrame())
        probe_g = probe[probe.split == "probe"] if not probe.empty else probe
        probe_c = const_eval[const_eval.split == "probe"] if not const_eval.empty else const_eval
        probe_mae = np.nan
        if not probe_g.empty and not probe_c.empty:
            pg, pc = probe_g.set_index("global_state_step"), probe_c.set_index("global_state_step")
            idx = pg.index.intersection(pc.index)
            probe_mae = (pg.loc[idx, "loss"] - pc.loc[idx, "loss"]).abs().mean()
        val_g = probe[probe.split == "validation"] if not probe.empty else probe
        summaries.append({
            "branch": branch, "loss_raw_mae_vs_constant": raw.mean(),
            "loss_ema_mae_vs_constant": ema.mean(),
            "loss_ema_max_abs_vs_constant": ema.max(),
            "probe_loss_mae_vs_constant": probe_mae,
            "nce": ema.sum() / (const.loc[common, "train_loss_ema"].abs().sum() + 1e-12),
            "mean_elr_relative_error": g.elr_relative_error_mean.mean(),
            "max_elr_relative_error": g.elr_relative_error_max.max(),
            "mean_target_norm_relative_error": g.target_norm_relative_error_mean.mean(),
            "max_target_norm_relative_error": g.target_norm_relative_error_max.max(),
            "mean_angular_step": g.angular_step_mean.mean(),
            "angular_step_mae_vs_constant": np.nan,
            "final_train_loss_ema": g.train_loss_ema.iloc[-1],
            "final_probe_loss": probe_g.loss.iloc[-1] if not probe_g.empty else np.nan,
            "final_val_loss": val_g.loss.iloc[-1] if not val_g.empty else np.nan,
            "final_val_top1": val_g.top1.iloc[-1] if not val_g.empty else np.nan,
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(analysis / "collapse_metrics.csv", index=False)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    keys = ["controlled_lr", "controlled_frobenius_norm", "tensorwise_elr_mean", "train_loss_ema"]
    labels = ["Controlled LR", "Controlled Frobenius norm", "Mean tensorwise ELR", "Train loss EMA"]
    for branch, group in data.groupby("branch"):
        residual = summary.loc[summary.branch == branch, "loss_ema_mae_vs_constant"].iloc[0]
        label = branch if branch == "constant" else f"{branch} (MAE={residual:.4g})"
        for ax, key in zip(axes, keys):
            ax.plot(group.global_state_step, group[key], color=COLORS[branch], label=label)
    for ax, label in zip(axes, labels):
        ax.set_title(label); ax.set_xlabel("State step"); ax.grid(alpha=.25)
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(analysis / "figure1_vit.png", dpi=180)
    plt.close(fig)
    # Focused diagnostics, all using fixed scales derived from the recorded data.
    fig, ax = plt.subplots(figsize=(8, 4))
    for branch, group in data.groupby("branch"):
        idx = group.global_state_step
        baseline = const.reindex(idx).train_loss_ema.to_numpy()
        ax.plot(idx, group.train_loss_ema.to_numpy() - baseline,
                color=COLORS[branch], label=branch)
    ax.axhline(0, color="gray", lw=.8); ax.set_title("EMA loss residual vs constant")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(analysis / "loss_residuals.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for branch, group in data.groupby("branch"):
        ax.plot(group.global_state_step, group.elr_relative_error_max,
                color=COLORS[branch], label=branch)
    ax.set_yscale("log"); ax.set_title("Maximum ELR matching error")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(analysis / "elr_matching_error.png", dpi=180); plt.close(fig)
    angular_rows = []
    fig, ax = plt.subplots(figsize=(8, 4))
    for branch in COLORS:
        path = root / "branches" / branch / "angular_metrics.csv"
        if path.exists():
            frame = pd.read_csv(path); angular_rows.append(frame)
            mean = frame.groupby("global_state_step").angular_step.mean()
            ax.plot(mean.index, mean.values, color=COLORS[branch], label=branch)
    ax.set_title("Mean angular step"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(analysis / "angular_step_collapse.png", dpi=180); plt.close(fig)
    if angular_rows:
        angles = pd.concat(angular_rows)
        angles.groupby(["branch", "tensor"], as_index=False).agg(
            numel=("numel", "first"), mean_angular_step=("angular_step", "mean"),
            final_cumulative_angular_step=("cumulative_angular_step", "max")
        ).to_csv(analysis / "per_tensor_summary.csv", index=False)
    else:
        pd.DataFrame(columns=["branch", "tensor", "numel", "mean_angular_step",
                              "final_cumulative_angular_step"]).to_csv(
                                  analysis / "per_tensor_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    found_probe = False
    for branch, frame in evaluations.items():
        probe = frame[frame.split == "probe"]
        if not probe.empty:
            found_probe = True
            ax.plot(probe.global_state_step, probe.loss, color=COLORS[branch], label=branch)
    ax.set_title("Fixed-probe loss"); ax.grid(alpha=.25)
    if found_probe:
        ax.legend()
    fig.tight_layout(); fig.savefig(analysis / "probe_loss_collapse.png", dpi=180); plt.close(fig)
    tensor_monitoring = analyze_tensor_monitoring(root)
    control_unit_monitoring = analyze_tensor_monitoring(
        root,
        metrics_filename="control_unit_metrics.csv",
        catalog_filename="control_unit_catalog.csv",
        analysis_directory="control_unit_monitoring",
        audit_stem="control_unit_monitoring_audit",
        require_both_scopes=False,
    )
    report = [
        "# ViT ELR Collapse Report", "",
        "1. Norm/ELR matching: see `collapse_metrics.csv` (engineering validity precedes interpretation).",
        "2. Training-loss collapse errors are listed in `collapse_metrics.csv`.",
        "3. Probe and validation collapse require completed periodic evaluations.",
        "4. Radial sensitivity is reported in `radial_audit/radial_sensitivity.csv`.",
        "5. Angular-step evidence is logged per tensor and summarized here.", "",
    ]
    if tensor_monitoring is not None:
        status = "passed" if tensor_monitoring["passed"] else "FAILED"
        report.extend([
            "6. All trainable tensors were monitored by component family.",
            f"   Tensor-monitoring audit: {status}; see `tensor_monitoring/`.",
            "",
        ])
    if control_unit_monitoring is not None:
        status = "passed" if control_unit_monitoring["passed"] else "FAILED"
        report.extend([
            "7. Every independently projected control unit was monitored.",
            f"   Control-unit audit: {status}; see `control_unit_monitoring/`.",
            "",
        ])
    best = float(summary.loss_ema_mae_vs_constant.max())
    label = "strong" if best < .01 else ("partial" if best < .02 else "weak/no collapse")
    report.append(f"Conclusion: {label}. This result applies only to this model, dataset, optimizer, and control scope.")
    (analysis / "report.md").write_text("\n".join(report), encoding="utf-8")
    return summary

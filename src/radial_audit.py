from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .utils import append_csv


def audit_groups(model):
    named = dict(model.named_parameters())
    pick = lambda fragment: {n: p for n, p in named.items() if fragment in n}
    return {
        "attn_qkv": pick(".attn.qkv.weight"), "attn_proj": pick(".attn.proj.weight"),
        "mlp_fc1": pick(".mlp.fc1.weight"), "mlp_fc2": pick(".mlp.fc2.weight"),
        "all_hidden": {n: p for n, p in named.items() if any(
            x in n for x in (".attn.qkv.weight", ".attn.proj.weight", ".mlp.fc1.weight", ".mlp.fc2.weight"))},
        "patch_embed": {"patch_embed.proj.weight": named["patch_embed.proj.weight"]},
        "classifier_head": {"head.weight": named["head.weight"]},
        "pos_embed": {"pos_embed": named["pos_embed"]}, "cls_token": {"cls_token": named["cls_token"]},
        "layernorm_gains": {n: p for n, p in named.items()
                            if n.endswith(".weight") and ("norm" in n)},
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum = correct = count = 0
    logits_all, probs_all = [], []
    for x, y in loader:
        logits = model(x.to(device))
        probs = logits.softmax(1)
        loss_sum += F.cross_entropy(logits, y.to(device), reduction="sum").item()
        correct += (logits.argmax(1).cpu() == y).sum().item()
        count += y.numel()
        logits_all.append(logits.cpu()); probs_all.append(probs.cpu())
    logits, probs = torch.cat(logits_all), torch.cat(probs_all)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(1).mean().item()
    return {"loss": loss_sum / count, "top1": correct / count,
            "logit_rms": logits.square().mean().sqrt().item(),
            "max_probability": probs.max(1).values.mean().item(), "entropy": entropy}, probs


def run_radial_audit(model, base_state, loader, device, scales, output, generate_plot=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "radial_sensitivity.csv"
    # A previous interrupted audit must not leave duplicated partial rows.
    if csv_path.exists():
        csv_path.unlink()
    original = copy.deepcopy(base_state)
    model.load_state_dict(original, strict=True)
    baseline, base_probs = evaluate(model, loader, device)
    for group_name in audit_groups(model):
        for scale in scales:
            model.load_state_dict(original, strict=True)
            with torch.no_grad():
                for p in audit_groups(model)[group_name].values():
                    p.mul_(scale)
            metrics, probs = evaluate(model, loader, device)
            kl = (base_probs * (base_probs.clamp_min(1e-12).log() -
                               probs.clamp_min(1e-12).log())).sum(1).mean().item()
            append_csv({
                "group": group_name, "scale": scale, "baseline_loss": baseline["loss"],
                "scaled_loss": metrics["loss"], "delta_loss": metrics["loss"] - baseline["loss"],
                "absolute_loss_change": abs(metrics["loss"] - baseline["loss"]),
                "relative_loss_change": abs(metrics["loss"] - baseline["loss"]) / max(baseline["loss"], 1e-12),
                "mean_kl": kl, "top1_change": metrics["top1"] - baseline["top1"],
                "logit_rms_ratio": metrics["logit_rms"] / max(baseline["logit_rms"], 1e-12),
                "max_probability_change": metrics["max_probability"] - baseline["max_probability"],
                "entropy_change": metrics["entropy"] - baseline["entropy"],
            }, csv_path)
    model.load_state_dict(original, strict=True)
    for k, v in original.items():
        if not torch.equal(model.state_dict()[k].detach().cpu(), v.detach().cpu()):
            raise AssertionError(f"radial audit failed to restore {k}")
    if not generate_plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    out = output
    table = pd.read_csv(out / "radial_sensitivity.csv").sort_values(
        ["absolute_loss_change", "mean_kl"], ascending=False)
    labels = table["group"] + " x" + table["scale"].astype(str)
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(table) * .25)))
    axes[0].barh(labels, table.absolute_loss_change.clip(lower=1e-12))
    axes[1].barh(labels, table.mean_kl.clip(lower=1e-12))
    axes[0].set_title("|delta loss|"); axes[1].set_title("Mean KL")
    for ax in axes:
        ax.set_xscale("log"); ax.invert_yaxis(); ax.grid(axis="x", alpha=.25)
    fig.tight_layout(); fig.savefig(out / "radial_sensitivity.png", dpi=180); plt.close(fig)

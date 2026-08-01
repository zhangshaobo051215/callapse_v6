from __future__ import annotations

from collections import Counter
import json
import math
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data import EpochBatchSampler
from src.model import ViTConfig, VisionTransformer
from src.norm_control import frobenius_norm, reference_norms_for
from src.policy_overlay_residual_stream_symmetry import POLICY, install
from src.train_branch import train_branch

install()

from run_pipeline import resolve_config, tensor_family  # noqa: E402
from src import param_groups  # noqa: E402


def _positive_model():
    model = VisionTransformer(ViTConfig(
        image_size=16,
        patch_size=8,
        embed_dim=12,
        depth=1,
        num_heads=3,
        num_classes=4,
    ))
    with torch.no_grad():
        for parameter in model.parameters():
            if torch.linalg.vector_norm(parameter.float()).item() == 0:
                parameter.fill_(0.01)
    return model


def test_target_policy_catalog_covers_all_152_tensors():
    cfg = resolve_config(
        "configs/vit_tiny_tinyimagenet_residual_stream_symmetry_v1_"
        "tensor_monitoring.yaml"
    )
    model = VisionTransformer(cfg["model"])
    controlled, uncontrolled = param_groups.classify_parameters(model, POLICY)
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    assert len(trainable) == 152
    assert len(controlled) == 52
    assert len(uncontrolled) == 100
    assert set(controlled) | set(uncontrolled) == set(trainable)
    assert sum(parameter.numel() for parameter in trainable.values()) == 5_427_080
    assert Counter(map(tensor_family, trainable)) == {
        "input_embedding": 4,
        "attention_qkv": 24,
        "attention_output": 24,
        "mlp_input": 24,
        "mlp_output": 24,
        "block_layernorm": 48,
        "final_layernorm": 2,
        "classifier_head": 2,
    }


def test_two_step_monitor_uses_actual_lr_and_pre_update_norm(tmp_path):
    torch.manual_seed(11)
    model = _positive_model()
    optimizer, controlled, uncontrolled = param_groups.build_optimizer(
        model, POLICY, 1e-4, weight_decay=0.0
    )
    controlled_references = reference_norms_for(
        controlled, split_fused_qkv=True
    )
    monitor_references = {
        name: frobenius_norm(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    dataset = TensorDataset(
        torch.randn(8, 3, 16, 16),
        torch.randint(0, 4, (8,)),
    )
    sampler = EpochBatchSampler(len(dataset), 2, 17)
    factory = lambda current: DataLoader(dataset, batch_sampler=current)
    cfg = {
        "data": {"global_batch_size": 2, "micro_batch_size": 2},
        "optimizer": {
            "peak_lr": 1e-4,
            "final_lr": 1e-5,
            "warmup_steps": 1,
            "control_start_step": 0,
            "decay_start_step": 1,
            "total_steps": 2,
        },
        "control": {
            "cyclic_period_steps": 2,
            "cyclic_amplitude": 0.5,
            "linear_up_final": 2.0,
            "linear_down_final": 1 / 3,
            "projection_eps": 1e-12,
            "split_fused_qkv": True,
        },
        "logging": {
            "ema_beta": 0.99,
            "angular_interval": 1,
            "tensor_monitor_interval": 1,
            "probe_interval": 100,
            "val_interval": 100,
            "checkpoint_interval": 2,
        },
    }

    train_branch(
        cfg,
        "linear_up",
        model,
        optimizer,
        controlled,
        sampler,
        factory,
        "cpu",
        tmp_path,
        0,
        None,
        controlled_references,
        monitor_reference_norms=monitor_references,
    )

    frame = pd.read_csv(tmp_path / "tensor_metrics.csv")
    n_tensors = len(monitor_references)
    assert len(frame) == 2 * n_tensors
    assert frame.groupby("global_state_step").tensor.nunique().to_dict() == {
        1: n_tensors,
        2: n_tensors,
    }
    assert not frame.duplicated(
        ["branch", "tensor", "global_state_step"]
    ).any()
    assert set(frame[frame.is_controlled].tensor) == set(controlled)
    assert set(frame[~frame.is_controlled].tensor) == set(uncontrolled)

    expected_lr = np.where(
        frame.is_controlled,
        frame.base_lr * frame.schedule_ratio,
        frame.base_lr,
    )
    np.testing.assert_allclose(frame.actual_lr, expected_lr, rtol=1e-12)
    np.testing.assert_allclose(
        frame.lr_over_frobenius_norm,
        frame.actual_lr / frame.pre_update_frobenius_norm,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        frame.lr_over_rms,
        frame.lr_over_frobenius_norm * np.sqrt(frame.numel),
        rtol=1e-12,
    )

    selected = frame[frame.is_controlled]
    np.testing.assert_allclose(
        selected.post_projection_frobenius_norm,
        selected.target_post_frobenius_norm,
        rtol=1e-5,
    )
    assert selected.post_projection_relative_error.max() < 1e-5
    assert (
        selected.lr_over_frobenius_ratio_to_reference.sub(1).abs().max()
        < 1e-5
    )
    assert (
        selected.lr_over_rms_ratio_to_reference.sub(1).abs().max()
        < 1e-5
    )

    free = frame[~frame.is_controlled]
    np.testing.assert_allclose(
        free.post_optimizer_pre_projection_frobenius_norm,
        free.post_projection_frobenius_norm,
        rtol=0,
        atol=0,
    )
    head = frame[frame.tensor.isin({"head.weight", "head.bias"})]
    assert not head.is_controlled.any()
    np.testing.assert_allclose(head.actual_lr, head.base_lr, rtol=0, atol=0)


def _analysis_row(branch, tensor, controlled, step, reference, numel, q_now, q_next):
    base_lr = 1e-3
    pre = reference * (q_now if controlled else 1 + 0.01 * (step - 2500))
    post = reference * (q_next if controlled else 1 + 0.008 * (step - 2500))
    actual_lr = base_lr * (q_now if controlled else 1.0)
    pre_rms = pre / math.sqrt(numel)
    reference_rms = reference / math.sqrt(numel)
    target = reference * q_next if controlled else math.nan
    return {
        "update_start_state_step": step,
        "global_state_step": step + 1,
        "branch": branch,
        "tensor": tensor,
        "optimizer_group": "controlled" if controlled else "uncontrolled",
        "is_controlled": controlled,
        "ndim": 2 if tensor.endswith("weight") else 1,
        "numel": numel,
        "base_lr": base_lr,
        "actual_lr": actual_lr,
        "schedule_ratio": q_now,
        "next_schedule_ratio": q_next,
        "reference_frobenius_norm": reference,
        "pre_update_frobenius_norm": pre,
        "post_optimizer_pre_projection_frobenius_norm":
            post * (1.001 if controlled else 1.0),
        "post_projection_frobenius_norm": post,
        "post_projection_norm_ratio_to_reference": post / reference,
        "lr_over_frobenius_norm": actual_lr / pre,
        "lr_over_rms": actual_lr / pre_rms,
        "reference_lr_over_frobenius_norm": base_lr / reference,
        "reference_lr_over_rms": base_lr / reference_rms,
        "lr_over_frobenius_ratio_to_reference":
            (actual_lr / pre) / (base_lr / reference),
        "lr_over_rms_ratio_to_reference":
            (actual_lr / pre_rms) / (base_lr / reference_rms),
        "target_post_frobenius_norm": target,
        "post_projection_relative_error":
            abs(post - target) / target if controlled else math.nan,
    }


def test_analyzer_creates_every_tensor_plot_and_strict_audit(tmp_path):
    tensors = (
        ("blocks.0.attn.proj.weight", True, 2.0, 4),
        ("head.bias", False, 1.0, 2),
    )
    catalog = pd.DataFrame([
        {
            "tensor": tensor,
            "shape": "[2,2]" if tensor.endswith("weight") else "[2]",
            "ndim": 2 if tensor.endswith("weight") else 1,
            "numel": numel,
            "family": tensor_family(tensor),
            "controlled": controlled,
            "optimizer_group": "controlled" if controlled else "uncontrolled",
            "reference_fro_norm": reference,
            "reference_rms_norm": reference / math.sqrt(numel),
        }
        for tensor, controlled, reference, numel in tensors
    ])
    catalog.to_csv(tmp_path / "tensor_catalog.csv", index=False)

    for branch in ("constant", "linear_up", "linear_down", "cyclic"):
        rows = []
        for step, q_now, q_next in ((2500, 1.0, 1.1), (2520, 1.1, 1.2)):
            rows.extend(
                _analysis_row(
                    branch, tensor, controlled, step, reference, numel,
                    q_now, q_next,
                )
                for tensor, controlled, reference, numel in tensors
            )
        path = tmp_path / "branches" / branch
        path.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(path / "tensor_metrics.csv", index=False)

    # The local Windows Conda environment bundles independent OpenMP runtimes
    # in Torch and Matplotlib. Run this plotting-only check in a clean process;
    # production Linux does not need a workaround.
    command = (
        "from src.tensor_monitoring_analysis import analyze_tensor_monitoring; "
        f"analyze_tensor_monitoring({str(tmp_path)!r})"
    )
    subprocess.run([sys.executable, "-c", command], check=True)
    audit = json.loads(
        (tmp_path / "analysis" / "tensor_monitoring_audit.json")
        .read_text(encoding="utf-8")
    )
    assert audit["passed"] is True
    assert audit["n_tensors"] == 2
    assert audit["n_controlled_tensors"] == 1
    assert audit["n_uncontrolled_tensors"] == 1
    manifest = pd.read_csv(
        tmp_path / "analysis" / "tensor_monitoring" / "plot_manifest.csv"
    )
    assert set(manifest.tensor) == {item[0] for item in tensors}
    assert len(list(
        (tmp_path / "analysis" / "tensor_monitoring" / "per_tensor")
        .glob("*.png")
    )) == 2
    assert (
        tmp_path / "analysis" / "tensor_monitoring_audit.json"
    ).is_file()

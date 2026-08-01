from __future__ import annotations

import csv
from pathlib import Path

import torch

from src import param_groups
from src.model import ViTConfig, VisionTransformer
from src.policy_overlay_residual_stream_symmetry import (
    V2_POLICY,
    expanded_block_parameter_names,
    install,
)
from src.reference_alignment import build_constant_reference_norm_aligner


install()

from run_pipeline import resolve_config  # noqa: E402


CONFIG = (
    "configs/"
    "vit_tiny_tinyimagenet_expanded_block_affine_rmsnorm_finalnorm_v6_"
    "tensor_monitoring.yaml"
)


def _unit_cfg():
    return {
        "optimizer": {"control_start_step": 0, "total_steps": 1},
        "control": {
            "policy": V2_POLICY,
            "projection_eps": 1.0e-12,
            "schedules": ["constant", "linear_up", "linear_down", "cyclic"],
            "constant_reference_alignment": {
                "enabled": True,
                "tensor": "norm.weight",
                "reference_branch": "constant",
                "mode": "frobenius_norm",
                "tolerance": 1.0e-5,
            },
        },
    }


def _components():
    model = VisionTransformer(ViTConfig(norm_type="rmsnorm"))
    optimizer, controlled, uncontrolled = param_groups.build_optimizer(
        model, V2_POLICY, 1.0e-3)
    return model, optimizer, controlled, uncontrolled


def test_v6_keeps_exact_v5_control_scope_and_aligns_only_final_gain():
    cfg = resolve_config(CONFIG)
    model, _, controlled, uncontrolled = _components()

    assert cfg["model"]["norm_type"] == "rmsnorm"
    assert cfg["control"]["policy"] == V2_POLICY
    assert set(controlled) == expanded_block_parameter_names(model)
    assert len(controlled) == 100
    assert len(uncontrolled) == 27
    assert "norm.weight" in uncontrolled
    assert "head.weight" in uncontrolled
    assert "head.bias" in uncontrolled
    assert cfg["control"]["constant_reference_alignment"] == {
        "enabled": True,
        "tensor": "norm.weight",
        "reference_branch": "constant",
        "mode": "frobenius_norm",
        "tolerance": 1.0e-5,
    }


def test_nonconstant_final_gain_is_projected_to_constant_norm(tmp_path):
    cfg = _unit_cfg()
    root = tmp_path / "branches"

    ref_model, ref_optimizer, ref_controlled, _ = _components()
    ref_output = root / "constant"
    ref_aligner = build_constant_reference_norm_aligner(
        cfg,
        branch="constant",
        model=ref_model,
        optimizer=ref_optimizer,
        controlled=ref_controlled,
        output=ref_output,
        start_step=0,
    )
    ref_pre = ref_aligner.pre_update_norm()
    with torch.no_grad():
        ref_model.norm.weight.mul_(1.2)
    ref_row = ref_aligner.apply(
        update_start_step=0,
        base_lr=1.0e-3,
        schedule_ratio=1.0,
        pre_update_norm=ref_pre,
    )

    branch_model, branch_optimizer, branch_controlled, _ = _components()
    branch_output = root / "linear_up"
    branch_aligner = build_constant_reference_norm_aligner(
        cfg,
        branch="linear_up",
        model=branch_model,
        optimizer=branch_optimizer,
        controlled=branch_controlled,
        output=branch_output,
        start_step=0,
    )
    branch_pre = branch_aligner.pre_update_norm()
    with torch.no_grad():
        branch_model.norm.weight.mul_(0.7)
    branch_row = branch_aligner.apply(
        update_start_step=0,
        base_lr=1.0e-3,
        schedule_ratio=1.5,
        pre_update_norm=branch_pre,
    )

    assert ref_row["was_projected"] is False
    assert branch_row["was_projected"] is True
    assert branch_row["actual_lr"] == ref_row["actual_lr"] == 1.0e-3
    assert torch.isclose(
        torch.linalg.vector_norm(branch_model.norm.weight),
        torch.tensor(ref_row["post_alignment_frobenius_norm"]),
        rtol=1.0e-5,
    )
    assert branch_row["alignment_relative_error"] <= 1.0e-5

    with (branch_output / "reference_alignment_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["tensor"] == "norm.weight"
    assert rows[0]["reference_branch"] == "constant"


def test_v6_runner_installs_alignment_overlay_and_launcher_is_fail_closed():
    runner = Path(
        "run_pipeline_expanded_block_affine_rmsnorm_finalnorm_v6_aligned.py"
    ).read_text(encoding="utf-8")
    launcher = Path("go_v6.sh").read_text(encoding="utf-8")

    assert "install_into(run_pipeline)" in runner
    assert "reference_alignment_audit.json" in launcher
    assert "n_rows\": 70000" in launcher
    assert launcher.rfind("scripts/iteration_gate.py") > launcher.rfind(
        "reference_alignment_audit")


def test_v6_preflight_module_imports():
    from scripts import expanded_block_affine_rmsnorm_finalnorm_v6_preflight

    assert callable(expanded_block_affine_rmsnorm_finalnorm_v6_preflight.main)

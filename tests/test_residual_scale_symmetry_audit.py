from dataclasses import asdict

import pytest
import torch

from scripts.residual_scale_symmetry_audit import (
    apply_transformation,
    run_audit,
    transformation_parameter_names,
)
from src.model import ViTConfig, VisionTransformer


CFG = ViTConfig(
    image_size=16,
    patch_size=8,
    embed_dim=12,
    depth=2,
    num_heads=3,
    num_classes=4,
    norm_eps=0.0,
)


def test_transformation_scopes_are_exact():
    model = VisionTransformer(CFG)
    groups = transformation_parameter_names(model)

    assert "head.weight" in groups["raw_pos_head"]
    assert "head.weight" not in groups["no_head"]
    assert "patch_embed.proj.bias" not in groups["no_head"]
    assert set(groups["input_family"]) == {
        "patch_embed.proj.weight",
        "patch_embed.proj.bias",
        "cls_token",
        "pos_embed",
    }
    residual = set(groups["residual_stream_symmetry"])
    assert set(groups["input_family"]) < residual
    for index in range(CFG.depth):
        assert f"blocks.{index}.attn.proj.weight" in residual
        assert f"blocks.{index}.attn.proj.bias" in residual
        assert f"blocks.{index}.mlp.fc2.weight" in residual
        assert f"blocks.{index}.mlp.fc2.bias" in residual
        assert f"blocks.{index}.attn.qkv.weight" not in residual
        assert f"blocks.{index}.attn.qkv.bias" not in residual
        assert f"blocks.{index}.mlp.fc1.weight" not in residual
        assert f"blocks.{index}.norm1.weight" not in residual
    assert "head.weight" not in residual
    assert "head.bias" not in residual
    assert "norm.weight" not in residual
    assert "norm.bias" not in residual


def test_residual_stream_case_is_functionally_invariant_and_restores(
        tmp_path):
    torch.manual_seed(101)
    model = VisionTransformer(CFG)
    checkpoint = tmp_path / "prefix.pt"
    torch.save({
        "global_state_step": 7,
        "config": {"model": asdict(CFG)},
        "model": model.state_dict(),
    }, checkpoint)

    result = run_audit(
        checkpoint,
        device="cpu",
        seed=303,
        probe_size=6,
        batch_size=3,
        scales=(0.5, 2.0),
    )

    assert result["restoration_verified"]
    assert result["global_state_step"] == 7
    assert len(result["results"]) == 8
    rows = {
        (row["case"], row["q"]): row
        for row in result["results"]
    }
    for scale in (0.5, 2.0):
        row = rows[("residual_stream_symmetry", scale)]
        assert row["logits_difference"]["rms_relative"] < 2e-5
        assert row["features_difference"]["rms_relative"] < 2e-5
        assert row["absolute_ce_loss_diff"] < 2e-6
        assert row["input_tokens"]["rms_ratio"] == pytest.approx(
            scale, rel=2e-6)
        assert row["max_block_relative_error_to_q"] < 2e-5
        assert (
            row["transformation_check"]
            ["scaled_parameter_max_abs_error"] == 0.0)
        assert (
            row["transformation_check"]
            ["unscaled_state_max_abs_error"] == 0.0)

        input_only = rows[("input_family", scale)]
        assert input_only["input_tokens"]["rms_ratio"] == pytest.approx(
            scale, rel=2e-6)
        assert (
            input_only["max_block_relative_error_to_q"]
            > row["max_block_relative_error_to_q"])


def test_each_application_starts_from_explicitly_restored_state():
    torch.manual_seed(505)
    model = VisionTransformer(CFG)
    baseline = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    selected = apply_transformation(model, "raw_pos_head", 0.5)
    assert selected
    assert torch.equal(
        model.head.weight, baseline["head.weight"] * 0.5)

    model.load_state_dict(baseline, strict=True)
    apply_transformation(model, "no_head", 2.0)
    assert torch.equal(model.head.weight, baseline["head.weight"])
    assert torch.equal(
        model.pos_embed, baseline["pos_embed"] * 2.0)

from __future__ import annotations

import math

import pytest
import torch

from src import param_groups
from src.model import VisionTransformer
from src.norm_control import (
    control_units,
    frobenius_norm,
    project_controlled,
    reference_norms_for,
)
from src.policy_overlay_residual_stream_symmetry import (
    V2_POLICY,
    expanded_block_parameter_names,
    expected_policy_delta,
    install,
)


install()

from run_pipeline import resolve_config  # noqa: E402


def test_v2_scope_has_100_parameters_and_148_control_units():
    model = VisionTransformer({})
    controlled, uncontrolled = param_groups.classify_parameters(
        model, V2_POLICY
    )
    units = control_units(controlled, split_fused_qkv=True)

    assert set(controlled) == expanded_block_parameter_names(model)
    assert len(controlled) == 100
    assert len(uncontrolled) == 52
    assert len(units) == 148
    assert set(controlled).isdisjoint(uncontrolled)
    assert set(controlled) | set(uncontrolled) == {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    for name in (
        "patch_embed.proj.weight",
        "patch_embed.proj.bias",
        "cls_token",
        "pos_embed",
    ):
        assert name in controlled

    for block in range(12):
        for suffix in (
            "attn.proj.weight",
            "attn.proj.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ):
            name = f"blocks.{block}.{suffix}"
            assert name in controlled
            assert name in units

        for kind in ("weight", "bias"):
            parent = f"blocks.{block}.attn.qkv.{kind}"
            assert parent in controlled
            assert parent not in units
            for label in ("q", "k", "v"):
                assert f"{parent}::{label}" in units

        for suffix in (
            "norm1.weight",
            "norm1.bias",
            "norm2.weight",
            "norm2.bias",
        ):
            assert f"blocks.{block}.{suffix}" in uncontrolled

    for name in ("norm.weight", "norm.bias", "head.weight", "head.bias"):
        assert name in uncontrolled


def test_qkv_weight_and_bias_are_split_and_projected_independently():
    weight = torch.arange(1, 25, dtype=torch.float64).reshape(6, 4)
    bias = torch.arange(1, 7, dtype=torch.float64)
    controlled = {
        "blocks.0.attn.qkv.weight": weight,
        "blocks.0.attn.qkv.bias": bias,
    }
    units = control_units(controlled, split_fused_qkv=True)

    expected_names = {
        f"blocks.0.attn.qkv.{kind}::{label}"
        for kind in ("weight", "bias")
        for label in ("q", "k", "v")
    }
    assert set(units) == expected_names
    assert all(
        unit.untyped_storage().data_ptr()
        == controlled[name.rpartition("::")[0]].untyped_storage().data_ptr()
        for name, unit in units.items()
    )
    assert {
        name: tuple(unit.shape) for name, unit in units.items()
    } == {
        **{
            f"blocks.0.attn.qkv.weight::{label}": (2, 4)
            for label in ("q", "k", "v")
        },
        **{
            f"blocks.0.attn.qkv.bias::{label}": (2,)
            for label in ("q", "k", "v")
        },
    }

    references = reference_norms_for(
        controlled, split_fused_qkv=True
    )
    for index, unit in enumerate(units.values(), start=1):
        unit.mul_(0.25 * index)
    directions_before_projection = {
        name: unit.detach().clone() for name, unit in units.items()
    }

    ratio = 1.7
    mean_error, max_error = project_controlled(
        controlled,
        references,
        ratio,
        split_fused_qkv=True,
    )

    assert mean_error < 2e-6
    assert max_error < 2e-6
    for name, unit in units.items():
        assert frobenius_norm(unit) == pytest.approx(
            ratio * references[name], rel=2e-6
        )
        cosine = torch.nn.functional.cosine_similarity(
            directions_before_projection[name].flatten(),
            unit.flatten(),
            dim=0,
        )
        assert cosine.item() == pytest.approx(1.0, abs=1e-12)


def test_v2_policy_delta_from_hidden_matrices_is_exactly_52_and_zero():
    model = VisionTransformer({})
    hidden, _ = param_groups.classify_parameters(
        model, "hidden_matrices"
    )
    controlled, _ = param_groups.classify_parameters(model, V2_POLICY)
    additions, removals = expected_policy_delta(model, V2_POLICY)

    expected_additions = {
        "patch_embed.proj.weight",
        "patch_embed.proj.bias",
        "cls_token",
        "pos_embed",
    }
    expected_additions.update(
        f"blocks.{block}.{family}.bias"
        for block in range(12)
        for family in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
    )

    assert additions == expected_additions
    assert removals == set()
    assert additions == set(controlled) - set(hidden)
    assert removals == set(hidden) - set(controlled)
    assert len(additions) == 52


def test_v2_monitoring_config_preserves_the_experiment_contract():
    cfg = resolve_config(
        "configs/"
        "vit_tiny_tinyimagenet_expanded_block_affine_v2_tensor_monitoring.yaml"
    )

    assert "_base_" not in cfg
    assert cfg["control"]["policy"] == V2_POLICY
    assert cfg["control"]["split_fused_qkv"] is True
    assert cfg["control"]["schedules"] == [
        "constant",
        "linear_up",
        "linear_down",
        "cyclic",
    ]
    assert cfg["optimizer"]["peak_lr"] == pytest.approx(6e-4)
    assert cfg["optimizer"]["control_start_step"] == 2500
    assert cfg["optimizer"]["total_steps"] == 20_000
    assert cfg["model"]["depth"] == 12
    assert cfg["model"]["embed_dim"] == 192
    assert cfg["model"]["qkv_bias"] is True
    assert cfg["data"]["global_batch_size"] == 128
    assert cfg["logging"]["tensor_monitor_interval"] == 20
    assert cfg["logging"]["control_unit_monitoring"] is True
    assert cfg["experiment"]["seed"] == 20260726
    assert cfg["experiment"]["data_seed"] == 1729
    assert cfg["experiment"]["deterministic"] is True
    assert math.isclose(
        cfg["control"]["linear_down_final"], 1 / 3, rel_tol=0, abs_tol=1e-15
    )

from __future__ import annotations

import pytest
import torch
from torch import nn

from src import param_groups
from src.model import RMSNorm, ViTConfig, VisionTransformer
from src.norm_control import control_units
from src.policy_overlay_residual_stream_symmetry import V2_POLICY, install


install()

from run_pipeline import resolve_config  # noqa: E402


def test_rmsnorm_matches_the_declared_formula_and_has_no_bias():
    layer = RMSNorm(3, eps=1e-6)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([1.0, 2.0, 3.0]))
    inputs = torch.tensor([[1.0, -2.0, 4.0], [0.5, 1.5, -3.0]])
    expected = (
        inputs
        * torch.rsqrt(inputs.square().mean(dim=-1, keepdim=True) + 1e-6)
        * layer.weight
    )
    assert torch.allclose(layer(inputs), expected)
    assert list(dict(layer.named_parameters())) == ["weight"]


def test_v5_has_25_rmsnorms_no_layernorms_and_expected_model_size():
    model = VisionTransformer(ViTConfig(norm_type="rmsnorm"))
    rmsnorms = [module for module in model.modules() if isinstance(module, RMSNorm)]
    assert len(rmsnorms) == 25
    assert not any(isinstance(module, nn.LayerNorm) for module in model.modules())

    parameters = dict(model.named_parameters())
    assert len(parameters) == 127
    assert sum(parameter.numel() for parameter in parameters.values()) == 5_422_280
    assert not any(
        name.endswith(".bias") and (".norm1." in name or ".norm2." in name or name.startswith("norm."))
        for name in parameters
    )
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 200)


def test_v5_preserves_v2_control_scope_but_has_27_uncontrolled_tensors():
    model = VisionTransformer(ViTConfig(norm_type="rmsnorm"))
    controlled, uncontrolled = param_groups.classify_parameters(model, V2_POLICY)
    units = control_units(controlled, split_fused_qkv=True)

    assert len(controlled) == 100
    assert len(uncontrolled) == 27
    assert len(units) == 148
    assert all(name.endswith(".weight") for name in uncontrolled if "norm" in name)
    assert "norm.weight" in uncontrolled
    assert all(f"blocks.{block}.norm1.weight" in uncontrolled for block in range(12))
    assert all(f"blocks.{block}.norm2.weight" in uncontrolled for block in range(12))
    assert not any("norm" in name and name.endswith(".bias") for name in controlled | uncontrolled)


def test_v5_configs_use_a_new_rmsnorm_prefix_and_the_v2_policy():
    prefix = resolve_config("configs/vit_tiny_tinyimagenet_rmsnorm_v5.yaml")
    target = resolve_config(
        "configs/vit_tiny_tinyimagenet_expanded_block_affine_rmsnorm_v5_tensor_monitoring.yaml"
    )
    assert prefix["model"]["norm_type"] == "rmsnorm"
    assert prefix["control"]["policy"] == "hidden_matrices"
    assert target["model"]["norm_type"] == "rmsnorm"
    assert target["control"]["policy"] == V2_POLICY
    assert target["control"]["split_fused_qkv"] is True
    assert prefix["experiment"]["output_dir"] == target["experiment"]["output_dir"]
    assert target["logging"]["tensor_monitor_interval"] == 20
    assert target["logging"]["control_unit_monitoring"] is True


def test_layernorm_state_dict_cannot_be_reused_as_the_v5_prefix():
    layernorm_model = VisionTransformer(ViTConfig(norm_type="layernorm"))
    rmsnorm_model = VisionTransformer(ViTConfig(norm_type="rmsnorm"))
    with pytest.raises(RuntimeError):
        rmsnorm_model.load_state_dict(layernorm_model.state_dict(), strict=True)


def test_unknown_norm_type_is_rejected():
    with pytest.raises(ValueError, match="unsupported norm_type"):
        VisionTransformer(ViTConfig(norm_type="not-a-norm"))

def test_v5_preflight_imports_torch_nn_namespace():
    from scripts import expanded_block_affine_rmsnorm_v5_preflight as preflight

    assert preflight.nn is nn

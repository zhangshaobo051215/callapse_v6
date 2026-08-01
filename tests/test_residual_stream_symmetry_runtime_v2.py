import copy

import pytest
import torch
from torch.nn import functional as F

from src import optimizer_migration, param_groups
from src.model import ViTConfig, VisionTransformer
from src.norm_control import (
    control_units,
    frobenius_norm,
    project_controlled,
    reference_norms_for,
    resolve_reference_norms,
)
from src.policy_overlay_residual_stream_symmetry import (
    POLICY,
    expected_policy_delta,
    install,
    residual_stream_parameter_names,
)


CFG = ViTConfig(
    image_size=16,
    patch_size=8,
    embed_dim=12,
    depth=2,
    num_heads=3,
    num_classes=4,
    norm_eps=0.0,
)


def _optimizer_step(model, optimizer, inputs, targets):
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(inputs), targets)
    loss.backward()
    optimizer.step()
    return loss.item()


def _assert_optimizer_states_equal(
        left, left_model, right, right_model):
    left_parameters = dict(left_model.named_parameters())
    right_parameters = dict(right_model.named_parameters())
    assert set(left_parameters) == set(right_parameters)
    for name in left_parameters:
        left_state = left.state[left_parameters[name]]
        right_state = right.state[right_parameters[name]]
        assert set(left_state) == set(right_state)
        for key in left_state:
            if isinstance(left_state[key], torch.Tensor):
                assert torch.equal(left_state[key], right_state[key])
            else:
                assert left_state[key] == right_state[key]


def test_runtime_v2_scope_is_exact():
    install()
    model = VisionTransformer({})
    controlled, uncontrolled = param_groups.classify_parameters(
        model, POLICY)
    units = control_units(controlled, split_fused_qkv=True)

    assert set(controlled) == residual_stream_parameter_names(model)
    assert len(controlled) == 52
    assert len(uncontrolled) == 100
    assert len(units) == 52
    assert set(units) == set(controlled)
    for index in range(12):
        for suffix in (
                "attn.proj.weight", "attn.proj.bias",
                "mlp.fc2.weight", "mlp.fc2.bias"):
            assert f"blocks.{index}.{suffix}" in controlled
        for suffix in (
                "attn.qkv.weight", "attn.qkv.bias",
                "mlp.fc1.weight", "mlp.fc1.bias",
                "norm1.weight", "norm1.bias",
                "norm2.weight", "norm2.bias"):
            assert f"blocks.{index}.{suffix}" in uncontrolled
    for name in (
            "patch_embed.proj.weight", "patch_embed.proj.bias",
            "cls_token", "pos_embed"):
        assert name in controlled
    assert "head.weight" in uncontrolled
    assert "head.bias" in uncontrolled
    assert "norm.weight" in uncontrolled
    assert "norm.bias" in uncontrolled


@pytest.mark.parametrize("ratio", (0.5, 2.0))
def test_runtime_v2_projection_realizes_forward_symmetry(ratio):
    install()
    torch.manual_seed(607)
    model = VisionTransformer(CFG)
    controlled, uncontrolled = param_groups.classify_parameters(
        model, POLICY)
    with torch.no_grad():
        for parameter in controlled.values():
            if torch.count_nonzero(parameter) == 0:
                parameter.fill_(0.01)
    baseline_state = copy.deepcopy(model.state_dict())
    references = reference_norms_for(
        controlled, split_fused_qkv=True)
    inputs = torch.randn(5, 3, 16, 16)
    model.eval()
    with torch.no_grad():
        baseline_logits = model(inputs)

    project_controlled(
        controlled,
        references,
        ratio,
        split_fused_qkv=True,
    )
    for name, parameter in controlled.items():
        expected = ratio * references[name]
        relative_error = abs(
            frobenius_norm(parameter) - expected) / expected
        assert relative_error < 2e-6
    for name, parameter in uncontrolled.items():
        assert torch.equal(parameter, baseline_state[name])
    with torch.no_grad():
        logits = model(inputs)
    relative_rms = (
        (logits - baseline_logits).double().square().mean().sqrt()
        / baseline_logits.double().square().mean().sqrt()
    ).item()
    assert relative_rms < 3e-5


def test_runtime_v2_prefix_reference_and_adam_migration_are_lossless():
    install()
    torch.manual_seed(701)
    source_model = VisionTransformer(CFG)
    hidden, _ = param_groups.classify_parameters(
        source_model, "hidden_matrices")
    target, _ = param_groups.classify_parameters(source_model, POLICY)
    additions, removals = expected_policy_delta(source_model)
    assert set(target) - set(hidden) == additions
    assert set(hidden) - set(target) == removals

    legacy_references = reference_norms_for(
        hidden, split_fused_qkv=False)
    upgraded = resolve_reference_norms(
        target,
        legacy_references,
        split_fused_qkv=True,
        allow_legacy_qkv=True,
        allow_prefix_upgrade=True,
    )
    assert set(upgraded) == set(target)

    source_optimizer, _, _ = param_groups.build_optimizer(
        source_model, "hidden_matrices", 1e-3, weight_decay=0)
    inputs = torch.randn(3, 3, 16, 16)
    targets = torch.randint(0, 4, (3,))
    _optimizer_step(source_model, source_optimizer, inputs, targets)

    migrated_model = VisionTransformer(CFG)
    migrated_model.load_state_dict(
        copy.deepcopy(source_model.state_dict()))
    migrated, controlled, uncontrolled = (
        optimizer_migration.rebuild_optimizer_with_policy(
            migrated_model,
            copy.deepcopy(source_optimizer.state_dict()),
            "hidden_matrices",
            POLICY,
            1e-3,
            weight_decay=0,
        )
    )
    assert set(controlled) == set(target)
    assert set(uncontrolled) == (
        set(dict(migrated_model.named_parameters())) - set(target))
    _assert_optimizer_states_equal(
        source_optimizer, source_model, migrated, migrated_model)


def test_runtime_v2_target_policy_resume_reproduces_updates_bitwise():
    install()
    torch.manual_seed(809)
    initial = VisionTransformer(CFG)
    source_optimizer, _, _ = param_groups.build_optimizer(
        initial, "hidden_matrices", 7e-4, weight_decay=0)
    prefix_inputs = torch.randn(4, 3, 16, 16)
    prefix_targets = torch.randint(0, 4, (4,))
    _optimizer_step(
        initial, source_optimizer, prefix_inputs, prefix_targets)

    branch_model = VisionTransformer(CFG)
    branch_model.load_state_dict(copy.deepcopy(initial.state_dict()))
    branch_optimizer, controlled, _ = (
        optimizer_migration.rebuild_optimizer_with_policy(
            branch_model,
            copy.deepcopy(source_optimizer.state_dict()),
            "hidden_matrices",
            POLICY,
            7e-4,
            weight_decay=0,
        )
    )
    references = reference_norms_for(
        controlled, split_fused_qkv=True)
    saved_model = copy.deepcopy(branch_model.state_dict())
    saved_optimizer = copy.deepcopy(branch_optimizer.state_dict())
    batches = [
        (torch.randn(3, 3, 16, 16), torch.randint(0, 4, (3,)))
        for _ in range(3)
    ]

    def continue_from_saved():
        model = VisionTransformer(CFG)
        model.load_state_dict(copy.deepcopy(saved_model))
        optimizer, resumed_controlled, _ = param_groups.build_optimizer(
            model, POLICY, 7e-4, weight_decay=0)
        optimizer.load_state_dict(copy.deepcopy(saved_optimizer))
        losses = []
        for inputs, targets in batches:
            for group in optimizer.param_groups:
                group["lr"] = (
                    3.5e-4
                    if group.get("name") == "controlled"
                    else 7e-4
                )
            losses.append(
                _optimizer_step(model, optimizer, inputs, targets))
            project_controlled(
                resumed_controlled,
                references,
                0.5,
                split_fused_qkv=True,
            )
        return losses, model, optimizer

    left_losses, left_model, left_optimizer = continue_from_saved()
    right_losses, right_model, right_optimizer = continue_from_saved()
    assert left_losses == right_losses
    for name, value in left_model.state_dict().items():
        assert torch.equal(value, right_model.state_dict()[name])
    _assert_optimizer_states_equal(
        left_optimizer, left_model, right_optimizer, right_model)

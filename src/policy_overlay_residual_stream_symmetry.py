from __future__ import annotations

import copy
import re

from torch import nn

from . import optimizer_migration, param_groups


POLICY = "residual_stream_symmetry_v1"
V2_POLICY = "expanded_block_affine_v2"
_INPUT_FAMILY = {
    "patch_embed.proj.weight",
    "patch_embed.proj.bias",
    "cls_token",
    "pos_embed",
}
_RESIDUAL_OUTPUT = re.compile(
    r"^blocks\.\d+\.(attn\.proj|mlp\.fc2)\.(weight|bias)$")
_EXPANDED_BLOCK_AFFINE = re.compile(
    r"^blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(fc1|fc2))\.(weight|bias)$")

_ORIGINAL_CLASSIFY = param_groups.classify_parameters
_ORIGINAL_REBUILD = optimizer_migration.rebuild_optimizer_with_policy
_INSTALLED = False


def residual_stream_parameter_names(model: nn.Module) -> set[str]:
    return {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (name in _INPUT_FAMILY or _RESIDUAL_OUTPUT.fullmatch(name))
    }


def expanded_block_parameter_names(model: nn.Module) -> set[str]:
    return {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (name in _INPUT_FAMILY or _EXPANDED_BLOCK_AFFINE.fullmatch(name))
    }


def expected_policy_delta(
        model: nn.Module, policy: str = POLICY) -> tuple[set[str], set[str]]:
    trainable = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    missing = set(_INPUT_FAMILY) - trainable
    if missing:
        raise ValueError(
            f"model is missing required input parameters: {sorted(missing)}")
    if policy == POLICY:
        target = residual_stream_parameter_names(model)
    elif policy == V2_POLICY:
        target = expanded_block_parameter_names(model)
    else:
        raise ValueError(f"unsupported overlay policy: {policy}")
    source_controlled, _ = _ORIGINAL_CLASSIFY(
        model, "hidden_matrices")
    return target - set(source_controlled), set(source_controlled) - target


def classify_parameters(model: nn.Module, policy: str = "hidden_matrices"):
    """Add experiment overlays without changing the core policy definitions."""
    if policy == POLICY:
        selected = residual_stream_parameter_names(model)
    elif policy == V2_POLICY:
        selected = expanded_block_parameter_names(model)
    else:
        return _ORIGINAL_CLASSIFY(model, policy)
    controlled, uncontrolled = {}, {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (controlled if name in selected else uncontrolled)[name] = parameter

    names = set(controlled) | set(uncontrolled)
    expected = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if names != expected or set(controlled) & set(uncontrolled):
        raise RuntimeError(
            "residual-stream trainable classification is not a partition")
    return controlled, uncontrolled


def rebuild_optimizer_with_policy(
    model: nn.Module,
    optimizer_state: dict,
    source_policy: str,
    target_policy: str,
    lr: float,
    betas=(0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
):
    """Losslessly regroup a hidden-matrices prefix for this intervention."""
    if target_policy not in {POLICY, V2_POLICY}:
        return _ORIGINAL_REBUILD(
            model,
            optimizer_state,
            source_policy,
            target_policy,
            lr,
            betas,
            eps,
            weight_decay,
        )
    if source_policy != "hidden_matrices":
        raise ValueError(
            f"unsupported optimizer policy migration: "
            f"{source_policy} -> {target_policy}")

    source, source_controlled, _ = param_groups.build_optimizer(
        model, source_policy, lr, betas, eps, weight_decay)
    source.load_state_dict(optimizer_state)
    target, controlled, uncontrolled = param_groups.build_optimizer(
        model, target_policy, lr, betas, eps, weight_decay)

    added = set(controlled) - set(source_controlled)
    removed = set(source_controlled) - set(controlled)
    expected_additions, expected_removals = expected_policy_delta(model, target_policy)
    if added != expected_additions or removed != expected_removals:
        raise ValueError(
            f"unexpected {target_policy} policy delta; "
            f"added={sorted(added)}, removed={sorted(removed)}")

    source_lrs = {float(group["lr"]) for group in source.param_groups}
    if len(source_lrs) != 1:
        raise ValueError(
            "prefix optimizer groups must have an identical learning rate")
    source_lr = source_lrs.pop()
    for group in target.param_groups:
        group["lr"] = source_lr

    missing_state = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter not in source.state
    ]
    if missing_state:
        raise ValueError(
            f"prefix optimizer state missing parameters: {missing_state}")
    for parameter, state in source.state.items():
        target.state[parameter] = copy.deepcopy(state)
    if len(target.state) != len(source.state):
        raise RuntimeError("optimizer state migration lost parameters")
    return target, controlled, uncontrolled


def install() -> None:
    """Install the overlay before importing run_pipeline."""
    global _INSTALLED
    if _INSTALLED:
        return
    param_groups.classify_parameters = classify_parameters
    optimizer_migration.rebuild_optimizer_with_policy = (
        rebuild_optimizer_with_policy)
    _INSTALLED = True

from __future__ import annotations

import copy

from torch import nn

from .param_groups import build_optimizer


def rebuild_optimizer_with_policy(model: nn.Module, optimizer_state: dict,
                                  source_policy: str, target_policy: str, lr: float,
                                  betas=(0.9, 0.95), eps: float = 1e-8,
                                  weight_decay: float = 0.0):
    """Preserve every parameter's Adam state while changing group membership."""
    expected_additions = {
        "all_2d": {"head.weight"},
        "all_matrix_like": {"head.weight", "patch_embed.proj.weight"},
        "all_matrix_like_pos_embed": {
            "head.weight",
            "patch_embed.proj.weight",
            "pos_embed",
        },
    }
    if source_policy != "hidden_matrices" or target_policy not in expected_additions:
        raise ValueError(
            f"unsupported optimizer policy migration: {source_policy} -> {target_policy}")

    source, source_controlled, _ = build_optimizer(
        model, source_policy, lr, betas, eps, weight_decay)
    source.load_state_dict(optimizer_state)
    target, controlled, uncontrolled = build_optimizer(
        model, target_policy, lr, betas, eps, weight_decay)

    added = set(controlled) - set(source_controlled)
    removed = set(source_controlled) - set(controlled)
    if added != expected_additions[target_policy] or removed:
        raise ValueError(
            f"unexpected policy delta; added={sorted(added)}, removed={sorted(removed)}")

    source_lrs = {float(group["lr"]) for group in source.param_groups}
    if len(source_lrs) != 1:
        raise ValueError("prefix optimizer groups must have an identical learning rate")
    source_lr = source_lrs.pop()
    for group in target.param_groups:
        group["lr"] = source_lr

    missing = [
        name for name, param in model.named_parameters()
        if param.requires_grad and param not in source.state
    ]
    if missing:
        raise ValueError(f"prefix optimizer state missing parameters: {missing}")
    for param, state in source.state.items():
        target.state[param] = copy.deepcopy(state)
    if len(target.state) != len(source.state):
        raise RuntimeError("optimizer state migration lost parameters")
    return target, controlled, uncontrolled

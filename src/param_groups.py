from __future__ import annotations

import re
from typing import Iterable

import torch
from torch import nn

_HIDDEN = re.compile(r"^blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(fc1|fc2))\.weight$")


def classify_parameters(model: nn.Module, policy: str = "hidden_matrices"):
    controlled, uncontrolled = {}, {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if policy == "hidden_matrices":
            selected = bool(_HIDDEN.fullmatch(name))
        elif policy == "all_2d":
            selected = param.ndim == 2
        elif policy == "all_matrix_like":
            selected = param.ndim == 2 or name == "patch_embed.proj.weight"
        elif policy == "all_matrix_like_pos_embed":
            selected = (
                param.ndim == 2
                or name in {"patch_embed.proj.weight", "pos_embed"}
            )
        else:
            selected = None
        if selected is None:
            raise ValueError(f"unknown control policy: {policy}")
        (controlled if selected else uncontrolled)[name] = param
    names = set(controlled) | set(uncontrolled)
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    if names != expected or set(controlled) & set(uncontrolled):
        raise RuntimeError("trainable parameter classification is not a partition")
    return controlled, uncontrolled


def parameter_audit(model: nn.Module, policy: str) -> dict:
    controlled, uncontrolled = classify_parameters(model, policy)
    def rows(group):
        return [{"name": n, "shape": list(p.shape), "numel": p.numel()} for n, p in group.items()]
    c_num = sum(p.numel() for p in controlled.values())
    u_num = sum(p.numel() for p in uncontrolled.values())
    return {
        "policy": policy, "controlled": rows(controlled), "uncontrolled": rows(uncontrolled),
        "controlled_numel": c_num, "uncontrolled_numel": u_num,
        "controlled_fraction": c_num / (c_num + u_num),
    }


def build_optimizer(model: nn.Module, policy: str, lr: float, betas=(0.9, 0.95),
                    eps: float = 1e-8, weight_decay: float = 0.0):
    controlled, uncontrolled = classify_parameters(model, policy)
    optimizer = torch.optim.AdamW(
        [{"params": list(controlled.values()), "name": "controlled"},
         {"params": list(uncontrolled.values()), "name": "uncontrolled"}],
        lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay,
    )
    return optimizer, controlled, uncontrolled

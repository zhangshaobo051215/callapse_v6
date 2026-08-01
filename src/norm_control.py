from __future__ import annotations

import math

import torch


_QKV_SUFFIXES = (".attn.qkv.weight", ".attn.qkv.bias")
_QKV_UNIT_LABELS = ("q", "k", "v")


def frobenius_norm(param: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(param.detach().float()).item())


def control_units(controlled, *, split_fused_qkv: bool = False):
    """Return the independently norm-controlled tensor views."""
    units = {}
    for name, param in controlled.items():
        if split_fused_qkv and name.endswith(_QKV_SUFFIXES):
            if param.ndim not in (1, 2) or param.shape[0] % 3:
                raise ValueError(f"invalid fused QKV shape for {name}: {tuple(param.shape)}")
            rows = param.shape[0] // 3
            for index, label in enumerate(_QKV_UNIT_LABELS):
                units[f"{name}::{label}"] = param.narrow(0, index * rows, rows)
        else:
            units[name] = param
    return units


def reference_norms_for(controlled, *, split_fused_qkv: bool = False):
    units = control_units(controlled, split_fused_qkv=split_fused_qkv)
    return {name: frobenius_norm(unit) for name, unit in units.items()}


def resolve_reference_norms(controlled, stored, *, split_fused_qkv: bool = False,
                            allow_legacy_qkv: bool = False,
                            allow_prefix_upgrade: bool = False):
    """Validate references or safely upgrade a ratio-one prefix checkpoint."""
    units = control_units(controlled, split_fused_qkv=split_fused_qkv)
    expected = set(units)
    if set(stored) == expected:
        return {name: float(stored[name]) for name in units}
    if not (allow_legacy_qkv or allow_prefix_upgrade):
        missing = sorted(expected - set(stored))
        extra = sorted(set(stored) - expected)
        raise ValueError(
            f"reference norm keys do not match control units; missing={missing}, extra={extra}")

    resolved = {}
    for name, unit in units.items():
        if name in stored:
            resolved[name] = float(stored[name])
            continue
        parent, separator, label = name.rpartition("::")
        is_legacy_qkv = (separator and label in _QKV_UNIT_LABELS
                         and parent in stored and allow_legacy_qkv)
        if not (is_legacy_qkv or allow_prefix_upgrade):
            raise ValueError(f"cannot upgrade prefix reference norm for {name}")
        resolved[name] = frobenius_norm(unit)
    return resolved


@torch.no_grad()
def project_frobenius_(param: torch.Tensor, target: float, eps: float = 1e-12) -> float:
    norm = torch.linalg.vector_norm(param.float())
    if not torch.isfinite(norm) or norm.item() <= eps:
        raise ValueError("cannot project a zero or non-finite tensor")
    param.mul_(target / norm.to(dtype=param.dtype))
    return abs(frobenius_norm(param) - target) / max(abs(target), eps)


def rms_from_norm(norm: float, numel: int) -> float:
    return norm / math.sqrt(numel)


@torch.no_grad()
def project_controlled(controlled, reference_norms, ratio: float, eps: float = 1e-12,
                       *, split_fused_qkv: bool = False):
    units = control_units(controlled, split_fused_qkv=split_fused_qkv)
    names = list(units)
    if not names:
        raise ValueError("no controlled tensors to project")
    norms = torch.stack([
        torch.linalg.vector_norm(units[name].float())
        for name in names
    ])
    valid = torch.isfinite(norms) & (norms > eps)
    if not bool(valid.all().item()):
        invalid = [
            name for name, ok in zip(names, valid.detach().cpu().tolist())
            if not ok
        ]
        raise ValueError(f"cannot project zero or non-finite tensors: {invalid}")
    targets = norms.new_tensor([
        ratio * float(reference_norms[name]) for name in names
    ])
    target_valid = torch.isfinite(targets) & (targets > eps)
    if not bool(target_valid.all().item()):
        raise ValueError("projection targets must be positive and finite")
    for index, name in enumerate(names):
        unit = units[name]
        scale = (targets[index] / norms[index]).to(dtype=unit.dtype)
        unit.mul_(scale)
    post_norms = torch.stack([
        torch.linalg.vector_norm(units[name].float())
        for name in names
    ])
    errors = (post_norms - targets).abs() / targets.abs().clamp_min(eps)
    return float(errors.mean().item()), float(errors.max().item())


def angular_steps(before, controlled):
    values = {}
    with torch.no_grad():
        for name, current in controlled.items():
            old = before[name].float().flatten()
            new = current.detach().float().flatten()
            cosine = torch.dot(old, new) / (old.norm() * new.norm())
            values[name] = float(torch.acos(cosine.clamp(-1, 1)).item())
    return values


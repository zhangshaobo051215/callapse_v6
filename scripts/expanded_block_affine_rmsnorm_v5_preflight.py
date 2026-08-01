from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import optimizer_migration, param_groups  # noqa: E402
from src.model import RMSNorm, VisionTransformer  # noqa: E402
from src.norm_control import (  # noqa: E402
    control_units,
    resolve_reference_norms,
)
from src.policy_overlay_residual_stream_symmetry import (  # noqa: E402
    V2_POLICY as POLICY,
    expanded_block_parameter_names,
    expected_policy_delta,
    install,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_entries_equal(left: dict, right: dict) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, torch.Tensor):
            if not isinstance(right_value, torch.Tensor):
                return False
            if not torch.equal(left_value, right_value):
                return False
        elif left_value != right_value:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    expected_sha = args.expected_sha256.strip().lower()
    if len(expected_sha) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha
    ):
        raise ValueError("--expected-sha256 must be 64 hex digits")
    actual_sha = sha256(args.checkpoint)
    if actual_sha != expected_sha:
        raise ValueError(
            f"shared prefix SHA256 mismatch: "
            f"actual={actual_sha}, expected={expected_sha}"
        )

    install()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["model"].get("norm_type") != "rmsnorm":
        raise ValueError(
            f"expected RMSNorm checkpoint, got {config['model'].get('norm_type')}"
        )
    if int(checkpoint["global_state_step"]) != int(
        config["optimizer"]["control_start_step"]
    ):
        raise ValueError("checkpoint is not the shared control-start prefix")
    source_policy = config["control"]["policy"]
    if source_policy != "hidden_matrices":
        raise ValueError(f"unexpected shared-prefix policy: {source_policy}")

    model = VisionTransformer(config["model"])
    norm_modules = [module for module in model.modules() if isinstance(module, RMSNorm)]
    expected_norm_modules = 2 * int(config["model"]["depth"]) + 1
    if len(norm_modules) != expected_norm_modules:
        raise ValueError(
            f"expected {expected_norm_modules} RMSNorm modules, got {len(norm_modules)}"
        )
    if any(isinstance(module, nn.LayerNorm) for module in model.modules()):
        raise ValueError("V5 must not contain LayerNorm modules")

    model.load_state_dict(checkpoint["model"], strict=True)
    controlled, uncontrolled = param_groups.classify_parameters(model, POLICY)
    units = control_units(controlled, split_fused_qkv=True)
    hidden, _ = param_groups.classify_parameters(model, "hidden_matrices")
    additions = set(controlled) - set(hidden)
    removals = set(hidden) - set(controlled)
    expected_additions, expected_removals = expected_policy_delta(model, POLICY)

    if set(controlled) != expanded_block_parameter_names(model):
        raise ValueError("controlled scope differs from expanded-block audit")
    if len(controlled) != 100 or len(uncontrolled) != 27:
        raise ValueError(
            f"expected controlled/uncontrolled=100/27, got "
            f"{len(controlled)}/{len(uncontrolled)}"
        )
    trainable_numel = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_numel != 5_422_280:
        raise ValueError(f"expected 5,422,280 trainable parameters, got {trainable_numel}")
    norm_biases = [name for name, _ in model.named_parameters() if "norm" in name and name.endswith(".bias")]
    if norm_biases:
        raise ValueError(f"RMSNorm V5 unexpectedly has norm biases: {norm_biases}")
    if len(units) != 148:
        raise ValueError(f"expected 148 independent control units, got {len(units)}")
    if additions != expected_additions or removals != expected_removals:
        raise ValueError(
            f"unexpected policy delta; additions={sorted(additions)}, "
            f"removals={sorted(removals)}"
        )
    if len(additions) != 52 or len(removals) != 0:
        raise ValueError(
            f"expected additions/removals=52/0, got {len(additions)}/{len(removals)}"
        )
    forbidden = {
        name for name in controlled if "norm" in name or name.startswith("head.")
    }
    if forbidden:
        raise ValueError(f"forbidden parameters are controlled: {sorted(forbidden)}")

    expected_qkv_units = {
        f"blocks.{index}.attn.qkv.{kind}::{label}"
        for index in range(int(config["model"]["depth"]))
        for kind in ("weight", "bias")
        for label in ("q", "k", "v")
    }
    observed_qkv_units = {name for name in units if ".attn.qkv." in name}
    if observed_qkv_units != expected_qkv_units:
        raise ValueError("Q/K/V weight and bias units are not split exactly")

    references = resolve_reference_norms(
        controlled,
        checkpoint["reference_norms"],
        split_fused_qkv=True,
        allow_legacy_qkv=True,
        allow_prefix_upgrade=True,
    )
    invalid_references = {
        name: value
        for name, value in references.items()
        if not math.isfinite(value) or value <= 1e-12
    }
    if invalid_references:
        raise ValueError(f"invalid reference norms: {invalid_references}")

    optimizer_cfg = config["optimizer"]
    optimizer_args = (
        optimizer_cfg["peak_lr"],
        optimizer_cfg["betas"],
        optimizer_cfg["eps"],
        optimizer_cfg["weight_decay"],
    )
    source_optimizer, _, _ = param_groups.build_optimizer(
        model, source_policy, *optimizer_args
    )
    source_optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    migrated, migrated_controlled, migrated_uncontrolled = (
        optimizer_migration.rebuild_optimizer_with_policy(
            model,
            copy.deepcopy(checkpoint["optimizer"]),
            source_policy,
            POLICY,
            *optimizer_args,
        )
    )
    if set(migrated_controlled) != set(controlled):
        raise ValueError("migrated controlled set differs from audit")
    if set(migrated_uncontrolled) != set(uncontrolled):
        raise ValueError("migrated uncontrolled set differs from audit")

    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if len(migrated.state) != len(trainable):
        raise ValueError(
            f"optimizer state count {len(migrated.state)} "
            f"!= trainable parameter count {len(trainable)}"
        )
    unequal_states = [
        name
        for name, parameter in trainable.items()
        if not _state_entries_equal(
            source_optimizer.state[parameter],
            migrated.state[parameter],
        )
    ]
    if unequal_states:
        raise ValueError(
            f"optimizer state values changed during migration: {unequal_states}"
        )

    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": actual_sha,
                "expected_checkpoint_sha256": expected_sha,
                "source_policy": source_policy,
                "target_policy": POLICY,
                "norm_type": config["model"]["norm_type"],
                "rmsnorm_modules": len(norm_modules),
                "trainable_tensors": len(controlled) + len(uncontrolled),
                "trainable_numel": trainable_numel,
                "controlled_tensors": len(controlled),
                "control_units": len(units),
                "uncontrolled_tensors": len(uncontrolled),
                "added_from_hidden": sorted(additions),
                "removed_from_hidden": sorted(removals),
                "reference_norm_min": min(references.values()),
                "reference_norm_max": max(references.values()),
                "optimizer_state_entries": len(migrated.state),
                "optimizer_state_values_bitwise_equal": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

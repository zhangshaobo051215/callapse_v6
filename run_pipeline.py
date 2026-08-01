from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.analyze import analyze_run
from src.checkpointing import latest_checkpoint, load_checkpoint
from src.data import EpochBatchSampler, TinyImageNet, ensure_dataset, select_probe_indices
from src.model import VisionTransformer
from src.norm_control import control_units, resolve_reference_norms
from src.optimizer_migration import rebuild_optimizer_with_policy
from src.param_groups import build_optimizer, parameter_audit
from src.radial_audit import run_radial_audit
from src.train_branch import train_branch
from src.train_prefix import train_prefix
from src.utils import (environment_info, load_config, restore_rng_states, save_json,
                       save_yaml, set_all_seeds, sha256)

ROOT = Path(__file__).resolve().parent


def resolve_config(path, _stack=()):
    path = Path(path).resolve()
    if path in _stack:
        chain = " -> ".join(str(item) for item in (*_stack, path))
        raise ValueError(f"cyclic config inheritance: {chain}")
    cfg = load_config(path)
    base = cfg.pop("_base_", None)
    if base:
        parent = resolve_config(path.parent / base, (*_stack, path))
        for key, value in cfg.items():
            parent[key] = value
        cfg = parent
    return cfg


def device_for(cfg):
    requested = cfg["experiment"]["device"]
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def preflight_microbatch(cfg, device):
    data = cfg["data"]
    if not data.get("auto_microbatch") or device.type != "cuda":
        return
    model = VisionTransformer(cfg["model"]).to(device)
    for size in (data["micro_batch_size"], 64, 32):
        if data["global_batch_size"] % size:
            continue
        try:
            model.zero_grad(set_to_none=True)
            model(torch.zeros(size, 3, cfg["model"]["image_size"],
                              cfg["model"]["image_size"], device=device)).sum().backward()
            data["micro_batch_size"] = size
            del model
            torch.cuda.empty_cache()
            return
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    raise RuntimeError("OOM even at micro-batch 32")


def datasets_and_loaders(cfg):
    dc = cfg["data"]
    root = ensure_dataset(ROOT / dc["root"] if not Path(dc["root"]).is_absolute()
                          else dc["root"], dc["download"])
    train, val = TinyImageNet(root, "train"), TinyImageNet(root, "val")
    if dc.get("max_train_samples"):
        indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(
            cfg["experiment"]["data_seed"]))[:dc["max_train_samples"]].tolist()
        train = Subset(train, indices)
    return train, val


def loader_factory(dataset, cfg):
    dc = cfg["data"]
    def factory(sampler):
        return DataLoader(dataset, batch_sampler=sampler, num_workers=dc["num_workers"],
                          pin_memory=dc["pin_memory"])
    return factory


def fresh_components(cfg, device):
    model = VisionTransformer(cfg["model"]).to(device)
    optimizer, controlled, uncontrolled = build_optimizer(
        model, cfg["control"]["policy"], cfg["optimizer"]["peak_lr"],
        cfg["optimizer"]["betas"], cfg["optimizer"]["eps"], cfg["optimizer"]["weight_decay"])
    return model, optimizer, controlled, uncontrolled


def load_components(cfg, ckpt_path, device):
    # Keep serialized RNG byte tensors on CPU. Model and optimizer state are
    # copied/cast to the parameter device by load_state_dict below.
    ckpt = load_checkpoint(ckpt_path, "cpu")
    model = VisionTransformer(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    target_policy = cfg["control"]["policy"]
    source_policy = ckpt.get("config", {}).get("control", {}).get(
        "policy", target_policy)
    oc = cfg["optimizer"]
    optimizer_args = (
        oc["peak_lr"], oc["betas"], oc["eps"], oc["weight_decay"])
    if source_policy == target_policy:
        optimizer, controlled, uncontrolled = build_optimizer(
            model, target_policy, *optimizer_args)
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        if ckpt["global_state_step"] != oc["control_start_step"]:
            raise ValueError(
                "optimizer policy may only change at the shared prefix checkpoint")
        optimizer, controlled, uncontrolled = rebuild_optimizer_with_policy(
            model, ckpt["optimizer"], source_policy, target_policy,
            *optimizer_args)
    restore_rng_states(ckpt)
    return ckpt, model, optimizer, controlled, uncontrolled


def tensor_family(name: str) -> str:
    if name in {"cls_token", "pos_embed"} or name.startswith("patch_embed."):
        return "input_embedding"
    if ".attn.qkv." in name:
        return "attention_qkv"
    if ".attn.proj." in name:
        return "attention_output"
    if ".mlp.fc1." in name:
        return "mlp_input"
    if ".mlp.fc2." in name:
        return "mlp_output"
    if ".norm1." in name or ".norm2." in name:
        return "block_layernorm"
    if name.startswith("norm."):
        return "final_layernorm"
    if name.startswith("head."):
        return "classifier_head"
    return "other"


def _write_catalog(rows, path):
    if not rows:
        raise ValueError(f"cannot write an empty tensor catalog: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".csv.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def tensor_monitor_context(cfg, prefix_path, output):
    """Load immutable all-tensor references from the shared control-start prefix."""
    prefix_path, output = Path(prefix_path), Path(output)
    checkpoint = load_checkpoint(prefix_path, "cpu")
    expected_step = int(cfg["optimizer"]["control_start_step"])
    prefix_step = int(checkpoint["global_state_step"])
    if prefix_step != expected_step:
        raise ValueError(
            f"tensor monitoring requires the shared step-{expected_step} prefix; "
            f"checkpoint reports step {prefix_step}")

    model = VisionTransformer(cfg["model"])
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer, controlled, uncontrolled = build_optimizer(
        model, cfg["control"]["policy"], cfg["optimizer"]["peak_lr"],
        cfg["optimizer"]["betas"], cfg["optimizer"]["eps"],
        cfg["optimizer"]["weight_decay"])
    del optimizer
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    missing = sorted(set(trainable) - set(checkpoint["model"]))
    if missing:
        raise ValueError(f"shared prefix is missing trainable tensors: {missing}")

    reference_norms = {
        name: float(torch.linalg.vector_norm(
            checkpoint["model"][name].detach().float()).item())
        for name in trainable
    }
    invalid = {
        name: norm
        for name, norm in reference_norms.items()
        if not math.isfinite(norm) or norm <= 0
    }
    if invalid:
        raise ValueError(
            f"all monitored tensors need positive finite prefix norms: {invalid}")

    try:
        prefix_label = str(prefix_path.resolve().relative_to(ROOT))
    except ValueError:
        prefix_label = str(prefix_path.resolve())
    prefix_config = checkpoint.get("config", {})
    prefix_metadata = {
        "prefix_checkpoint": prefix_label,
        "prefix_sha256": sha256(prefix_path),
        "prefix_global_state_step": prefix_step,
        "prefix_git_commit": checkpoint.get("git_commit"),
        "prefix_source_policy": prefix_config.get("control", {}).get("policy"),
        "target_policy": cfg["control"]["policy"],
    }
    rows = []
    for name, parameter in trainable.items():
        is_controlled = name in controlled
        if not is_controlled and name not in uncontrolled:
            raise RuntimeError(
                f"optimizer classification omitted trainable tensor: {name}")
        reference_norm = reference_norms[name]
        rows.append({
            "tensor": name,
            "shape": json.dumps(list(parameter.shape), separators=(",", ":")),
            "ndim": parameter.ndim,
            "numel": parameter.numel(),
            "family": tensor_family(name),
            "controlled": is_controlled,
            "optimizer_group": "controlled" if is_controlled else "uncontrolled",
            "reference_fro_norm": reference_norm,
            "reference_rms_norm": reference_norm / math.sqrt(parameter.numel()),
            **prefix_metadata,
        })
    if set(controlled) | set(uncontrolled) != set(trainable):
        raise RuntimeError("optimizer classification is not a trainable-tensor partition")

    _write_catalog(rows, output / "tensor_catalog.csv")
    unit_reference_norms = None
    if bool(cfg["logging"].get("control_unit_monitoring", False)):
        split_qkv = bool(cfg["control"].get("split_fused_qkv", False))
        if not split_qkv:
            raise ValueError("control-unit monitoring requires split_fused_qkv=true")
        units = control_units(
            controlled, split_fused_qkv=split_qkv)
        unit_reference_norms = {
            name: float(torch.linalg.vector_norm(unit.detach().float()).item())
            for name, unit in units.items()
        }
        invalid_units = {
            name: value
            for name, value in unit_reference_norms.items()
            if not math.isfinite(value) or value <= 0
        }
        if invalid_units:
            raise ValueError(
                f"control units need positive finite prefix norms: {invalid_units}")
        unit_rows = []
        for name, unit in units.items():
            parent, separator, label = name.partition("::")
            reference_norm = unit_reference_norms[name]
            unit_rows.append({
                "tensor": name,
                "parent_tensor": parent,
                "unit_label": label if separator else "",
                "shape": json.dumps(list(unit.shape), separators=(",", ":")),
                "ndim": unit.ndim,
                "numel": unit.numel(),
                "family": tensor_family(name),
                "controlled": True,
                "optimizer_group": "controlled",
                "reference_fro_norm": reference_norm,
                "reference_rms_norm": reference_norm / math.sqrt(unit.numel()),
                **prefix_metadata,
            })
        _write_catalog(
            unit_rows, output / "control_unit_catalog.csv")
    return reference_norms, unit_reference_norms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["all", "test", "prefix", "radial", "branches", "analyze"],
                        default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-prechecks", action="store_true",
                        help="Developer option; full runs should not use this.")
    args = parser.parse_args()
    cfg = resolve_config(args.config)
    output = ROOT / cfg["experiment"]["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    device = device_for(cfg)
    preflight_microbatch(cfg, device)
    save_yaml(cfg, output / "resolved_config.yaml")
    save_json(environment_info(), output / "environment.json")
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"stages": {}}
    if args.stage in ("all", "test") and not args.skip_prechecks:
        subprocess.run([sys.executable, "-m", "pytest", "-q",
                        "--basetemp=.pytest-tmp"], cwd=ROOT, check=True)
        manifest["stages"]["tests"] = "complete"
        save_json(manifest, manifest_path)
        if args.stage == "test":
            return
        if Path(args.config).resolve().name != "smoke.yaml":
            subprocess.run([
                sys.executable, "run_pipeline.py", "--config", "configs/smoke.yaml",
                "--stage", "all", "--resume"
            ], cwd=ROOT, check=True)
            manifest["stages"]["smoke_precheck"] = "complete"
            save_json(manifest, manifest_path)
    train, val = datasets_and_loaders(cfg)
    set_all_seeds(cfg["experiment"]["seed"], cfg["experiment"]["deterministic"])
    prefix_dir = output / "prefix"
    prefix_path = prefix_dir / f"checkpoint_step_{cfg['optimizer']['control_start_step']:06d}.pt"
    if args.stage in ("all", "prefix") and not (args.resume and prefix_path.exists()):
        model, optimizer, controlled, _ = fresh_components(cfg, device)
        save_json(parameter_audit(model, cfg["control"]["policy"]), output / "parameter_audit.json")
        sampler = EpochBatchSampler(len(train), cfg["data"]["micro_batch_size"],
                                    cfg["experiment"]["data_seed"], cfg["data"]["drop_last"])
        prefix_path = train_prefix(cfg, model, optimizer, controlled, loader_factory(train, cfg),
                                   sampler, device, prefix_dir)
        manifest["stages"]["prefix"] = "complete"; save_json(manifest, manifest_path)
    if not prefix_path.exists():
        raise FileNotFoundError(f"prefix checkpoint missing: {prefix_path}")
    tensor_monitor_interval = cfg["logging"].get("tensor_monitor_interval")
    monitor_reference_norms = None
    control_unit_monitor_reference_norms = None
    if tensor_monitor_interval is not None:
        if int(tensor_monitor_interval) <= 0:
            raise ValueError("logging.tensor_monitor_interval must be positive")
        (monitor_reference_norms,
         control_unit_monitor_reference_norms) = tensor_monitor_context(
            cfg, prefix_path, output)
    probe_path = output / "radial_audit" / "probe_indices.json"
    probe_indices = select_probe_indices(len(val), cfg["data"]["probe_size"],
                                         cfg["experiment"]["data_seed"], probe_path)
    probe_loader = DataLoader(Subset(val, probe_indices), batch_size=cfg["data"]["micro_batch_size"],
                              shuffle=False, num_workers=cfg["data"]["num_workers"])
    val_loader = DataLoader(val, batch_size=cfg["data"]["micro_batch_size"], shuffle=False,
                            num_workers=cfg["data"]["num_workers"])
    radial_csv = output / "radial_audit" / "radial_sensitivity.csv"
    radial_png = output / "radial_audit" / "radial_sensitivity.png"
    if args.stage in ("all", "radial") and not (
            args.resume and radial_csv.exists() and radial_png.exists()):
        ckpt, model, _, _, _ = load_components(cfg, prefix_path, device)
        run_radial_audit(model, ckpt["model"], probe_loader, device,
                         cfg["radial_audit"]["scales"], output / "radial_audit")
        manifest["stages"]["radial_audit"] = "complete"; save_json(manifest, manifest_path)
    if args.stage in ("all", "branches"):
        for branch in cfg["control"]["schedules"]:
            branch_dir = output / "branches" / branch
            final = branch_dir / f"checkpoint_step_{cfg['optimizer']['total_steps']:06d}.pt"
            if args.resume and final.exists():
                continue
            resume_ckpt = latest_checkpoint(branch_dir) if args.resume else None
            source = resume_ckpt or prefix_path
            ckpt, model, optimizer, controlled, _ = load_components(cfg, source, device)
            sampler = EpochBatchSampler(len(train), cfg["data"]["micro_batch_size"],
                                        cfg["experiment"]["data_seed"], cfg["data"]["drop_last"])
            sampler.load_state_dict(ckpt["sampler_state"])
            split_qkv = bool(cfg["control"].get("split_fused_qkv", False))
            is_shared_prefix = (
                source == prefix_path
                and ckpt["global_state_step"] == cfg["optimizer"]["control_start_step"])
            if is_shared_prefix and set(ckpt["reference_norms"]) != set(
                    ckpt["controlled_names"]):
                raise ValueError("prefix reference norms and controlled names do not match")
            reference_norms = resolve_reference_norms(
                controlled, ckpt["reference_norms"],
                split_fused_qkv=split_qkv,
                allow_legacy_qkv=split_qkv and is_shared_prefix,
                allow_prefix_upgrade=is_shared_prefix)
            train_branch(cfg, branch, model, optimizer, controlled, sampler,
                         loader_factory(train, cfg), device, branch_dir,
                         ckpt["global_state_step"], ckpt["train_loss_ema"], reference_norms,
                         probe_loader, val_loader,
                         monitor_reference_norms=monitor_reference_norms,
                         control_unit_monitor_reference_norms=(
                             control_unit_monitor_reference_norms))
            manifest["stages"][branch] = "complete"; save_json(manifest, manifest_path)
    if args.stage in ("all", "analyze"):
        analyze_run(output)
        manifest["stages"]["analysis"] = "complete"
    files = [p for p in output.rglob("*") if p.is_file() and p != manifest_path]
    manifest["files"] = {str(p.relative_to(output)): sha256(p) for p in files}
    save_json(manifest, manifest_path)


if __name__ == "__main__":
    main()

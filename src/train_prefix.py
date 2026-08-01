from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F

from .checkpointing import save_checkpoint
from .data import infinite_batches
from .norm_control import control_units, reference_norms_for
from .schedules import base_lr_at
from .utils import append_csv, finite_gradients


def optimizer_step(model, optimizer, batches, accumulation, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = correct = count = 0
    for _ in range(accumulation):
        x, y = next(batches)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y) / accumulation
        loss.backward()
        loss_sum += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        count += y.numel()
    if not finite_gradients(model):
        raise FloatingPointError("non-finite gradient")
    optimizer.step()
    return loss_sum, correct / count


def train_prefix(cfg, model, optimizer, controlled, loader_factory, sampler, device, output,
                 start_step=0, ema=None):
    oc = cfg["optimizer"]
    output = Path(output)
    batches = infinite_batches(loader_factory, sampler)
    accumulation = cfg["data"]["global_batch_size"] // cfg["data"]["micro_batch_size"]
    for step in range(start_step, oc["control_start_step"]):
        lr = base_lr_at(step, peak_lr=oc["peak_lr"], final_lr=oc["final_lr"],
                        warmup_steps=oc["warmup_steps"],
                        decay_start_step=oc["decay_start_step"], total_steps=oc["total_steps"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss, acc = optimizer_step(model, optimizer, batches, accumulation, device)
        ema = loss if ema is None else cfg["logging"]["ema_beta"] * ema + (
            1 - cfg["logging"]["ema_beta"]) * loss
        append_csv({"global_state_step": step + 1, "train_loss_raw": loss,
                    "train_loss_ema": ema, "train_top1": acc, "base_lr": lr},
                   output / "prefix_metrics.csv")
    split_qkv = bool(cfg["control"].get("split_fused_qkv", False))
    refs = reference_norms_for(controlled, split_fused_qkv=split_qkv)
    path = output / f"checkpoint_step_{oc['control_start_step']:06d}.pt"
    save_checkpoint(path, model=model, optimizer=optimizer,
                    global_state_step=oc["control_start_step"],
                    sampler_state=sampler.state_dict(), train_loss_ema=ema,
                    reference_norms=refs, config=cfg,
                    controlled_names=control_units(
                        controlled, split_fused_qkv=split_qkv))
    return path

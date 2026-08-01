from __future__ import annotations

import os
from pathlib import Path

import torch

from .utils import capture_rng_states, git_commit


def save_checkpoint(path, *, model, optimizer, global_state_step, sampler_state,
                    train_loss_ema, reference_norms, config, controlled_names):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "global_state_step": int(global_state_step),
        "epoch": int(sampler_state["epoch"]),
        "batch_offset": int(sampler_state["batch_offset"]),
        "sampler_state": sampler_state, "train_loss_ema": float(train_loss_ema),
        "reference_norms": reference_norms, "config": config,
        "controlled_names": list(controlled_names), "git_commit": git_commit(),
        **capture_rng_states(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=False)


def latest_checkpoint(directory):
    candidates = sorted(Path(directory).glob("checkpoint_step_*.pt"))
    return candidates[-1] if candidates else None


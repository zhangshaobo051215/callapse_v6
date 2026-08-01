from __future__ import annotations

import math


def base_lr_at(step: int, *, peak_lr: float, final_lr: float, warmup_steps: int,
               decay_start_step: int, total_steps: int) -> float:
    if not 0 <= step < total_steps:
        raise ValueError(f"step {step} outside [0, {total_steps})")
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if step < decay_start_step:
        return peak_lr
    return final_lr + (peak_lr - final_lr) * (total_steps - step) / (
        total_steps - decay_start_step
    )


def schedule_ratio(name: str, step: int, *, control_start_step: int, total_steps: int,
                   cyclic_period_steps: int = 5000, cyclic_amplitude: float = 0.5,
                   linear_up_final: float = 2.0,
                   linear_down_final: float = 1 / 3) -> float:
    if not control_start_step <= step <= total_steps:
        raise ValueError("ratio step outside controlled interval")
    p = (step - control_start_step) / (total_steps - control_start_step)
    if name == "constant":
        return 1.0
    if name == "linear_up":
        return 1.0 + p * (linear_up_final - 1.0)
    if name == "linear_down":
        return 1.0 + p * (linear_down_final - 1.0)
    if name == "cyclic":
        return 1.0 + cyclic_amplitude * math.sin(
            2 * math.pi * (step - control_start_step) / cyclic_period_steps
        )
    raise ValueError(f"unknown schedule: {name}")


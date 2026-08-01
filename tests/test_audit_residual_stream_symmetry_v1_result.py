from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Callable

import pytest

from scripts.audit_residual_stream_symmetry_v1_result import (
    AuditError,
    METRICS_HEADER,
    _base_lr_at,
    _schedule_ratio,
    audit_metrics_file,
)


def _config() -> dict:
    return {
        "optimizer": {
            "peak_lr": 0.1,
            "final_lr": 0.01,
            "total_steps": 5,
            "warmup_steps": 1,
            "control_start_step": 2,
            "decay_start_step": 4,
        },
        "control": {
            "cyclic_period_steps": 4,
            "cyclic_amplitude": 0.5,
            "linear_up_final": 2.0,
            "linear_down_final": 1.0 / 3.0,
        },
        "logging": {
            "ema_beta": 0.9,
            "angular_interval": 2,
        },
    }


def _write_valid_metrics(
    path: Path,
    *,
    branch: str = "linear_down",
    tamper: Callable[[list[dict]], None] | None = None,
) -> None:
    cfg = _config()
    references = {"a": 2.0, "b": 4.0}
    combined = math.sqrt(sum(value * value for value in references.values()))
    previous_ema = 1.25
    rows = []
    for step in range(3, 6):
        optimizer_step = step - 1
        raw_loss = 1.0 + step / 100.0
        ema = 0.9 * previous_ema + 0.1 * raw_loss
        previous_ema = ema
        lr = _base_lr_at(optimizer_step, cfg["optimizer"])
        ratio = _schedule_ratio(
            branch, optimizer_step, cfg["optimizer"], cfg["control"]
        )
        post_ratio = _schedule_ratio(
            branch, step, cfg["optimizer"], cfg["control"]
        )
        angular = 0.01 if optimizer_step % 2 == 0 else math.nan
        row = {
            "global_state_step": step,
            "branch": branch,
            "train_loss_raw": raw_loss,
            "train_loss_ema": ema,
            "train_top1": 0.25,
            "base_lr": lr,
            "controlled_lr": lr * ratio,
            "schedule_ratio": ratio,
            "controlled_frobenius_norm": combined * post_ratio,
            "target_norm_relative_error_mean": 0.0,
            "target_norm_relative_error_max": 0.0,
            "tensorwise_elr_mean": lr,
            "tensorwise_elr_median": lr,
            "tensorwise_elr_min": lr,
            "tensorwise_elr_max": lr,
            "elr_relative_error_mean": 0.0,
            "elr_relative_error_max": 0.0,
            "gradient_finite": True,
            "angular_step_mean": angular,
            "angular_step_median": angular,
        }
        rows.append(row)
    if tamper is not None:
        tamper(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def _audit(path: Path):
    return audit_metrics_file(
        path,
        _config(),
        "linear_down",
        1.25,
        {"a": 2.0, "b": 4.0},
        {"a": 4, "b": 16},
        1e-5,
    )


def test_read_only_raw_audit_accepts_exact_steps_and_ema(tmp_path):
    path = tmp_path / "metrics.csv"
    _write_valid_metrics(path)
    before = path.read_bytes()
    rows, report = _audit(path)
    assert [int(row["global_state_step"]) for row in rows] == [3, 4, 5]
    assert report["steps_unique_and_complete"]
    assert report["ema_recurrence_verified_from_prefix"]
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "tamper, message",
    [
        (lambda rows: rows.pop(), "expected 3 metrics rows"),
        (
            lambda rows: rows.__setitem__(
                1, {**rows[1], "global_state_step": 3}
            ),
            "step 3 != expected 4",
        ),
        (
            lambda rows: rows[1].__setitem__(
                "train_loss_ema", rows[1]["train_loss_ema"] + 0.01
            ),
            "EMA recurrence",
        ),
        (
            lambda rows: rows[1].__setitem__(
                "elr_relative_error_max", float("nan")
            ),
            "non-finite",
        ),
        (
            lambda rows: rows[1].__setitem__("gradient_finite", False),
            "gradient_finite",
        ),
    ],
)
def test_raw_audit_rejects_false_pass_inputs(tmp_path, tamper, message):
    path = tmp_path / "metrics.csv"
    _write_valid_metrics(path, tamper=tamper)
    with pytest.raises(AuditError, match=message):
        _audit(path)

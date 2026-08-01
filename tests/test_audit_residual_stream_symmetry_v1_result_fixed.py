from __future__ import annotations

import math

import pytest
import torch

from scripts import audit_residual_stream_symmetry_v1_result as base
from scripts.audit_residual_stream_symmetry_v1_result_fixed import (
    _float32_norm_roundoff_bound,
    _norm64,
    _reference_error,
    _runtime_reference_maps,
)


def test_float64_baseline_accepts_reduction_order_difference():
    torch.manual_seed(0)
    tensor = torch.randn(147_456, dtype=torch.float32)
    cpu32 = float(torch.linalg.vector_norm(tensor).item())
    mathematical = _norm64(tensor)
    simulated_gpu32 = float(
        torch.tensor(mathematical, dtype=torch.float32).item()
    )

    # This is the exact false-negative shape: CPU32 differs enough to fail
    # the old 2e-7 comparison even though the serialized float32 scalar is
    # a high-quality approximation to the float64 mathematical norm.
    assert abs(cpu32 - simulated_gpu32) / mathematical > 2e-7
    report = _reference_error(
        simulated_gpu32,
        mathematical,
        tensor.numel(),
        "adversarial reduction fixture",
    )
    assert report["within_bound"]
    assert report["relative_error"] < report["float32_roundoff_bound"]


def test_float64_baseline_rejects_reference_outside_derived_bound():
    tensor = torch.linspace(
        -1.0, 1.0, 147_456, dtype=torch.float32
    )
    mathematical = _norm64(tensor)
    bound = _float32_norm_roundoff_bound(tensor.numel())
    corrupted = mathematical * (1.0 + 1.05 * bound)
    with pytest.raises(base.AuditError, match="differs from the float64 norm"):
        _reference_error(
            corrupted,
            mathematical,
            tensor.numel(),
            "corrupted reference",
        )


def test_roundoff_bound_is_size_aware_and_below_validity_threshold():
    small = _float32_norm_roundoff_bound(192)
    large = _float32_norm_roundoff_bound(147_456)
    assert 0.0 < small < large < 1e-5
    expected_budget = 2 * math.ceil(math.log2(147_456)) + 8
    u = torch.finfo(torch.float32).eps / 2.0
    assert large == pytest.approx(
        expected_budget * u / (1.0 - expected_budget * u),
        rel=0.0,
        abs=0.0,
    )


def test_runtime_source_reference_requires_exact_inheritance(
    tmp_path, monkeypatch
):
    mathematical = {"inherited": 2.0, "added": 3.0}
    source = {"inherited": 2.0}
    numels = {"inherited": 4, "added": 4}
    canonical = {"inherited": 2.0, "added": 3.0}
    checkpoints = {}
    for branch in base.BRANCHES:
        path = (
            tmp_path
            / "branches"
            / branch
            / "checkpoint_step_020000.pt"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        checkpoints[str(path)] = {
            "global_state_step": 20_000,
            "config": {"approved": True},
            "reference_norms": dict(canonical),
        }

    def fake_load(path, **_kwargs):
        return checkpoints[str(path)]

    monkeypatch.setattr(torch, "load", fake_load)
    monkeypatch.setattr(base, "sha256", lambda _path: "0" * 64)
    runtime, report = _runtime_reference_maps(
        tmp_path,
        {"approved": True},
        mathematical,
        source,
        numels,
    )
    assert runtime == canonical
    assert report["all_four_runtime_maps_exactly_identical"]

    corrupted = math.nextafter(2.0, 3.0)
    for checkpoint in checkpoints.values():
        checkpoint["reference_norms"]["inherited"] = corrupted
    with pytest.raises(base.AuditError, match="did not exactly inherit"):
        _runtime_reference_maps(
            tmp_path,
            {"approved": True},
            mathematical,
            source,
            numels,
        )


def test_runtime_added_reference_requires_four_branch_exact_equality(
    tmp_path, monkeypatch
):
    mathematical = {"added": 3.0}
    checkpoints = {}
    for branch in base.BRANCHES:
        path = (
            tmp_path
            / "branches"
            / branch
            / "checkpoint_step_020000.pt"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        checkpoints[str(path)] = {
            "global_state_step": 20_000,
            "config": {"approved": True},
            "reference_norms": {"added": 3.0},
        }
    cyclic_path = str(
        tmp_path
        / "branches"
        / "cyclic"
        / "checkpoint_step_020000.pt"
    )
    checkpoints[cyclic_path]["reference_norms"]["added"] = math.nextafter(
        3.0, 4.0
    )

    monkeypatch.setattr(
        torch, "load", lambda path, **_kwargs: checkpoints[str(path)]
    )
    monkeypatch.setattr(base, "sha256", lambda _path: "0" * 64)
    with pytest.raises(
        base.AuditError, match="not exactly identical to constant"
    ):
        _runtime_reference_maps(
            tmp_path,
            {"approved": True},
            mathematical,
            {},
            {"added": 4},
        )

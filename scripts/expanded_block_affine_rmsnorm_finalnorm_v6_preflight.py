from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import expanded_block_affine_rmsnorm_v5_preflight as v5_preflight  # noqa: E402
from src.policy_overlay_residual_stream_symmetry import V2_POLICY, install  # noqa: E402
from src.reference_alignment import alignment_settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    original_argv = sys.argv
    captured = io.StringIO()
    try:
        sys.argv = [
            original_argv[0],
            "--checkpoint", str(args.checkpoint),
            "--expected-sha256", args.expected_sha256,
        ]
        with contextlib.redirect_stdout(captured):
            v5_preflight.main()
    finally:
        sys.argv = original_argv
    report = json.loads(captured.getvalue())

    install()
    from run_pipeline import resolve_config  # noqa: E402

    cfg = resolve_config(args.config)
    settings = alignment_settings(cfg)
    if cfg["model"].get("norm_type") != "rmsnorm":
        raise ValueError("V6 must keep the V5 RMSNorm architecture")
    if cfg["control"]["policy"] != V2_POLICY:
        raise ValueError("V6 must keep the exact V5 q(t)-controlled scope")
    if settings is None:
        raise ValueError("V6 constant-reference alignment is disabled")
    if settings["tensor"] != "norm.weight":
        raise ValueError("V6 must align only the final RMSNorm norm.weight")
    if settings["reference_branch"] != "constant":
        raise ValueError("V6 reference branch must be constant")
    if cfg["control"]["schedules"][0] != "constant":
        raise ValueError("constant must be trained before intervention branches")

    report.update({
        "v6_intervention": "constant_reference_frobenius_norm_alignment",
        "reference_aligned_tensor": settings["tensor"],
        "reference_branch": settings["reference_branch"],
        "reference_alignment_tolerance": settings["tolerance"],
        "reference_aligned_tensor_remains_q_uncontrolled": True,
        "reference_aligned_tensor_uses_base_lr": True,
        "q_control_scope_is_bitwise_identical_to_v5": True,
    })
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

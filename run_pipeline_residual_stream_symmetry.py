from __future__ import annotations

from src.policy_overlay_residual_stream_symmetry import install


install()

import run_pipeline  # noqa: E402


if __name__ == "__main__":
    run_pipeline.main()

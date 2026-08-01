from __future__ import annotations

from src.policy_overlay_residual_stream_symmetry import install


install()

import run_pipeline  # noqa: E402
from src.reference_alignment_overlay import install_into  # noqa: E402


install_into(run_pipeline)


if __name__ == "__main__":
    run_pipeline.main()

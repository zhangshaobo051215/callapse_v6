#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
BASE_CONFIG="configs/vit_tiny_tinyimagenet.yaml"
TARGET_CONFIG="configs/vit_tiny_tinyimagenet_residual_stream_symmetry_v1_tensor_monitoring.yaml"
SOURCE_PREFIX="$PROJECT_ROOT/outputs/full/prefix/checkpoint_step_002500.pt"
TARGET_OUTPUT="$PROJECT_ROOT/outputs/residual_stream_symmetry_v1_tensor_monitoring"
TARGET_PREFIX_DIR="$TARGET_OUTPUT/prefix"
TARGET_PREFIX="$TARGET_PREFIX_DIR/checkpoint_step_002500.pt"
PREFLIGHT_JSON="$TARGET_OUTPUT/residual_stream_preflight.json"
AUDIT_JSON="$TARGET_OUTPUT/analysis/tensor_monitoring_audit.json"

cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=20260726
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLBACKEND=Agg

if [[ ! -x "$PYTHON" ]]; then
  echo "Python interpreter is not executable: $PYTHON" >&2
  exit 1
fi

"$PYTHON" -m pytest -q

# Reuse a supplied, verified shared prefix. On a clean machine this creates
# the same full-run step-2500 prefix before any branch intervention begins.
if [[ ! -f "$SOURCE_PREFIX" ]]; then
  "$PYTHON" run_pipeline.py \
    --config "$BASE_CONFIG" \
    --stage prefix \
    --resume
fi

if [[ ! -f "$SOURCE_PREFIX" ]]; then
  echo "Shared hidden-matrices prefix was not created: $SOURCE_PREFIX" >&2
  exit 1
fi

mkdir -p "$TARGET_PREFIX_DIR"
if [[ ! -f "$TARGET_PREFIX" ]]; then
  cp "$SOURCE_PREFIX" "$TARGET_PREFIX"
fi

SOURCE_SHA="$(sha256sum "$SOURCE_PREFIX" | awk '{print $1}')"
TARGET_SHA="$(sha256sum "$TARGET_PREFIX" | awk '{print $1}')"
if [[ "$SOURCE_SHA" != "$TARGET_SHA" ]]; then
  echo "Shared-prefix copy mismatch: source=$SOURCE_SHA target=$TARGET_SHA" >&2
  exit 1
fi
echo "SHARED_PREFIX_SHA256=$TARGET_SHA"

"$PYTHON" scripts/residual_stream_symmetry_preflight.py \
  --checkpoint "$TARGET_PREFIX" \
  --expected-sha256 "$TARGET_SHA" \
  > "$PREFLIGHT_JSON"
cat "$PREFLIGHT_JSON"

"$PYTHON" run_pipeline_residual_stream_symmetry.py \
  --config "$TARGET_CONFIG" \
  --stage branches \
  --resume
"$PYTHON" run_pipeline_residual_stream_symmetry.py \
  --config "$TARGET_CONFIG" \
  --stage analyze \
  --resume
"$PYTHON" scripts/iteration_gate.py \
  --output "$TARGET_OUTPUT" \
  --threshold 0.03 \
  --validity-threshold 1e-5

"$PYTHON" - "$AUDIT_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"tensor monitoring audit is missing: {path}")
audit = json.loads(path.read_text(encoding="utf-8"))
if audit.get("passed") is not True:
    raise SystemExit(
        "tensor monitoring audit failed:\n"
        + json.dumps(audit, indent=2, ensure_ascii=False)
    )
expected_counts = {
    "n_rows": 532608,
    "n_tensors": 152,
    "n_controlled_tensors": 52,
    "n_uncontrolled_tensors": 100,
}
observed = {key: audit.get(key) for key in expected_counts}
if observed != expected_counts:
    raise SystemExit(
        f"unexpected tensor coverage: expected={expected_counts}, observed={observed}"
    )
print(json.dumps(audit, indent=2, ensure_ascii=False))
PY

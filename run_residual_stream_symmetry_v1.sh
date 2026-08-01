#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
BASE_CONFIG="configs/vit_tiny_tinyimagenet.yaml"
TARGET_CONFIG="configs/vit_tiny_tinyimagenet_residual_stream_symmetry_v1.yaml"
SOURCE_PREFIX="$PROJECT_ROOT/outputs/full/prefix/checkpoint_step_002500.pt"
TARGET_PREFIX_DIR="$PROJECT_ROOT/outputs/residual_stream_symmetry_v1/prefix"
TARGET_PREFIX="$TARGET_PREFIX_DIR/checkpoint_step_002500.pt"
PREFLIGHT_JSON="$PROJECT_ROOT/outputs/residual_stream_symmetry_v1/residual_stream_preflight.json"

cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=20260726
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLBACKEND=Agg

"$PYTHON" -m pytest -q

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
  --output "$PROJECT_ROOT/outputs/residual_stream_symmetry_v1" \
  --threshold 0.03 \
  --validity-threshold 1e-5

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
BASE_CONFIG="configs/vit_tiny_tinyimagenet.yaml"
TARGET_CONFIG="configs/vit_tiny_tinyimagenet_expanded_block_affine_v2_tensor_monitoring.yaml"
SOURCE_PREFIX="${SOURCE_PREFIX:-$PROJECT_ROOT/outputs/full/prefix/checkpoint_step_002500.pt}"
TARGET_OUTPUT="$PROJECT_ROOT/outputs/expanded_block_affine_v2_tensor_monitoring"
TARGET_PREFIX_DIR="$TARGET_OUTPUT/prefix"
TARGET_PREFIX="$TARGET_PREFIX_DIR/checkpoint_step_002500.pt"
PREFLIGHT_JSON="$TARGET_OUTPUT/expanded_block_affine_v2_preflight.json"
AUDIT_JSON="$TARGET_OUTPUT/analysis/tensor_monitoring_audit.json"
UNIT_AUDIT_JSON="$TARGET_OUTPUT/analysis/control_unit_monitoring_audit.json"

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

"$PYTHON" scripts/expanded_block_affine_v2_preflight.py \
  --checkpoint "$TARGET_PREFIX" \
  --expected-sha256 "$TARGET_SHA" \
  > "$PREFLIGHT_JSON"
cat "$PREFLIGHT_JSON"

"$PYTHON" run_pipeline_expanded_block_affine_v2.py \
  --config "$TARGET_CONFIG" \
  --stage branches \
  --resume
"$PYTHON" run_pipeline_expanded_block_affine_v2.py \
  --config "$TARGET_CONFIG" \
  --stage analyze \
  --resume
"$PYTHON" scripts/iteration_gate.py \
  --output "$TARGET_OUTPUT" \
  --threshold 0.03 \
  --validity-threshold 1e-5

"$PYTHON" - "$AUDIT_JSON" "$UNIT_AUDIT_JSON" <<'PY'
import json
import sys
from pathlib import Path

def load_and_validate(path_arg, label, expected_counts):
    path = Path(path_arg)
    if not path.is_file():
        raise SystemExit(f"{label} audit is missing: {path}")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("passed") is not True:
        raise SystemExit(
            f"{label} audit failed:\n"
            + json.dumps(audit, indent=2, ensure_ascii=False)
        )
    observed = {key: audit.get(key) for key in expected_counts}
    if observed != expected_counts:
        raise SystemExit(
            f"unexpected {label} coverage: "
            f"expected={expected_counts}, observed={observed}"
        )
    return audit


named_audit = load_and_validate(
    sys.argv[1],
    "named-tensor monitoring",
    {
        "n_rows": 532608,
        "n_tensors": 152,
        "n_controlled_tensors": 100,
        "n_uncontrolled_tensors": 52,
    },
)
unit_audit = load_and_validate(
    sys.argv[2],
    "control-unit monitoring",
    {
        "n_rows": 518592,
        "n_tensors": 148,
        "n_controlled_tensors": 148,
        "n_uncontrolled_tensors": 0,
    },
)
print(json.dumps(
    {
        "named_tensor_audit": named_audit,
        "control_unit_audit": unit_audit,
    },
    indent=2,
    ensure_ascii=False,
))
PY

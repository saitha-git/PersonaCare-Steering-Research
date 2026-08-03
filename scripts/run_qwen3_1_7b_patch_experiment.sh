#!/usr/bin/env bash
# Extract a Qwen3-1.7B steering vector, patch qai-hub-models, and export
# baseline + steered variants. Run from WSL/Linux after setup_wsl_qwen3_patch.sh.
set -euo pipefail

cd "$(dirname "$0")/.."

QHM_DIR="${QHM_DIR:-external/qai-hub-models}"
PATCH_FILE="${PATCH_FILE:-patches/qai_hub_models_qwen3_1_7b_steering.patch}"
CONFIG="${CONFIG:-configs/qwen3_1_7b.yaml}"
DATA="${DATA:-data/contrast_pairs.jsonl}"
LAYER="${LAYER:-7}"
TAG="${TAG:-qwen3_1_7b}"
VECTOR="${VECTOR:-artifacts/vector_layer_${LAYER}_${TAG}.pt}"
DEVICE="${DEVICE:-Samsung Galaxy S25 (Family)}"
RUNTIME="${RUNTIME:-geniex_qairt}"
EXPORT_MODULE="${EXPORT_MODULE:-qai_hub_models.models.qwen3_1_7b.export}"
OUT_ROOT="${OUT_ROOT:-artifacts/qwen3_1_7b_patch}"
SEQUENCE_LENGTHS="${SEQUENCE_LENGTHS:-1,128}"
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-512}"

if [ ! -d "$QHM_DIR/.git" ]; then
    echo "$QHM_DIR missing; run scripts/setup_wsl_qwen3_patch.sh first." >&2
    exit 2
fi

if ! grep -q "STEERING_POC_QWEN3_VECTOR" "$QHM_DIR/src/qai_hub_models/models/_shared/qwen3/model.py"; then
    git -C "$QHM_DIR" apply "$(pwd)/$PATCH_FILE"
fi

mkdir -p "$OUT_ROOT"

if [ ! -f "$VECTOR" ]; then
    python -m steering_poc.extract \
        --config "$CONFIG" \
        --data "$DATA" \
        --layers "$LAYER" \
        --tag "$TAG"
fi

python - <<EOF
import torch
payload = torch.load("$VECTOR", map_location="cpu")
vector = payload["vector"] if isinstance(payload, dict) else payload
print("vector", "$VECTOR", "shape", tuple(vector.shape), "norm", float(vector.float().norm()))
assert vector.numel() == 2048, "Qwen3-1.7B steering vector must be 2048-dim"
EOF

echo "Exporting baseline Qwen3-1.7B..."
env -u STEERING_POC_QWEN3_VECTOR \
    -u STEERING_POC_QWEN3_LAYER \
    -u STEERING_POC_QWEN3_ALPHA \
    -u STEERING_POC_QWEN3_NORMALIZE \
    python -m "$EXPORT_MODULE" \
        --device "$DEVICE" \
        --runtime "$RUNTIME" \
        --sequence-lengths "$SEQUENCE_LENGTHS" \
        --context-lengths "$CONTEXT_LENGTHS" \
        --skip-profiling \
        --output-dir "$OUT_ROOT/baseline"

for ALPHA in 0 4 -4; do
    NAME="${ALPHA/-/neg}"
    echo "Exporting steered Qwen3-1.7B alpha=$ALPHA..."
    STEERING_POC_QWEN3_VECTOR="$(pwd)/$VECTOR" \
    STEERING_POC_QWEN3_LAYER="$LAYER" \
    STEERING_POC_QWEN3_ALPHA="$ALPHA" \
    STEERING_POC_QWEN3_NORMALIZE=1 \
    python -m "$EXPORT_MODULE" \
        --device "$DEVICE" \
        --runtime "$RUNTIME" \
        --sequence-lengths "$SEQUENCE_LENGTHS" \
        --context-lengths "$CONTEXT_LENGTHS" \
        --skip-profiling \
        --output-dir "$OUT_ROOT/alpha_${NAME}"
done

cat <<EOF

Qwen3-1.7B patch experiment exports complete.
Outputs:
  $OUT_ROOT/baseline
  $OUT_ROOT/alpha_0
  $OUT_ROOT/alpha_4
  $OUT_ROOT/alpha_neg4

Generated artifacts are intentionally gitignored.
EOF

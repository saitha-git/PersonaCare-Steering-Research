#!/usr/bin/env bash
# Milestone B: export the injection op to ONNX and validate numerics across
# backends and precisions. No Qualcomm tooling required.
set -euo pipefail
cd "$(dirname "$0")/.."

HIDDEN=${HIDDEN:-1024}   # Qwen3-0.6B hidden size

python -m steering_poc.export_onnx --hidden-size "$HIDDEN"
if [ -f artifacts/vector_layer_14.pt ]; then
    python -m steering_poc.compare_backends --hidden-size "$HIDDEN" \
        --vector artifacts/vector_layer_14.pt
else
    python -m steering_poc.compare_backends --hidden-size "$HIDDEN"
fi
python -m steering_poc.qualcomm.detect_environment
echo "Done. See artifacts/backend_comparison.json"

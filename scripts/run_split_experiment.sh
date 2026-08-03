#!/usr/bin/env bash
# Split-compute experiment: partition Qwen3-0.6B across two devices and verify
# on Qualcomm AI Hub (phone NPU runs layers 0-13, laptop NPU runs 14-27).
set -euo pipefail
cd "$(dirname "$0")/.."

# 1. Export the two ONNX halves (fp32, fixed T=32, boundary = hidden state)
python -m split_compute.export_split --config configs/qwen3_0_6b.yaml \
    --split-layer 14 --seq-len 32

# 2. Free local parity gate: ORT chain must match full HF fp32 before upload
python -m split_compute.verify_local --config configs/qwen3_0_6b.yaml

# 3. Cloud verification (requires configured AI Hub token; dry-run without --submit)
python -m split_compute.submit_split --submit --profile \
    --phone "Snapdragon 8 Elite QRD" --laptop "Snapdragon X Elite CRD"

echo "Results -> artifacts/split/ai_hub_split_jobs.json"

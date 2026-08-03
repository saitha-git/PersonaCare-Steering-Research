#!/usr/bin/env bash
# Milestone A end-to-end: extract -> generate -> evaluate -> plot.
# Works in any venv where `pip install -e .` has been run (Windows Git Bash or WSL).
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${CONFIG:-configs/qwen3_0_6b.yaml}
LAYERS=${LAYERS:-"7 14 21"}

python -m steering_poc.extract --config "$CONFIG" --data data/contrast_pairs.jsonl
python -m steering_poc.generate --config "$CONFIG" \
    --vector artifacts/vector_layer_7.pt --layer 7 \
    --alpha -4 -2 -1 0 1 2 4 --prompt "How do I make coffee?"
python -m steering_poc.evaluate --config "$CONFIG" --layers $LAYERS \
    --prompts data/eval_prompts.jsonl --max-prompts 32
python -m steering_poc.plots
echo "Done. See artifacts/eval_metrics.csv and artifacts/dose_response.png"

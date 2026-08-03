# Milestone A end-to-end on native Windows (PowerShell).
# Assumes: .venv created and `pip install -e .` done (see README quick start).
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"

& $py -m steering_poc.extract --config configs/qwen3_0_6b.yaml --data data/contrast_pairs.jsonl
& $py -m steering_poc.generate --config configs/qwen3_0_6b.yaml `
    --vector artifacts/vector_layer_7.pt --layer 7 `
    --alpha -4 -2 -1 0 1 2 4 --prompt "How do I make coffee?"
& $py -m steering_poc.evaluate --config configs/qwen3_0_6b.yaml --layers 7 14 21 `
    --prompts data/eval_prompts.jsonl --max-prompts 32
& $py -m steering_poc.plots
Write-Host "Done. See artifacts/eval_metrics.csv and artifacts/dose_response.png"

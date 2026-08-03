# Reproduce the POC

This repo keeps heavyweight generated files out of git. Source, tests, configs,
input data, and small result summaries are committed; model downloads and
generated artifacts are produced locally.

## 1. Local Python setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e . --no-deps
python -m pytest tests -q
```

The unit tests use tiny random-init models and do not require Qualcomm tooling
or Hugging Face model downloads.

## 2. Steering demo

```powershell
python -m steering_poc.extract --config configs/qwen3_0_6b.yaml --data data/contrast_pairs.jsonl
python -m steering_poc.generate --config configs/qwen3_0_6b.yaml `
  --vector artifacts/vector_layer_7.pt --layer 7 --alpha -4 0 4 `
  --prompt "How do I make coffee?"
python -m steering_poc.evaluate --config configs/qwen3_0_6b.yaml --layers 7 14 21
python -m steering_poc.plots
```

This downloads Qwen3-0.6B to the Hugging Face cache.

## 3. ONNX and backend validation

```powershell
python -m steering_poc.export_onnx --hidden-size 1024
python -m steering_poc.compare_backends --hidden-size 1024 --vector artifacts/vector_layer_14.pt
```

Optional Qualcomm cloud submission requires a configured AI Hub token:

```powershell
python -m steering_poc.qualcomm.submit_ai_hub
python -m steering_poc.qualcomm.submit_ai_hub --submit --infer --profile
python -m steering_poc.qualcomm.prove_steering_ai_hub
python -m steering_poc.qualcomm.prove_steering_ai_hub --submit --profile
```

`prove_steering_ai_hub` is the strongest standalone on-device proof: it runs
the same compiled graph with alpha=0 and alpha=4, then verifies the device
delta against `4 * mask * steering`. The committed result summary is
`docs/results/steering_device_proof.json`.

Optional local QNN/QAIRT validation requires ONNX Runtime QNN EP or a QAIRT SDK:

```powershell
python -m steering_poc.qualcomm.qnn_emulation --model artifacts/steering_injection_prefill.onnx
```

## 4. Split-compute experiment

```powershell
python -m split_compute.export_split --config configs/qwen3_0_6b.yaml --split-layer 14 --seq-len 32
python -m split_compute.verify_local --config configs/qwen3_0_6b.yaml
python -m split_compute.submit_split
```

The split export writes large ONNX files under `artifacts/split/`. Do not commit
those files; committed summaries are in `docs/results/split/`.

## 5. Qwen3-1.7B steering patch experiment

This is the first step toward steering inside Qualcomm's deployed LLM export
path. Run it from WSL/Linux because the Qwen3-1.7B `qai-hub-models` recipe uses
Linux/Python-3.10 AIMET dependencies.

```bash
bash scripts/setup_wsl_qwen3_patch.sh
source ~/.venvs/qwen3_patch310/bin/activate
bash scripts/run_qwen3_1_7b_patch_experiment.sh
```

For long exports, use the logged step runner and monitor from two WSL terminals:

```bash
# terminal 1
source ~/.venvs/qwen3_patch310/bin/activate
bash scripts/run_qwen3_1_7b_patch_step.sh baseline

# terminal 2
bash scripts/monitor_qwen3_1_7b_patch.sh
```

After export/link succeeds, AI Hub can profile individual multi-graph context
binaries by selecting the prompt or token graph explicitly:

```bash
python -m steering_poc.qualcomm.profile_qwen3_patch_links --phases alpha_4 --graphs prompt
python -m steering_poc.qualcomm.profile_qwen3_patch_links --phases alpha_4 --graphs prompt --submit
```

Runtime text comparison requires a target device with the Genie SDK:

```bash
GENIE_BIN=/path/to/genie-t2t-run bash scripts/run_qwen3_1_7b_genie_compare.sh
```

The scripts clone `qai-hub-models` into ignored `external/qai-hub-models`, apply
`patches/qai_hub_models_qwen3_1_7b_steering.patch`, extract a 2048-dim steering
vector, and export baseline/steered Qwen3-1.7B variants under
`artifacts/qwen3_1_7b_patch/`.

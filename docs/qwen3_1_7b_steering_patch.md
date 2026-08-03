# Qwen3-1.7B Steering Patch Experiment

Goal: patch the activation-steering Add into Qualcomm's editable
`qai-hub-models` Qwen3-1.7B export and recompile/export steered variants of the
deployed LLM path. This avoids Meta-gated Llama weights and uses Qwen3's
Apache-2.0 model family.

## Why Qwen3-1.7B

Qwen3-0.6B is useful for the local behavior demo, but Qualcomm's catalog entry
is a prebuilt GGUF/llama.cpp artifact rather than an editable QAIRT graph.
Qwen3-1.7B has a `qai-hub-models` recipe (`qwen3_1_7b`) with Genie/GenieX-QAIRT
support and hidden size 2048.

## Patch design

The committed patch file is:

```text
patches/qai_hub_models_qwen3_1_7b_steering.patch
```

It patches `qai_hub_models.models._shared.qwen3.model.Qwen3Base.monkey_patch()`
so normal exports are unchanged unless `STEERING_POC_QWEN3_VECTOR` is set.
When enabled, it wraps `modeling_qwen3.Qwen3DecoderLayer.forward` and modifies
only one decoder layer output:

```text
hidden = hidden + alpha * unit_steering_vector
```

Defaults:

```text
STEERING_POC_QWEN3_LAYER=7
STEERING_POC_QWEN3_ALPHA=4.0
STEERING_POC_QWEN3_NORMALIZE=1
```

The vector must be extracted from Qwen3-1.7B so it has hidden dimension 2048.

## Run from WSL/Linux

```bash
bash scripts/setup_wsl_qwen3_patch.sh
source ~/.venvs/qwen3_patch310/bin/activate
bash scripts/run_qwen3_1_7b_patch_experiment.sh
```

For manual monitoring/debugging, prefer the step runner. In terminal 1:

```bash
source ~/.venvs/qwen3_patch310/bin/activate
bash scripts/run_qwen3_1_7b_patch_step.sh baseline
```

In terminal 2:

```bash
bash scripts/monitor_qwen3_1_7b_patch.sh
```

The step runner writes a durable log and status files under
`artifacts/qwen3_1_7b_patch/runs/<timestamp>_<phase>/`. It also updates
`artifacts/qwen3_1_7b_patch/latest_run_dir.txt`, which the monitor reads by
default. Useful follow-up phases after baseline succeeds:

```bash
bash scripts/run_qwen3_1_7b_patch_step.sh alpha0
bash scripts/run_qwen3_1_7b_patch_step.sh alpha4
```

You can reduce the export shape for debugging by setting environment variables:

```bash
SEQUENCE_LENGTHS=1 CONTEXT_LENGTHS=128 bash scripts/run_qwen3_1_7b_patch_step.sh baseline
```

The setup script clones `qai-hub-models` into `external/qai-hub-models`, which is
ignored by git. The run script:

1. applies the patch if it is not already present;
2. extracts `artifacts/vector_layer_7_qwen3_1_7b.pt`;
3. exports a baseline Qwen3-1.7B model;
4. exports steered variants for alpha `0`, `4`, and `-4`.

Generated exports are written under `artifacts/qwen3_1_7b_patch/` and are not
committed.

## Known limitations

This first milestone uses compiled constants. It proves the steering Add can be
inserted into the deployed model export path, but it does not yet expose runtime
alpha/vector inputs through Genie. That is the next risk item after a compiled
variant exports and compiles cleanly.

## Export result

Baseline, alpha=0, and alpha=4 Qwen3-1.7B GenieX QAIRT `w4a16` bundles exported
and linked successfully on 2026-07-22. The alpha=4 prompt graphs for all four
context binaries also profiled successfully with all recorded ops on NPU. The
generated multi-GB bundles stay under ignored `artifacts/qwen3_1_7b_patch/`;
the shareable result summaries are committed under
`docs/results/qwen3_1_7b_patch/`.

This is a full-model export/compile/link/profile proof for the patched path. It
is not yet a Genie runtime behavior proof; the next step is to run the three
bundles with the same prompt and compare baseline vs alpha=0 vs alpha=4 outputs
using `scripts/run_qwen3_1_7b_genie_compare.sh` on a device with the Genie SDK.

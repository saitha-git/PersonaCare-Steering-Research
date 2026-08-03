# Activation Steering POC — with a Qualcomm Snapdragon deployment path

> Part of **PersonaCare — Persona-Steered Medical Companion**, built for the
> Snapdragon Multiverse Hackathon. This repo is the research track: extracting
> and validating the steering mechanism used to control the companion's tone.
> The on-device inference pipeline that pairs with it lives in
> [PersonaCare](https://github.com/saitha-git/PersonaCare).

Extract a steering vector from contrast pairs, inject it into the residual
stream of a small causal LM (Qwen3-0.6B), demonstrate a measurable
dose-response, export the injection op as an NPU-friendly ONNX graph
(`steered = hidden + alpha * mask * steering`), and validate its numerics under
FP16/INT8. Includes an opt-in Qualcomm AI Hub submission script and a concrete
(unimplemented) plan for full integration into Qualcomm's LLM export pipeline.

**Honesty up front — what this repo demonstrates vs. what it does not:**

| Claim | Status |
|---|---|
| Steering vectors change Qwen3-0.6B behavior with an interpretable dose-response | **Demonstrated** (results below) |
| alpha=0 injection is numerically identical to no injection | **Demonstrated** (tests + eval) |
| The injection op exports to ONNX as pure `Mul`/`Add` and matches PyTorch | **Demonstrated** |
| The op survives FP16 / INT8 quantization with signal ≫ quantization noise | **Demonstrated** (standalone op only — NOT full-LLM quantization) |
| The op compiles and runs on a Snapdragon device via AI Hub | **Demonstrated** on Snapdragon X Elite CRD. Strongest proof run: compile `jp2v2z265`, link `jp0vnxy0g`, inference alpha=0 `j56dk06np`, inference alpha=4 `jg9x67xlg`, profile `jp1vrkj8p`. **All 5 graph ops on the NPU**, est. 167 µs, ~14.7 MB peak; on-device alpha=4 delta matches `4 * mask * steering` with max error 1.07e-3 and cosine 0.9999986 |
| Steering injected inside Qualcomm's optimized on-device LLM (Genie/QNN) | **Export/compile/link/profile demonstrated; runtime behavior not yet validated.** Qwen3-1.7B baseline, alpha=0, and alpha=4 compiled-constant GenieX QAIRT bundles exported successfully; alpha=4 prompt graphs profile with all ops on NPU; see `docs/results/qwen3_1_7b_patch/` |
| Full-LLM Snapdragon latency/power numbers | **None measured.** The 167/180 µs figures are standalone injection-op timings only |
| Two-device pipeline split (phone NPU + laptop NPU) of one LLM | **Demonstrated** (functional): Qwen3-0.6B layers 0–13 on a Snapdragon 8 Elite QRD, layers 14–27 on an X Elite CRD, hidden state as the only boundary tensor; 100% greedy-token agreement with local fp32. Live transport/streaming decode **not** demonstrated — see "Split-compute experiment" |

## Quick start (Windows native, Python 3.10+; WSL: `scripts/setup_wsl.sh`)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or cpu wheel
pip install -r requirements.txt      # exact tested versions: requirements.lock.txt
pip install -e . --no-deps

# Unit tests (no model download needed — uses a tiny random-init model)
python -m pytest tests -q

# Milestone A: extract vectors (downloads Qwen3-0.6B, ~1.5 GB, to the HF cache)
python -m steering_poc.extract --config configs/qwen3_0_6b.yaml --data data/contrast_pairs.jsonl

# Steered generation at one layer
python -m steering_poc.generate --config configs/qwen3_0_6b.yaml `
    --vector artifacts/vector_layer_7.pt --layer 7 --alpha -4 -2 -1 0 1 2 4 `
    --prompt "How do I make coffee?"

# Layer x alpha sweep: 32 prompts, censoring-aware stats, bootstrap CIs
python -m steering_poc.evaluate --config configs/qwen3_0_6b.yaml --layers 7 14 21
python -m steering_poc.plots

# Dedicated Qualcomm tooling env (WSL Ubuntu, Python 3.10, qai-hub pinned)
# wsl bash scripts/setup_wsl_qualcomm.sh

# Milestone B: ONNX export + cross-backend/quantization validation
python -m steering_poc.export_onnx --hidden-size 1024
python -m steering_poc.compare_backends --hidden-size 1024 --vector artifacts/vector_layer_14.pt

# Qualcomm tooling detection (works without any Qualcomm hardware)
python -m steering_poc.qualcomm.detect_environment
# AI Hub submission is DRY-RUN unless you pass --submit:
python -m steering_poc.qualcomm.submit_ai_hub                # dry run
python -m steering_poc.qualcomm.submit_ai_hub --submit --infer --profile
python -m steering_poc.qualcomm.prove_steering_ai_hub        # dry run
python -m steering_poc.qualcomm.prove_steering_ai_hub --submit --profile
```

Fallback model: `configs/qwen2_5_0_5b.yaml` (Qwen2.5-0.5B-Instruct). Optional:
`configs/llama32_1b.yaml` (gated Meta weights; not required for the demo).

## Method

**Concept:** verbose vs. concise responses (benign, easy to score: response
length in words). `data/contrast_pairs.jsonl` holds 24 content-matched pairs;
a deterministic 20% holdout is excluded from extraction so evaluation prompts
and extraction text never coincide.

**Extraction** (`steering_poc/extract.py`): both members of each pair are
tokenized identically; hidden states are captured at the **output of decoder
layer i** (the residual stream after that layer's attention + MLP), at the
**last non-padding token**; the vector is
`mean(act(verbose) - act(concise))` over training pairs. Saved with full
reproducibility metadata (model/tokenizer revision, layer, capture point,
dataset SHA-256, dtype, holdout indices, norms).

**Injection** (`steering_poc/inject.py`): a forward hook on the chosen decoder
layer replaces its output hidden state with
`hidden + alpha * mask * unit_vector` — out-of-place, preserving KV-cache and
other tuple outputs. Under HF `generate()`, prefill sees `[B, T, D]` and each
cached decode step sees `[B, 1, D]`; `positions="all"` steers every token,
`positions="last"` steers only the final prompt token during prefill and every
new token during decode. Hooks are installed/removed by a context manager
(exception-safe; verified by tests).

## Results (Milestone A) — Qwen3-0.6B, greedy decoding

Sweep: layers {7, 14, 21} x alpha {-4,-2,-1,0,1,2,4}, **32 diverse prompts**,
unit-normalized vectors, `positions=all`, `max_new_tokens=512`. Evaluation is
**censoring-aware**: every generation records its token IDs,
`generated_tokens`, `reached_eos`, and `hit_max_new_tokens`; generations that
hit the 512-token cap (0–3 of 32 per cell) are flagged censored and excluded
from word-count statistics — a truncated generation is never reported as its
true response length. Full generated data is written locally under
`artifacts/`; committed summary copies live under `docs/results/`
(`docs/results/eval_doseresponse.json`, `docs/results/dose_response.png`).

**Zero-dose identity:** with the injection hook installed and alpha=0, the
next-token logits match the hook-free baseline with **max abs deviation
0.00e+00** at all 3 layers over the full 32-prompt batch (tolerance 1e-5;
FP32 `hidden + 0*v` is bit-exact).

**Paired word-count change vs alpha=0** (same prompt, both sides uncensored;
mean with percentile bootstrap 95% CI over prompts, 10k resamples):

| layer | α=−4 | α=−2 | α=−1 | α=+1 | α=+2 | α=+4 | Spearman ρ(α, words) |
|---|---|---|---|---|---|---|---|
| **7** | **−46.4 [−64.5, −28.8]** | **−31.5 [−46.8, −17.6]** | −7.3 [−18.1, 2.6] | +7.9 [−4.4, 21.3] | +9.5 [−4.9, 24.9] | **+25.6 [7.4, 43.1]** | **+0.251** (n=214) |
| 14 | −14.1 [−30.1, 0.5] | −5.6 [−18.8, 7.9] | −2.7 [−13.6, 7.3] | −2.3 [−16.8, 11.5] | −1.8 [−16.9, 13.1] | +0.1 [−14.3, 14.4] | +0.049 (n=207) |
| 21 | −5.9 [−15.2, 1.7] | −0.7 [−7.0, 4.2] | −1.5 [−7.5, 2.7] | +1.8 [−1.4, 6.3] | −2.3 [−7.8, 3.3] | +1.9 [−2.8, 7.3] | +0.015 (n=219) |

* **Layer 7** (the shipped default): monotone mean trend across the full
  alpha range; the CIs at α=−4, −2, and +4 exclude zero. Baseline responses
  average 147 words (uncensored); α=−4 removes ~46 of them, α=+4 adds ~26.
* **Layers 14 and 21 are weak-to-null** at unit vector norm (late-layer
  residual norms dwarf a unit vector: raw diff norms grow 8→175 from layer 3
  to 25). Reported as measured — no cherry-picking.
* Logit metrics at layer 7 scale smoothly with dose (mean over prompts):
  KL(baseline‖steered) 0 → 0.0015 → 0.03 and top-10 agreement 1.00 → 0.97 →
  0.90 for α = 0 → ±1 → ±4; the fluency proxy (NLL of the steered text under
  the unsteered model) stays in 0.38–0.42 nats/token across all alphas —
  steering changes style, not coherence.

Example (layer 7, "What is the boiling point of water?", both `reached_eos`):

> **α=−4 (12 words):** The boiling point of water is **100°C (212°F)** at standard atmospheric pressure.
>
> **α=0 (27 words):** The boiling point of water is **100°C (212°F)** at standard atmospheric pressure (1 atm). This is the temperature at which water turns into steam at sea level.

The verbose direction is modest on prompts whose baseline answer is already
minimal and shows up in the paired aggregate (+25.6 words at α=+4); it does
not manufacture content on single-fact questions.

## ONNX + quantization (Milestone B)

`steering_poc/export_onnx.py` exports three graphs (opset 17, classic
exporter), each containing **only `Mul` and `Add` nodes** with `hidden`,
`steering`, `alpha`, `mask` as runtime inputs:

* `steering_injection_prefill.onnx` — fixed `[1, 16, 1024]`
* `steering_injection_decode.onnx` — fixed `[1, 1, 1024]`
* `steering_injection_dynamic.onnx` — dynamic sequence length (works in ONNX
  Runtime; the fixed-shape variants exist because Qualcomm HTP compilation
  strongly prefers static shapes)

Measured on the real layer-14 vector (`compare_backends.py`, full table in
`artifacts/backend_comparison.json`):

| Condition | max abs err | mean abs err | cosine |
|---|---|---|---|
| ONNX Runtime CPU vs PyTorch FP32 | **0.0 (bit-exact)** | 0.0 | 1.000000 |
| PyTorch FP16 | 2.3e-03 | 1.9e-04 | ≥0.999999 |
| INT8 steering vector (per-tensor, alpha=4) | 2.5e-03 | 1.2e-03 | 0.999999 |
| INT8 activations (QDQ-style, alpha=4) | 3.3e-02 | 1.1e-02 | 0.999917 |
| Real QDQ INT8 graph (`onnxruntime.quantization`, alpha=4) | 3.8e-02 | 1.2e-02 | 0.999902 |

* alpha=0 is a **bit-exact identity** in FP32 ORT and with an INT8 steering
  vector; under full activation quantization (QDQ) the alpha=0 error (~2e-02)
  is the hidden-state quantization noise floor itself, not a steering artifact.
* The steering signal (unit vector, |alpha|≥1) is **~25x larger** than the
  INT8-vector quantization error — steering remains clearly distinguishable.
* Caveat: these numbers characterize the standalone op. They do **not**
  predict end-to-end error in a fully quantized LLM, where residual-stream
  scales are set by the surrounding network.

## On-device result (Snapdragon X Elite via AI Hub Workbench)

The full opt-in pipeline has been executed once against a cloud-hosted
Snapdragon X Elite CRD (job record: `artifacts/ai_hub_jobs.json`):

* **Compile + link**: ONNX → `qnn_dlc` → QNN context binary — accepted without
  modification (jobs `jgn7qjnmp`, `jp2vd2wm5`).
* **On-device inference** (`jgk90j4o5`): max abs diff vs local FP32 ONNX
  Runtime **3.879e-3**, mean **4.7e-4** — exactly fp16 rounding (the HTP
  executes fp16; 1 ulp at \|x\|≈4 is 3.9e-3). α, steering vector, and mask
  were fed as runtime inputs.
* **Profile** (`j5w1zj7jg`): **every op placed on the NPU** (no CPU
  fallback), estimated inference 180 µs, peak memory ~14.4 MB, first/warm
  load 407/240 ms.

The stronger proof run in `docs/results/steering_device_proof.json` executes
the same compiled graph twice on the Snapdragon X Elite NPU with identical
hidden state, steering vector, and mask:

* **alpha=0 identity** (`j56dk06np`): device output matches the input hidden
  state within fp16 rounding, max abs error **1.13e-3**, mean **1.40e-4**.
* **alpha=4 steering delta** (`jg9x67xlg`): device
  `output(alpha=4) - output(alpha=0)` matches the expected
  `4 * mask * steering` update with max abs error **1.07e-3**, mean
  **7.37e-5**, cosine on active positions **0.9999986**.
* **Mask behavior**: the first 8 prompt positions had mask=0 and exactly zero
  measured delta; the last 8 positions had mask=1 and L2 delta **11.3137**.
* **Profile** (`jp1vrkj8p`): **all 5 ops placed on the NPU**, estimated
  inference **167 µs**, peak memory **14.7 MB**.
* Caveats: numbers are for the **standalone injection op**, not steering
  inside a full LLM; a CRD is real silicon but cloud-hosted — thermals and
  concurrent-workload behavior are not characterized.

## Qualcomm path (Milestone C)

`detect_environment.py` reports which tools are present (qai-hub,
qai-hub-models, ONNX Runtime QNN EP, QAIRT SDK converters, ExecuTorch Qualcomm
backend, AI Hub token — presence only, values never printed).

* **Cloud (recommended for this POC):** `submit_ai_hub.py` (tested against
  `qai-hub==0.52.0`, pinned) uses `qai_hub.Client()` and the current
  `submit_compile_and_link_jobs()` route — ONNX is compiled as `qnn_dlc` and
  linked into a QNN context binary (the old direct `qnn_context_binary`
  compile target is deprecated); inference/profile jobs run on the **LinkJob's**
  target model. `--multi-graph` links prefill `[1,16,D]` + decode `[1,1,D]`
  as two graphs of one weight-shared context binary, mirroring how Qualcomm
  LLMs package prompt/token graphs. Devices are resolved from the **live**
  device list. Dry-run by default; `--submit` required to spend quota.
* **Local (optional):** `qnn_emulation.py` is discovery-driven: it reads the
  installed QAIRT SDK version, checks each tool's `--help` for the flags it
  actually supports (`--dlc_path` for context generation on current releases),
  runs the generated context via `qnn-net-run --retrieve_context` (a DLC is not
  a directly executable QNN model), and compares against ONNX Runtime. Without
  an SDK it prints a command template explicitly marked **unvalidated**. The
  ONNX Runtime QNN EP path sets `session.disable_cpu_ep_fallback=1` so
  unsupported nodes fail hard instead of silently running on CPU. This path is
  not required for the core tests or AI Hub cloud demo.
* **Environments:** model/eval work runs in the main venv (Windows, Python
  3.13, `requirements.lock.txt`); Qualcomm tooling has a dedicated WSL
  Ubuntu Python 3.10 env (`scripts/setup_wsl_qualcomm.sh`,
  `requirements-qualcomm.lock.txt`) matching Qualcomm's Linux-first,
  Python-3.10-era tooling.
* Desktop emulation validates **graph acceptance and functional numerics
  only** — not Snapdragon latency, memory pressure, power, scheduling, driver
  behavior, or device-specific fusions.

Full-model integration (where exactly the op would go in Qualcomm's
`qai-hub-models` LLM export, which artifacts are Genie/QNN vs GGUF, and what
must be recompiled) is analyzed in **`docs/qualcomm_integration_plan.md`**.

## Repository layout

```
configs/            model configs (Qwen3-0.6B primary, Qwen2.5-0.5B fallback, Llama-3.2-1B optional)
data/               contrast pairs + eval prompts (JSONL)
src/steering_poc/   activation steering: extract / inject / generate / evaluate / export_onnx / plots
src/steering_poc/qualcomm/   detect_environment / submit_ai_hub / prove_steering_ai_hub / qnn_emulation
src/split_compute/  SEPARATE experiment: two-device pipeline split (export_split /
                    verify_local / submit_split) — see "Split-compute experiment" below
scripts/            setup_wsl*.sh, run_local_poc.{sh,ps1}, run_onnx_validation.sh, run_split_experiment.sh
tests/              injection, zero-alpha, hook cleanup, ONNX equivalence, split-model parity
artifacts/          generated vectors/graphs/results (gitignored); artifacts/split/ for the split experiment
docs/results/       small committed result summaries and plots for teammates
patches/            patch files applied to external dependencies for experiments
```

## Artifact policy

This repository is source-first. Heavy/generated files stay local under
`artifacts/` and are intentionally gitignored: steering vectors (`.pt`), ONNX
graphs, NumPy tensors, DLC/context binaries, model weights, and full generation
logs. Small result summaries suitable for review are committed under
`docs/results/`.

To regenerate local artifacts:

```powershell
python -m steering_poc.export_onnx --hidden-size 1024
python -m steering_poc.compare_backends --hidden-size 1024 --vector artifacts/vector_layer_14.pt
python -m split_compute.export_split --config configs/qwen3_0_6b.yaml --split-layer 14 --seq-len 32
python -m split_compute.verify_local --config configs/qwen3_0_6b.yaml
```

The split export loads Qwen3-0.6B and writes large ONNX files, so teammates
should regenerate it only when they need to rerun the split pipeline.

## Qwen3-1.7B steering-in-deployed-LLM experiment

The next integration target is Qwen3-1.7B rather than Llama-3.2-1B, avoiding
Meta-gated weights while using an editable Qualcomm `qai-hub-models` QAIRT/Genie
recipe. This repo does not vendor `qai-hub-models`; it clones an ignored
external checkout and applies `patches/qai_hub_models_qwen3_1_7b_steering.patch`.

```bash
bash scripts/setup_wsl_qwen3_patch.sh
source ~/.venvs/qwen3_patch310/bin/activate
bash scripts/run_qwen3_1_7b_patch_experiment.sh
```

Details and limitations are in `docs/qwen3_1_7b_steering_patch.md`.

Current result: baseline, alpha=0, and alpha=4 compiled-constant Qwen3-1.7B
GenieX QAIRT `w4a16` bundles export, compile, link, and download successfully.
The alpha=4 prompt graphs for all four context binaries also profile with all
recorded ops on NPU. The shareable job summary is in
`docs/results/qwen3_1_7b_patch/`. The remaining gate is Genie runtime comparison
on identical prompts.

## Split-compute experiment (`src/split_compute/`)

Question: can decoder layers 0..13 of Qwen3-0.6B run on a **phone** NPU and
layers 14..27 on a **laptop** NPU, with only the residual hidden state
crossing the device boundary? This mirrors how Qualcomm's own LLM exports
split models into sequential context binaries with the hidden state as
boundary graph I/O.

```powershell
python -m split_compute.export_split --config configs/qwen3_0_6b.yaml --split-layer 14 --seq-len 32
python -m split_compute.verify_local --config configs/qwen3_0_6b.yaml   # free local parity gate
python -m split_compute.submit_split                                    # dry run
python -m split_compute.submit_split --submit --profile                 # phone + laptop CRDs
```

Scope, stated precisely: fixed-shape prompt-processor split (T=32, no KV
cache); AI Hub has no live device-to-device transport, so the cloud round
trip stands in for the network — this verifies **functional/numerical
viability on real silicon**, not live transport latency or streaming decode.
Per-token boundary payload at fp16 is `hidden_size * 2` bytes (2 KB for this
model). Result summaries are committed under `docs/results/split/`; generated
runtime artifacts are written under `artifacts/split/`.

### Split-compute results (run 2026-07-21, job URLs in the JSON record)

Local gate first: the ONNX PartA→PartB chain matches the full HF fp32 model
to max logit diff **7.9e-5** with 100% argmax agreement — the split itself is
numerically clean before any hardware enters the picture.

On real silicon (phone = Snapdragon 8 Elite QRD running layers 0–13 + embed;
laptop = Snapdragon X Elite CRD running layers 14–27 + head, fed the phone's
actual activations):

| Check | Result |
|---|---|
| Compile + link, both halves, both devices | ✅ (jobs `jp49702v5`/`jpelq3n1g`, `jgk91mjw5`/`j577m3wv5`) |
| Boundary hidden state (phone) vs local fp32 | max abs err 4.16 **on a 6,520-magnitude outlier channel** → relative 6.4e-4 ≈ fp16 precision; mean abs err 3.2e-3 |
| Chained logits vs local single-device fp32 | max abs diff 0.120, mean 6.4e-3, KL(last position) 2.1e-5 |
| **Greedy argmax agreement, all 32 positions** | **100%** — the two-device pipeline predicts the identical tokens |
| NPU placement | PartA: 829/829 ops NPU (phone) · PartB: 832/832 ops NPU (laptop) |
| Compute latency (32-token prefill) | PartA 6.11 ms on the phone · PartB 16.38 ms on the laptop (LM head dominates) · peak mem 118 MB / 36 MB |

What this demonstrates: layer-partitioned execution of one LLM across a
phone NPU and a laptop NPU is **functionally exact to fp16 precision**, with
a 2 KB/token boundary payload. What it does not demonstrate: live transport
latency (the AI Hub cloud round trip stood in for the network), streaming
KV-cached decode (these are prompt-processor graphs), or power. A real-time
demo needs an orchestrator + socket on both devices; end-to-end per-token
time would be roughly `t_A + t_B + network RTT`.

## Five-minute demo flow

Slide deck: `docs/demo_slides.pptx` (claims mirror this README — measured
results and not-yet-demonstrated items are kept on separate slides).

1. `pytest -q` — 31 green tests, including zero-alpha identity, hook cleanup,
   and censoring accounting (30 s).
2. `python -m steering_poc.generate ... --alpha -4 0 4` — live: same prompt,
   concise / baseline / verbose answers, with token counts and truncation
   flags printed (1 min).
3. Show `artifacts/dose_response.png` — paired Δwords vs α with bootstrap 95%
   CIs over 32 prompts; layer 7's CIs exclude zero at α=−4/−2/+4, Spearman
   ρ=+0.25; weak layers shown too, not a cherry-picked sample (1 min).
4. Show `steering_injection_prefill.onnx` (Netron or `export_onnx` output):
   the whole intervention is two NPU-trivial ops, `Mul` + `Add`, with runtime
   `alpha`/`steering` inputs; point at the bit-exact ORT parity and INT8 table (1 min).
5. Open the AI Hub job pages (`artifacts/ai_hub_jobs.json`): the op compiled,
   linked, ran, and profiled on a real Snapdragon X Elite — 100% NPU
   placement, 180 µs, fp16-exact — then show the honest full-model
   integration plan (`docs/qualcomm_integration_plan.md`) (1 min).

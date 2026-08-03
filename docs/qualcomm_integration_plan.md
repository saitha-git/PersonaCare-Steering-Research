# Milestone C: Full Qualcomm integration plan (analysis — NOT implemented)

This document records what we verified in Qualcomm's maintained sources
(`github.com/qualcomm/ai-hub-models`, `main` branch as of 2026-07-18, package
under `src/`; `github.com/quic/ai-hub-apps` llm_on_genie tutorial) and derives
a concrete patch plan. Nothing here has been executed on Qualcomm hardware.

## 1. What Qualcomm's LLM artifacts actually are (they are NOT interchangeable)

| Model | On-device format | Editable graph? |
|---|---|---|
| **Qwen3-0.6B** (our POC model) | **Prebuilt GGUF Q4_0** (`unsloth/Qwen3-0.6B-GGUF`, 429 MB) run by the **GenieX llama.cpp** backend (`orchestrator_runtimes: [geniex_llamacpp]`, `genie_compatible: false`). Fetched via `qai-hub-models fetch Qwen3-0.6B --runtime geniex_llamacpp --precision q4_0`. | **No.** The model directory (`src/qai_hub_models/models/qwen3_0_6b/`) contains only metadata YAMLs — no `model.py`/`export.py`. Presence in the AI Hub catalog does *not* mean an editable QNN graph is downloadable. A forward hook obviously cannot attach to a GGUF; steering there would mean patching llama.cpp's graph build (a different project). |
| **Qwen3-1.7B and larger** | Self-exportable via the shared recipe `src/qai_hub_models/models/_shared/qwen3/model.py` (`class Qwen3Base(LLMBase)`, wrapping HF `Qwen3ForCausalLM`, with QC adaptations in `model_adaptations.py`); runtimes `genie`, `geniex_qairt`, `geniex_llamacpp`; precision `w4a16`. | **Yes** (QAIRT/Genie path). |
| **Llama-3.2-1B-Instruct** | Best-documented maintained recipe: `src/qai_hub_models/models/llama_v3_2_1b_instruct/model.py` — `NUM_LAYERS=16`, `HIDDEN_SIZE=2048`, `NUM_SPLITS=3`, `NUM_LAYERS_PER_SPLIT=8`; precisions `w4`/`w4a16`; QNN context binaries + Genie config (`llm_on_genie` tutorial: `genie_bundle` with `ctx-bins: [..._part_1_of_3.bin, ...]`, run by `genie-t2t-run`). | **Yes** (gated Meta weights required). |

Consequence for this POC: **our behavioral demo model (Qwen3-0.6B) and the
cleanest Qualcomm integration target are different artifacts.** The preferred
next path is now **Qwen3-1.7B** through the QAIRT/Genie recipe because it avoids
Meta-gated weights while staying in the Qwen family. See
`docs/qwen3_1_7b_steering_patch.md` and
`patches/qai_hub_models_qwen3_1_7b_steering.patch`.

## 2. Verified structure of the QAIRT/Genie LLM export

* **Prompt processor vs token generator:** not separate components but separate
  **graphs per split**, named by `DynamicSplitPartBase._build_graph_names`
  (`_shared/llm/model.py`): e.g. `prompt_ar128_cl4096_1_of_3` and
  `token_ar1_cl4096_1_of_3`, from `DEFAULT_EXPORT_SEQUENCE_LENGTHS = [128, 1]`.
  Shapes are **fixed**: prompt seq len 128, token-gen seq len 1, context 4096 —
  matching our fixed-shape prefill/decode ONNX exports.
* **The residual hidden state IS exposed at split boundaries.**
  `LLMPartBase.get_graph_input_spec` (`_shared/llm/model.py`): every non-KV,
  non-mask, non-rope input of parts 2..N is "an intermediate hidden state from
  the previous part", shape `(1, sequence_length, hidden_size)`, float32 at
  the spec level (float16 on device for `w4`, uint16 quantized for `w4a16`,
  per `_infer_output_specs` in `_shared/llm/export.py`).
* **Where splits are cut:** `_shared/llm/split_onnx_utils/utils.py` —
  `get_split_tensors()` documents that valid splitting points are the
  **post-FFN residual Add outputs** between decoder layers; `is_residual_add()`
  pattern-matches them and `split_onnx()` cuts every `num_layers_per_split`
  layers.
* **The PyTorch wrapper that gets exported:** `class LLMBase` in
  `_shared/llm/model.py`; its `forward()` calls the wrapped HF
  `*ForCausalLM`; ONNX export happens in `get_onnx_model()` (torch.onnx
  dynamo exporter, opset 18). Architecture adaptations are applied by
  monkey-patching (`Qwen3Base.monkey_patch()`, `QCQwen3ForCausalLM`,
  `SHAQwen3Attention`; analogous `Llama3Base` classes).

## 3. Where the steering op would be inserted

Target: `steered = hidden + alpha * steering_vector` on the residual stream
after decoder layer `i` (for Qwen3-1.7B, layer 7 is the first target; hidden
dim 2048).

**Patch point (pre-export, PyTorch level):** in the monkey-patch layer of the
chosen recipe — e.g. wrap decoder layer `i`'s forward in
`Qwen3Base.monkey_patch()` / the `QCQwen3ForCausalLM` adaptation (Qwen3) or
`Llama3Base`'s equivalent (Llama) — so the exported ONNX contains our exact
`Mul`+`Add` subgraph with `steering` and `alpha` as extra graph inputs.
This is the same op we already validated standalone (bit-exact in ORT,
INT8-robust).

**Verified pitfall:** `split_onnx` finds split points by counting residual-Add
pairs (`is_residual_add(strict=True)`, every second match). A bare extra Add
on the residual stream between layers could be miscounted as a residual add
and silently shift the split boundaries. Two safe options:
  1. add the steering delta **inside** layer `i` (onto the FFN output, before
     its residual add) so the residual-Add pattern count is unchanged; or
  2. insert exactly at a split boundary and instead patch the **part
     boundary**, see CPU-side option below.

**Runtime inputs:** every graph input of a QNN context binary is fixed at
compile time, so `steering` `[1,1,D]` and `alpha` (scalar) can be *declared*
as inputs and fed per-inference — on the QNN/QAIRT API level. However, the
**Genie runner wires graph I/O by naming convention** (KV cache, attention
mask, rope, inter-part hidden states); we found **no documented mechanism to
feed arbitrary extra inputs** through `genie_config.json` / `genie-t2t-run`.
GenieX (`geniex_qairt`) is newer; same caveat. Driving the patched binaries
directly through the QNN API (bypassing Genie's dialog layer) is the fallback,
at the cost of reimplementing the sampling loop.

**Recompilation scope:** the modified ONNX for the affected split must be
re-converted and re-compiled to a context binary (per-SoC). Untouched splits'
binaries remain valid, but the `genie_bundle` must be reassembled. For `w4a16`
the new Add sits in the quantized activation domain — the steering input would
need matching quantization encodings (our INT8 experiments are the small-scale
rehearsal of exactly this issue).

## 4. Deployment design options (evaluated)

| # | Design | On-NPU? | Flexibility | Effort | Notes |
|---|---|---|---|---|---|
| 1 | Runtime `steering` + runtime `alpha` as graph inputs | Yes (extra DMA of D floats/step is negligible) | Full | Medium | Blocked at the *Genie* layer (no documented extra-input plumbing); fine via raw QNN API |
| 2 | Compiled-in constant vector, runtime `alpha` | Yes | Change vector ⇒ recompile split | Low-Medium | Alpha as a 1-element input; simplest graph change |
| 3 | K compiled vectors + runtime coefficients `hidden + Σ coeff_i · v_i` | Yes | K behaviors + blending | Medium | **Recommended hackathon target**: one `[K]` input, MatMul `[K]×[K,D]` or K Mul/Adds; still trivial ops |
| 4 | CPU-side intervention at the split boundary (part1 → host add → part2) | Boundary op on CPU | Full, no recompilation *if it works* | Low-Medium | **PROPOSED, UNVERIFIED.** The hidden state *is* exposed between parts (verified in source), but two prerequisites are unproven: (a) driving the per-part context binaries directly via the raw QNN API with a hand-written orchestration loop (Genie does not support host interception), and (b) intercepting w4a16 activations means the boundary tensor is uint16-quantized — the host add must requantize with the correct encodings, which we have not extracted. Costs an NPU→CPU→NPU round-trip per step |
| 5 | Patch source model pre-export + recompile (this plan's §3) | Yes | Per design 1-3 | High | The clean end state |

**Recommendation:** implement a compiled-constant Qwen3-1.7B patch first. This
proves the Add can survive the `qai-hub-models` export/recompile path without
depending on Genie runtime extra-input plumbing. Then graduate to option 3 for
runtime coefficients.

## 5. Exact next commands (once credentials/hardware exist)

```bash
bash scripts/setup_wsl_qwen3_patch.sh
source ~/.venvs/qwen3_patch310/bin/activate
bash scripts/run_qwen3_1_7b_patch_experiment.sh
```

## 6. What remains unproven until real hardware

* Genie/GenieX accepting a rebuilt bundle with extra inputs (or QNN-API driving).
* w4a16 quantization encodings for the steering input at full-model scale.
* Any latency/power/memory impact on Snapdragon (x86 emulation cannot show this).
* Unverified detail from the source read: exact per-part layer distribution
  (embedding-only vs embedding+8-layers in part 1) — check before picking the
  patch layer.

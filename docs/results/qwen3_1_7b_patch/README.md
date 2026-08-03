# Qwen3-1.7B GenieX QAIRT Steering Patch Export Results

Run date: 2026-07-22.

This records the first successful full-model export/compile/link of Qwen3-1.7B
through Qualcomm `qai-hub-models` with the steering patch applied.

## What Succeeded

Three GenieX QAIRT `w4a16` bundles were exported for Qualcomm Snapdragon 8
Elite for Galaxy with sequence lengths `1,128` and context length `512`:

| phase | status | link jobs |
|---|---|---|
| baseline | success | `j5mdj2dq5`, `jgn7jy7mp`, `jprnzqne5`, `jp2v26vm5` |
| alpha_0 patched | success | `jpv9303j5`, `jgjwxzxx5`, `jpel9e91g`, `jgz4eoekp` |
| alpha_4 steered | success | `jprnk2305`, `jp2v89yr5`, `jpy7ej38p`, `jp0vy209g` |

Each bundle contains four downloaded QNN context binaries:

| file | bytes |
|---|---:|
| `part1_of_4.bin` | 622391296 |
| `part2_of_4.bin` | 263032832 |
| `part3_of_4.bin` | 263024640 |
| `part4_of_4.bin` | 525799424 |

AI Hub reported QAIRT `2.45.0.260326154327`.

## Local Artifact Paths

Generated bundles are intentionally gitignored:

```text
artifacts/qwen3_1_7b_patch/baseline/qwen3_1_7b-geniex_qairt-w4a16-qualcomm_snapdragon_8_elite_for_galaxy
artifacts/qwen3_1_7b_patch/alpha_0/qwen3_1_7b-geniex_qairt-w4a16-qualcomm_snapdragon_8_elite_for_galaxy
artifacts/qwen3_1_7b_patch/alpha_4/qwen3_1_7b-geniex_qairt-w4a16-qualcomm_snapdragon_8_elite_for_galaxy
```

Detailed job IDs and file sizes are in `export_summary.json`.

## Alpha=4 Prompt Profile

The alpha=4 steered bundle was profiled on AI Hub for the prompt graph in each
of the four context binaries. AI Hub requires selecting a graph explicitly for
these multi-graph context binaries; the helper uses
`--qnn_option context_enable_graphs=<graph_name>`.

| part | graph | profile job | time us | peak memory bytes | ops |
|---|---|---|---:|---:|---|
| 1 | `prompt_ar128_cl512_1_of_4` | `jgn724mvp` | 77 | 127205376 | `NPU: 3` |
| 2 | `prompt_ar128_cl512_2_of_4` | `jgk9z1925` | 6750 | 159121408 | `NPU: 5063` |
| 3 | `prompt_ar128_cl512_3_of_4` | `jp3wkdo35` | 6740 | 159223808 | `NPU: 5063` |
| 4 | `prompt_ar128_cl512_4_of_4` | `jgz4e6exp` | 13448 | 192528384 | `NPU: 4055` |

All profiled prompt graphs report `all_ops_on_npu: true`. The full profile
record is in `profile_alpha4_prompt.json`.

## What This Proves

The Qwen3-1.7B QAIRT/Genie export path accepts the patched model and produces
complete context-binary Genie bundles for baseline, alpha=0, and alpha=4
compiled-constant steering variants. The alpha=4 prompt graphs also profile
successfully on a Snapdragon 8 Elite class device with all recorded ops on NPU.

## What Is Still Missing

This does not yet prove behavioral steering through Genie runtime execution.
The next gate is to run baseline, alpha=0, and alpha=4 bundles through Genie on
the same prompt and compare outputs. The expected checks are:

1. baseline and alpha=0 should match or be very close;
2. alpha=4 should differ in the steering direction;
3. profiling should confirm full-bundle device execution characteristics.

Use the committed helper on a device with the Genie SDK:

```bash
GENIE_BIN=/path/to/genie-t2t-run bash scripts/run_qwen3_1_7b_genie_compare.sh
```

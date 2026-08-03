"""Layer x alpha sweep with censoring-aware quantitative evaluation.

Per (layer, alpha, prompt) we record generated token IDs, decoded text,
generated_tokens, reached_eos, and hit_max_new_tokens. A generation that hit
max_new_tokens is CENSORED: its length is a lower bound and is excluded from
word-count statistics (it is never reported as a true response length).

Statistics per (layer, alpha):
  * mean words over uncensored generations (+ how many were censored);
  * mean PAIRED word-count change vs the alpha=0 baseline on the same prompt
    (pairs where either side is censored are dropped), with a percentile
    bootstrap 95% CI over prompts;
  * per layer: Spearman rank correlation between alpha and word count.

Logit metrics (max/mean abs diff, cosine, top-k, KL) come from a batched
prefill forward vs the unsteered baseline. Zero-dose identity is checked with
the hook installed at alpha=0 against the hook-free baseline.

Baseline generations are produced ONCE hook-free and reused as the alpha=0
row for every layer (the hook at alpha=0 is verified bit-exact separately).

Usage:
    python -m steering_poc.evaluate --config configs/qwen3_0_6b.yaml \
        --layers 7 14 21 --prompts data/eval_prompts.jsonl
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import load_config, load_jsonl, save_json, vector_path
from .generate import build_prompt_ids, generate_batch, load_vector
from .inject import SteeringInjector
from .metrics import bootstrap_mean_ci, logit_metrics, spearman, verbosity_score
from .model_adapter import load_model_and_tokenizer


def _left_pad_batch(tokenizer, prompts, gen_cfg, device):
    id_batches = [build_prompt_ids(tokenizer, p, gen_cfg, device) for p in prompts]
    pad_id = tokenizer.pad_token_id
    max_len = max(ids.shape[1] for ids in id_batches)
    input_ids = torch.full((len(prompts), max_len), pad_id, dtype=torch.long,
                           device=device)
    attention_mask = torch.zeros_like(input_ids)
    for i, ids in enumerate(id_batches):
        input_ids[i, max_len - ids.shape[1]:] = ids[0]
        attention_mask[i, max_len - ids.shape[1]:] = 1
    return input_ids, attention_mask, id_batches


@torch.no_grad()
def _batch_next_token_logits(model, input_ids, attention_mask):
    logits = model(input_ids, attention_mask=attention_mask).logits
    return logits[:, -1, :].float().cpu()  # left padding: -1 is the last real token


@torch.no_grad()
def _batched_nll(model, prompt_id_batches, records, device, chunk=4):
    """Mean per-token NLL of each generated continuation under the CURRENT
    (unsteered) model, conditioned on its prompt. Empty continuations -> nan."""
    out = []
    for start in range(0, len(records), chunk):
        chunk_prompts = prompt_id_batches[start:start + chunk]
        chunk_recs = records[start:start + chunk]
        fulls, cont_lens, prompt_lens = [], [], []
        for pids, rec in zip(chunk_prompts, chunk_recs):
            cont = torch.tensor(rec["token_ids"], dtype=torch.long, device=device)
            fulls.append(torch.cat([pids[0], cont]))
            prompt_lens.append(pids.shape[1])
            cont_lens.append(len(rec["token_ids"]))
        max_len = max(f.shape[0] for f in fulls)
        batch = torch.zeros(len(fulls), max_len, dtype=torch.long, device=device)
        mask = torch.zeros_like(batch)
        for i, f in enumerate(fulls):  # right padding for scoring
            batch[i, :f.shape[0]] = f
            mask[i, :f.shape[0]] = 1
        logits = model(batch, attention_mask=mask).logits.float()
        for i in range(len(fulls)):
            P, C = prompt_lens[i], cont_lens[i]
            if C == 0:
                out.append(float("nan"))
                continue
            pred = logits[i, P - 1 : P + C - 1, :]
            tgt = batch[i, P : P + C]
            out.append(F.cross_entropy(pred, tgt).item())
        del logits
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--alphas", type=float, nargs="*", default=None)
    parser.add_argument("--prompts", default="data/eval_prompts.jsonl")
    parser.add_argument("--max-prompts", type=int, default=32)
    parser.add_argument("--positions", default=None, choices=["all", "last"])
    parser.add_argument("--out-prefix", default="artifacts/eval")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    alphas = [float(a) for a in
              (args.alphas if args.alphas is not None else cfg["injection"]["alphas"])]
    positions = args.positions or cfg["injection"]["positions"]
    tol = float(cfg["evaluation"]["zero_alpha_tolerance"])
    top_k = int(cfg["evaluation"]["top_k"])
    gen_cfg = cfg["generation"]
    max_new = int(gen_cfg["max_new_tokens"])

    prompts = [r["prompt"] for r in load_jsonl(args.prompts)][: args.max_prompts]
    model, tokenizer, adapter = load_model_and_tokenizer(cfg["model"])
    device = adapter.device
    print(f"{len(prompts)} prompts, layers {args.layers}, alphas {alphas}, "
          f"max_new_tokens={max_new}, positions={positions}", flush=True)

    input_ids, attention_mask, prompt_id_batches = _left_pad_batch(
        tokenizer, prompts, gen_cfg, device
    )

    # Hook-free baseline: generations + next-token logits (reused for alpha=0).
    base_records = generate_batch(model, tokenizer, prompts, gen_cfg, device)
    base_logits = _batch_next_token_logits(model, input_ids, attention_mask)
    base_nll = _batched_nll(model, prompt_id_batches, base_records, device)
    n_base_censored = sum(r["hit_max_new_tokens"] for r in base_records)
    print(f"baseline: {n_base_censored}/{len(prompts)} censored at "
          f"{max_new} tokens", flush=True)

    rows, texts = [], []
    for layer in args.layers:
        vec_file = vector_path(layer)
        if not vec_file.exists():
            raise FileNotFoundError(
                f"{vec_file} not found — run steering_poc.extract for layer {layer}"
            )
        vector, vmeta = load_vector(
            str(vec_file), adapter, cfg["injection"]["normalize_vector"]
        )

        # Zero-dose identity: hook installed, alpha=0, whole batch.
        with SteeringInjector(adapter, vector, layer, 0.0, positions=positions):
            zero_logits = _batch_next_token_logits(model, input_ids, attention_mask)
        zero_dev = (zero_logits - base_logits).abs().max().item()
        print(f"layer {layer:>2}: zero-dose max dev = {zero_dev:.2e} "
              f"({'OK' if zero_dev <= tol else 'FAIL'})", flush=True)

        for alpha in alphas:
            if alpha == 0.0:
                records, logits, nlls = base_records, base_logits, base_nll
            else:
                with SteeringInjector(adapter, vector, layer, alpha,
                                      positions=positions):
                    records = generate_batch(model, tokenizer, prompts, gen_cfg,
                                             device)
                    logits = _batch_next_token_logits(model, input_ids,
                                                      attention_mask)
                nlls = _batched_nll(model, prompt_id_batches, records, device)

            for i, (prompt, rec) in enumerate(zip(prompts, records)):
                lm = logit_metrics(base_logits[i], logits[i], top_k=top_k)
                censored = rec["hit_max_new_tokens"]
                base_censored = base_records[i]["hit_max_new_tokens"]
                words = verbosity_score(rec["text"])
                base_words = verbosity_score(base_records[i]["text"])
                rows.append({
                    "layer": layer,
                    "alpha": alpha,
                    "prompt_id": i,
                    "prompt": prompt,
                    "generated_tokens": rec["generated_tokens"],
                    "reached_eos": rec["reached_eos"],
                    "hit_max_new_tokens": censored,
                    "words_uncensored": words if not censored else None,
                    "words_lower_bound": words,
                    "paired_delta_words": (words - base_words)
                    if not (censored or base_censored) else None,
                    "nll_under_baseline": nlls[i],
                    "zero_alpha_max_dev": zero_dev,
                    "zero_alpha_ok": zero_dev <= tol,
                    "vector_raw_norm": vmeta.get("raw_norm"),
                    **lm,
                })
                texts.append({
                    "layer": layer, "alpha": alpha, "prompt": prompt,
                    "text": rec["text"], "token_ids": rec["token_ids"],
                    "generated_tokens": rec["generated_tokens"],
                    "reached_eos": rec["reached_eos"],
                    "hit_max_new_tokens": censored,
                })
            n_cens = sum(r["hit_max_new_tokens"] for r in records)
            print(f"  alpha={alpha:+.1f}: {n_cens}/{len(prompts)} censored",
                  flush=True)

    out_csv = Path(f"{args.out_prefix}_metrics.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_json(texts, f"{args.out_prefix}_texts.json")

    # ---- Aggregates -------------------------------------------------------
    agg = []
    print("\nPaired word-count change vs alpha=0 (uncensored pairs only):")
    print("layer alpha  n_ok n_cens  mean_words  delta  [95% CI]")
    for layer in args.layers:
        for alpha in alphas:
            sel = [r for r in rows if r["layer"] == layer and r["alpha"] == alpha]
            unc = [r["words_uncensored"] for r in sel
                   if r["words_uncensored"] is not None]
            deltas = [r["paired_delta_words"] for r in sel
                      if r["paired_delta_words"] is not None]
            mean_d, lo, hi = bootstrap_mean_ci(deltas)
            n_cens = sum(1 for r in sel if r["hit_max_new_tokens"])
            mean_w = sum(unc) / len(unc) if unc else float("nan")
            agg.append({
                "layer": layer, "alpha": alpha,
                "n_uncensored": len(unc), "n_censored": n_cens,
                "mean_words_uncensored": mean_w,
                "mean_paired_delta_words": mean_d,
                "delta_ci95_low": lo, "delta_ci95_high": hi,
                "n_pairs": len(deltas),
            })
            print(f"{layer:>5} {alpha:>5.1f} {len(unc):>5} {n_cens:>6} "
                  f"{mean_w:>11.1f} {mean_d:>6.1f}  [{lo:.1f}, {hi:.1f}]")

    spearman_by_layer = {}
    for layer in args.layers:
        pairs = [(r["alpha"], r["words_uncensored"]) for r in rows
                 if r["layer"] == layer and r["words_uncensored"] is not None]
        rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
        spearman_by_layer[layer] = {"rho": rho, "n": len(pairs)}
        print(f"layer {layer}: Spearman(alpha, words) = {rho:+.3f} "
              f"(n={len(pairs)} uncensored generations)")

    save_json({"cells": agg, "spearman_by_layer": spearman_by_layer,
               "max_new_tokens": max_new, "n_prompts": len(prompts)},
              f"{args.out_prefix}_doseresponse.json")
    print(f"\nSaved -> {out_csv}, {args.out_prefix}_texts.json, "
          f"{args.out_prefix}_doseresponse.json")


if __name__ == "__main__":
    main()

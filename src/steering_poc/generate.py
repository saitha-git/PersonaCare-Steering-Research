"""Steered generation: sweep alpha values and print/save generations.

Usage:
    python -m steering_poc.generate --config configs/qwen3_0_6b.yaml \
        --vector artifacts/vector_layer_14.pt --layer 14 \
        --alpha -4 -2 -1 0 1 2 4 [--prompt "..."] [--positions all|last]
"""

from __future__ import annotations

import argparse

import torch

from .common import load_config, save_json
from .inject import SteeringInjector
from .model_adapter import load_model_and_tokenizer


def load_vector(path: str, adapter, normalize: bool):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    vector = payload["vector"].float()
    meta = payload.get("metadata", {})
    if meta.get("hidden_dim") not in (None, adapter.hidden_size):
        raise ValueError(
            f"Vector was extracted for hidden_dim={meta.get('hidden_dim')} but "
            f"model has hidden_size={adapter.hidden_size}"
        )
    if normalize:
        vector = vector / vector.norm()
    return vector, meta


def build_prompt_ids(tokenizer, prompt: str, gen_cfg: dict, device):
    if gen_cfg.get("chat_template", True) and tokenizer.chat_template:
        kwargs = {}
        if "enable_thinking" in (tokenizer.chat_template or ""):
            kwargs["enable_thinking"] = bool(gen_cfg.get("enable_thinking", False))
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            **kwargs,
        )
        # transformers v5 returns a BatchEncoding; v4 returned a bare tensor.
        if not isinstance(ids, torch.Tensor):
            ids = ids["input_ids"]
    else:
        ids = tokenizer(prompt, return_tensors="pt").input_ids
    return ids.to(device)


def _eos_ids(model, tokenizer) -> set[int]:
    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    return set(eos) if isinstance(eos, (list, tuple)) else {int(eos)}


def _censoring_record(gen_ids: list[int], eos_ids: set[int], pad_id, max_new: int,
                      tokenizer) -> dict:
    """Token-level accounting for one generated continuation.

    reached_eos / hit_max_new_tokens are mutually exclusive; a generation that
    hit the cap is CENSORED — its length is a lower bound, not a measurement.
    """
    # Strip right padding that batched generate() adds after a sequence finishes.
    if pad_id is not None and pad_id not in eos_ids:
        while gen_ids and gen_ids[-1] == pad_id:
            gen_ids = gen_ids[:-1]
    n = len(gen_ids)
    eos_pos = next((i for i, t in enumerate(gen_ids) if t in eos_ids), None)
    if eos_pos is not None:
        kept = gen_ids[:eos_pos]
        reached_eos = True
    else:
        kept = gen_ids
        reached_eos = False
    text = tokenizer.decode(kept, skip_special_tokens=True)
    return {
        "token_ids": kept,
        "text": text,
        "generated_tokens": len(kept),
        "reached_eos": reached_eos,
        "hit_max_new_tokens": (not reached_eos) and n >= max_new,
    }


@torch.no_grad()
def generate_batch(model, tokenizer, prompts: list[str], gen_cfg: dict,
                   device) -> list[dict]:
    """Batched greedy generation with left padding.

    Returns one record per prompt with token_ids / text / generated_tokens /
    reached_eos / hit_max_new_tokens (see _censoring_record).
    """
    max_new = int(gen_cfg.get("max_new_tokens", 512))
    id_batches = [build_prompt_ids(tokenizer, p, gen_cfg, device) for p in prompts]
    pad_id = tokenizer.pad_token_id
    max_len = max(ids.shape[1] for ids in id_batches)
    input_ids = torch.full((len(prompts), max_len), pad_id, dtype=torch.long,
                           device=device)
    attention_mask = torch.zeros_like(input_ids)
    for i, ids in enumerate(id_batches):  # left padding
        input_ids[i, max_len - ids.shape[1]:] = ids[0]
        attention_mask[i, max_len - ids.shape[1]:] = 1

    do_sample = bool(gen_cfg.get("do_sample", False))
    if do_sample:
        torch.manual_seed(int(gen_cfg.get("seed", 0)))
    out = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new,
        do_sample=do_sample,
        temperature=1.0 if do_sample else None,
        top_p=1.0 if do_sample else None,
        top_k=None,
        pad_token_id=pad_id,
    )
    eos_ids = _eos_ids(model, tokenizer)
    records = []
    for i in range(len(prompts)):
        gen_ids = out[i, max_len:].tolist()
        records.append(
            _censoring_record(gen_ids, eos_ids, pad_id, max_new, tokenizer)
        )
    return records


@torch.no_grad()
def generate_once(model, tokenizer, input_ids, gen_cfg: dict) -> str:
    do_sample = bool(gen_cfg.get("do_sample", False))
    if do_sample:
        torch.manual_seed(int(gen_cfg.get("seed", 0)))
    out = model.generate(
        input_ids,
        max_new_tokens=int(gen_cfg.get("max_new_tokens", 100)),
        do_sample=do_sample,
        temperature=1.0 if do_sample else None,
        top_p=1.0 if do_sample else None,
        top_k=None,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)


def steered_generate(
    model, tokenizer, adapter, vector, layer, alpha, prompt, gen_cfg, positions="all"
) -> dict:
    """Single-prompt steered generation; returns a full censoring record."""
    if alpha == 0.0:
        # Explicit no-hook baseline; SteeringInjector(alpha=0) is numerically
        # identical (verified by tests) but this keeps the baseline hook-free.
        return generate_batch(model, tokenizer, [prompt], gen_cfg, adapter.device)[0]
    with SteeringInjector(adapter, vector, layer, alpha, positions=positions):
        return generate_batch(model, tokenizer, [prompt], gen_cfg, adapter.device)[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--vector", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--alpha", type=float, nargs="+", default=None)
    parser.add_argument("--prompt", default="How do I make coffee?")
    parser.add_argument("--positions", default=None, choices=["all", "last"])
    parser.add_argument("--out", default="artifacts/generations.json")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    alphas = args.alpha if args.alpha is not None else cfg["injection"]["alphas"]
    positions = args.positions or cfg["injection"]["positions"]

    model, tokenizer, adapter = load_model_and_tokenizer(cfg["model"])
    vector, meta = load_vector(
        args.vector, adapter, cfg["injection"]["normalize_vector"]
    )
    print(
        f"Vector: layer={meta.get('layer')} raw_norm={meta.get('raw_norm', 0):.3f} "
        f"(normalized={cfg['injection']['normalize_vector']}), "
        f"injecting at layer {args.layer}, positions={positions}"
    )

    rows = []
    for alpha in alphas:
        rec = steered_generate(
            model, tokenizer, adapter, vector, args.layer, alpha,
            args.prompt, cfg["generation"], positions,
        )
        n_words = len(rec["text"].split())
        censored = " [TRUNCATED at max_new_tokens — length is a lower bound]" \
            if rec["hit_max_new_tokens"] else ""
        rows.append({"alpha": alpha, "n_words": n_words, **rec})
        print(f"\n=== alpha={alpha:+.1f}  ({n_words} words, "
              f"{rec['generated_tokens']} tokens{censored}) ===\n{rec['text']}")

    save_json(
        {
            "prompt": args.prompt,
            "layer": args.layer,
            "positions": positions,
            "vector_metadata": meta,
            "generations": rows,
        },
        args.out,
    )
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()

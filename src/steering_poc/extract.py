"""Extract steering vectors from contrast pairs.

For each requested layer:
    vector[layer] = mean_over_pairs( act(positive)[last_tok] - act(negative)[last_tok] )

where act() is the OUTPUT hidden state of that decoder layer (see model_adapter
docstring) and last_tok is the last non-padding token.

Usage:
    python -m steering_poc.extract --config configs/qwen3_0_6b.yaml \
        --data data/contrast_pairs.jsonl [--layers 6 14 22]
"""

from __future__ import annotations

import argparse

import torch

from .capture import capture_layer_outputs, last_token_activations
from .common import file_sha256, load_config, load_jsonl, split_pairs, vector_path
from .model_adapter import load_model_and_tokenizer, resolved_revision


@torch.no_grad()
def extract_vectors(model, tokenizer, adapter, pairs, layers, batch_size=8):
    """Returns dict layer -> dict(vector [D], raw_norm, per-pair diff norms)."""
    device = adapter.device
    diffs = {L: [] for L in layers}

    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        pos_texts = [p["positive"] for p in chunk]
        neg_texts = [p["negative"] for p in chunk]

        for texts, sign_key in ((pos_texts, "pos"), (neg_texts, "neg")):
            enc = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True,
                max_length=256,
            ).to(device)
            with capture_layer_outputs(adapter, layers) as captured:
                model(**enc)
            for L in layers:
                acts = last_token_activations(
                    captured[L][0], enc["attention_mask"]
                )  # [B, D]
                diffs[L].append((sign_key, acts.float().cpu()))

    results = {}
    for L in layers:
        pos = torch.cat([a for k, a in diffs[L] if k == "pos"], dim=0)
        neg = torch.cat([a for k, a in diffs[L] if k == "neg"], dim=0)
        assert pos.shape == neg.shape
        delta = pos - neg                      # [N_pairs, D]
        vector = delta.mean(dim=0)             # [D]
        results[L] = {
            "vector": vector,
            "raw_norm": vector.norm().item(),
            "mean_pair_diff_norm": delta.norm(dim=1).mean().item(),
            "num_pairs": pos.shape[0],
        }
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="Override extraction layers from config")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tag", default="",
                        help="Optional artifact suffix, e.g. qwen3_1_7b -> "
                             "artifacts/vector_layer_7_qwen3_1_7b.pt")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    layers = args.layers if args.layers else cfg["extraction"]["layers"]

    pairs = load_jsonl(args.data)
    train_idx, holdout_idx = split_pairs(
        len(pairs),
        cfg["extraction"]["holdout_fraction"],
        cfg["extraction"]["seed"],
    )
    train_pairs = [pairs[i] for i in train_idx]
    print(
        f"{len(pairs)} pairs -> {len(train_pairs)} extraction / "
        f"{len(holdout_idx)} holdout (indices {holdout_idx})"
    )

    model, tokenizer, adapter = load_model_and_tokenizer(cfg["model"])
    print(
        f"Loaded {cfg['model']['model_id']} on {adapter.device} "
        f"({adapter.num_layers} layers, hidden {adapter.hidden_size}, "
        f"layers at '{adapter.layer_path}')"
    )

    results = extract_vectors(
        model, tokenizer, adapter, train_pairs, layers, args.batch_size
    )

    dataset_hash = file_sha256(args.data)
    for L, res in results.items():
        meta = {
            "model_id": cfg["model"]["model_id"],
            "model_revision": resolved_revision(model),
            "tokenizer_revision": getattr(tokenizer, "_commit_hash", None)
            or resolved_revision(model),
            "layer": L,
            "hidden_dim": adapter.hidden_size,
            "capture_point": f"output of {adapter.layer_path}[{L}]",
            "extraction_position": "last_non_padding_token",
            "normalization": "raw_mean_difference (normalize at injection time)",
            "concept": cfg["concept"],
            "dataset_path": args.data,
            "dataset_sha256": dataset_hash,
            "num_pairs": res["num_pairs"],
            "holdout_indices": holdout_idx,
            "seed": cfg["extraction"]["seed"],
            "dtype": "float32",
            "raw_norm": res["raw_norm"],
            "mean_pair_diff_norm": res["mean_pair_diff_norm"],
        }
        out = vector_path(L, args.tag)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"vector": res["vector"], "metadata": meta}, out)
        unit = res["vector"] / res["vector"].norm()
        print(
            f"layer {L:>2}: raw_norm={res['raw_norm']:.4f} "
            f"mean_pair_diff_norm={res['mean_pair_diff_norm']:.4f} "
            f"unit_norm={unit.norm().item():.4f} -> {out}"
        )


if __name__ == "__main__":
    main()

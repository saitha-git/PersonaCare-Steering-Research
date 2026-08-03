"""Export the two pipeline halves of Qwen3-0.6B (or config sibling) to ONNX.

Usage:
    python -m split_compute.export_split --config configs/qwen3_0_6b.yaml \
        [--split-layer 14] [--seq-len 32] [--out-dir artifacts/split]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from steering_poc.common import load_config
from steering_poc.model_adapter import resolved_revision

from .split_model import build_parts


def load_fp32_eager(model_cfg: dict):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_id"])
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_id"],
        revision=model_cfg.get("revision", "main"),
        dtype=torch.float32,
        attn_implementation="eager",  # explicit additive mask; ONNX-safe
    )
    model.eval()
    return model, tokenizer


def export_part(module, example, path: Path, input_name: str, output_name: str):
    torch.onnx.export(
        module,
        (example,),
        str(path),
        input_names=[input_name],
        output_names=[output_name],
        opset_version=17,
        dynamo=False,
    )
    import onnx

    onnx.checker.check_model(str(path))
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen3_0_6b.yaml")
    parser.add_argument("--split-layer", type=int, default=14)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--out-dir", default="artifacts/split")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_fp32_eager(cfg["model"])
    n_layers = len(model.model.layers)
    hidden = model.config.hidden_size
    print(f"{cfg['model']['model_id']}: {n_layers} layers, hidden {hidden}; "
          f"split at {args.split_layer} -> A=[0,{args.split_layer}) "
          f"B=[{args.split_layer},{n_layers}), T={args.seq_len}")

    part_a, part_b = build_parts(model, args.split_layer, args.seq_len)

    ids = torch.randint(0, model.config.vocab_size, (1, args.seq_len),
                        dtype=torch.int32)
    a_path = export_part(part_a, ids, out / "part_a.onnx", "input_ids", "hidden")
    print(f"PartA -> {a_path} ({a_path.stat().st_size / 1e6:.0f} MB)")

    with torch.no_grad():
        boundary = part_a(ids)
    b_path = export_part(part_b, boundary, out / "part_b.onnx", "hidden", "logits")
    print(f"PartB -> {b_path} ({b_path.stat().st_size / 1e6:.0f} MB)")

    meta = {
        "model_id": cfg["model"]["model_id"],
        "model_revision": resolved_revision(model),
        "num_layers": n_layers,
        "split_layer": args.split_layer,
        "seq_len": args.seq_len,
        "hidden_size": hidden,
        "vocab_size": model.config.vocab_size,
        "boundary_tensor": f"[1, {args.seq_len}, {hidden}] float32",
        "boundary_bytes_fp16_per_token": hidden * 2,
        "dtype": "float32 (ONNX); HTP executes fp16",
        "scope": "fixed-shape prompt-processor split; no KV cache",
    }
    with open(out / "split_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"metadata -> {out / 'split_meta.json'}")


if __name__ == "__main__":
    main()

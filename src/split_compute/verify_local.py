"""Local (free) parity check: ONNX PartA -> PartB chain vs the full HF model.

Must pass before any cloud submission. Also produces the reference tensors
that the AI Hub chained run is later compared against.

Usage:
    python -m split_compute.verify_local --config configs/qwen3_0_6b.yaml \
        [--dir artifacts/split] [--prompt "..."]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from steering_poc.common import load_config, save_json

DEFAULT_PROMPT = (
    "The engineers connected the phone and the laptop over the local network "
    "so that the first half of the language model could run on one device "
    "while the second half ran on the other, passing activations between them."
)


def token_ids(tokenizer, prompt: str, seq_len: int) -> np.ndarray:
    ids = tokenizer(prompt, return_tensors="np").input_ids[0]
    if len(ids) < seq_len:
        raise SystemExit(
            f"Prompt tokenizes to {len(ids)} tokens; need >= {seq_len}."
        )
    return ids[:seq_len].astype(np.int32).reshape(1, -1)


def chain_metrics(ref_logits: np.ndarray, got_logits: np.ndarray) -> dict:
    ref = torch.from_numpy(ref_logits).float()
    got = torch.from_numpy(got_logits).float()
    diff = (ref - got).abs()
    top1_ref = ref.argmax(-1)
    top1_got = got.argmax(-1)
    lp = torch.log_softmax(ref[0, -1], -1)
    lq = torch.log_softmax(got[0, -1], -1)
    kl = torch.nn.functional.kl_div(lq, lp, log_target=True,
                                    reduction="sum").item()
    return {
        "max_abs_logit_diff": diff.max().item(),
        "mean_abs_logit_diff": diff.mean().item(),
        "argmax_agreement_all_positions": (top1_ref == top1_got)
        .float().mean().item(),
        "next_token_match": bool(top1_ref[0, -1] == top1_got[0, -1]),
        "kl_last_position": kl,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen3_0_6b.yaml")
    parser.add_argument("--dir", default="artifacts/split")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args(argv)

    import onnxruntime as ort
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config(args.config)
    d = Path(args.dir)
    meta = json.loads((d / "split_meta.json").read_text())
    seq_len = meta["seq_len"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["model_id"])
    ids = token_ids(tokenizer, args.prompt, seq_len)

    # Reference: full HF model, fp32, single device.
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["model_id"], dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    with torch.no_grad():
        ref_logits = model(torch.from_numpy(ids).long()).logits.numpy()

    # Chain: ONNX PartA -> PartB on ORT CPU.
    sess_a = ort.InferenceSession(str(d / "part_a.onnx"),
                                  providers=["CPUExecutionProvider"])
    sess_b = ort.InferenceSession(str(d / "part_b.onnx"),
                                  providers=["CPUExecutionProvider"])
    boundary = sess_a.run(None, {"input_ids": ids})[0]
    chain_logits = sess_b.run(None, {"hidden": boundary})[0]

    m = chain_metrics(ref_logits, chain_logits)
    print("Local ORT chain (PartA -> PartB) vs full HF fp32:")
    for k, v in m.items():
        print(f"  {k}: {v}")

    next_tok = int(np.argmax(chain_logits[0, -1]))
    print(f"  next token (chain): {next_tok!r} -> {tokenizer.decode([next_tok])!r}")

    np.save(d / "input_ids.npy", ids)
    np.save(d / "ref_logits.npy", ref_logits)
    np.save(d / "local_boundary.npy", boundary)
    save_json({"prompt": args.prompt, "metrics": m,
               "next_token_id": next_tok,
               "next_token_text": tokenizer.decode([next_tok])},
              d / "local_verify.json")
    print(f"Saved reference tensors + {d / 'local_verify.json'}")

    ok = m["max_abs_logit_diff"] < 1e-2 and m["next_token_match"]
    if not ok:
        raise SystemExit("LOCAL PARITY FAILED — do not submit to AI Hub.")
    print("LOCAL PARITY OK")


if __name__ == "__main__":
    main()

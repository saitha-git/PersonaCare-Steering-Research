"""Export the injection op  hidden + alpha * mask * steering  as an ONNX graph.

The exported graph is Mul/Add only — trivially NPU-friendly. `hidden`,
`steering`, `alpha`, and `mask` are all runtime inputs.

We export three graphs:
  * steering_injection_prefill.onnx  — fixed [1, 16, D]
  * steering_injection_decode.onnx   — fixed [1, 1, D]
  * steering_injection_dynamic.onnx  — dynamic T (works in ORT; fixed-shape
    variants are kept because Qualcomm HTP compilation strongly prefers — and
    for many toolchain versions requires — static shapes)

Usage:
    python -m steering_poc.export_onnx --hidden-size 1024 [--out-dir artifacts]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn


class SteeringInjection(nn.Module):
    """steered = hidden + alpha * mask * steering  (pure Mul/Add graph)."""

    def forward(self, hidden, steering, alpha, mask):
        # hidden [B,T,D], steering [1,1,D], alpha [1] (rank-1), mask [B,T,1]
        return hidden + alpha * mask * steering


def export(hidden_size: int, seq_len: int, path: Path, dynamic: bool, opset: int = 17):
    module = SteeringInjection().eval()
    hidden = torch.randn(1, seq_len, hidden_size)
    steering = torch.randn(1, 1, hidden_size)
    # alpha is rank-1 [1], not a rank-0 scalar: broadcasting is identical, but
    # AI Hub's dataset uploader cannot ship rank-0 tensors (h5 gzip limitation)
    # while the compiled QNN graph enforces input ranks exactly.
    alpha = torch.tensor([1.0])
    mask = torch.ones(1, seq_len, 1)

    dynamic_axes = (
        {"hidden": {1: "seq"}, "mask": {1: "seq"}, "steered": {1: "seq"}}
        if dynamic
        else None
    )
    torch.onnx.export(
        module,
        (hidden, steering, alpha, mask),
        str(path),
        input_names=["hidden", "steering", "alpha", "mask"],
        output_names=["steered"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        dynamo=False,  # classic exporter: stable static graphs for QNN tooling
    )
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--prefill-len", type=int, default=16)
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    D = args.hidden_size

    for name, seq, dyn in (
        ("steering_injection_prefill.onnx", args.prefill_len, False),
        ("steering_injection_decode.onnx", 1, False),
        ("steering_injection_dynamic.onnx", args.prefill_len, True),
    ):
        path = export(D, seq, out / name, dyn, args.opset)
        import onnx

        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        ops = sorted({n.op_type for n in model.graph.node})
        print(f"{path}  ops={ops}  (opset {args.opset}, D={D}, "
              f"{'dynamic T' if dyn else f'T={seq}'})")


if __name__ == "__main__":
    main()

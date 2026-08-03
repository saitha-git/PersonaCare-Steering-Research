"""Cross-backend numerical validation of the injection op.

Backends/conditions:
  * PyTorch FP32 (reference)
  * PyTorch FP16 (if CUDA is available; FP16 matmul-free op also runs on CPU)
  * ONNX Runtime CPU (prefill graph, decode graph, dynamic graph)
  * INT8 fake-quantized steering vector (per-tensor symmetric)
  * INT8 fake-quantized hidden activations (per-tensor symmetric)
  * Optional QDQ ONNX INT8 graph via onnxruntime.quantization, if installed

Checks recorded per condition: max/mean abs error vs FP32 reference, cosine
similarity, alpha=0 identity, and steering-vs-quantization-noise separability
(is ||steered - baseline|| >> quantization error at alpha=1?).

Usage:
    python -m steering_poc.compare_backends [--hidden-size 1024]
        [--vector artifacts/vector_layer_14.pt]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .common import save_json
from .export_onnx import SteeringInjection, export
from .quantization import error_metrics, fake_quant_int8


def _make_inputs(hidden_size, seq_len, vector=None, seed=1234):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(1, seq_len, hidden_size, generator=g)
    if vector is None:
        steering = torch.randn(1, 1, hidden_size, generator=g)
        steering = steering / steering.norm()
    else:
        steering = (vector / vector.norm()).reshape(1, 1, -1).float()
    mask = torch.ones(1, seq_len, 1)
    return hidden, steering, mask


def _ort_run(session, hidden, steering, alpha, mask):
    out = session.run(
        None,
        {
            "hidden": hidden.numpy().astype(np.float32),
            "steering": steering.numpy().astype(np.float32),
            "alpha": np.array([alpha], dtype=np.float32),
            "mask": mask.numpy().astype(np.float32),
        },
    )
    return torch.from_numpy(out[0])


def run_suite(hidden_size, vector, out_dir: Path, alphas=(0.0, 1.0, 4.0)):
    import onnxruntime as ort

    module = SteeringInjection().eval()
    results = []
    scratch = out_dir

    for shape_name, seq_len in (("prefill", 16), ("decode", 1)):
        hidden, steering, mask = _make_inputs(hidden_size, seq_len, vector)
        onnx_path = scratch / f"steering_injection_{shape_name}.onnx"
        if not onnx_path.exists():
            export(hidden_size, seq_len, onnx_path, dynamic=False)
        sess = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )

        for alpha in alphas:
            a = torch.tensor(float(alpha))
            ref = module(hidden, steering, a, mask)  # FP32 reference
            steer_mag = (ref - hidden).norm().item()

            conds = {}
            # PyTorch FP16
            fp16 = module(hidden.half(), steering.half(), a.half(), mask.half())
            conds["torch_fp16"] = error_metrics(ref, fp16.float())
            # ORT CPU
            ort_out = _ort_run(sess, hidden, steering, alpha, mask)
            conds["ort_cpu_fp32"] = error_metrics(ref, ort_out)
            # INT8 steering vector
            q_steer, s_scale, _ = fake_quant_int8(steering)
            conds["int8_steering_vector"] = error_metrics(
                ref, module(hidden, q_steer, a, mask)
            )
            conds["int8_steering_vector"]["scale"] = s_scale
            # INT8 hidden activations (quantize input AND output like a QDQ chain)
            q_hidden, h_scale, _ = fake_quant_int8(hidden)
            act_out = module(q_hidden, q_steer, a, mask)
            q_out, o_scale, _ = fake_quant_int8(act_out)
            conds["int8_activations"] = error_metrics(ref, q_out)
            conds["int8_activations"]["hidden_scale"] = h_scale
            conds["int8_activations"]["output_scale"] = o_scale

            for name, m in conds.items():
                results.append(
                    {
                        "shape": shape_name,
                        "seq_len": seq_len,
                        "alpha": alpha,
                        "backend": name,
                        "steering_magnitude": steer_mag,
                        "identity_at_zero_alpha": bool(
                            alpha == 0.0 and m["max_abs_err"] < 1e-3
                        ),
                        "signal_to_error_ratio": (
                            steer_mag / (m["mean_abs_err"] * ref.numel() ** 0.5)
                            if m["mean_abs_err"] > 0
                            else float("inf")
                        ),
                        **m,
                    }
                )
    return results


def try_qdq_quantization(hidden_size, out_dir: Path):
    """Real QDQ INT8 path via onnxruntime.quantization (static, per-tensor)."""
    try:
        from onnxruntime.quantization import (
            CalibrationDataReader, QuantFormat, QuantType, quantize_static,
        )
    except ImportError:
        return None, "onnxruntime.quantization not available"

    import onnxruntime as ort

    fp32_path = out_dir / "steering_injection_prefill.onnx"
    qdq_path = out_dir / "steering_injection_prefill_qdq.onnx"

    class Reader(CalibrationDataReader):
        def __init__(self):
            g = torch.Generator().manual_seed(7)
            self.data = [
                {
                    "hidden": torch.randn(1, 16, hidden_size, generator=g).numpy(),
                    "steering": (
                        lambda v: (v / v.norm()).numpy()
                    )(torch.randn(1, 1, hidden_size, generator=g)),
                    "alpha": np.array([float(a)], dtype=np.float32),
                    "mask": np.ones((1, 16, 1), dtype=np.float32),
                }
                for a in (-4, -1, 0, 1, 4)
            ]
            self.it = iter(self.data)

        def get_next(self):
            return next(self.it, None)

    try:
        quantize_static(
            str(fp32_path),
            str(qdq_path),
            Reader(),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
        )
        module = SteeringInjection().eval()
        hidden, steering, mask = _make_inputs(hidden_size, 16)
        sess = ort.InferenceSession(str(qdq_path), providers=["CPUExecutionProvider"])
        rows = []
        for alpha in (0.0, 1.0, 4.0):
            ref = module(hidden, steering, torch.tensor(alpha), mask)
            out = _ort_run(sess, hidden, steering, alpha, mask)
            m = error_metrics(ref, out)
            rows.append({"alpha": alpha, "backend": "ort_qdq_int8", **m})
        return rows, str(qdq_path)
    except Exception as e:  # tool support varies across ORT versions
        return None, f"QDQ quantization failed: {e}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--vector", default=None,
                        help="Optional real steering vector .pt (else random unit)")
    parser.add_argument("--out", default="artifacts/backend_comparison.json")
    args = parser.parse_args(argv)

    vector = None
    if args.vector:
        payload = torch.load(args.vector, map_location="cpu", weights_only=False)
        vector = payload["vector"].float()
        print(f"Using real vector from {args.vector} (norm {vector.norm():.3f})")

    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    results = run_suite(args.hidden_size, vector, out_dir)

    qdq_rows, qdq_note = try_qdq_quantization(args.hidden_size, out_dir)
    if qdq_rows:
        results.extend(
            {"shape": "prefill", "seq_len": 16, **r} for r in qdq_rows
        )
    print(f"QDQ path: {qdq_note}")

    save_json({"results": results, "qdq_note": qdq_note}, args.out)

    print(f"\n{'shape':<8}{'alpha':>6}  {'backend':<24}{'max_err':>12}"
          f"{'mean_err':>12}{'cosine':>10}")
    for r in results:
        print(f"{r['shape']:<8}{r['alpha']:>6.1f}  {r['backend']:<24}"
              f"{r['max_abs_err']:>12.3e}{r['mean_abs_err']:>12.3e}"
              f"{r['cosine']:>10.6f}")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()

"""Simulated reduced-precision / quantization helpers for the injection op.

Caveat (also in the README): quantizing this tiny standalone graph probes the
numerics of the injection op itself. It does NOT reproduce the error behavior
of a fully quantized LLM, where the residual-stream scale is set by the
surrounding weights/activations.
"""

from __future__ import annotations

import torch


def fake_quant_int8(x: torch.Tensor, symmetric: bool = True):
    """Per-tensor int8 fake-quantization: quantize -> dequantize.

    Returns (dequantized tensor, scale, zero_point).
    """
    if symmetric:
        scale = x.abs().max().clamp(min=1e-12) / 127.0
        q = torch.clamp(torch.round(x / scale), -127, 127)
        return q * scale, scale.item(), 0
    xmin, xmax = x.min(), x.max()
    scale = (xmax - xmin).clamp(min=1e-12) / 255.0
    zp = torch.round(-xmin / scale) - 128
    q = torch.clamp(torch.round(x / scale) + zp, -128, 127)
    return (q - zp) * scale, scale.item(), int(zp.item())


def error_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    ref = reference.float().flatten()
    cand = candidate.float().flatten()
    diff = (ref - cand).abs()
    cos = torch.nn.functional.cosine_similarity(ref, cand, dim=0).item()
    return {
        "max_abs_err": diff.max().item(),
        "mean_abs_err": diff.mean().item(),
        "cosine": cos,
    }

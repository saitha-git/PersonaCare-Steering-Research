"""Steering-vector injection into the residual stream via forward hooks.

Core operation (identical math to the exported ONNX graph):

    steered_hidden = hidden + alpha * mask * steering_vector

Behavior under Hugging Face ``generate()``:
  * The PREFILL forward sees the whole prompt: hidden is [B, T, D].
  * Each cached DECODE step sees only the newest token: hidden is [B, 1, D].
  * positions="all":  every prefill token and every decode token is steered.
  * positions="last": only the last prompt token is steered during prefill; each
    decode step has T==1, so its single (last) token is steered too. Net effect:
    the prompt body is untouched, generation is steered.
  * An explicit mask only makes sense for single-forward analysis (prefill
    shapes); it is rejected during decode steps with mismatched T.

The hook never mutates tensors in place — KV-cache and other aux outputs from
the layer tuple are passed through untouched; only the hidden-state tensor is
replaced by a new tensor.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

from .model_adapter import ModelAdapter


def inject_steering(
    hidden: torch.Tensor,
    steering: torch.Tensor,
    alpha: Union[float, torch.Tensor],
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pure functional form of the injection op.

    hidden:   [B, T, D]
    steering: [D] or [1, 1, D]
    alpha:    python scalar, 0-dim tensor, or [B, 1, 1]
    mask:     optional [B, T, 1] (1.0 = steer this position)
    """
    steering = steering.reshape(1, 1, -1).to(dtype=hidden.dtype, device=hidden.device)
    if steering.shape[-1] != hidden.shape[-1]:
        raise ValueError(
            f"Steering dim {steering.shape[-1]} != hidden dim {hidden.shape[-1]}"
        )
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.to(dtype=hidden.dtype, device=hidden.device)
    delta = alpha * steering
    if mask is not None:
        delta = delta * mask.to(dtype=hidden.dtype, device=hidden.device)
    return hidden + delta  # out-of-place: never modifies `hidden`


class SteeringInjector:
    """Context manager that installs injection hooks on one or more layers.

    Example:
        with SteeringInjector(adapter, vector, layer=14, alpha=2.0):
            out = model.generate(...)
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        vector: torch.Tensor,
        layer: Union[int, Sequence[int]],
        alpha: float,
        positions: str = "all",
        mask: Optional[torch.Tensor] = None,
    ):
        if positions not in ("all", "last", "mask"):
            raise ValueError(f"positions must be all|last|mask, got {positions!r}")
        if positions == "mask" and mask is None:
            raise ValueError("positions='mask' requires an explicit mask tensor")

        vector = vector.reshape(-1)
        if vector.shape[0] != adapter.hidden_size:
            raise ValueError(
                f"Vector dim {vector.shape[0]} != model hidden size "
                f"{adapter.hidden_size}"
            )

        self.adapter = adapter
        self.vector = vector.to(device=adapter.device, dtype=adapter.dtype)
        self.layer_indices = [layer] if isinstance(layer, int) else list(layer)
        self.alpha = float(alpha)
        self.positions = positions
        self.mask = mask
        self._handles = []

    def _make_hook(self):
        def hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                hidden, rest = output, None
            elif isinstance(output, tuple) and len(output) > 0:
                hidden, rest = output[0], output[1:]
            else:
                raise TypeError(f"Unsupported layer output type: {type(output)}")

            B, T, _ = hidden.shape
            if self.positions == "all":
                mask = None
            elif self.positions == "last":
                # During cached decode T == 1, so this steers each new token.
                mask = torch.zeros(B, T, 1, device=hidden.device, dtype=hidden.dtype)
                mask[:, -1, :] = 1.0
            else:  # explicit mask
                if self.mask.shape[1] != T:
                    raise ValueError(
                        f"Explicit mask has T={self.mask.shape[1]} but forward pass "
                        f"has T={T}; explicit masks are for single-forward analysis, "
                        "not cached decode."
                    )
                mask = self.mask

            steered = inject_steering(hidden, self.vector, self.alpha, mask)
            if rest is None:
                return steered
            return (steered, *rest)

        return hook

    def __enter__(self):
        try:
            for idx in self.layer_indices:
                layer = self.adapter.get_layer(idx)
                self._handles.append(layer.register_forward_hook(self._make_hook()))
        except Exception:
            self._remove()
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        self._remove()
        return False

    def _remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

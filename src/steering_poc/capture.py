"""Hidden-state capture at decoder-layer outputs via forward hooks."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, List

import torch

from .model_adapter import ModelAdapter


def _hidden_from_output(output):
    """Decoder layers may return a Tensor or a tuple whose first element is the
    hidden state; normalize to the hidden-state tensor."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(
        output[0], torch.Tensor
    ):
        return output[0]
    raise TypeError(f"Unsupported decoder layer output type: {type(output)}")


@contextmanager
def capture_layer_outputs(adapter: ModelAdapter, layers: Iterable[int]):
    """Context manager that records each requested layer's output hidden state
    ([B, T, D], detached) for every forward pass inside the context.

    Yields a dict: layer_index -> list of captured tensors (one per forward call).
    """
    captured: Dict[int, List[torch.Tensor]] = {i: [] for i in layers}
    handles = []
    try:
        for idx in captured:
            layer = adapter.get_layer(idx)

            def hook(_module, _inputs, output, _idx=idx):
                captured[_idx].append(_hidden_from_output(output).detach())

            handles.append(layer.register_forward_hook(hook))
        yield captured
    finally:
        for h in handles:
            h.remove()


def last_token_activations(
    hidden: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Select the activation at the last NON-PADDING token of each sequence.

    hidden: [B, T, D]; attention_mask: [B, T] with 1 on real tokens.
    Works with both right-padding and left-padding. Returns [B, D].
    """
    # Index of the last position where attention_mask == 1.
    positions = (
        attention_mask * torch.arange(attention_mask.shape[1], device=hidden.device)
    ).argmax(dim=1)
    batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch_idx, positions]

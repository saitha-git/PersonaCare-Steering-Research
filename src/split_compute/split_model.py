"""Two-part pipeline split of a HF causal LM (Llama/Qwen2/Qwen3 family).

PartA: input_ids [1,T] (int32)  -> hidden [1,T,D]   (embed + layers 0..k-1)
PartB: hidden    [1,T,D]        -> logits [1,T,V]   (layers k..N-1 + norm + head)

The ONLY tensor crossing the A->B boundary is the residual hidden state —
the same interchange Qualcomm's own multi-split LLM exports use. Rotary
cos/sin are deterministic functions of position, so PartB recomputes them
locally instead of shipping them.

Scope (documented, deliberate): fixed sequence length, single forward, no KV
cache — this is the prompt-processor shape. It verifies numerical viability
of the cross-device boundary; a streaming decode loop would need KV-cached
[1,1,D] graphs per part, exactly like Qualcomm's token_ar1 exports.

The causal mask is an additive [1,1,T,T] constant baked at export. Masked
positions use -30000.0 (not -inf): fp16-safe on HTP, and exp(-30000)
underflows to exactly 0.
"""

from __future__ import annotations

import torch
from torch import nn

MASK_VALUE = -30000.0


def _causal_mask(seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), MASK_VALUE, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.reshape(1, 1, seq_len, seq_len)


def _layer_out(out):
    return out[0] if isinstance(out, (tuple, list)) else out


class _PartBase(nn.Module):
    def __init__(self, model, seq_len: int):
        super().__init__()
        inner = model.model  # LlamaModel / Qwen3Model
        self.rotary = inner.rotary_emb
        self.seq_len = seq_len
        self.register_buffer(
            "mask", _causal_mask(seq_len, next(model.parameters()).dtype)
        )
        self.register_buffer(
            "position_ids", torch.arange(seq_len, dtype=torch.long).reshape(1, -1)
        )

    def _run_layers(self, layers, hidden):
        cos, sin = self.rotary(hidden, self.position_ids)
        for layer in layers:
            hidden = _layer_out(
                layer(
                    hidden,
                    attention_mask=self.mask,
                    position_ids=self.position_ids,
                    position_embeddings=(cos, sin),
                    use_cache=False,
                )
            )
        return hidden


class PartA(_PartBase):
    """Embedding + decoder layers [0, split)."""

    def __init__(self, model, split: int, seq_len: int):
        super().__init__(model, seq_len)
        self.embed = model.model.embed_tokens
        self.layers = model.model.layers[:split]

    def forward(self, input_ids):
        hidden = self.embed(input_ids.long())
        return self._run_layers(self.layers, hidden)


class PartB(_PartBase):
    """Decoder layers [split, N) + final norm + LM head."""

    def __init__(self, model, split: int, seq_len: int):
        super().__init__(model, seq_len)
        self.layers = model.model.layers[split:]
        self.norm = model.model.norm
        self.lm_head = model.lm_head

    def forward(self, hidden):
        hidden = self._run_layers(self.layers, hidden)
        return self.lm_head(self.norm(hidden))


def build_parts(model, split: int, seq_len: int):
    n = len(model.model.layers)
    if not (0 < split < n):
        raise ValueError(f"split must be in (0, {n}), got {split}")
    return (
        PartA(model, split, seq_len).eval(),
        PartB(model, split, seq_len).eval(),
    )

"""Architecture adapter: a single place that knows how to find decoder layers and
load models, so the rest of the project stays architecture-agnostic.

Capture/injection point (documented contract):
    We hook the OUTPUT of ``DecoderLayer i`` — the residual stream AFTER that
    layer's attention block, MLP block, and both residual additions. In HF
    Llama/Qwen2/Qwen3 code this is the tensor returned by
    ``model.model.layers[i].forward(...)`` (first element if a tuple).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import torch
from torch import nn


# Known attribute paths to the decoder-layer ModuleList, tried in order.
_KNOWN_LAYER_PATHS = (
    "model.layers",          # LlamaForCausalLM, Qwen2/Qwen2.5, Qwen3
    "transformer.h",         # GPT-2 family
    "gpt_neox.layers",       # GPT-NeoX / Pythia
    "model.decoder.layers",  # OPT
)


def _resolve_attr(obj: Any, dotted: str) -> Optional[Any]:
    cur = obj
    for part in dotted.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


@dataclasses.dataclass
class ModelAdapter:
    """Wraps a HF causal LM and exposes its decoder layers uniformly."""

    model: nn.Module
    layers: nn.ModuleList
    hidden_size: int
    num_layers: int
    layer_path: str

    @classmethod
    def from_model(cls, model: nn.Module) -> "ModelAdapter":
        config = getattr(model, "config", None)
        expected_n = getattr(config, "num_hidden_layers", None) or getattr(
            config, "n_layer", None
        )

        layers = None
        layer_path = None
        for path in _KNOWN_LAYER_PATHS:
            candidate = _resolve_attr(model, path)
            if isinstance(candidate, nn.ModuleList) and len(candidate) > 0:
                layers, layer_path = candidate, path
                break

        if layers is None:
            # Generic fallback: the ModuleList whose length matches num_hidden_layers.
            for name, module in model.named_modules():
                if isinstance(module, nn.ModuleList) and len(module) == expected_n:
                    layers, layer_path = module, name
                    break

        if layers is None:
            raise ValueError(
                f"Could not locate decoder layers on {type(model).__name__}. "
                f"Tried {_KNOWN_LAYER_PATHS} and ModuleList length matching."
            )
        if expected_n is not None and len(layers) != expected_n:
            raise ValueError(
                f"Found ModuleList at '{layer_path}' with {len(layers)} entries but "
                f"config says {expected_n} hidden layers — refusing to guess."
            )

        hidden_size = getattr(config, "hidden_size", None) or getattr(
            config, "n_embd", None
        )
        if hidden_size is None:
            raise ValueError("Config exposes neither hidden_size nor n_embd.")

        return cls(
            model=model,
            layers=layers,
            hidden_size=int(hidden_size),
            num_layers=len(layers),
            layer_path=layer_path,
        )

    def get_layer(self, index: int) -> nn.Module:
        if not (0 <= index < self.num_layers):
            raise IndexError(
                f"Layer {index} out of range for model with {self.num_layers} layers."
            )
        return self.layers[index]

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype


def resolve_device(spec: str) -> str:
    if spec == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return spec


def load_model_and_tokenizer(model_cfg: dict):
    """Load a HF model + tokenizer from a config dict (see configs/*.yaml)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = model_cfg["model_id"]
    revision = model_cfg.get("revision", "main")
    dtype = getattr(torch, model_cfg.get("dtype", "float32"))
    device = resolve_device(model_cfg.get("device", "auto"))

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=dtype,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter = ModelAdapter.from_model(model)
    return model, tokenizer, adapter


def resolved_revision(model) -> str:
    """Best-effort resolved commit hash of the loaded checkpoint."""
    for attr in ("_commit_hash", "commit_hash"):
        value = getattr(getattr(model, "config", None), attr, None)
        if value:
            return str(value)
    return "unknown"

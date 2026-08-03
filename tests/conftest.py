import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def tiny_model():
    """Tiny randomly-initialized Llama-architecture model: exercises the real HF
    decoder-layer code path (tuple outputs, KV cache) without any download."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


@pytest.fixture(scope="session")
def tiny_adapter(tiny_model):
    from steering_poc.model_adapter import ModelAdapter

    return ModelAdapter.from_model(tiny_model)


@pytest.fixture()
def input_ids():
    torch.manual_seed(1)
    return torch.randint(0, 128, (2, 10))

import pytest
import torch

from split_compute.split_model import build_parts


@pytest.fixture(scope="module")
def eager_tiny():
    """Tiny Llama with eager attention (what export uses)."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, attn_implementation="eager",
    )
    return LlamaForCausalLM(config).eval()


@pytest.mark.parametrize("split", [1, 2, 3])
def test_split_chain_matches_full_model(eager_tiny, split):
    T = 12
    torch.manual_seed(1)
    ids = torch.randint(0, 128, (1, T), dtype=torch.int32)
    part_a, part_b = build_parts(eager_tiny, split, T)
    with torch.no_grad():
        full = eager_tiny(ids.long()).logits
        chain = part_b(part_a(ids))
    assert chain.shape == full.shape
    max_diff = (full - chain).abs().max().item()
    assert max_diff < 1e-4, f"split={split}: chain diverges by {max_diff}"
    assert torch.equal(full.argmax(-1), chain.argmax(-1))


def test_boundary_is_only_interchange(eager_tiny):
    """PartB must work from the hidden state alone (positions recomputed)."""
    T = 8
    part_a, part_b = build_parts(eager_tiny, 2, T)
    hidden = torch.randn(1, T, 64)  # arbitrary boundary tensor
    with torch.no_grad():
        logits = part_b(hidden)
    assert logits.shape == (1, T, 128)


def test_causal_mask_in_parts(eager_tiny):
    """Changing a future token must not affect earlier positions' logits."""
    T = 10
    torch.manual_seed(2)
    ids = torch.randint(0, 128, (1, T), dtype=torch.int32)
    ids2 = ids.clone()
    ids2[0, -1] = (ids2[0, -1] + 1) % 128
    part_a, part_b = build_parts(eager_tiny, 2, T)
    with torch.no_grad():
        l1 = part_b(part_a(ids))
        l2 = part_b(part_a(ids2))
    assert torch.allclose(l1[:, :-1], l2[:, :-1], atol=1e-5)
    assert not torch.allclose(l1[:, -1], l2[:, -1])


def test_invalid_split_rejected(eager_tiny):
    with pytest.raises(ValueError):
        build_parts(eager_tiny, 0, 8)
    with pytest.raises(ValueError):
        build_parts(eager_tiny, 4, 8)


def test_split_parts_export_to_onnx(eager_tiny, tmp_path):
    """Both halves must survive ONNX export and reproduce the torch chain."""
    import numpy as np
    import onnxruntime as ort

    from split_compute.export_split import export_part

    T = 8
    part_a, part_b = build_parts(eager_tiny, 2, T)
    ids = torch.randint(0, 128, (1, T), dtype=torch.int32)
    with torch.no_grad():
        boundary = part_a(ids)
        ref = part_b(boundary)

    pa = export_part(part_a, ids, tmp_path / "a.onnx", "input_ids", "hidden")
    pb = export_part(part_b, boundary, tmp_path / "b.onnx", "hidden", "logits")

    sa = ort.InferenceSession(str(pa), providers=["CPUExecutionProvider"])
    sb = ort.InferenceSession(str(pb), providers=["CPUExecutionProvider"])
    h = sa.run(None, {"input_ids": ids.numpy()})[0]
    logits = sb.run(None, {"hidden": h})[0]
    assert np.abs(logits - ref.numpy()).max() < 1e-4

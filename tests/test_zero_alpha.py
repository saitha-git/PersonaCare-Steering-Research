import torch

from steering_poc.inject import SteeringInjector, inject_steering


def test_zero_alpha_pure_function():
    hidden = torch.randn(2, 5, 8)
    out = inject_steering(hidden, torch.randn(8), 0.0)
    assert torch.equal(out, hidden + 0.0 * torch.zeros(8))
    assert torch.allclose(out, hidden, atol=0)


def test_zero_alpha_full_model(tiny_model, tiny_adapter, input_ids):
    """alpha=0 with the hook installed must reproduce the hook-free baseline."""
    torch.manual_seed(0)
    vector = torch.randn(64)
    with torch.no_grad():
        base = tiny_model(input_ids).logits
        with SteeringInjector(tiny_adapter, vector, layer=2, alpha=0.0):
            zero = tiny_model(input_ids).logits
    max_dev = (base - zero).abs().max().item()
    assert max_dev <= 1e-5, f"zero-dose deviation {max_dev} exceeds 1e-5"


def test_zero_alpha_generation(tiny_model, tiny_adapter, input_ids):
    torch.manual_seed(0)
    vector = torch.randn(64)
    base = tiny_model.generate(input_ids[:1], max_new_tokens=8, do_sample=False,
                               pad_token_id=0)
    with SteeringInjector(tiny_adapter, vector, layer=1, alpha=0.0):
        zero = tiny_model.generate(input_ids[:1], max_new_tokens=8, do_sample=False,
                                   pad_token_id=0)
    assert torch.equal(base, zero), "greedy generation must be identical at alpha=0"

import pytest
import torch

from steering_poc.capture import capture_layer_outputs
from steering_poc.inject import SteeringInjector


def _total_hooks(adapter):
    return sum(len(layer._forward_hooks) for layer in adapter.layers)


def test_injector_removes_hooks(tiny_adapter):
    before = _total_hooks(tiny_adapter)
    with SteeringInjector(tiny_adapter, torch.randn(64), layer=[1, 3], alpha=1.0):
        assert _total_hooks(tiny_adapter) == before + 2
    assert _total_hooks(tiny_adapter) == before


def test_injector_removes_hooks_on_exception(tiny_model, tiny_adapter, input_ids):
    before = _total_hooks(tiny_adapter)
    with pytest.raises(RuntimeError):
        with SteeringInjector(tiny_adapter, torch.randn(64), layer=2, alpha=1.0):
            raise RuntimeError("boom")
    assert _total_hooks(tiny_adapter) == before


def test_capture_removes_hooks(tiny_adapter):
    before = _total_hooks(tiny_adapter)
    with capture_layer_outputs(tiny_adapter, [0, 2]):
        assert _total_hooks(tiny_adapter) == before + 2
    assert _total_hooks(tiny_adapter) == before


def test_no_effect_after_context_exit(tiny_model, tiny_adapter, input_ids):
    with torch.no_grad():
        base = tiny_model(input_ids).logits
        with SteeringInjector(tiny_adapter, torch.randn(64), layer=2, alpha=8.0):
            steered = tiny_model(input_ids).logits
        after = tiny_model(input_ids).logits
    assert not torch.allclose(base, steered)
    assert torch.equal(base, after), "model must be pristine after context exit"

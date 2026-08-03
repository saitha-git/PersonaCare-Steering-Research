import numpy as np
import pytest
import torch

from steering_poc.export_onnx import SteeringInjection, export
from steering_poc.quantization import error_metrics, fake_quant_int8

D = 64


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    import onnxruntime as ort

    tmp = tmp_path_factory.mktemp("onnx")
    paths = {
        "prefill": export(D, 16, tmp / "prefill.onnx", dynamic=False),
        "decode": export(D, 1, tmp / "decode.onnx", dynamic=False),
        "dynamic": export(D, 16, tmp / "dynamic.onnx", dynamic=True),
    }
    return {
        k: ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        for k, p in paths.items()
    }


def _run(sess, hidden, steering, alpha, mask):
    out = sess.run(None, {
        "hidden": hidden.numpy(), "steering": steering.numpy(),
        "alpha": np.array([alpha], dtype=np.float32), "mask": mask.numpy(),
    })
    return torch.from_numpy(out[0])


@pytest.mark.parametrize("name,T", [("prefill", 16), ("decode", 1)])
@pytest.mark.parametrize("alpha", [-4.0, 0.0, 2.0])
def test_ort_matches_pytorch(sessions, name, T, alpha):
    torch.manual_seed(0)
    hidden = torch.randn(1, T, D)
    steering = torch.randn(1, 1, D)
    mask = torch.ones(1, T, 1)
    ref = SteeringInjection()(hidden, steering, torch.tensor(alpha), mask)
    out = _run(sessions[name], hidden, steering, alpha, mask)
    assert (ref - out).abs().max().item() < 1e-6


def test_dynamic_graph_handles_both_shapes(sessions):
    torch.manual_seed(1)
    for T in (1, 7, 16, 33):
        hidden = torch.randn(1, T, D)
        steering = torch.randn(1, 1, D)
        mask = torch.ones(1, T, 1)
        ref = SteeringInjection()(hidden, steering, torch.tensor(1.5), mask)
        out = _run(sessions["dynamic"], hidden, steering, 1.5, mask)
        assert (ref - out).abs().max().item() < 1e-6


def test_zero_alpha_identity_in_onnx(sessions):
    torch.manual_seed(2)
    hidden = torch.randn(1, 16, D)
    out = _run(sessions["prefill"], hidden, torch.randn(1, 1, D), 0.0,
               torch.ones(1, 16, 1))
    assert torch.equal(out, hidden) or (out - hidden).abs().max() < 1e-7


def test_mask_respected_in_onnx(sessions):
    torch.manual_seed(3)
    hidden = torch.randn(1, 16, D)
    steering = torch.randn(1, 1, D)
    mask = torch.zeros(1, 16, 1)
    mask[:, -1] = 1.0
    out = _run(sessions["prefill"], hidden, steering, 2.0, mask)
    assert (out[:, :-1] - hidden[:, :-1]).abs().max() < 1e-7
    assert (out[:, -1] - hidden[:, -1]).abs().max() > 0.01


def test_int8_steering_keeps_signal():
    """Quantization error of an int8 steering vector must be far smaller than
    the steering signal itself at alpha=1."""
    torch.manual_seed(4)
    hidden = torch.randn(1, 16, D)
    steering = torch.randn(1, 1, D)
    steering = steering / steering.norm()
    q_steer, _, _ = fake_quant_int8(steering)
    mod = SteeringInjection()
    ones = torch.ones(1, 16, 1)
    ref = mod(hidden, steering, torch.tensor(1.0), ones)
    quant = mod(hidden, q_steer, torch.tensor(1.0), ones)
    m = error_metrics(ref, quant)
    signal = (ref - hidden).abs().mean().item()
    assert m["mean_abs_err"] < 0.05 * signal, (
        f"quantization noise {m['mean_abs_err']:.2e} not << signal {signal:.2e}"
    )

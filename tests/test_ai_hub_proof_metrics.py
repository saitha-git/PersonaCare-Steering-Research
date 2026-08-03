import numpy as np

from steering_poc.qualcomm.prove_steering_ai_hub import _inputs, _metrics


def test_proof_metrics_capture_identity_delta_and_mask():
    hidden_size = 4
    seq_len = 4
    alpha = 4.0
    vector = np.ones((1, 1, hidden_size), dtype=np.float32)
    inputs0 = _inputs(hidden_size, seq_len, 0.0, vector)
    a0 = inputs0["hidden"][0]
    aN = a0 + alpha * inputs0["mask"][0] * vector

    metrics = _metrics(a0, aN, inputs0, alpha)

    assert metrics["overall_passed"] is True
    assert metrics["alpha0_identity_vs_input"]["max_abs_err"] == 0.0
    assert metrics["alpha_delta_vs_expected"]["max_abs_err"] == 0.0
    assert metrics["mask_check"]["masked_delta_l2"] == 0.0
    assert metrics["mask_check"]["active_delta_l2"] > 0.0

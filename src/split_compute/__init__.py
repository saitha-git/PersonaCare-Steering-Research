"""Split-compute experiment: pipeline-parallel partitioning of a causal LM
across two devices (phone NPU + laptop NPU), verified via Qualcomm AI Hub.

Separate from steering_poc — this package answers one question: can layers
0..k-1 run on device A and layers k..N-1 on device B, with the residual
hidden state as the only tensor crossing the boundary, and still produce the
right logits on real Snapdragon silicon?
"""

__version__ = "0.1.0"

"""Deterministic training mode — the foundation of spot-recompute verification.

A verifier reproduces a worker's weight update bit-for-bit from (batch, weights,
seed); that requires the training itself to be deterministic. The kill-test
confirmed this is bit-exact run-to-run ON THE SAME GPU. (Cross-architecture is
the open question the Discord harness answers.)

NOTE: CUBLAS_WORKSPACE_CONFIG must be set in the environment BEFORE torch
initializes CUDA — do it at the top of the entrypoint, not here.
"""

from contextlib import contextmanager

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def enable_determinism(seed: int = 0):
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)


@contextmanager
def deterministic_attention():
    """Force the MATH SDPA backend. Flash/mem-efficient backends have
    non-deterministic reductions; MATH is reproducible (at ~3x the memory —
    that is why the verifiable model/context ceiling is lower)."""
    with sdpa_kernel(SDPBackend.MATH):
        yield

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
def deterministic_attention(dtype: torch.dtype = torch.bfloat16):
    """Force the FLASH SDPA backend, which is bit-exact once determinism is on.

    This used to force MATH, on the grounds that flash and mem-efficient backends
    have non-deterministic reductions. That is true by default and false under
    `torch.use_deterministic_algorithms(True)` — which `enable_determinism` above
    sets before any of this runs. So the ceiling on context length was being paid
    for a guarantee the process already had by another means.

    Measured on an RTX 5090 (sm_120), bf16, 16 heads of 128, causal, forward and
    backward, repeated runs compared byte for byte:

        default                             FLASH  differs in dq
        use_deterministic_algorithms(True)  FLASH  bit-exact

    and the memory, which is the point:

        seq_len     FLASH      MATH
           8192   0.69 GiB  16.38 GiB
          32768   2.82 GiB       OOM
         131072  11.08 GiB       OOM

    MATH materialises the n x n attention matrix, so it cannot go past 8192 on a
    24 GB card; FLASH is linear and reaches 128K there. Same arithmetic, same
    bytes, twenty-four times less memory at the length we already train at.

    ONE BACKEND PER DTYPE, NOT A PREFERENCE LIST. Flash and math are each
    internally reproducible and are not bit-identical to each other, so choosing
    between them by what happens to be available would let two workers compute
    different updates and each conclude the other was cheating.

    The choice is made by dtype, which is a property of the round rather than of
    the machine: flash kernels exist only for 16-bit inputs, and a real round runs
    bf16 — that is what makes a billion parameters fit a consumer card. fp32 is a
    test and simulation convenience, never a consensus configuration, so sending
    it to MATH cannot split a network. A 16-bit run that cannot reach flash raises
    rather than falling back, because that is a machine whose arithmetic would not
    match anyone else's.
    """
    if dtype in (torch.bfloat16, torch.float16):
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            yield
    else:
        with sdpa_kernel(SDPBackend.MATH):
            yield

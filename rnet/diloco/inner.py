"""The inner loop: what a worker actually does with a GPU for twenty minutes.

Take the weights the round starts from, run `inner_steps` optimizer steps on
batches the worker did not choose, and report the difference. That difference,
quantised, is the contribution.

WHY THE DIFFERENCE AND NOT THE WEIGHTS. A 30-billion-parameter model is 59 GB;
its update, at one byte per parameter, is 29. More importantly the difference is
what aggregates: several workers' differences add up to a sensible step, while
several workers' weights do not add up to anything.

DETERMINISM IS SET UP HERE AND NOWHERE ELSE. Seeds, deterministic kernels, and
the attention backend from the round's numerics. A verifier replaying this on
the same determinism class must reach byte-identical output, so anything that
introduces machine-dependent behaviour has to be turned off explicitly rather
than left to a default that varies by torch version.

CUBLAS_WORKSPACE_CONFIG MUST BE SET BEFORE TORCH INITIALISES CUDA, which is at
import. This module cannot do it — by the time it runs, the decision is made —
so it CHECKS, and refuses to train rather than producing an update that will not
replay. That check exists because the alternative is a worker that looks fine
and is quietly unverifiable.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch

from ..consensus.model_spec import ModelSpec
from ..consensus.numerics import Numerics
from ..dataset.scheduler import synthetic_window, window_seed
from ..model import weights as W
from ..model.transformer import Transformer
from .quantize import quantize_update, quantize_update_chunked


class InnerError(Exception):
    pass


def enable_determinism(seed: int = 0, device: str = "cuda") -> None:
    """Everything that has to be true before a replayable step can be taken.

    `device` because the cuBLAS requirement is about cuBLAS: a run on CPU does
    not need it, and demanding it there would make a machine that happens to
    have a card refuse work it could do perfectly well.
    """
    if str(device).startswith("cuda") and not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        raise InnerError(
            "CUBLAS_WORKSPACE_CONFIG is unset. It must be ':4096:8' in the "
            "environment BEFORE python starts, because torch reads it when CUDA "
            "initialises at import. Without it cuBLAS picks workspaces by "
            "heuristic and the same batch gives different bytes on the same card.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    # TF32 silently drops ten mantissa bits from every matmul. It is a large
    # speedup and it is not the arithmetic the round pinned.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def tune_cpu_threads(cap: int = 8) -> int:
    """Cap intra-op threads, and return what was chosen.

    PERFORMANCE ONLY. This does not change a single byte of the result —
    measured across 1, 4, 8 and 24 threads, identical loss to ten decimal places
    and an identical hash over every gradient, by
    RoundTests.ThreadCountDoesNotChangeTheBytes. It is here because it changes
    the speed by more than an order of magnitude in the wrong direction.

    Measured on a 24-core hybrid CPU with a 256x256 @ 256x1024 matmul, which is
    the shape this model's feed-forward actually uses:

        1 thread    139 GFLOPS
        4 threads   286 GFLOPS
        8 threads   397 GFLOPS
        24 threads   10.9 GFLOPS      <- thirty-six times slower than eight

    The collapse is oversubscription: the matrices are small enough that
    synchronising twenty-four workers costs far more than the arithmetic. torch
    defaults to one thread per core, which is right for a datacentre-sized model
    and catastrophic for this one.
    """
    import torch as _torch
    chosen = max(1, min(cap, os.cpu_count() or 1))
    _torch.set_num_threads(chosen)
    return chosen


@dataclass
class RoundResult:
    payload: np.ndarray
    scale_exp: int
    final_loss: float
    steps: int
    # The weights the round STARTED from, kept because the outer step must be
    # applied to those and not to where training left the model. Costs a second
    # copy — 8 MB on regtest, 800 MB on the dense 400M — which is the price of
    # not having to fetch the base back over a socket.
    base_weights: dict = None


def derive_batch(spec: ModelSpec, dataset_root: bytes, round_id: int,
                 worker_id: int, outer_step: int, inner_index: int,
                 micro_batch: int, corpus=None) -> torch.Tensor:
    """The tokens for one inner step, from protocol state alone.

    `corpus` is anything with `tokens_for_chunk(index)`. Without one the round
    has no corpus pinned and the windows are synthetic — which is a real answer
    for regtest and not a fallback for a corpus that failed to load: a round
    that pins a root and cannot read it must fail, not invent.
    """
    if corpus is None and dataset_root != bytes(32):
        # The rule this module already stated, now enforced. Falling through to
        # synthetic tokens here is not a graceful degradation: the round pins a
        # corpus, the daemon accepts the contribution, the chain advances on it,
        # and a verifier replaying from the same protocol state reproduces the
        # identical noise and AGREES — so the anti-cheating machinery certifies
        # work that could not possibly have learned anything. Measured: 50 inner
        # steps on the 397M model with loss flat at 10.53 against ln(32000) =
        # 10.37, submitted and accepted without a word.
        raise InnerError(
            f"inner: this round pins a corpus ({dataset_root.hex()[:16]}…) and no "
            f"corpus is attached. Training would silently use random tokens, the "
            f"daemon would accept it, and the verifier would confirm it — so it "
            f"stops here instead.")

    windows = []
    for b in range(micro_batch):
        seed = window_seed(dataset_root, round_id, worker_id, outer_step,
                           inner_index * micro_batch + b)
        if corpus is None:
            windows.append(synthetic_window(seed, spec.seq_len + 1, spec.vocab_size))
        else:
            windows.append(corpus.window_for_seed(seed, spec.seq_len + 1))
    return torch.tensor(windows, dtype=torch.long)


def run(model: Transformer, spec: ModelSpec, numerics: Numerics, *,
        dataset_root: bytes, round_id: int, worker_id: int, outer_step: int,
        inner_steps: int, micro_batch: int, lr: float, device: str,
        corpus=None, on_step=None) -> RoundResult:
    """One round's worth of training, returning the quantised difference.

    The weights before are captured as canonical words rather than as a tensor
    clone: the difference has to be taken in the same representation the network
    agreed on, and a clone in the compute dtype would make the subtraction
    depend on how the model happened to be stored.
    """
    before = W.save_weights(model)
    order = sorted(before)
    total = int(sum(before[name].size for name in order))

    optimizer = _adafactor(model, lr)
    model.train()
    loss_value = float("nan")

    for step in range(inner_steps):
        tokens = derive_batch(spec, dataset_root, round_id, worker_id,
                              outer_step, step, micro_batch, corpus).to(device)
        loss = model.loss(tokens)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss_value = float(loss.detach())
        if not np.isfinite(loss_value):
            raise InnerError(
                f"inner: loss became {loss_value} at step {step}; the run diverged "
                f"and its update is not worth submitting")
        if on_step is not None:
            on_step(step, loss_value)

    after = W.save_weights(model)

    # before - after: a worker reports how far it moved, and the network moves
    # the same way when it subtracts.
    #
    # A TENSOR AT A TIME, in the same order the concatenation used, so the
    # bytes are identical to the version that built one array of 397,728,768
    # float64s and subtracted another from it. That version peaked at 19.9 GB
    # and was OOM-killed at the submit, having already spent eighteen minutes
    # computing the update it was about to send.
    def deltas():
        for name in order:
            yield _to_float(before[name]) - _to_float(after[name])

    payload, exp = quantize_update_chunked(deltas, total,
                                           numerics.contribution_format)
    return RoundResult(payload=payload, scale_exp=exp, final_loss=loss_value,
                       steps=inner_steps, base_weights=before)


def _to_float(words: np.ndarray) -> np.ndarray:
    from ..consensus.init import bf16_to_float32
    return bf16_to_float32(words).astype(np.float64)


def _adafactor(model: Transformer, lr: float):
    """Adafactor without momentum, which is what fits.

    Adam keeps two full-size moments — 8 bytes per parameter on top of the
    weights. Adafactor keeps row and column sums instead, which is O(n+m) per
    matrix rather than O(nm), and that difference is what leaves room for a
    16384-token context on a 24 GB card.
    """
    try:
        from torch.optim import Adafactor
    except ImportError as exc:      # pragma: no cover - torch < 2.5
        raise InnerError(
            "inner: this torch has no Adafactor; the round's optimizer_id names it"
        ) from exc
    return Adafactor(model.parameters(), lr=lr, beta2_decay=-0.8,
                     weight_decay=0.0, foreach=False)

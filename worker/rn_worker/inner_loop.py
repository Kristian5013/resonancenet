"""The deterministic inner loop: what one worker does between outer syncs.

A worker starts from the round's base weights, trains for exactly `inner_steps`
on batches the protocol assigned it, and submits what changed:

    pseudo_gradient = weights_before - weights_after

Determinism is not a nicety here — it is the whole verification model. A verifier
with the same base weights, the same schedule and the same seed must reproduce
this byte for byte. Every source of nondeterminism is therefore pinned: the math
attention backend (flash has nondeterministic reductions), TF32 off, fixed seeds,
and no data the worker chose for itself.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before torch CUDA init

import numpy as np
import torch

from .consensus import canon, diloco, scheduler


def enable_determinism(seed: int) -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)


def flat_parameters(model, model_spec) -> list[tuple[str, torch.nn.Parameter]]:
    """Parameters in CONSENSUS layout order.

    Not alphabetical. `sorted(model.named_parameters())` is the obvious thing to
    write and produces an entirely different permutation of the same tensors — on
    regtest zero of 46 positions agree, and with sixteen layers it also orders them
    0, 1, 10, 11, ..., 2 because the sort is textual.

    That mismatch has no symptom: byte *i* of a payload would mean one parameter
    here and a different one to every C++ node, aggregation would still run, and
    the model would train into noise. `model_spec` is required rather than
    optional precisely so this ordering cannot be reached by accident.

    Tied embeddings share storage and appear under two names; the consensus layout
    emits such a tensor once, so the deduplication that used to live here is now a
    property of the layout itself.
    """
    from .consensus_order import consensus_parameters

    return consensus_parameters(model, model_spec)


def snapshot(model, model_spec) -> list[torch.Tensor]:
    return [p.detach().to(torch.float32).cpu().clone()
            for _, p in flat_parameters(model, model_spec)]


def pseudo_gradient(before: list[torch.Tensor], after: list[torch.Tensor]) -> np.ndarray:
    """weights_before - weights_after, flattened in canonical parameter order."""
    deltas = [(b - a).reshape(-1) for b, a in zip(before, after)]
    return torch.cat(deltas).numpy().astype(np.float32)


def load_windows(corpus, dataset_root: bytes, round_id: int, worker_id: int, step: int,
                 micro_batch: int, seq_len: int, device) -> tuple:
    """Assembles a batch the PROTOCOL chose. The worker never selects its own data
    — that is what closes data poisoning by construction.

    Two draws per window, from one seed: which chunk of the corpus, and where
    inside its tokenization the window starts. The second needs the chunk's token
    count, which is why it happens here and not in the node: only whoever
    tokenized the chunk knows it, and the worker and its verifier both did, with
    the artifact tokenizer_hash pins.
    """
    xs, ys = [], []
    for i in range(micro_batch):
        seed = scheduler.window_seed(dataset_root, round_id, worker_id, step, i)
        chunk_index = scheduler.chunk_for_window(seed, corpus.n_chunks)
        tokens = corpus.tokens_for_chunk(chunk_index)
        start = scheduler.offset_in_chunk(seed, len(tokens), seq_len)
        window = np.asarray(tokens[start : start + seq_len + 1], dtype=np.int64)
        xs.append(window[:-1])
        ys.append(window[1:])
    return (torch.from_numpy(np.stack(xs)).to(device),
            torch.from_numpy(np.stack(ys)).to(device))


def run_inner_loop(model, model_spec, optimizer_factory, corpus,
                   dataset_root: bytes, round_id: int, worker_id: int, outer_step: int,
                   inner_steps: int, micro_batch: int, seq_len: int, lr: float, device,
                   poison: "PoisonPolicy | None" = None) -> dict:
    """Runs one worker's local training and returns its quantised contribution.

    `poison` exists so the simulation can inject a dishonest worker and verify
    that recomputation catches it. An honest run passes None.
    """
    from .determinism import deterministic_attention

    before = snapshot(model, model_spec)
    model.train()
    session = optimizer_factory(model, lr)

    try:
        for step in range(inner_steps):
            x, y = load_windows(corpus, dataset_root, round_id, worker_id,
                                outer_step * inner_steps + step, micro_batch, seq_len, device)
            if poison is not None:
                x, y = poison.corrupt_batch(x, y, step)
            # From determinism.py, not hardcoded here. This used to force MATH with
            # its own import, which meant the module that decides the attention
            # backend had no caller and changing it changed nothing — the flash
            # measurement was real, the claim that the simulation verified it was
            # not, because the simulation was still running MATH.
            with deterministic_attention(next(model.parameters()).dtype):
                _, loss = model(x, y)
                loss.backward()
    finally:
        # Always detach: a leaked hook would corrupt every subsequent run on this
        # model instance.
        if hasattr(session, "close"):
            session.close()

    after = snapshot(model, model_spec)
    delta = pseudo_gradient(before, after)
    if poison is not None:
        delta = poison.corrupt_delta(delta)

    # Vectorised path: at a billion parameters the scalar reference would need
    # tens of gigabytes for a Python list. Proven equal to the reference in
    # test_cross_language.
    scale_exp = diloco.choose_scale_exp_array(delta)
    values, clamped = diloco.quantize_array(delta, scale_exp)
    payload = values.tobytes()

    return {
        "values": values,
        "scale_exp": scale_exp,
        "clamped": clamped,
        "payload": payload,
        "payload_hash": canon.sha3_256(payload),
        "n_params": len(values),
        "final_loss": float(loss.item()),
    }


def apply_update(model, model_spec, update, scale_exp: int) -> None:
    """Applies an outer-step update, expressed in fixed-point units of 2^-scale_exp.

    Sign convention: the pseudo-gradient is (before - after), so it already points
    downhill and the update is SUBTRACTED, exactly like a gradient.
    """
    inv = 2.0 ** (-scale_exp)
    update = np.asarray(update, dtype=np.int64)
    cursor = 0
    with torch.no_grad():
        for _, p in flat_parameters(model, model_spec):
            n = p.numel()
            chunk = torch.from_numpy(update[cursor : cursor + n].astype(np.float64))
            delta = (chunk * inv).to(torch.float32).reshape(p.shape).to(p.device, p.dtype)
            p.sub_(delta)
            cursor += n
    assert cursor == len(update), "update length does not match the model"


def weights_hash(model, model_spec) -> bytes:
    """The consensus weights hash: every tensor in layout order as big-endian bf16.

    The previous form interleaved utf-8 parameter names with native-endian float32
    and could never equal the value a C++ node computes over the same weights — so
    it could not be compared against a checkpoint or against the genesis anchor,
    which is the only thing such a hash is for.
    """
    from .consensus_order import consensus_weights_hash

    return consensus_weights_hash(model, model_spec)


class PoisonPolicy:
    """A dishonest worker, for testing verification on a real model.

    Two flavours, matching the two things a worker could lie about:
      * `data`   — trains on something other than its assigned batches,
      * `delta`  — reports an update it did not actually compute.
    Both must be caught by recomputation, because both change the payload.
    """

    def __init__(self, kind: str, magnitude: float = 0.01):
        if kind not in ("data", "delta"):
            raise ValueError("poison kind must be 'data' or 'delta'")
        self.kind = kind
        self.magnitude = magnitude

    def corrupt_batch(self, x, y, step: int):
        if self.kind != "data":
            return x, y
        # Train on a shifted target: the loss still falls, the update is not the
        # one the protocol asked for.
        return x, torch.roll(y, shifts=1, dims=-1)

    def corrupt_delta(self, delta: np.ndarray) -> np.ndarray:
        if self.kind != "delta":
            return delta
        tampered = delta.copy()
        # A small, coherent perturbation — deliberately the shape that hid under a
        # loss-based gate in the earlier poison experiment.
        tampered[: max(1, len(tampered) // 100)] += self.magnitude * float(np.abs(delta).max())
        return tampered

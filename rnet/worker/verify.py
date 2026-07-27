"""Replaying somebody else's round, to find out whether they did it.

This is the whole anti-cheating scheme in one function. A worker claimed that
starting from these weights, with this id, at this step, for this many inner
steps, it produced an update hashing to X. Every one of those inputs is public
and every one of them is derivable, so a verifier can do the same work and see
whether it gets X.

WHAT MAKES IT DECIDABLE. The batches are a pure function of protocol state, so
the verifier trains on exactly the same text without being told what it is —
which is why a worker cannot pick data it likes. And the arithmetic is pinned by
the round's numerics, so two machines in the same determinism class produce
identical bytes rather than similar ones. Comparable networks check similarity
against empirical thresholds and call it an open problem; here the comparison is
equality.

WHAT MAKES IT HONEST. A verifier that cannot reproduce the work — wrong base
weights, different determinism class — returns INDETERMINATE, not MISMATCH.
Reporting a mismatch it cannot distinguish from its own circumstances would
slash an honest worker for the verifier's hardware, and a scheme that does that
is worse than no scheme: it punishes the wrong people while looking rigorous.
"""

from __future__ import annotations

import hashlib

import numpy as np

from ..consensus.init import hash_of_values
from ..consensus.model_spec import ModelSpec
from ..consensus.numerics import Numerics
from ..consensus.objects import Verdict as VerdictKind
from ..diloco import inner
from ..model import weights as W
from . import ipc


def payload_hash(values: np.ndarray, scale_exp: int) -> bytes:
    """The same digest the daemon computes over an accepted contribution.

    Kept identical on purpose: a verifier and a daemon that hashed differently
    would disagree about every contribution while both being internally
    consistent, which is the hardest kind of bug to see.
    """
    return hashlib.sha3_256(
        scale_exp.to_bytes(8, "big", signed=True)
        + np.ascontiguousarray(values, dtype=np.int64).astype(">i8").tobytes()
    ).digest()


def replay(model, spec: ModelSpec, numerics: Numerics, message: ipc.Verify, *,
           dataset_root: bytes, base_weights: dict | None, device: str,
           lr: float, corpus=None) -> ipc.Verdict:
    """Re-run the challenged round and compare.

    `base_weights` is the state the challenged round started from. Without it
    there is nothing to replay against, and saying so is the correct answer
    rather than an error.
    """
    if numerics.determinism_class != message.determinism_class:
        return ipc.Verdict(
            challenge_id=message.challenge_id,
            verdict=int(VerdictKind.INDETERMINATE),
            replay_payload_hash=bytes(32),
            determinism_class=numerics.determinism_class,
            note=f"class {numerics.determinism_class:#010x} cannot judge "
                 f"{message.determinism_class:#010x}")

    if base_weights is None:
        return ipc.Verdict(
            challenge_id=message.challenge_id,
            verdict=int(VerdictKind.INDETERMINATE),
            replay_payload_hash=bytes(32),
            determinism_class=numerics.determinism_class,
            note="the weights that round started from are not held here")

    if hash_of_values(spec, base_weights) != message.base_weights_hash:
        # Holding different weights than the round started from is the verifier's
        # problem, not the worker's. Anything else here would slash a worker for
        # the verifier having fallen behind.
        return ipc.Verdict(
            challenge_id=message.challenge_id,
            verdict=int(VerdictKind.INDETERMINATE),
            replay_payload_hash=bytes(32),
            determinism_class=numerics.determinism_class,
            note="the base weights held here are not the ones challenged")

    W.load_weights(model, base_weights)
    result = inner.run(
        model, spec, numerics, dataset_root=dataset_root,
        round_id=message.round_id,
        # The TARGET's id, not the verifier's: the id feeds the schedule, so
        # replaying under our own would train on entirely different text and
        # produce a mismatch that means nothing.
        worker_id=message.target_worker_id,
        outer_step=message.outer_step, inner_steps=message.inner_steps,
        micro_batch=message.micro_batch, lr=lr, device=device, corpus=corpus)

    mine = payload_hash(result.payload, result.scale_exp)
    matched = mine == message.claimed_payload_hash
    return ipc.Verdict(
        challenge_id=message.challenge_id,
        verdict=int(VerdictKind.MATCH if matched else VerdictKind.MISMATCH),
        replay_payload_hash=mine,
        determinism_class=numerics.determinism_class,
        note="" if matched else "replay produced different bytes")

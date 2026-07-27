"""Which text a worker trains on — derived, never chosen.

This is the mechanism that closes data poisoning. A worker's batches are a pure
function of protocol state: the corpus root, the round, the worker's own id, the
outer step and the index of the inner step. Anyone holding the same corpus
recomputes them exactly, so "you trained on data you picked yourself" stops
being an accusation nobody can settle and becomes an arithmetic check.

TWO DRAWS, DOMAIN SEPARATED. Which chunk, and where in the chunk, come from two
different hashes of the same seed rather than from two halves of one. Halves
would be cheaper and would leak: a worker learning the chunk index would learn
something about the offset, and a worker who could search over one draw could
steer the other. Separate domains make the second draw independent of anything
learned about the first.

THE WORKER ID IS AN INPUT, WHICH IS WHY IT IS ASSIGNED. Two workers in a round
must see different data or the round learns nothing new from the second. The
daemon hands out the id; a worker that could name itself could take another's
assignment, or fish for an id whose data it likes.

see: docs/corpus-addressing.md
"""

from __future__ import annotations

import hashlib

from ..consensus.objects import BatchSeed

# Domain for the second draw. A constant string rather than a counter, so the
# two draws cannot be made to collide by any choice of the inputs.
OFFSET_DOMAIN = b"rnet/window-offset"

# A micro-batch beyond this is not a schedule, it is a request to allocate. The
# value is read from policy, which is hashed, but the ceiling is here so a
# malformed policy cannot ask for a billion windows.
MAX_MICRO_BATCH = 4096


class ScheduleError(Exception):
    pass


def window_seed(dataset_root: bytes, round_id: int, worker_id: int,
                outer_step: int, inner_index: int) -> bytes:
    """The seed for one window, as the hash of a canonical pre-image.

    Built through the same container machinery as everything else that gets
    hashed, so a verifier reproduces the bytes by the same route rather than by
    a private convention that happens to agree today.
    """
    seed = BatchSeed(dataset_root=dataset_root, round_id=round_id,
                     worker_id=worker_id, outer_step=outer_step,
                     inner_index=inner_index)
    return hashlib.sha3_256(seed.to_container()).digest()


def chunk_for_window(seed: bytes, n_chunks: int) -> int:
    """Which chunk of the corpus this window comes from."""
    if n_chunks <= 0:
        raise ScheduleError("schedule: a corpus with no chunks has no windows")
    return int.from_bytes(seed[:8], "big") % n_chunks


def offset_in_chunk(seed: bytes, tokens_in_chunk: int, seq_len: int) -> int:
    """Where in the chunk the window starts.

    The window needs `seq_len` inputs and one more token to shift against, so a
    chunk of exactly `seq_len + 1` tokens has precisely one legal offset and a
    shorter one has none. Refused rather than clamped: the manifest is meant to
    make a short chunk impossible, so arriving here means something upstream is
    wrong, and saying so beats quietly training on a truncated window.
    """
    span = tokens_in_chunk - seq_len
    if span <= 0:
        raise ScheduleError(
            f"schedule: a chunk of {tokens_in_chunk} tokens cannot hold a window of "
            f"{seq_len} plus its shifted target")
    draw = hashlib.sha3_256(OFFSET_DOMAIN + seed).digest()
    return int.from_bytes(draw[:8], "big") % span


def batch_chunks(dataset_root: bytes, round_id: int, worker_id: int,
                 outer_step: int, micro_batch: int, n_chunks: int) -> list[int]:
    """The chunks for one inner step's micro-batch.

    Each element of the micro-batch is its own draw, so a batch of four is four
    independent windows rather than four slices of one — which is what keeps a
    larger micro-batch from simply seeing the same document repeatedly.
    """
    if micro_batch <= 0 or micro_batch > MAX_MICRO_BATCH:
        raise ScheduleError(f"schedule: micro_batch {micro_batch} is out of range")
    return [chunk_for_window(
                window_seed(dataset_root, round_id, worker_id, outer_step, i),
                n_chunks)
            for i in range(micro_batch)]


def synthetic_window(seed: bytes, length: int, vocab_size: int) -> list[int]:
    """Tokens for a round with no corpus pinned.

    Regtest trains on noise, and this is where the noise comes from. It says so
    plainly rather than pretending: a network without a corpus cannot teach a
    model anything, and the reason to have one is to exercise the protocol —
    quantise, aggregate, produce, verify — in seconds instead of twenty minutes.

    Derived from the same seed as a real window, so the property that matters
    for the protocol still holds: the data is a function of protocol state, and
    a verifier reaches the same tokens.
    """
    if vocab_size <= 0:
        raise ScheduleError("schedule: a vocabulary of no tokens has no windows")
    out: list[int] = []
    block = 0
    while len(out) < length:
        digest = hashlib.shake_256(
            b"rnet/synthetic/v2" + seed + block.to_bytes(4, "big")
        ).digest(max(1024, (length - len(out)) * 4))
        for i in range(0, len(digest), 4):
            if len(out) == length:
                break
            out.append(int.from_bytes(digest[i:i + 4], "big") % vocab_size)
        block += 1
    return out

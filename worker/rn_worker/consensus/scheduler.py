"""Deterministic batch derivation — Python mirror of src/dataset/scheduler.

The worker does not choose what it trains on. Its assignment is a pure function
of protocol state, so a verifier reproduces any claimed update exactly:

    seed_i   = SHA3-256(canon(BatchSeed{root, round, worker, step, i}))
    chunk_i  = seed_i mod n_chunks
    start_i  = SHA3-256("rnet/window-offset" ‖ seed_i) mod (tokens_in_chunk - seq_len)

The corpus is raw UTF-8 text, not tokens, so the schedule names a CHUNK and the
worker tokenizes it itself with the artifact tokenizer_hash pins. The offset
within the chunk therefore depends on how many tokens that chunk holds, which
only whoever tokenized it knows — the node never computes it, the worker and the
verifier both do, and they agree because they tokenized the same bytes with the
same artifact.

This must agree with the C++ implementation to the last bit — a divergence would
mean workers and verifiers disagree about what was trained, which is a network
split. worker/tests/test_cross_language.py asserts the agreement against
rnet-tool.

see: docs/corpus-addressing.md
"""

from __future__ import annotations

from . import canon

MAX_MICRO_BATCH = 4096

# Domain separation between the two draws. Without it the chunk and the position
# inside it would come from one value used twice, which correlates them — and a
# correlation is something to grind against.
OFFSET_DOMAIN = "rnet/window-offset"


class ScheduleError(Exception):
    pass


def batch_seed_container(dataset_root: bytes, round_id: int, worker_id: int, step: int,
                         index: int) -> bytes:
    if len(dataset_root) != 32:
        raise ScheduleError("schedule: dataset_root must be 32 bytes")
    content = (
        canon.Writer()
        .raw(dataset_root)
        .u64(round_id)
        .u64(worker_id)
        .u64(step)
        .u32(index)
        .bytes()
    )
    return canon.wrap_container(canon.OBJ_BATCH_SEED, content)


def window_seed(dataset_root: bytes, round_id: int, worker_id: int, step: int,
                index: int) -> bytes:
    """The seed a window is drawn from. Reconstructible from a contribution header."""
    return canon.sha3_256(batch_seed_container(dataset_root, round_id, worker_id, step, index))


def chunk_for_window(seed: bytes, n_chunks: int) -> int:
    """Which chunk of the corpus this window comes from."""
    if n_chunks <= 0:
        raise ScheduleError("schedule: corpus has no chunks")
    # Big-endian read of the first 8 bytes, then reduced. The modulo bias is below
    # 2^-24 for any realistic corpus; rejection sampling is deliberately avoided so
    # the derivation stays trivially reimplementable in any language.
    return int.from_bytes(seed[:8], "big") % n_chunks


def offset_in_chunk(seed: bytes, tokens_in_chunk: int, seq_len: int) -> int:
    """Where inside that chunk's tokenization the window starts."""
    if seq_len <= 0:
        raise ScheduleError("schedule: seq_len must be non-zero")
    need = seq_len + 1                      # inputs plus the shifted target
    if tokens_in_chunk < need:
        raise ScheduleError(
            f"schedule: a chunk of {tokens_in_chunk} tokens cannot hold a window of {need}; "
            f"the manifest should have refused this corpus when it was built"
        )
    digest = canon.sha3_256(canon.Writer().string(OFFSET_DOMAIN).raw(seed).bytes())
    return int.from_bytes(digest[:8], "big") % (tokens_in_chunk - seq_len)


def batch_chunks(dataset_root: bytes, round_id: int, worker_id: int, step: int, micro_batch: int,
                 n_chunks: int) -> list[int]:
    """Every chunk index for one inner step's micro-batch."""
    if micro_batch <= 0:
        raise ScheduleError("schedule: micro_batch must be non-zero")
    if micro_batch > MAX_MICRO_BATCH:
        raise ScheduleError("schedule: micro_batch too large")
    return [
        chunk_for_window(window_seed(dataset_root, round_id, worker_id, step, i), n_chunks)
        for i in range(micro_batch)
    ]

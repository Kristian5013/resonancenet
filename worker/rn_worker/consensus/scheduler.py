"""Deterministic batch derivation — Python mirror of src/dataset/scheduler.

The worker does not choose what it trains on. Offsets are a pure function of
protocol state, so a verifier can reproduce any claimed update exactly:

    offset_i = SHA3-256(canon(BatchSeed{root, round, worker, step, i})) mod W

This must agree with the C++ implementation to the last bit — a divergence would
mean workers and verifiers disagree about what was trained, which is a network
split. worker/tests/test_cross_language.py asserts the agreement against
rnet-tool.
"""

from __future__ import annotations

from . import canon

MAX_MICRO_BATCH = 4096


class ScheduleError(Exception):
    pass


def window_count(n_tokens: int, seq_len: int) -> int:
    """Valid window start positions: a window needs seq_len inputs plus one
    shifted target token."""
    if seq_len <= 0:
        raise ScheduleError("schedule: seq_len must be non-zero")
    if n_tokens <= seq_len + 1:
        raise ScheduleError(f"schedule: corpus too small ({n_tokens} tokens) for seq_len {seq_len}")
    return n_tokens - seq_len - 1


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


def window_offset(dataset_root: bytes, round_id: int, worker_id: int, step: int, index: int,
                  n_tokens: int, seq_len: int) -> int:
    windows = window_count(n_tokens, seq_len)
    digest = canon.sha3_256(batch_seed_container(dataset_root, round_id, worker_id, step, index))
    # Big-endian read of the first 8 bytes, then reduced. The modulo bias is below
    # 2^-24 for any realistic corpus; rejection sampling is deliberately avoided so
    # the derivation stays trivially reimplementable in any language.
    return int.from_bytes(digest[:8], "big") % windows


def batch_offsets(dataset_root: bytes, round_id: int, worker_id: int, step: int, micro_batch: int,
                  n_tokens: int, seq_len: int) -> list[int]:
    if micro_batch <= 0:
        raise ScheduleError("schedule: micro_batch must be non-zero")
    if micro_batch > MAX_MICRO_BATCH:
        raise ScheduleError("schedule: micro_batch too large")
    return [
        window_offset(dataset_root, round_id, worker_id, step, i, n_tokens, seq_len)
        for i in range(micro_batch)
    ]

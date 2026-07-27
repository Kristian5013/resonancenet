"""Where the first weights come from: derived, not distributed.

A network that shipped its initial weights as a file would be asking every
participant to trust whoever produced the file. Here they are a pure function of
the genesis hash, so anyone can recompute them and check that the network
started where it says it did. That is the fourth anchor, alongside the round,
the policy and the corpus.

HOW A VALUE IS DERIVED. Each tensor gets its own keyed stream, seeded from the
genesis hash and the tensor's NAME rather than from a running counter over the
whole model. Two consequences, both wanted:

  * Adding, removing or reordering a tensor changes only that tensor. A global
    counter would reshuffle every value after the edit, which turns a small
    architecture change into an unreviewable diff.
  * Any tensor can be derived alone. A holder of one expert shard reproduces
    its own experts without walking 29 billion parameters it does not have —
    which is what makes a sharded mixture checkable by the people holding it.

Values are uniform in ±1/sqrt(fan_in) — the classic scaling, chosen because it
is the one that keeps activations from growing or vanishing through depth, and
because it depends only on a shape everyone already agrees on. Norms start at
one and consume no stream.

EVERYTHING IS DONE IN FLOAT32 AND ROUNDED ONCE. The stream produces a uint32,
one division puts it in range, and a single round-half-to-even lands it in
bfloat16. Rounding twice, or in a different order, gives different weights on
different machines and there is no way to tell afterwards which was right.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..model.layout import Kind, TensorSpec, layout, shard_layout
from .model_spec import ModelSpec

# Bumped with the container format. A node deriving weights under a different
# domain gets different weights and fails at the first hash comparison rather
# than training something subtly wrong.
INIT_DOMAIN = b"rnet/init/v2"

_WORDS_PER_DIGEST = 8   # SHA3-256 is 32 bytes


def _stream(name: str, genesis_hash: bytes, count: int) -> np.ndarray:
    """`count` uint32 words, keyed by the genesis hash and the tensor name.

    Counter-based rather than sequential so that any slice is addressable
    without producing the ones before it.
    """
    n_blocks = (count + _WORDS_PER_DIGEST - 1) // _WORDS_PER_DIGEST
    prefix = (INIT_DOMAIN + genesis_hash
              + len(name).to_bytes(4, "big") + name.encode("utf-8"))
    buf = bytearray(n_blocks * 32)
    for b in range(n_blocks):
        buf[b * 32:(b + 1) * 32] = hashlib.sha3_256(
            prefix + b.to_bytes(8, "big")).digest()
    # Big-endian, matching every other integer this project puts on a wire.
    return np.frombuffer(bytes(buf), dtype=">u4", count=n_blocks * _WORDS_PER_DIGEST)[:count]


def float32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Round-half-to-even into bfloat16, returned as raw uint16.

    numpy has no bfloat16, and going through float16 would be a different
    rounding of a different range. Doing it on the bits is the only way to state
    exactly which rounding happened. Proved by
    InitTests.RoundingIsHalfToEvenAtTheTie.
    """
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    # Add half an ulp, plus one more when the retained bit is odd — which is
    # what makes exact ties go to the even neighbour instead of always up.
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    out = (rounded >> np.uint32(16)).astype(np.uint16)
    # A NaN whose surviving mantissa bits are all zero would come out as
    # infinity, turning "no answer" into a very large answer.
    nan = np.isnan(x)
    if nan.any():
        out[nan] = np.uint16(0x7FC0)
    return out


def bf16_to_float32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def derive_tensor(tensor: TensorSpec, genesis_hash: bytes) -> np.ndarray:
    """One tensor's initial value, as raw bf16 words in row-major order."""
    if tensor.kind is Kind.NORM:
        # Ones, and no stream consumed. A scale that starts anywhere else makes
        # the first forward pass depend on a random draw for no benefit.
        return np.full(tensor.numel, 0x3F80, dtype=np.uint16)   # 1.0 in bf16

    words = _stream(tensor.name, genesis_hash, tensor.numel)
    bound = np.float32(1.0) / np.float32(np.sqrt(tensor.fan_in))
    # uint32 -> [0, 1) -> [-bound, +bound). The division is exact in float64 and
    # the result is narrowed once, so the sequence of roundings is fixed.
    unit = words.astype(np.float64) / np.float64(1 << 32)
    values = ((unit * 2.0 - 1.0) * float(bound)).astype(np.float32)
    return float32_to_bf16(values)


def derive_all(spec: ModelSpec, genesis_hash: bytes, *, shard: int | None = None,
               workers: int = 0) -> dict[str, np.ndarray]:
    """Every tensor this holder needs, as raw bf16.

    `shard` restricts the work to what one holder of a sharded mixture stores.
    Threads help because hashlib releases the interpreter lock above a couple of
    kilobytes, so this really does use the cores it asks for.
    """
    tensors = layout(spec) if shard is None else shard_layout(spec, shard)
    if workers == 0:
        workers = min(32, (len(tensors) or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        derived = list(pool.map(lambda t: derive_tensor(t, genesis_hash), tensors))
    return {t.name: d for t, d in zip(tensors, derived)}


def weights_hash(spec: ModelSpec, genesis_hash: bytes, *, workers: int = 0,
                 progress=None) -> bytes:
    """The fourth anchor: SHA3-256 over every tensor, in canonical order.

    Streamed rather than concatenated — the mixture is 59 GB of bf16 and does
    not fit anywhere it could be concatenated. Hashing in layout order is what
    makes the digest a statement about the model rather than about one node's
    memory.

    Derivation runs in parallel, hashing does not and cannot: the digest is
    defined by the order. A bounded look-ahead keeps the workers fed without
    materialising the whole model, which for the mixture would be 59 GB of
    lookahead. Proved by InitTests.ParallelDerivationMatchesSequential.
    """
    tensors = layout(spec)
    if workers == 0:
        workers = min(24, len(tensors))

    h = hashlib.sha3_256()
    # Enough in flight to keep every worker busy, few enough that the queue is
    # bounded by worker count rather than by model size.
    depth = max(2, workers * 2)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: list = []
        for i, tensor in enumerate(tensors):
            pending.append(pool.submit(derive_tensor, tensor, genesis_hash))
            if len(pending) < depth and i + 1 < len(tensors):
                continue
            h.update(pending.pop(0).result().astype(">u2").tobytes())
            if progress is not None:
                progress(i + 1 - len(pending), len(tensors))
        for future in pending:
            h.update(future.result().astype(">u2").tobytes())
    if progress is not None:
        progress(len(tensors), len(tensors))
    return h.digest()


def hash_of_values(spec: ModelSpec, tensors: dict[str, np.ndarray]) -> bytes:
    """The same digest, over weights that came from somewhere else.

    What a node uses to check that a checkpoint it received is the checkpoint
    the network agreed on, and what a worker uses to report where it ended up.
    """
    h = hashlib.sha3_256()
    for tensor in layout(spec):
        value = tensors.get(tensor.name)
        if value is None:
            raise KeyError(f"init: {tensor.name} is missing from the weights")
        if value.size != tensor.numel:
            raise ValueError(
                f"init: {tensor.name} has {value.size} elements, the layout says "
                f"{tensor.numel}")
        h.update(np.ascontiguousarray(value, dtype=np.uint16).astype(">u2").tobytes())
    return h.digest()

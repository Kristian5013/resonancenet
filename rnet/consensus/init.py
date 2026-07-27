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
    its own experts without walking 29 billion parameters it does not have.

The stream is SHAKE-256, the extendable-output function of the same family, in
blocks of a fixed size. It is the right primitive for "produce a long keyed
stream" — and it is 6.6x faster per byte than calling SHA3-256 in counter mode,
measured, which for a 29-billion-parameter mixture is the difference between
half a minute and half an hour.

WHY THE ANCHOR IS A MERKLE ROOT AND NOT A FLAT DIGEST. A flat hash over the
tensors concatenated must be computed in order, start to finish, on one core;
hashlib holds the interpreter lock for inputs this size, so threads do not help
and measurably hurt. A root over per-tensor leaves computes each leaf
independently — 32 bytes come back from a worker instead of a gigabyte — and it
buys something a flat digest cannot offer at all: an INCLUSION PROOF FOR ONE
TENSOR. A shard holder can prove its experts are the ones the network agreed on
without producing a model nobody has in one place.

Values are uniform in ±1/sqrt(fan_in) — the scaling that keeps activations from
growing or vanishing through depth, and which depends only on a shape everyone
already agrees on. Norms start at one and consume no stream. Everything is done
in float32 and rounded ONCE into bfloat16 with round-half-to-even; rounding
twice, or in a different order, gives different weights on different machines
and there is no way to tell afterwards which was right.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..crypto import merkle
from ..model.layout import Kind, TensorSpec, layout, shard_layout
from .model_spec import ModelSpec

# Bumped with the container format. A node deriving under a different domain
# gets different weights and fails at the first hash comparison rather than
# training something subtly wrong.
INIT_DOMAIN = b"rnet/init/v2"

# Elements per stream block. Bounds the largest single allocation: the mixture's
# embedding table is 98 million elements, and asking SHAKE for all of it at once
# would be a 393 MB buffer in every worker that happened to draw it. At this size
# a block is 16 MB, and the block index keeps any offset addressable.
BLOCK_ELEMS = 1 << 22


def _stream(name: str, genesis_hash: bytes, count: int) -> np.ndarray:
    """`count` uint32 words, keyed by the genesis hash and the tensor name."""
    prefix = (INIT_DOMAIN + genesis_hash
              + len(name).to_bytes(4, "big") + name.encode("utf-8"))
    out = np.empty(count, dtype=">u4")
    for start in range(0, count, BLOCK_ELEMS):
        n = min(BLOCK_ELEMS, count - start)
        raw = hashlib.shake_256(
            prefix + (start // BLOCK_ELEMS).to_bytes(4, "big")).digest(n * 4)
        # Big-endian, matching every other integer this project puts on a wire.
        out[start:start + n] = np.frombuffer(raw, dtype=">u4", count=n)
    return out


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
    bound = 1.0 / np.sqrt(tensor.fan_in)
    # uint32 -> [0, 1) -> [-bound, +bound). Computed in float64 and narrowed
    # once, so the sequence of roundings is fixed rather than incidental.
    unit = words.astype(np.float64) / np.float64(1 << 32)
    return float32_to_bf16(((unit * 2.0 - 1.0) * bound).astype(np.float32))


def tensor_bytes(tensor: TensorSpec, genesis_hash: bytes) -> bytes:
    """A tensor's canonical bytes: big-endian bf16, row-major."""
    return derive_tensor(tensor, genesis_hash).astype(">u2").tobytes()


def tensor_leaf(tensor: TensorSpec, genesis_hash: bytes) -> bytes:
    """The Merkle leaf for one tensor. What a worker process returns."""
    return merkle.leaf_hash(tensor_bytes(tensor, genesis_hash))


def derive_all(spec: ModelSpec, genesis_hash: bytes, *,
               shard: int | None = None) -> dict[str, np.ndarray]:
    """Every tensor this holder needs, as raw bf16.

    `shard` restricts the work to what one holder of a sharded mixture stores.
    Single-process on purpose: the values themselves are the result, and moving
    gigabytes of them between processes costs more than deriving them.
    """
    tensors = layout(spec) if shard is None else shard_layout(spec, shard)
    return {t.name: derive_tensor(t, genesis_hash) for t in tensors}


def _leaf_job(args):
    return tensor_leaf(*args)


def leaves(spec: ModelSpec, genesis_hash: bytes, *, workers: int = 0,
           progress=None) -> list[bytes]:
    """A Merkle leaf per tensor, in layout order.

    Processes rather than threads, because hashlib holds the interpreter lock
    at these input sizes — measured at 83 MB/s on one thread and 76 on eight,
    which is why the previous design's concurrency was decorative. Only 32
    bytes come back per tensor, so the usual reason to avoid processes does not
    apply here.
    """
    tensors = layout(spec) if spec else []
    if workers == 0:
        import os
        workers = max(1, min(len(tensors), (os.cpu_count() or 2) - 1))
    if workers == 1 or len(tensors) < 4:
        out = []
        for i, t in enumerate(tensors):
            out.append(tensor_leaf(t, genesis_hash))
            if progress:
                progress(i + 1, len(tensors))
        return out

    with ProcessPoolExecutor(max_workers=workers) as pool:
        out = []
        # Ordered: the root is defined by layout order, so results must not be
        # taken as they finish.
        for i, leaf in enumerate(pool.map(
                _leaf_job, [(t, genesis_hash) for t in tensors], chunksize=1)):
            out.append(leaf)
            if progress:
                progress(i + 1, len(tensors))
        return out


def weights_hash(spec: ModelSpec, genesis_hash: bytes, *, workers: int = 0,
                 progress=None) -> bytes:
    """The fourth anchor: the Merkle root over every tensor, in canonical order."""
    return merkle.root(leaves(spec, genesis_hash, workers=workers, progress=progress))


def hash_of_values(spec: ModelSpec, tensors: dict[str, np.ndarray]) -> bytes:
    """The same root, over weights that came from somewhere else.

    What a node uses to check that a checkpoint it received is the one the
    network agreed on, and what a worker uses to report where it ended up.
    """
    out = []
    for tensor in layout(spec):
        value = tensors.get(tensor.name)
        if value is None:
            raise KeyError(f"init: {tensor.name} is missing from the weights")
        if value.size != tensor.numel:
            raise ValueError(
                f"init: {tensor.name} has {value.size} elements, the layout says "
                f"{tensor.numel}")
        out.append(merkle.leaf_hash(
            np.ascontiguousarray(value, dtype=np.uint16).astype(">u2").tobytes()))
    return merkle.root(out)


def tensor_proof(spec: ModelSpec, genesis_hash: bytes, name: str) -> merkle.Proof:
    """An inclusion proof that one tensor belongs to the agreed weights.

    What a flat digest could not give at any price: a shard holder proving its
    experts are the ones the network settled on, without producing a model that
    exists nowhere in one place.
    """
    tensors = layout(spec)
    index = next((i for i, t in enumerate(tensors) if t.name == name), None)
    if index is None:
        raise KeyError(f"init: {name} is not a tensor of this model")
    return merkle.build_proof(leaves(spec, genesis_hash), index)

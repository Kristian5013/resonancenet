"""Where training starts — the Python mirror of src/consensus/init.cpp.

Every node must begin from the same weights, byte for byte. This file and its C++
counterpart are two implementations of one specification, and the cross-language
test is what keeps them one specification rather than two that resemble each other.

The arithmetic here is deliberately awkward-looking in places. Each awkwardness is
load-bearing:

  * The stream is COUNTER-BASED, so any position is reachable without replaying
    what came before it. A verifier can re-derive one tensor of a 1B model.

  * The distribution is UNIFORM. Box-Muller would need log and cos, and only sqrt
    is required by IEEE-754 to be correctly rounded — the others differ between
    libm implementations, which would put a cross-platform disagreement at the
    very root of the chain.

  * Everything is float32 (numpy float32, not Python float), because Python's
    native float is a double and would round differently at the last step.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np

from .canon import Writer

# Domain separation: the same descriptor bytes must not produce this stream for
# any other purpose in the protocol.
INIT_DOMAIN = "rnet/init/v1"

# Each 32-byte digest yields eight 32-bit words, and each word yields one value.
VALUES_PER_BLOCK = 8


@dataclass(frozen=True)
class TensorSpec:
    """One parameter tensor in canonical order.

    The order is consensus, not convenience: the weights hash is over the
    concatenation, so two nodes that lay tensors out differently produce different
    hashes from identical numbers.
    """

    name: str
    rows: int   # fan_out
    cols: int   # fan_in; 1 for vectors
    is_norm: bool = False

    @property
    def elements(self) -> int:
        return self.rows * self.cols


def tensor_layout(model) -> list[TensorSpec]:
    """The canonical tensor layout for a model spec."""
    d = model.d_model
    head_dim = d // model.n_heads
    q_dim = model.n_heads * head_dim
    kv_dim = model.n_kv_heads * head_dim

    tensors = [TensorSpec("tok_embeddings", model.vocab_size, d)]
    for layer in range(model.n_layers):
        p = f"layers.{layer}."
        tensors.append(TensorSpec(p + "attention_norm", d, 1, True))
        tensors.append(TensorSpec(p + "attention.wq", q_dim, d))
        tensors.append(TensorSpec(p + "attention.wk", kv_dim, d))
        tensors.append(TensorSpec(p + "attention.wv", kv_dim, d))
        tensors.append(TensorSpec(p + "attention.wo", d, q_dim))
        if model.qk_norm:
            tensors.append(TensorSpec(p + "attention.q_norm", head_dim, 1, True))
            tensors.append(TensorSpec(p + "attention.k_norm", head_dim, 1, True))
        tensors.append(TensorSpec(p + "ffn_norm", d, 1, True))
        tensors.append(TensorSpec(p + "feed_forward.w_gate", model.d_ff, d))
        tensors.append(TensorSpec(p + "feed_forward.w_up", model.d_ff, d))
        tensors.append(TensorSpec(p + "feed_forward.w_down", d, model.d_ff))

    tensors.append(TensorSpec("norm", d, 1, True))
    # A tied head is the embedding matrix read the other way round. Emitting it
    # again would both double the storage and let the two copies drift.
    if not model.tie_embeddings:
        tensors.append(TensorSpec("output", model.vocab_size, d))
    return tensors


def init_seed(round_id_hash: bytes) -> bytes:
    """The seed for a round, from the round descriptor's own container hash.

    Depending on the whole descriptor means the seed depends on every consensus
    value the round fixes, including the network magic — so two networks running
    the same architecture start from different weights.
    """
    w = Writer()
    # Mirrors canon::Writer::String — a u32 length prefix, then utf-8.
    encoded = INIT_DOMAIN.encode("utf-8")
    w.u32(len(encoded)).raw(encoded)
    w.raw(round_id_hash)
    return hashlib.sha3_256(w.bytes()).digest()


def _stream_block(seed: bytes, tensor_index: int, counter: int) -> bytes:
    w = Writer()
    w.raw(seed)
    w.u32(tensor_index)
    w.u64(counter)
    return hashlib.sha3_256(w.bytes()).digest()


def init_tensor_slice(
    seed: bytes, tensor_index: int, tensor: TensorSpec, element_offset: int, count: int
) -> np.ndarray:
    """The initial values of one tensor slice, in row-major order.

    `element_offset` lets a caller generate a slice without generating what comes
    before it — what makes a 1B model initialisable in bounded memory.
    """
    if element_offset + count > tensor.elements:
        raise ValueError(
            f"init: slice [{element_offset}, {element_offset + count}) exceeds tensor "
            f"'{tensor.name}' of {tensor.elements} elements"
        )
    if tensor.is_norm:
        # Norm weights start at one. Random noise in the one place the model has
        # no capacity to correct it would make the first forward pass depend on
        # chance.
        return np.ones(count, dtype=np.float32)
    if tensor.cols == 0:
        raise ValueError("init: tensor has zero fan-in")

    # 1/sqrt(fan_in). Division and square root are both correctly rounded by
    # IEEE-754, so this is the same float32 on every conforming platform. Computed
    # in float32 throughout: Python's float is a double and would round elsewhere.
    bound = np.float32(1.0) / np.sqrt(np.float32(tensor.cols))

    first_block = element_offset // VALUES_PER_BLOCK
    last_block = (element_offset + count - 1) // VALUES_PER_BLOCK
    words = np.empty((last_block - first_block + 1) * VALUES_PER_BLOCK, dtype=np.uint32)
    for i, block in enumerate(range(first_block, last_block + 1)):
        digest = _stream_block(seed, tensor_index, block)
        words[i * VALUES_PER_BLOCK : (i + 1) * VALUES_PER_BLOCK] = struct.unpack(">8I", digest)

    start = element_offset - first_block * VALUES_PER_BLOCK
    words = words[start : start + count]

    # `word >> 8` keeps 24 bits, which converts to float32 exactly (the mantissa
    # holds 24 bits); the multiply by 2^-24 only changes the exponent, so it is
    # exact; `2x - 1` is exact for the same reason; the final multiply is one
    # correctly-rounded operation.
    unit = (words >> np.uint32(8)).astype(np.float32) * np.float32(2.0 ** -24)
    return (np.float32(2.0) * unit - np.float32(1.0)) * bound


def float_to_bf16(values: np.ndarray) -> np.ndarray:
    """float32 to bfloat16 bits, rounding half to even.

    The same rule the hardware follows, so a node converting in software and one
    converting on a GPU agree.
    """
    bits = values.astype(np.float32).view(np.uint32)

    # A NaN must stay a NaN rather than rounding into infinity, which is what a
    # plain truncate-and-round does to a NaN whose payload is only in the low bits.
    is_nan = ((bits & np.uint32(0x7F800000)) == np.uint32(0x7F800000)) & (
        (bits & np.uint32(0x007FFFFF)) != 0
    )

    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = ((bits + np.uint32(0x7FFF) + lsb) >> np.uint32(16)).astype(np.uint16)
    nan_result = ((bits >> np.uint32(16)) | np.uint32(0x0040)).astype(np.uint16)
    return np.where(is_nan, nan_result, rounded)


def genesis_weights_hash(model, round_id_hash: bytes, slice_elements: int = 1 << 20) -> bytes:
    """SHA3-256 of the canonical initial weight bytes.

    Every tensor in layout order, each element as big-endian bf16. Streamed, so a
    1B-parameter model can be hashed on a machine that cannot hold it.
    """
    layout = tensor_layout(model)
    total = sum(t.elements for t in layout)
    if total != model.parameter_count():
        raise ValueError(
            f"init: the tensor layout holds {total} parameters but the model spec says "
            f"{model.parameter_count()} — the two descriptions of the same model disagree"
        )

    seed = init_seed(round_id_hash)
    hasher = hashlib.sha3_256()
    for index, tensor in enumerate(layout):
        done = 0
        while done < tensor.elements:
            count = min(slice_elements, tensor.elements - done)
            values = init_tensor_slice(seed, index, tensor, done, count)
            # Big-endian, as everywhere on the wire and in every hashed artifact.
            hasher.update(float_to_bf16(values).astype(">u2").tobytes())
            done += count
    return hasher.digest()

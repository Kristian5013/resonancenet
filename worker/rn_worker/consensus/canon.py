"""Python mirror of src/canon — read and verify canonical objects.

Byte-for-byte identical to the C++ authority. The worker only ever CONSUMES
consensus objects, so this module parses and verifies; it never authors them.

Fail-closed throughout: a bad magic, version, length, hash or CRC raises rather
than returning a partially trusted object.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field

MAGIC = b"RNET"
# 1.1 — CheckpointHeader gained optimizer_state_hash. Must track the C++ constant
# in src/canon/canon.h exactly: a mirror that lags is a worker that cannot read
# what the network writes, and it fails at the first container rather than
# somewhere subtle, which is the one mercy here.
PROTOCOL_VERSION = 0x0001_0002  # 1.2

OBJ_ROUND_DESCRIPTOR = 1
OBJ_DATASET_MANIFEST = 2
OBJ_BATCH_SEED = 3

NORM_RMSNORM = 1
ACT_SWIGLU = 1
OPT_ADAFACTOR = 1

DTYPE_UINT16 = 1
DTYPE_UINT32 = 2

_CRC32C_POLY = 0x82F63B78  # Castagnoli, reflected


class CanonError(Exception):
    """Any failure to parse or verify a consensus object."""


def sha3_256(data: bytes) -> bytes:
    return hashlib.sha3_256(data).digest()


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ _CRC32C_POLY if crc & 1 else crc >> 1
    return crc ^ 0xFFFFFFFF


class Writer:
    """Big-endian, fixed-width. Used only to rebuild pre-images (BatchSeed) whose
    hash the worker must reproduce — never to author objects for other nodes."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def u8(self, v: int) -> "Writer":
        self._buf += struct.pack(">B", v)
        return self

    def u16(self, v: int) -> "Writer":
        self._buf += struct.pack(">H", v)
        return self

    def u32(self, v: int) -> "Writer":
        self._buf += struct.pack(">I", v)
        return self

    def u64(self, v: int) -> "Writer":
        self._buf += struct.pack(">Q", v)
        return self

    def string(self, s: str) -> "Writer":
        """u32 length prefix then UTF-8, mirroring canon::Writer::String."""
        encoded = s.encode("utf-8")
        self.u32(len(encoded))
        self._buf += encoded
        return self

    def raw(self, data: bytes) -> "Writer":
        self._buf += data
        return self

    def bytes(self) -> bytes:
        return bytes(self._buf)


class Reader:
    """Bounds-checked reads; every failure raises CanonError."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def _take(self, n: int) -> bytes:
        if self._pos + n > len(self._data):
            raise CanonError(f"canon: truncated ({n} bytes wanted, {self.remaining} left)")
        chunk = self._data[self._pos : self._pos + n]
        self._pos += n
        return chunk

    def u8(self) -> int:
        return struct.unpack(">B", self._take(1))[0]

    def u16(self) -> int:
        return struct.unpack(">H", self._take(2))[0]

    def u32(self) -> int:
        return struct.unpack(">I", self._take(4))[0]

    def u64(self) -> int:
        return struct.unpack(">Q", self._take(8))[0]

    def hash(self) -> bytes:
        return self._take(32)

    @property
    def remaining(self) -> int:
        return len(self._data) - self._pos

    def expect_exhausted(self) -> None:
        if self._pos != len(self._data):
            raise CanonError(f"canon: {self.remaining} trailing byte(s)")


@dataclass
class Container:
    obj_type: int
    version: int
    content: bytes
    content_hash: bytes


def wrap_container(obj_type: int, content: bytes) -> bytes:
    body = (
        MAGIC
        + struct.pack(">I", PROTOCOL_VERSION)
        + struct.pack(">H", obj_type)
        + struct.pack(">I", len(content))
        + content
        + sha3_256(content)
    )
    return body + struct.pack(">I", crc32c(body))


def parse_container(raw: bytes) -> Container:
    header, trailer = 4 + 4 + 2 + 4, 32 + 4
    if len(raw) < header + trailer:
        raise CanonError("canon: container too short")
    if raw[:4] != MAGIC:
        raise CanonError("canon: bad magic")

    r = Reader(raw)
    r._take(4)
    version = r.u32()
    if version != PROTOCOL_VERSION:
        raise CanonError(f"canon: unsupported protocol version {version:#x}")
    obj_type = r.u16()
    length = r.u32()
    if length + header + trailer != len(raw):
        raise CanonError("canon: declared content length does not match container size")
    content = r._take(length)
    content_hash = r.hash()
    crc = r.u32()
    r.expect_exhausted()

    if crc32c(raw[:-4]) != crc:
        raise CanonError("canon: crc32c mismatch (corrupt artifact)")
    if sha3_256(content) != content_hash:
        raise CanonError("canon: content hash mismatch")
    return Container(obj_type, version, content, content_hash)


def container_id(raw: bytes) -> str:
    """SHA3-256 of the whole container — the artifact's identity, as hex."""
    return hashlib.sha3_256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Typed objects
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    vocab_size: int
    seq_len: int
    rope_theta: int
    tie_embeddings: bool
    qk_norm: bool
    norm_id: int
    activation_id: int

    def parameter_count(self) -> int:
        """Mirrors ModelSpec::ParameterCount in src/canon/objects.cpp.

        Written from the same formula rather than derived from the tensor layout,
        so the two remain independent descriptions that can disagree and be caught
        disagreeing.
        """
        d = self.d_model
        head_dim = d // self.n_heads
        attn = (
            d * (self.n_heads * head_dim)      # wq
            + d * (self.n_kv_heads * head_dim)  # wk
            + d * (self.n_kv_heads * head_dim)  # wv
            + (self.n_heads * head_dim) * d     # wo
        )
        ffn = 3 * d * self.d_ff                 # gate, up, down
        norms = 2 * d + (2 * head_dim if self.qk_norm else 0)
        per_layer = attn + ffn + norms
        embed = self.vocab_size * d
        head = 0 if self.tie_embeddings else embed
        return per_layer * self.n_layers + embed + head + d   # + final norm


@dataclass
class RoundDescriptor:
    protocol_version: int
    network_magic: int
    round_id: int
    model: ModelSpec
    determinism_class: int
    optimizer_id: int
    tokenizer_hash: bytes
    dataset_root: bytes
    dataset_chunks: int
    genesis_hash: str = field(default="")

    @property
    def byte_level_tokenizer(self) -> bool:
        return self.tokenizer_hash == bytes(32)


def parse_round_descriptor(content: bytes) -> RoundDescriptor:
    r = Reader(content)
    protocol_version = r.u32()
    network_magic = r.u32()
    round_id = r.u64()
    model = ModelSpec(
        d_model=r.u32(),
        n_layers=r.u32(),
        n_heads=r.u32(),
        n_kv_heads=r.u32(),
        d_ff=r.u32(),
        vocab_size=r.u32(),
        seq_len=r.u32(),
        rope_theta=r.u32(),
        tie_embeddings=bool(r.u8()),
        qk_norm=bool(r.u8()),
        norm_id=r.u16(),
        activation_id=r.u16(),
    )
    determinism_class = r.u16()
    optimizer_id = r.u16()
    tokenizer_hash = r.hash()
    dataset_root = r.hash()
    dataset_chunks = r.u32()
    r.expect_exhausted()

    if protocol_version != PROTOCOL_VERSION:
        raise CanonError("round: unsupported protocol version")
    if model.d_model % model.n_heads != 0:
        raise CanonError("round: d_model not divisible by n_heads")
    if model.n_heads % model.n_kv_heads != 0:
        raise CanonError("round: n_heads not divisible by n_kv_heads (GQA)")

    return RoundDescriptor(
        protocol_version,
        network_magic,
        round_id,
        model,
        determinism_class,
        optimizer_id,
        tokenizer_hash,
        dataset_root,
        dataset_chunks,
    )


@dataclass
class DatasetManifest:
    """What a corpus is: raw UTF-8 text, addressed in document-aligned chunks.

    No token count and no dtype, because nothing is tokenized in advance — the
    worker tokenizes the chunk it is assigned. Chunk boundaries are derived by
    scanning for document separators rather than stored, so the manifest stays
    three numbers instead of an offset table.
    """

    dataset_root: bytes
    tokenizer_hash: bytes
    n_bytes: int
    target_chunk_bytes: int
    n_chunks: int


def parse_dataset_manifest(content: bytes) -> DatasetManifest:
    r = Reader(content)
    m = DatasetManifest(
        dataset_root=r.hash(),
        tokenizer_hash=r.hash(),
        n_bytes=r.u64(),
        target_chunk_bytes=r.u32(),
        n_chunks=r.u32(),
    )
    r.expect_exhausted()
    if m.n_bytes == 0:
        raise CanonError("dataset: empty corpus")
    if m.target_chunk_bytes == 0:
        raise CanonError("dataset: target_chunk_bytes must be non-zero")
    if m.n_chunks == 0:
        raise CanonError("dataset: n_chunks must be non-zero")
    # Chunks end at the first separator at or after the target, so each is at
    # least that long except the last. More than that allows means the boundaries
    # do not follow the rule, and a node that scans the corpus would disagree.
    if m.n_chunks > m.n_bytes // m.target_chunk_bytes + 1:
        raise CanonError(
            f"dataset: {m.n_chunks} chunks over {m.n_bytes} bytes at a target of "
            f"{m.target_chunk_bytes} — at least one is shorter than the target"
        )
    if m.n_chunks > m.n_bytes:
        raise CanonError("dataset: more chunks than bytes")
    return m


def load_container_file(path: str, expected_type: int) -> Container:
    with open(path, "rb") as f:
        raw = f.read()
    container = parse_container(raw)
    if container.obj_type != expected_type:
        raise CanonError(f"canon: expected object type {expected_type}, got {container.obj_type}")
    return container


OBJ_POLICY_DESCRIPTOR = 9


@dataclass
class PolicyDescriptor:
    """Operational policy — the second hashed artifact. Separate from the round
    because a challenge rate may be tuned from real data while the model's
    identity stays frozen."""

    protocol_version: int
    network_magic: int
    policy_version: int
    inner_steps: int
    micro_batch: int
    min_contributors: int
    checkpoint_interval: int
    dataset_chunk_tokens: int
    round_deadline_ms: int
    outer_momentum_q16: int
    outer_lr_q16: int
    nesterov: bool
    challenge_percent: int
    challenge_deadline_steps: int
    retained_checkpoints: int
    slash_quorum: int
    shadow_mode: bool
    policy_hash: str = field(default="")


def parse_policy_descriptor(content: bytes) -> PolicyDescriptor:
    r = Reader(content)
    p = PolicyDescriptor(
        protocol_version=r.u32(),
        network_magic=r.u32(),
        policy_version=r.u64(),
        inner_steps=r.u32(),
        micro_batch=r.u32(),
        min_contributors=r.u32(),
        checkpoint_interval=r.u32(),
        dataset_chunk_tokens=r.u32(),
        round_deadline_ms=r.u32(),
        outer_momentum_q16=r.u32(),
        outer_lr_q16=r.u32(),
        nesterov=bool(r.u8()),
        challenge_percent=r.u32(),
        challenge_deadline_steps=r.u32(),
        retained_checkpoints=r.u32(),
        slash_quorum=r.u32(),
        shadow_mode=bool(r.u8()),
    )
    r.expect_exhausted()
    if p.protocol_version != PROTOCOL_VERSION:
        raise CanonError("policy: unsupported protocol version")
    # The invariant that ties verification to storage: a challenge must expire
    # before the weights needed to answer it are pruned.
    if p.challenge_deadline_steps >= p.retained_checkpoints:
        raise CanonError(
            f"policy: challenge_deadline_steps {p.challenge_deadline_steps} >= "
            f"retained_checkpoints {p.retained_checkpoints} — unanswerable challenges"
        )
    return p

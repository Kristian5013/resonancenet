"""The envelope every consensus object travels in.

    "RNET" | version u32 | type u16 | length u32 | content | SHA3-256 | CRC32C

Two integrity fields, doing two different jobs, and conflating them is a
mistake worth naming:

  * The SHA3-256 is the object's IDENTITY. It is what the network agrees on,
    what a genesis anchor pins, and what an attacker would have to break.
  * The CRC32C is an ACCIDENT DETECTOR. It catches a truncated download or a
    bad sector and says so cheaply, before anyone spends a hash on it. It
    defends against nothing deliberate and is not meant to.

A container is refused unless magic, version, declared length, content hash and
CRC all agree AND nothing trails the end. Fail-closed: there is no partially
trusted object, because a caller handed one would have to decide which half to
believe.

VERSION 2.0. The format this replaces could not express a mixture-of-experts
layout or a per-round arithmetic contract, both of which are now consensus
fields rather than compiled-in assumptions. The version is the first thing after
the magic so that a node meeting a newer artifact says exactly that, instead of
misreading its fields and disagreeing about a hash later.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import IntEnum

from .stream import CanonError, Reader, Writer

MAGIC = b"RNET"
PROTOCOL_VERSION = 0x0002_0000  # 2.0

_HEADER = 4 + 4 + 2 + 4
_TRAILER = 32 + 4

# A consensus object is a description, never a payload. The largest legitimate
# one is a few hundred bytes; a megabyte ceiling is four orders of headroom and
# still refuses a length field that says four gigabytes.
MAX_CONTENT = 1 << 20

_CRC32C_POLY = 0x82F63B78  # Castagnoli, reflected


class ObjectType(IntEnum):
    """Every kind of object that can be hashed and agreed on.

    Adding a member here and nowhere else does not create an object type — the
    codec table in `objects.py` must know it too, and a test walks this enum to
    prove there is no member without one. That test exists because the previous
    implementation shipped a message type that was in the enum, absent from the
    dispatch tables, and failed at runtime with "unknown message type 0x0060".
    """

    ROUND_DESCRIPTOR = 1
    DATASET_MANIFEST = 2
    BATCH_SEED = 3
    CONTRIBUTION_HEADER = 4
    CHECKPOINT_HEADER = 5
    CHALLENGE_ORDER = 6
    VERIFICATION_VERDICT = 7
    SLASH_EVIDENCE = 8
    POLICY_DESCRIPTOR = 9
    EXPERT_SHARD = 10


def sha3_256(data: bytes) -> bytes:
    return hashlib.sha3_256(data).digest()


def crc32c(data: bytes) -> int:
    """Castagnoli CRC-32, bit at a time.

    Deliberately the obvious loop rather than a table: this runs once per
    artifact, the artifacts are hundreds of bytes, and a reader checking this
    implementation against the polynomial should not have to also check a
    generated table. Proved against a known vector by
    CanonTests.Crc32cMatchesTheReferenceVector.
    """
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ _CRC32C_POLY if crc & 1 else crc >> 1
    return crc ^ 0xFFFFFFFF


@dataclass(frozen=True)
class Container:
    obj_type: ObjectType
    version: int
    content: bytes

    @property
    def id(self) -> bytes:
        """SHA3-256 of the whole container — the artifact's identity."""
        return sha3_256(wrap(self.obj_type, self.content))


def wrap(obj_type: ObjectType, content: bytes) -> bytes:
    if len(content) > MAX_CONTENT:
        raise CanonError(f"canon: content of {len(content)} bytes exceeds {MAX_CONTENT}")
    body = (Writer()
            .raw(MAGIC)
            .u32(PROTOCOL_VERSION)
            .u16(obj_type)
            .u32(len(content))
            .raw(content)
            .hash(sha3_256(content))
            .take())
    return body + crc32c(body).to_bytes(4, "big")


def parse(raw: bytes) -> Container:
    if len(raw) < _HEADER + _TRAILER:
        raise CanonError(f"canon: container of {len(raw)} bytes is too short to hold a header")
    if raw[:4] != MAGIC:
        raise CanonError("canon: not a ResonanceNet container (bad magic)")

    r = Reader(raw)
    r._take(4)  # the magic, already checked
    version = r.u32()
    if version != PROTOCOL_VERSION:
        raise CanonError(
            f"canon: container is protocol {version >> 16}.{version & 0xFFFF}, "
            f"this build speaks {PROTOCOL_VERSION >> 16}.{PROTOCOL_VERSION & 0xFFFF}"
        )
    try:
        obj_type = ObjectType(r.u16())
    except ValueError as exc:
        raise CanonError(f"canon: unknown object type ({exc})") from exc

    length = r.u32()
    if length + _HEADER + _TRAILER != len(raw):
        raise CanonError(
            f"canon: declared content length {length} does not match the "
            f"{len(raw) - _HEADER - _TRAILER} bytes present"
        )
    content = r._take(length)
    content_hash = r.hash()
    crc = r.u32()
    r.expect_exhausted()

    # CRC first: it is the cheap check, and if the bytes arrived damaged there is
    # no point spending a SHA3 to find out they are also the wrong bytes.
    if crc32c(raw[:-4]) != crc:
        raise CanonError("canon: crc32c mismatch — the artifact is corrupt, not merely wrong")
    if sha3_256(content) != content_hash:
        raise CanonError("canon: content hash mismatch")

    return Container(obj_type=obj_type, version=version, content=content)


def load(path: str, expected: ObjectType) -> Container:
    """Read and verify an artifact from disk, insisting on its type.

    The type is a required argument because "parse whatever this is" is how a
    policy descriptor gets read as a round descriptor by a caller that then
    trusts the wrong fields.
    """
    with open(path, "rb") as f:
        container = parse(f.read())
    if container.obj_type is not expected:
        raise CanonError(
            f"canon: {path} holds a {container.obj_type.name}, expected a {expected.name}"
        )
    return container

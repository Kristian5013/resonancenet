"""The wire: how two nodes frame and name what they say to each other.

    magic u32 | command u16 | length u32 | checksum u32 | payload

The magic is the network's, so a node that somehow reaches a peer on another
network fails at the first four bytes rather than after a handshake. The
checksum is the first four bytes of SHA3-256 over the payload — an accident
detector, like the one on a consensus container and for the same reason: it is
cheap, it catches a truncated read, and it defends against nothing deliberate,
because anything deliberate would have recomputed it.

EVERY LIMIT IS A CONSTANT HERE, NOT A JUDGEMENT AT THE CALL SITE. A length read
off a socket is an attacker-chosen allocation. The header is fixed-width and the
length is checked against `MAX_PAYLOAD` before a single byte of body is read, so
a peer claiming four gigabytes costs fourteen bytes and a disconnection.

THE HANDSHAKE CARRIES THE ANCHORS. A peer whose genesis or policy hash differs
is on a different network however similar it looks, and finding that out at the
handshake costs one message; finding it out later means having relayed objects
nobody can use and having to explain why.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import IntEnum

from ..canon.stream import CanonError, Reader, Writer
from . import framing
from .address import NetAddress, Services, TimestampedAddress

PROTOCOL_VERSION = 2

# Framing itself lives in framing.py, shared with the worker channel: two
# layouts would mean two places to get a length check wrong.
HEADER_SIZE = framing.HEADER_SIZE

# Nothing this protocol says is large. Bulk — weights, corpus chunks — moves in
# explicitly chunked messages, so the ceiling is about what a header may claim,
# not about what a node may transfer.
MAX_PAYLOAD = 2 << 20

MAX_ADDR = 1000
MAX_INV = 50_000
MAX_HEADERS = 2000
MAX_LOCATORS = 64
MAX_USER_AGENT = 64
MAX_CHUNK_BYTES = 1 << 20


class Command(IntEnum):
    """Every message this protocol has.

    A command added here and to no dispatch table is not a command. The
    implementation this replaces shipped exactly that and found out on a live
    node, as "unknown message type 0x0060" — so ProtocolTests.EveryCommandHasA
    Codec walks this enum and requires one.
    """

    VERSION = 1
    VERACK = 2
    PING = 3
    PONG = 4
    GETADDR = 5
    ADDR = 6
    INV = 7
    GETDATA = 8
    OBJECT = 9
    NOTFOUND = 10
    GETHEADERS = 11
    HEADERS = 12
    GETCHUNK = 13
    CHUNK = 14
    REJECT = 15


class InvType(IntEnum):
    """What kind of thing an inventory entry names."""

    CONTRIBUTION = 1
    CHECKPOINT = 2
    CHALLENGE = 3
    VERDICT = 4
    SLASH_EVIDENCE = 5
    EXPERT_SHARD = 6


class RejectCode(IntEnum):
    MALFORMED = 1
    INVALID = 2
    OBSOLETE = 3
    DUPLICATE = 4
    UNAVAILABLE = 5
    RATE_LIMITED = 6


class ProtocolError(Exception):
    """A peer said something that ends the conversation."""


Header = framing.Header
checksum = framing.checksum


def frame(magic: int, command: Command, payload: bytes) -> bytes:
    try:
        return framing.frame(magic, command, payload, max_payload=MAX_PAYLOAD)
    except framing.FramingError as exc:
        raise ProtocolError(str(exc)) from exc


def parse_header(raw: bytes, expected_magic: int) -> Header:
    try:
        return framing.parse_header(raw, expected_magic, Command,
                                    max_payload=MAX_PAYLOAD)
    except framing.FramingError as exc:
        raise ProtocolError(str(exc)) from exc


def check_payload(header: Header, payload: bytes) -> None:
    try:
        framing.check_payload(header, payload)
    except framing.FramingError as exc:
        raise ProtocolError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

class Message:
    COMMAND: Command

    def serialize(self, w: Writer) -> Writer:      # pragma: no cover - interface
        raise NotImplementedError

    @classmethod
    def parse(cls, r: Reader):                     # pragma: no cover - interface
        raise NotImplementedError

    def payload(self) -> bytes:
        return self.serialize(Writer()).take()

    def to_frame(self, magic: int) -> bytes:
        return frame(magic, self.COMMAND, self.payload())


@dataclass(frozen=True)
class Version(Message):
    """The first thing either side says.

    `nonce` is how a node notices it has dialled itself: every outbound
    connection carries a fresh one, and a VERSION coming back with a nonce the
    node itself issued means the far end is this node. Without it a node with
    its own address in its table connects to itself forever, and the
    implementation this replaces spent three rounds of debugging on exactly
    that.

    The anchors are here rather than in a later message because a peer on
    another network has nothing to offer and finding out early is free.
    """

    COMMAND = Command.VERSION

    version: int
    services: Services
    timestamp_ms: int
    receiver: NetAddress
    sender: NetAddress
    nonce: int
    user_agent: str
    start_height: int
    genesis_hash: bytes
    policy_hash: bytes

    def serialize(self, w: Writer) -> Writer:
        w.u32(self.version).u64(self.services).u64(self.timestamp_ms)
        self.receiver.serialize(w)
        self.sender.serialize(w)
        return (w.u64(self.nonce).string(self.user_agent).u64(self.start_height)
                 .hash(self.genesis_hash).hash(self.policy_hash))

    @classmethod
    def parse(cls, r: Reader) -> "Version":
        return cls(
            version=r.u32(), services=Services(r.u64()), timestamp_ms=r.u64(),
            receiver=NetAddress.parse_wire(r), sender=NetAddress.parse_wire(r),
            nonce=r.u64(), user_agent=r.string(limit=MAX_USER_AGENT),
            start_height=r.u64(), genesis_hash=r.hash(), policy_hash=r.hash())


@dataclass(frozen=True)
class Verack(Message):
    COMMAND = Command.VERACK

    def serialize(self, w: Writer) -> Writer:
        return w

    @classmethod
    def parse(cls, r: Reader) -> "Verack":
        return cls()


@dataclass(frozen=True)
class Ping(Message):
    COMMAND = Command.PING
    nonce: int

    def serialize(self, w: Writer) -> Writer:
        return w.u64(self.nonce)

    @classmethod
    def parse(cls, r: Reader) -> "Ping":
        return cls(nonce=r.u64())


@dataclass(frozen=True)
class Pong(Message):
    """Echoes the ping's nonce, which is what makes it an answer rather than
    a message that merely arrived afterwards."""

    COMMAND = Command.PONG
    nonce: int

    def serialize(self, w: Writer) -> Writer:
        return w.u64(self.nonce)

    @classmethod
    def parse(cls, r: Reader) -> "Pong":
        return cls(nonce=r.u64())


@dataclass(frozen=True)
class GetAddr(Message):
    COMMAND = Command.GETADDR

    def serialize(self, w: Writer) -> Writer:
        return w

    @classmethod
    def parse(cls, r: Reader) -> "GetAddr":
        return cls()


@dataclass(frozen=True)
class Addr(Message):
    COMMAND = Command.ADDR
    addresses: tuple[TimestampedAddress, ...] = field(default=())

    def serialize(self, w: Writer) -> Writer:
        if len(self.addresses) > MAX_ADDR:
            raise ProtocolError(f"wire: {len(self.addresses)} addresses over {MAX_ADDR}")
        w.u32(len(self.addresses))
        for a in self.addresses:
            a.serialize(w)
        return w

    @classmethod
    def parse(cls, r: Reader) -> "Addr":
        n = r.u32()
        if n > MAX_ADDR:
            raise CanonError(f"wire: {n} addresses over the {MAX_ADDR} limit")
        return cls(addresses=tuple(TimestampedAddress.parse_wire(r) for _ in range(n)))


@dataclass(frozen=True)
class InvEntry:
    type: InvType
    hash: bytes

    def serialize(self, w: Writer) -> Writer:
        return w.u16(self.type).hash(self.hash)

    @classmethod
    def parse_wire(cls, r: Reader) -> "InvEntry":
        try:
            kind = InvType(r.u16())
        except ValueError as exc:
            raise CanonError(f"wire: unknown inventory type ({exc})") from exc
        return cls(type=kind, hash=r.hash())


class _InvList(Message):
    """Shared shape for the three messages that carry a list of inventory."""

    def __init__(self, entries=()):
        self.entries = tuple(entries)

    def __eq__(self, other):
        return type(self) is type(other) and self.entries == other.entries

    def __repr__(self):
        return f"{type(self).__name__}({len(self.entries)} entries)"

    def serialize(self, w: Writer) -> Writer:
        if len(self.entries) > MAX_INV:
            raise ProtocolError(f"wire: {len(self.entries)} entries over {MAX_INV}")
        w.u32(len(self.entries))
        for e in self.entries:
            e.serialize(w)
        return w

    @classmethod
    def parse(cls, r: Reader):
        n = r.u32()
        if n > MAX_INV:
            raise CanonError(f"wire: {n} inventory entries over the {MAX_INV} limit")
        return cls(tuple(InvEntry.parse_wire(r) for _ in range(n)))


class Inv(_InvList):
    """"I have these." Announcements, never data."""
    COMMAND = Command.INV


class GetData(_InvList):
    """"Send me these." """
    COMMAND = Command.GETDATA


class NotFound(_InvList):
    """"I said I had these and now I do not."

    Answering explicitly matters: a requester that got silence could not tell a
    peer that has gone quiet from one that never had the object, and would wait
    out its timeout for both.
    """
    COMMAND = Command.NOTFOUND


@dataclass(frozen=True)
class Object(Message):
    """One consensus object, as its canonical container.

    The bytes are passed through untouched — the receiver parses and verifies
    them itself, because an object that arrived over the network has exactly the
    standing of an object found on disk, which is none until it hashes right.
    """

    COMMAND = Command.OBJECT
    container: bytes = b""

    def serialize(self, w: Writer) -> Writer:
        return w.blob(self.container)

    @classmethod
    def parse(cls, r: Reader) -> "Object":
        return cls(container=r.blob(limit=MAX_PAYLOAD - 8))


@dataclass(frozen=True)
class GetHeaders(Message):
    """Ask for the chain from the first locator the peer recognises.

    A locator is a list of checkpoint ids going back exponentially, which is how
    two nodes find their common ancestor in one round trip instead of walking
    back one step at a time.
    """

    COMMAND = Command.GETHEADERS
    locators: tuple[bytes, ...] = field(default=())
    stop: bytes = bytes(32)

    def serialize(self, w: Writer) -> Writer:
        if len(self.locators) > MAX_LOCATORS:
            raise ProtocolError(f"wire: {len(self.locators)} locators over {MAX_LOCATORS}")
        w.u32(len(self.locators))
        for h in self.locators:
            w.hash(h)
        return w.hash(self.stop)

    @classmethod
    def parse(cls, r: Reader) -> "GetHeaders":
        n = r.u32()
        if n > MAX_LOCATORS:
            raise CanonError(f"wire: {n} locators over the {MAX_LOCATORS} limit")
        return cls(locators=tuple(r.hash() for _ in range(n)), stop=r.hash())


@dataclass(frozen=True)
class Headers(Message):
    COMMAND = Command.HEADERS
    headers: tuple[bytes, ...] = field(default=())

    def serialize(self, w: Writer) -> Writer:
        if len(self.headers) > MAX_HEADERS:
            raise ProtocolError(f"wire: {len(self.headers)} headers over {MAX_HEADERS}")
        w.u32(len(self.headers))
        for h in self.headers:
            w.blob(h)
        return w

    @classmethod
    def parse(cls, r: Reader) -> "Headers":
        n = r.u32()
        if n > MAX_HEADERS:
            raise CanonError(f"wire: {n} headers over the {MAX_HEADERS} limit")
        return cls(headers=tuple(r.blob(limit=4096) for _ in range(n)))


@dataclass(frozen=True)
class GetChunk(Message):
    """Ask for one corpus chunk by index, against the round's pinned root."""

    COMMAND = Command.GETCHUNK
    dataset_root: bytes = bytes(32)
    index: int = 0

    def serialize(self, w: Writer) -> Writer:
        return w.hash(self.dataset_root).u64(self.index)

    @classmethod
    def parse(cls, r: Reader) -> "GetChunk":
        return cls(dataset_root=r.hash(), index=r.u64())


@dataclass(frozen=True)
class Chunk(Message):
    """Corpus bytes with the proof that they belong to the pinned root.

    The proof travels with the data rather than being fetched separately: a
    chunk without one is bytes nobody can place, and making the receiver ask
    twice would double the round trips on the one message type that carries real
    volume.
    """

    COMMAND = Command.CHUNK
    dataset_root: bytes = bytes(32)
    index: int = 0
    leaf_count: int = 0
    proof: tuple[bytes, ...] = field(default=())
    data: bytes = b""

    def serialize(self, w: Writer) -> Writer:
        if len(self.data) > MAX_CHUNK_BYTES:
            raise ProtocolError(
                f"wire: a {len(self.data)}-byte chunk exceeds {MAX_CHUNK_BYTES}")
        w.hash(self.dataset_root).u64(self.index).u64(self.leaf_count)
        w.u32(len(self.proof))
        for h in self.proof:
            w.hash(h)
        return w.blob(self.data)

    @classmethod
    def parse(cls, r: Reader) -> "Chunk":
        dataset_root, index, leaf_count = r.hash(), r.u64(), r.u64()
        depth = r.u32()
        # A Merkle path over 2^40 leaves is forty hashes; sixty-four is generous
        # and still refuses a peer that wants a million.
        if depth > 64:
            raise CanonError(f"wire: a proof of {depth} siblings is not a proof")
        proof = tuple(r.hash() for _ in range(depth))
        return cls(dataset_root=dataset_root, index=index, leaf_count=leaf_count,
                   proof=proof, data=r.blob(limit=MAX_CHUNK_BYTES))


@dataclass(frozen=True)
class Reject(Message):
    """Why the last thing was refused.

    Advisory and never load-bearing: a node must behave identically whether or
    not this arrives, because a peer that wanted to mislead would simply not
    send it. It exists so an operator can see why two nodes will not talk.
    """

    COMMAND = Command.REJECT
    command: Command = Command.VERSION
    code: RejectCode = RejectCode.INVALID
    reason: str = ""

    def serialize(self, w: Writer) -> Writer:
        return w.u16(self.command).u16(self.code).string(self.reason[:200])

    @classmethod
    def parse(cls, r: Reader) -> "Reject":
        try:
            command, code = Command(r.u16()), RejectCode(r.u16())
        except ValueError as exc:
            raise CanonError(f"wire: unknown reject field ({exc})") from exc
        return cls(command=command, code=code, reason=r.string(limit=256))


CODECS: dict[Command, type[Message]] = {
    Command.VERSION: Version,
    Command.VERACK: Verack,
    Command.PING: Ping,
    Command.PONG: Pong,
    Command.GETADDR: GetAddr,
    Command.ADDR: Addr,
    Command.INV: Inv,
    Command.GETDATA: GetData,
    Command.OBJECT: Object,
    Command.NOTFOUND: NotFound,
    Command.GETHEADERS: GetHeaders,
    Command.HEADERS: Headers,
    Command.GETCHUNK: GetChunk,
    Command.CHUNK: Chunk,
    Command.REJECT: Reject,
}


def decode(command: Command, payload: bytes) -> Message:
    codec = CODECS.get(command)
    if codec is None:
        raise ProtocolError(f"wire: {command.name} has no codec")
    r = Reader(payload)
    try:
        message = codec.parse(r)
        r.expect_exhausted()
    except CanonError as exc:
        raise ProtocolError(f"wire: {command.name}: {exc}") from exc
    return message

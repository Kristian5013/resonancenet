"""Message framing, shared by every socket in the project.

    magic u32 | command u16 | length u32 | checksum u32 | payload

One layout, used by the peer-to-peer wire and by the worker channel alike.
Having two would mean two places to get a length check wrong, and the length
check is the whole reason the header is fixed-width: a length read off a socket
is an attacker-chosen allocation, so it is validated before a single byte of
body is read.

The checksum is the first four bytes of SHA3-256 over the payload. An accident
detector, not a defence: it catches a truncated read cheaply and stops nothing
deliberate, because anything deliberate would have recomputed it.

The command enum is a parameter rather than a fixed set, because the two
channels carry different vocabularies and there is no sense in either knowing
the other's.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..canon.stream import Reader, Writer

HEADER_SIZE = 4 + 2 + 4 + 4


class FramingError(Exception):
    """A frame that cannot be read. Always ends the conversation."""


def checksum(payload: bytes) -> int:
    return int.from_bytes(hashlib.sha3_256(payload).digest()[:4], "big")


def frame(magic: int, command: int, payload: bytes, *, max_payload: int) -> bytes:
    if len(payload) > max_payload:
        raise FramingError(
            f"wire: a {len(payload)}-byte payload exceeds the {max_payload} limit")
    return (Writer().u32(magic).u16(int(command)).u32(len(payload))
            .u32(checksum(payload)).raw(payload).take())


@dataclass(frozen=True)
class Header:
    magic: int
    command: object
    length: int
    checksum: int


def parse_header(raw: bytes, expected_magic: int, commands, *,
                 max_payload: int) -> Header:
    """Read the fixed bytes and refuse before any body is read."""
    if len(raw) != HEADER_SIZE:
        raise FramingError(f"wire: a header is {HEADER_SIZE} bytes, got {len(raw)}")
    r = Reader(raw)
    magic = r.u32()
    if magic != expected_magic:
        raise FramingError(
            f"wire: magic {magic:#010x} is not this channel's {expected_magic:#010x}")
    try:
        command = commands(r.u16())
    except ValueError as exc:
        raise FramingError(f"wire: unknown command ({exc})") from exc
    length = r.u32()
    if length > max_payload:
        raise FramingError(
            f"wire: {command.name} claims {length} bytes, over the "
            f"{max_payload} limit")
    return Header(magic, command, length, r.u32())


def check_payload(header: Header, payload: bytes) -> None:
    if len(payload) != header.length:
        raise FramingError(
            f"wire: {header.command.name} declared {header.length} bytes, got "
            f"{len(payload)}")
    if checksum(payload) != header.checksum:
        raise FramingError(f"wire: {header.command.name} failed its checksum")


async def read_frame(reader, magic: int, commands, *, max_payload: int):
    """One whole frame from an asyncio stream. Returns (header, payload)."""
    head = await reader.readexactly(HEADER_SIZE)
    header = parse_header(head, magic, commands, max_payload=max_payload)
    payload = await reader.readexactly(header.length) if header.length else b""
    check_payload(header, payload)
    return header, payload

"""CBOR restricted to exactly one encoding per value — the Python mirror of src/ipc/cbor.cpp.

This is deliberately not `cbor2`. A general CBOR library accepts every spelling
the format permits, and this channel carries consensus work: a decoder that reads
two different byte strings as the same message is a decoder that lets meaning
hide. What is implemented here is a profile with one legal encoding of every
value, and it must agree with the C++ side byte for byte.

The Python-specific traps, each of which has bitten someone:

  * `bool` IS a subclass of `int`. `isinstance(True, int)` is True, so a type
    dispatch that checks int first encodes True as 1 and the two languages
    disagree about the type of every flag. bool is checked first, always.

  * Python integers are unbounded. A value that cannot be encoded is rejected here
    rather than silently truncated into something the C++ side would read as a
    different number.

  * `str` and `bytes` are different types with different major types. Passing one
    where the other belongs is an error, not a coercion.

  * dict preserves insertion order, which is NOT the canonical order. Maps are
    sorted by encoded key bytes on the way out and verified on the way in.
"""

from __future__ import annotations

import struct

# Nesting past this is refused: our messages are a map of scalars with at most an
# array of maps inside, so anything deeper is a bug or an attempt to exhaust the
# parser's stack.
MAX_DEPTH = 8

# Items in one container.
MAX_ITEMS = 65_536

_MAJOR_UINT = 0
_MAJOR_NINT = 1
_MAJOR_BYTES = 2
_MAJOR_TEXT = 3
_MAJOR_ARRAY = 4
_MAJOR_MAP = 5
_MAJOR_TAG = 6
_MAJOR_SIMPLE = 7

_SIMPLE_FALSE = 20
_SIMPLE_TRUE = 21

_UINT64_MAX = 0xFFFF_FFFF_FFFF_FFFF
_INT64_MAX = 0x7FFF_FFFF_FFFF_FFFF
_INT64_MIN = -0x8000_0000_0000_0000


class CborError(ValueError):
    """Every rejection raises this. There is no lenient mode."""


def _head(major: int, value: int) -> bytes:
    """The one legal head for a value. Anything longer is a second spelling."""
    if value < 0 or value > _UINT64_MAX:
        raise CborError(f"cbor: {value} is out of range for a head argument")
    prefix = major << 5
    if value < 24:
        return bytes([prefix | value])
    if value <= 0xFF:
        return bytes([prefix | 24, value])
    if value <= 0xFFFF:
        return bytes([prefix | 25]) + struct.pack(">H", value)
    if value <= 0xFFFF_FFFF:
        return bytes([prefix | 26]) + struct.pack(">I", value)
    return bytes([prefix | 27]) + struct.pack(">Q", value)


def _minimal_head_length(value: int) -> int:
    if value < 24:
        return 1
    if value <= 0xFF:
        return 2
    if value <= 0xFFFF:
        return 3
    if value <= 0xFFFF_FFFF:
        return 5
    return 9


def encode_key(key: str) -> bytes:
    """The encoded form of a text key — what the canonical ordering compares."""
    if not isinstance(key, str):
        raise CborError("cbor: map keys must be text strings")
    raw = key.encode("utf-8")
    return _head(_MAJOR_TEXT, len(raw)) + raw


def key_sort_order(key: str) -> bytes:
    """Sort key for canonical map ordering: the encoded bytes themselves.

    Shorter encoded keys sort first, so "z" comes before "aa". That surprises
    people, which is why it is a named function with a test rather than an inline
    `sorted(d)`.
    """
    return encode_key(key)


def dumps(value) -> bytes:
    """Encodes one value in the canonical profile."""
    out = bytearray()
    _encode(value, out, 0)
    return bytes(out)


def _encode(value, out: bytearray, depth: int) -> None:
    if depth > MAX_DEPTH:
        raise CborError(f"cbor: nesting deeper than {MAX_DEPTH}")

    # bool BEFORE int: bool is a subclass of int in Python, and checking int first
    # would encode every flag as a number.
    if isinstance(value, bool):
        out += _head(_MAJOR_SIMPLE, _SIMPLE_TRUE if value else _SIMPLE_FALSE)
        return

    if isinstance(value, int):
        if value >= 0:
            if value > _UINT64_MAX:
                raise CborError(f"cbor: {value} exceeds what an unsigned 64-bit value can hold")
            out += _head(_MAJOR_UINT, value)
        else:
            if value < _INT64_MIN:
                raise CborError(f"cbor: {value} exceeds what a signed 64-bit value can hold")
            out += _head(_MAJOR_NINT, -1 - value)
        return

    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        out += _head(_MAJOR_BYTES, len(raw))
        out += raw
        return

    if isinstance(value, str):
        raw = value.encode("utf-8")
        out += _head(_MAJOR_TEXT, len(raw))
        out += raw
        return

    if isinstance(value, (list, tuple)):
        if len(value) > MAX_ITEMS:
            raise CborError("cbor: array exceeds the item limit")
        out += _head(_MAJOR_ARRAY, len(value))
        for item in value:
            _encode(item, out, depth + 1)
        return

    if isinstance(value, dict):
        if len(value) > MAX_ITEMS:
            raise CborError("cbor: map exceeds the item limit")
        # Insertion order is not canonical order. Sorted by encoded key bytes, so
        # the order is a property of the keys rather than of whoever built the
        # dict.
        keys = sorted(value.keys(), key=key_sort_order)
        out += _head(_MAJOR_MAP, len(keys))
        for key in keys:
            out += encode_key(key)
            _encode(value[key], out, depth + 1)
        return

    # Floats land here deliberately: half, single and double all encode 1.0, NaN
    # carries a payload, and -0.0 is a distinct bit pattern that compares equal to
    # 0.0. Anything needing a fraction crosses this boundary as a scaled integer.
    if isinstance(value, float):
        raise CborError(
            "cbor: floats are not permitted — send a scaled integer, as the protocol does elsewhere"
        )
    if value is None:
        raise CborError("cbor: null is not permitted — omit the key instead")
    raise CborError(f"cbor: cannot encode {type(value).__name__}")


def loads(raw: bytes):
    """Parses one complete value and requires the input to be exhausted.

    Trailing bytes are an error for the same reason they are in the canon
    container format: an attacker who can append data that some parsers ignore can
    make two readers see different messages in one frame.
    """
    value, offset = _decode(raw, 0, 0)
    if offset != len(raw):
        raise CborError(f"cbor: {len(raw) - offset} trailing bytes after the value")
    return value


def _read_head(raw: bytes, offset: int) -> tuple[int, int, int]:
    if offset >= len(raw):
        raise CborError("cbor: truncated at a head byte")
    initial = raw[offset]
    offset += 1
    major = initial >> 5
    info = initial & 0x1F

    if info < 24:
        return major, info, offset
    if info == 31:
        # A second way to write what a definite length already says, and it lets a
        # sender stream past any bound checked on the head.
        raise CborError("cbor: indefinite-length items are not permitted")
    if info not in (24, 25, 26, 27):
        raise CborError(f"cbor: reserved additional information {info}")

    extra = {24: 1, 25: 2, 26: 4, 27: 8}[info]
    if offset + extra > len(raw):
        raise CborError("cbor: truncated argument")
    argument = int.from_bytes(raw[offset : offset + extra], "big")
    offset += extra

    # 1 must be 0x01, never 0x1801: two byte strings that mean the same thing let
    # any check made on bytes rather than meaning be walked around.
    if _minimal_head_length(argument) != extra + 1:
        raise CborError(f"cbor: {argument} is not encoded in its shortest form")
    return major, argument, offset


def _decode(raw: bytes, offset: int, depth: int):
    if depth > MAX_DEPTH:
        raise CborError(f"cbor: nesting deeper than {MAX_DEPTH}")
    major, argument, offset = _read_head(raw, offset)
    remaining = len(raw) - offset

    if major == _MAJOR_UINT:
        return argument, offset

    if major == _MAJOR_NINT:
        if argument > _INT64_MAX:
            raise CborError("cbor: negative integer out of range")
        return -1 - argument, offset

    if major in (_MAJOR_BYTES, _MAJOR_TEXT):
        # The length is a claim by the sender, checked against the bytes actually
        # present before anything is allocated.
        if argument > remaining:
            raise CborError(f"cbor: string claims {argument} bytes, {remaining} remain")
        chunk = raw[offset : offset + argument]
        offset += argument
        if major == _MAJOR_BYTES:
            return chunk, offset
        try:
            # strict=True rejects surrogates; the codec already rejects overlong
            # forms and truncated sequences.
            return chunk.decode("utf-8", errors="strict"), offset
        except UnicodeDecodeError as exc:
            raise CborError(f"cbor: invalid utf-8: {exc}") from exc

    if major == _MAJOR_ARRAY:
        if argument > MAX_ITEMS:
            raise CborError(f"cbor: array of {argument} items exceeds the limit")
        if argument > remaining:
            raise CborError("cbor: array declares more items than the input can hold")
        items = []
        for _ in range(argument):
            item, offset = _decode(raw, offset, depth + 1)
            items.append(item)
        return items, offset

    if major == _MAJOR_MAP:
        if argument > MAX_ITEMS:
            raise CborError(f"cbor: map of {argument} pairs exceeds the limit")
        if argument * 2 > remaining:
            raise CborError("cbor: map declares more pairs than the input can hold")
        entries: dict[str, object] = {}
        previous_key: bytes | None = None
        for _ in range(argument):
            key_start = offset
            key, offset = _decode(raw, offset, depth + 1)
            if not isinstance(key, str):
                # Integer keys would be a second way to name the same field, and
                # mixed key types make the ordering rule depend on which types
                # happen to be present.
                raise CborError("cbor: map keys must be text strings")
            encoded_key = raw[key_start:offset]
            if previous_key is not None:
                if encoded_key == previous_key:
                    # "Last wins" and "first wins" are both defensible, which is
                    # exactly why two implementations disagree.
                    raise CborError("cbor: duplicate map key")
                if encoded_key < previous_key:
                    raise CborError("cbor: map keys are not in canonical order")
            previous_key = encoded_key
            entries[key], offset = _decode(raw, offset, depth + 1)
        return entries, offset

    if major == _MAJOR_SIMPLE:
        if argument in (_SIMPLE_FALSE, _SIMPLE_TRUE):
            return argument == _SIMPLE_TRUE, offset
        # Floats, null and undefined all land here and are all refused.
        raise CborError("cbor: only true and false are permitted in major type 7")

    if major == _MAJOR_TAG:
        # A tag changes what a value means without changing its bytes.
        raise CborError("cbor: tags are not permitted")

    raise CborError(f"cbor: unknown major type {major}")


# --- typed field readers -----------------------------------------------------
#
# A message parser should read as a list of requirements, not as a chain of checks
# that is easy to leave incomplete. Each of these fails rather than converting: a
# silent conversion is how a length gets read out of a boolean.


def require_map(value) -> dict:
    if not isinstance(value, dict):
        raise CborError("cbor: value is not a map")
    return value


def get_uint(message: dict, key: str) -> int:
    value = _require(message, key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CborError(f"cbor: field '{key}' is not an unsigned integer")
    return value


def get_int(message: dict, key: str) -> int:
    value = _require(message, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CborError(f"cbor: field '{key}' is not an integer")
    return value


def get_bytes(message: dict, key: str) -> bytes:
    value = _require(message, key)
    if not isinstance(value, bytes):
        raise CborError(f"cbor: field '{key}' is not a byte string")
    return value


def get_hash(message: dict, key: str) -> bytes:
    """A hash field of the wrong length is not a hash.

    Padding or truncating one would turn a malformed message into a valid-looking
    identity.
    """
    value = get_bytes(message, key)
    if len(value) != 32:
        raise CborError(f"cbor: field '{key}' is {len(value)} bytes, a hash is 32")
    return value


def get_text(message: dict, key: str) -> str:
    value = _require(message, key)
    if not isinstance(value, str):
        raise CborError(f"cbor: field '{key}' is not a text string")
    return value


def get_bool(message: dict, key: str) -> bool:
    value = _require(message, key)
    if not isinstance(value, bool):
        raise CborError(f"cbor: field '{key}' is not a boolean")
    return value


def _require(message: dict, key: str):
    if not isinstance(message, dict):
        raise CborError("cbor: value is not a map")
    if key not in message:
        # A missing field is an error rather than a default: a message that forgot
        # to say which step it refers to must not be read as referring to step
        # zero.
        raise CborError(f"cbor: required field '{key}' is missing")
    return message[key]

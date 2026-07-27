"""The one way this project turns values into bytes.

Every hash the network agrees on is a hash of bytes produced here, so the rules
are deliberately few and deliberately rigid:

  * BIG-ENDIAN, FIXED-WIDTH. No native sizes, no padding, no alignment. A u32 is
    four bytes on every machine that will ever run this.
  * LENGTH-PREFIXED, NEVER TERMINATED. A reader always knows how much to take
    before it takes it, so a truncated stream is an error rather than a shorter
    value that happens to parse.
  * NOTHING IMPLICIT. No optional fields, no defaults filled in on read. If a
    field is not in the bytes, it does not exist.
  * A READ THAT LEAVES BYTES BEHIND IS A FAILED READ. `expect_exhausted` is not
    politeness; trailing bytes mean the writer and the reader disagree about the
    shape of the object, and the reader must not guess which of them is right.

One deliberate difference from the C++ this replaces. There, `Writer::U32` took
a `static_cast<uint32_t>` and a value too large for the field was silently
truncated — a payload of 2^32 + 7 bytes wrote a length of 7. Here `int.to_bytes`
raises OverflowError, so the same mistake is a crash at the writer rather than a
corrupt artifact at every reader. Proved by CanonTests.OversizedFieldIsRefused.
"""

from __future__ import annotations


class CanonError(Exception):
    """Any failure to produce or consume canonical bytes.

    One exception type on purpose: callers must not branch on the kind of
    malformation, because "which way was this object wrong" is exactly the
    question an attacker gets to choose the answer to.
    """


class Writer:
    """Accumulates canonical bytes. Every method returns self so field order
    reads as a single expression and a reordering is visible in review."""

    __slots__ = ("_buf",)

    def __init__(self) -> None:
        self._buf = bytearray()

    def _fixed(self, value: int, width: int, *, signed: bool = False) -> "Writer":
        if not isinstance(value, int) or isinstance(value, bool):
            # bool is an int subclass, and `w.u8(True)` writing 1 is fine while
            # `w.u32(True)` writing 0x00000001 is a field-width confusion waiting
            # to happen. Callers say what they mean via .boolean().
            raise CanonError(f"canon: {type(value).__name__} is not an integer field")
        try:
            self._buf += value.to_bytes(width, "big", signed=signed)
        except OverflowError as exc:
            raise CanonError(
                f"canon: {value} does not fit in {'i' if signed else 'u'}{width * 8}"
            ) from exc
        return self

    def u8(self, v: int) -> "Writer":
        return self._fixed(v, 1)

    def u16(self, v: int) -> "Writer":
        return self._fixed(v, 2)

    def u32(self, v: int) -> "Writer":
        return self._fixed(v, 4)

    def u64(self, v: int) -> "Writer":
        return self._fixed(v, 8)

    def i64(self, v: int) -> "Writer":
        """Two's complement, so a negative momentum term round-trips as itself.

        The C++ wrote signed values by reinterpreting the bits through
        `static_cast<uint64_t>`, which is the same bytes reached by a rule that
        only holds because the platform is two's complement. Saying `signed=True`
        states the intent instead of relying on it.
        """
        return self._fixed(v, 8, signed=True)

    def boolean(self, v: bool) -> "Writer":
        """One byte, 0 or 1, and nothing else is a boolean.

        Accepting any non-zero byte on read would give an object two encodings
        and therefore two hashes, which is the whole failure this module exists
        to prevent. Proved by CanonTests.BooleanRejectsNonCanonicalByte.
        """
        if v is not True and v is not False:
            raise CanonError(f"canon: {v!r} is not a boolean")
        return self._fixed(1 if v else 0, 1)

    def raw(self, data: bytes) -> "Writer":
        """Bytes whose length is known from context — a 32-byte hash, mostly."""
        self._buf += data
        return self

    def hash(self, digest: bytes) -> "Writer":
        if len(digest) != 32:
            raise CanonError(f"canon: a hash is 32 bytes, got {len(digest)}")
        self._buf += digest
        return self

    def blob(self, data: bytes) -> "Writer":
        """u32 length then the bytes."""
        self.u32(len(data))
        self._buf += data
        return self

    def string(self, s: str) -> "Writer":
        """u32 length then UTF-8. The length counts BYTES, not characters."""
        return self.blob(s.encode("utf-8"))

    def take(self) -> bytes:
        return bytes(self._buf)

    def __len__(self) -> int:
        return len(self._buf)


class Reader:
    """Bounds-checked consumption. Every short read raises rather than returning
    a truncated or default value."""

    __slots__ = ("_data", "_pos")

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def _take(self, n: int) -> bytes:
        if n < 0:
            raise CanonError(f"canon: negative read of {n}")
        end = self._pos + n
        if end > len(self._data):
            raise CanonError(f"canon: wanted {n} bytes, {self.remaining} remain")
        chunk = self._data[self._pos : end]
        self._pos = end
        return chunk

    def _fixed(self, width: int, *, signed: bool = False) -> int:
        return int.from_bytes(self._take(width), "big", signed=signed)

    def u8(self) -> int:
        return self._fixed(1)

    def u16(self) -> int:
        return self._fixed(2)

    def u32(self) -> int:
        return self._fixed(4)

    def u64(self) -> int:
        return self._fixed(8)

    def i64(self) -> int:
        return self._fixed(8, signed=True)

    def boolean(self) -> bool:
        v = self._fixed(1)
        if v > 1:
            raise CanonError(f"canon: {v} is not a canonical boolean")
        return v == 1

    def hash(self) -> bytes:
        return self._take(32)

    def blob(self, *, limit: int) -> bytes:
        """A length-prefixed blob, refused above `limit`.

        The limit is mandatory and has no default. A length read off the wire is
        an attacker-chosen allocation size, and the one place that must not be
        convenient is the place that decides how much memory to reserve.
        """
        n = self.u32()
        if n > limit:
            raise CanonError(f"canon: blob of {n} bytes exceeds the {limit}-byte limit")
        return self._take(n)

    def string(self, *, limit: int) -> str:
        raw = self.blob(limit=limit)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CanonError("canon: string is not valid UTF-8") from exc

    @property
    def remaining(self) -> int:
        return len(self._data) - self._pos

    def expect_exhausted(self) -> None:
        if self.remaining:
            raise CanonError(f"canon: {self.remaining} trailing byte(s)")

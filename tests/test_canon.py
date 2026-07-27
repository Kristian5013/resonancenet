"""Tests for the byte layer: the stream primitives and the container envelope.

Named to match the comments that assert them, so a claim in the source can be
found here and checked. Standard library only — verifying what this network
claims must not require the hardware to participate in it.
"""

import unittest

from rnet.canon.container import (MAGIC, MAX_CONTENT, ObjectType, crc32c, parse,
                                  sha3_256, wrap)
from rnet.canon.stream import CanonError, Reader, Writer


class CanonTests(unittest.TestCase):
    # -- stream ------------------------------------------------------------

    def test_OversizedFieldIsRefused(self):
        """The C++ this replaces silently truncated; here it raises."""
        Writer().u32(0xFFFF_FFFF)
        with self.assertRaises(CanonError):
            Writer().u32(0x1_0000_0000)
        with self.assertRaises(CanonError):
            Writer().u8(256)
        with self.assertRaises(CanonError):
            Writer().u16(-1)

    def test_BooleanRejectsNonCanonicalByte(self):
        """Two encodings of one value would be two hashes of one object."""
        self.assertEqual(Writer().boolean(True).take(), b"\x01")
        self.assertEqual(Writer().boolean(False).take(), b"\x00")
        self.assertTrue(Reader(b"\x01").boolean())
        self.assertFalse(Reader(b"\x00").boolean())
        with self.assertRaises(CanonError):
            Reader(b"\x02").boolean()
        # bool is an int subclass; an integer field must not silently take one.
        with self.assertRaises(CanonError):
            Writer().u32(True)

    def test_EveryFixedWidthRoundTrips(self):
        w = (Writer().u8(0xFF).u16(0xFFFF).u32(0xFFFF_FFFF)
             .u64(0xFFFF_FFFF_FFFF_FFFF).i64(-(1 << 63)).i64(-1).boolean(True))
        r = Reader(w.take())
        self.assertEqual(r.u8(), 0xFF)
        self.assertEqual(r.u16(), 0xFFFF)
        self.assertEqual(r.u32(), 0xFFFF_FFFF)
        self.assertEqual(r.u64(), 0xFFFF_FFFF_FFFF_FFFF)
        self.assertEqual(r.i64(), -(1 << 63))
        self.assertEqual(r.i64(), -1)
        self.assertTrue(r.boolean())
        r.expect_exhausted()

    def test_ABigEndianU32IsFourNamedBytes(self):
        """Pinned against the literal so an endianness change cannot pass."""
        self.assertEqual(Writer().u32(0x0102_0304).take(), b"\x01\x02\x03\x04")
        self.assertEqual(Writer().u16(0x0102).take(), b"\x01\x02")

    def test_AShortReadRaisesRatherThanPadding(self):
        with self.assertRaises(CanonError):
            Reader(b"\x01\x02").u32()
        with self.assertRaises(CanonError):
            Reader(b"").u8()

    def test_TrailingBytesAreAFailedRead(self):
        r = Reader(b"\x01\x02\x03\x04\x05")
        r.u32()
        with self.assertRaises(CanonError):
            r.expect_exhausted()

    def test_ABlobNeedsAnExplicitLimit(self):
        w = Writer().blob(b"hello")
        self.assertEqual(Reader(w.take()).blob(limit=16), b"hello")
        with self.assertRaises(CanonError):
            Reader(w.take()).blob(limit=4)
        # A length field is an attacker-chosen allocation; the limit is checked
        # before the read, so a huge declared length costs nothing.
        with self.assertRaises(CanonError):
            Reader(b"\xff\xff\xff\xff").blob(limit=1024)

    def test_StringLengthCountsBytesNotCharacters(self):
        w = Writer().string("привет")
        self.assertEqual(len(w.take()), 4 + 12)
        self.assertEqual(Reader(w.take()).string(limit=64), "привет")

    def test_AStringThatIsNotUtf8IsRefused(self):
        raw = Writer().u32(2).raw(b"\xff\xfe").take()
        with self.assertRaises(CanonError):
            Reader(raw).string(limit=64)

    def test_AHashIsExactlyThirtyTwoBytes(self):
        with self.assertRaises(CanonError):
            Writer().hash(b"\x00" * 31)

    # -- container ---------------------------------------------------------

    def test_Crc32cMatchesTheReferenceVector(self):
        """The check value every CRC-32C implementation publishes."""
        self.assertEqual(crc32c(b"123456789"), 0xE306_9283)
        self.assertEqual(crc32c(b""), 0x0000_0000)

    def test_AContainerRoundTrips(self):
        content = b"round descriptor bytes"
        c = parse(wrap(ObjectType.ROUND_DESCRIPTOR, content))
        self.assertEqual(c.obj_type, ObjectType.ROUND_DESCRIPTOR)
        self.assertEqual(c.content, content)
        self.assertEqual(c.id, sha3_256(wrap(ObjectType.ROUND_DESCRIPTOR, content)))

    def test_EveryMutationOfAContainerIsCaught(self):
        """One flipped bit anywhere must be refused, wherever it lands."""
        raw = bytearray(wrap(ObjectType.POLICY_DESCRIPTOR, b"policy bytes"))
        for i in range(len(raw)):
            mutated = bytearray(raw)
            mutated[i] ^= 0x01
            with self.assertRaises(CanonError, msg=f"byte {i} mutated and accepted"):
                parse(bytes(mutated))

    def test_BadMagicIsRefusedBeforeAnythingElse(self):
        raw = wrap(ObjectType.BATCH_SEED, b"x")
        with self.assertRaises(CanonError):
            parse(b"XXXX" + raw[4:])

    def test_AnUnknownObjectTypeIsRefused(self):
        """A type this build does not know is a newer protocol, not a guess."""
        raw = bytearray(wrap(ObjectType.BATCH_SEED, b"x"))
        raw[8:10] = (0xFFFE).to_bytes(2, "big")
        # Repair the CRC so the type check is what refuses it, not the checksum.
        body = bytes(raw[:-4])
        with self.assertRaises(CanonError) as ctx:
            parse(body + crc32c(body).to_bytes(4, "big"))
        self.assertIn("object type", str(ctx.exception))

    def test_ALyingLengthFieldIsRefused(self):
        raw = bytearray(wrap(ObjectType.BATCH_SEED, b"12345678"))
        raw[10:14] = (7).to_bytes(4, "big")
        body = bytes(raw[:-4])
        with self.assertRaises(CanonError) as ctx:
            parse(body + crc32c(body).to_bytes(4, "big"))
        self.assertIn("length", str(ctx.exception))

    def test_TrailingBytesAfterAContainerAreRefused(self):
        with self.assertRaises(CanonError):
            parse(wrap(ObjectType.BATCH_SEED, b"x") + b"\x00")

    def test_AContainerTooLargeIsRefusedAtTheWriter(self):
        with self.assertRaises(CanonError):
            wrap(ObjectType.ROUND_DESCRIPTOR, b"\x00" * (MAX_CONTENT + 1))

    def test_TheMagicIsWhereItIsDocumented(self):
        self.assertTrue(wrap(ObjectType.BATCH_SEED, b"x").startswith(MAGIC))


if __name__ == "__main__":
    unittest.main(verbosity=2)

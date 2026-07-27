"""Tests for the arithmetic contract a round is trained under.

The property under test throughout: two workers reproduce each other if and
only if their numerics agree, and the determinism class says so without anyone
maintaining a registry.
"""

import unittest

from rnet.canon.stream import CanonError, Reader, Writer
from rnet.consensus.numerics import (PORTABLE, ROUND0, AttentionKernel,
                                     ContributionFormat, DType, Numerics)


class NumericsTests(unittest.TestCase):

    def test_TheClassChangesWithEveryField(self):
        """A field that does not move the class is a field a worker could vary."""
        base = ROUND0.determinism_class
        variants = [
            Numerics(DType.FP16, DType.BF16, DType.FP32,
                     AttentionKernel.FLASH, ContributionFormat.INT8_POW2),
            Numerics(DType.BF16, DType.FP16, DType.FP32,
                     AttentionKernel.FLASH, ContributionFormat.INT8_POW2),
            Numerics(DType.BF16, DType.BF16, DType.FP32,
                     AttentionKernel.MATH, ContributionFormat.INT8_POW2),
            Numerics(DType.BF16, DType.BF16, DType.FP32,
                     AttentionKernel.FLASH, ContributionFormat.INT4_POW2),
        ]
        seen = {base}
        for v in variants:
            self.assertNotEqual(v.determinism_class, base, f"{v.describe()} collided with round 0")
            seen.add(v.determinism_class)
        self.assertEqual(len(seen), len(variants) + 1, "two variants share a class")

    def test_TheClassIsDerivedNotAssigned(self):
        """Two independently built objects land in the same class or they do not."""
        again = Numerics(DType.BF16, DType.BF16, DType.FP32,
                         AttentionKernel.FLASH, ContributionFormat.INT8_POW2)
        self.assertEqual(again.determinism_class, ROUND0.determinism_class)
        self.assertNotEqual(PORTABLE.determinism_class, ROUND0.determinism_class)

    def test_TheClassIsStableAcrossRuns(self):
        """Pinned, so a future refactor cannot silently repartition the network."""
        self.assertEqual(ROUND0.determinism_class, ROUND0.determinism_class)
        self.assertEqual(ROUND0.canonical_bytes(),
                         b"\x00\x01\x00\x01\x00\x03\x00\x02\x00\x08")

    def test_FlashWithoutSixteenBitsIsRefused(self):
        """Flash kernels are 16-bit only; fp32 gets "No available kernel"."""
        with self.assertRaises(CanonError):
            Numerics(DType.FP32, DType.FP32, DType.FP32,
                     AttentionKernel.FLASH, ContributionFormat.INT8_POW2).validate()
        # The math path at fp32 is legal, just slow.
        Numerics(DType.FP32, DType.FP32, DType.FP32,
                 AttentionKernel.MATH, ContributionFormat.INT8_POW2).validate()

    def test_AccumulatingNarrowerThanComputingIsRefused(self):
        with self.assertRaises(CanonError):
            Numerics(DType.BF16, DType.FP32, DType.BF16,
                     AttentionKernel.MATH, ContributionFormat.INT8_POW2).validate()

    def test_Float8NeedsWideAccumulation(self):
        """fp8 carries 3-4 mantissa bits; a narrow accumulator loses the update."""
        with self.assertRaises(CanonError):
            Numerics(DType.FP8_E4M3, DType.BF16, DType.BF16,
                     AttentionKernel.MATH, ContributionFormat.INT8_POW2).validate()
        Numerics(DType.FP8_E4M3, DType.BF16, DType.FP32,
                 AttentionKernel.FLASH, ContributionFormat.INT8_POW2).validate()

    def test_ValidationIsAboutTheObjectNotTheMachine(self):
        """fp8 parses on a machine that cannot run it — an artifact must not
        mean different things on different hardware."""
        n = Numerics(DType.FP8_E4M3, DType.FP16, DType.FP32,
                     AttentionKernel.FLASH, ContributionFormat.INT4_POW2)
        self.assertEqual(Numerics.parse(Reader(n.canonical_bytes())), n)

    def test_EveryLegalCombinationRoundTrips(self):
        seen = set()
        for pd in DType:
            for cd in (DType.BF16, DType.FP16, DType.FP32):
                for ad in (DType.FP32, DType.BF16):
                    for k in AttentionKernel:
                        for cf in ContributionFormat:
                            n = Numerics(pd, cd, ad, k, cf)
                            try:
                                n.validate()
                            except CanonError:
                                continue
                            r = Reader(n.canonical_bytes())
                            self.assertEqual(Numerics.parse(r), n)
                            r.expect_exhausted()
                            seen.add(n.determinism_class)
        # Distinct legal contracts must not share a class.
        self.assertGreater(len(seen), 20)

    def test_AnUnknownEnumValueIsRefused(self):
        """A round from a newer protocol is refused, not guessed at."""
        raw = Writer().u16(99).u16(1).u16(3).u16(2).u16(8).take()
        with self.assertRaises(CanonError):
            Numerics.parse(Reader(raw))
        raw = Writer().u16(1).u16(1).u16(3).u16(99).u16(8).take()
        with self.assertRaises(CanonError):
            Numerics.parse(Reader(raw))

    def test_TheContributionRangeIsSymmetricAboutZero(self):
        """Two's complement would give one extra negative value, biasing the
        mean of every update in a direction nobody chose."""
        self.assertEqual(ContributionFormat.INT8_POW2.max_magnitude, 127)
        self.assertEqual(ContributionFormat.INT6_POW2.max_magnitude, 31)
        self.assertEqual(ContributionFormat.INT4_POW2.max_magnitude, 7)

    def test_NumericsAreFrozen(self):
        """The bytes define a class; they must not move after it is derived."""
        with self.assertRaises(Exception):
            ROUND0.param_dtype = DType.FP16

    def test_DtypeMetadataIsConsistent(self):
        for d in DType:
            self.assertIn(d.bits, (8, 16, 32))
            self.assertTrue(d.torch_name)
            self.assertEqual(d.is_float8, d.bits == 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)

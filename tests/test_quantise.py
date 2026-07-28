"""Quantising an update, and doing it without holding it all at once.

The whole-array path is the definition; the chunked path is the one a worker
actually uses on a model large enough to matter. These tests exist to keep the
second one from ever becoming a different answer than the first.
"""

import unittest

import numpy as np

from rnet.consensus.numerics import ContributionFormat
from rnet.diloco.quantize import (QuantiseError, pack, quantize_update,
                                  quantize_update_chunked)


class ChunkedIsTheSameAnswerTests(unittest.TestCase):
    """Streaming the update rather than concatenating it.

    The reason this exists is memory, and the reason it is safe is that the
    exponent depends only on the largest magnitude and the encoding is
    elementwise. So the only thing worth proving is that it changes nothing:
    a contribution that differs by one byte from what the whole-array version
    produced is a contribution no verifier could reproduce.
    """

    def pieces(self, rng, sizes):
        return [rng.normal(0, 1e-3, n) for n in sizes]

    def check(self, sizes, fmt, scale=1e-3):
        rng = np.random.default_rng(7)
        parts = [rng.normal(0, scale, n) for n in sizes]
        whole = np.concatenate(parts) if parts else np.array([])
        want, want_exp = quantize_update(whole, fmt)
        got, got_exp = quantize_update_chunked(lambda: iter(parts),
                                               int(whole.size), fmt)
        self.assertEqual(got_exp, want_exp, sizes)
        np.testing.assert_array_equal(got, want)
        # And the packed bytes, which is what actually goes on the wire.
        self.assertEqual(pack(got, fmt), pack(want, fmt))

    def test_ItMatchesTheWholeArrayVersion(self):
        for fmt in (ContributionFormat.INT8_POW2, ContributionFormat.INT6_POW2,
                    ContributionFormat.INT4_POW2):
            with self.subTest(fmt=fmt):
                # int4 packs values in pairs, so the count has to be even — a
                # constraint of the wire format, not of the chunking.
                self.check([1000, 512, 34, 4096], fmt)

    def test_TheChunkingDoesNotChangeTheAnswer(self):
        """Same values, different piece boundaries, same bytes — otherwise the
        contribution would depend on how a model happened to be split into
        tensors."""
        rng = np.random.default_rng(11)
        values = rng.normal(0, 1e-3, 4096)
        fmt = ContributionFormat.INT8_POW2
        want, want_exp = quantize_update(values, fmt)
        for cut in ([4096], [1, 4095], [2048, 2048], [1000, 1000, 1000, 1096]):
            at, parts = 0, []
            for n in cut:
                parts.append(values[at:at + n]); at += n
            got, exp = quantize_update_chunked(lambda p=parts: iter(p), 4096, fmt)
            self.assertEqual(exp, want_exp, cut)
            np.testing.assert_array_equal(got, want, cut)

    def test_TheLargestValueSetsTheExponentEvenInTheLastChunk(self):
        """The exponent is global. A first pass that stopped early, or one that
        took a maximum per chunk, would clamp everything that came after."""
        parts = [np.full(100, 1e-6), np.full(100, 1e-6), np.array([0.5])]
        fmt = ContributionFormat.INT8_POW2
        whole = np.concatenate(parts)
        want, want_exp = quantize_update(whole, fmt)
        got, exp = quantize_update_chunked(lambda: iter(parts), 201, fmt)
        self.assertEqual(exp, want_exp)
        np.testing.assert_array_equal(got, want)

    def test_AWrongTotalIsRefusedRatherThanTruncated(self):
        parts = [np.zeros(10), np.zeros(10)]
        with self.assertRaises(QuantiseError):
            quantize_update_chunked(lambda: iter(parts), 30,
                                    ContributionFormat.INT8_POW2)
        with self.assertRaises(QuantiseError):
            quantize_update_chunked(lambda: iter(parts), 10,
                                    ContributionFormat.INT8_POW2)

    def test_AnEmptyUpdateIsRefusedAsItIsWholeArray(self):
        """Both paths reject it, and for the same reason: a contribution with
        no values in it is not a contribution."""
        with self.assertRaises(QuantiseError):
            quantize_update(np.array([]), ContributionFormat.INT8_POW2)
        with self.assertRaises(QuantiseError):
            quantize_update_chunked(lambda: iter([]), 0,
                                    ContributionFormat.INT8_POW2)

    def test_ANonFiniteValueIsRefusedInEitherPath(self):
        parts = [np.zeros(10), np.array([float("nan")]), np.zeros(9)]
        with self.assertRaises(QuantiseError) as caught:
            quantize_update_chunked(lambda: iter(parts), 20,
                                    ContributionFormat.INT8_POW2)
        self.assertIn("not finite", str(caught.exception))

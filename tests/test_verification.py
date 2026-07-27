"""Who gets checked, and what a check concludes.

The selection tests are about a property that is easy to state and easy to get
subtly wrong: the choice must be unpredictable to the people it judges,
checkable by everyone afterwards, and made by nobody.
"""

import hashlib
import unittest

from rnet.consensus.objects import Verdict as Kind
from rnet.verification.select import (challenge_draw, challenged_in,
                                      should_challenge, verifier_for)


def ids(n: int, salt: bytes = b"") -> list:
    return [hashlib.sha3_256(salt + i.to_bytes(4, "big")).digest() for i in range(n)]


class SelectionTests(unittest.TestCase):

    def test_TheDrawIsDeterministic(self):
        checkpoint, contribution = ids(2)
        self.assertEqual(challenge_draw(checkpoint, contribution),
                         challenge_draw(checkpoint, contribution))

    def test_TheRateIsRoughlyThePolicy(self):
        """Not exactly, because a hash is not a quota — but close enough that a
        node cannot quietly check far fewer than it says."""
        checkpoint = ids(1)[0]
        for percent in (0, 10, 25, 50, 100):
            picked = sum(1 for c in ids(4000, b"rate")
                         if should_challenge(checkpoint, c, percent))
            self.assertAlmostEqual(picked / 4000, percent / 100, delta=0.03,
                                   msg=f"{percent}%")

    def test_ZeroAndOneHundredAreExact(self):
        checkpoint = ids(1)[0]
        contributions = ids(200, b"exact")
        self.assertEqual(challenged_in(checkpoint, contributions, 0), [])
        self.assertEqual(len(challenged_in(checkpoint, contributions, 100)), 200)

    def test_ADifferentCheckpointDrawsDifferently(self):
        """Which is what makes it unpredictable: the checkpoint is a hash over
        every contribution in it, so it does not exist until they all do."""
        a, b = ids(2, b"cp")
        contributions = ids(500, b"same")
        first = {c for c in contributions if should_challenge(a, c, 25)}
        second = {c for c in contributions if should_challenge(b, c, 25)}
        self.assertNotEqual(first, second)
        self.assertTrue(first & second)         # not disjoint either: it is a draw

    def test_ItIsCheckableByAnyone(self):
        """"Nobody challenged it" and "the challenge was quietly dropped" must
        not look the same."""
        checkpoint = ids(1)[0]
        contributions = ids(50, b"check")
        chosen = challenged_in(checkpoint, contributions, 25)
        for c in contributions:
            self.assertEqual(c in chosen, should_challenge(checkpoint, c, 25))

    def test_AnOutOfRangePercentIsRefused(self):
        checkpoint, contribution = ids(2)
        with self.assertRaises(ValueError):
            should_challenge(checkpoint, contribution, 101)
        with self.assertRaises(ValueError):
            should_challenge(checkpoint, contribution, -1)

    def test_MalformedIdsAreRefused(self):
        with self.assertRaises(ValueError):
            challenge_draw(b"short", bytes(32))

    def test_TheVerifierIsDerivedNotVolunteered(self):
        """A verifier that volunteered could volunteer for its accomplice's
        work; one assigned by a node would be assigned by whoever wanted the
        answer."""
        checkpoint, contribution = ids(2, b"v")
        candidates = [b"a", b"b", b"c", b"d"]
        first = verifier_for(checkpoint, contribution, candidates)
        self.assertIn(first, candidates)
        self.assertEqual(first, verifier_for(checkpoint, contribution, candidates))
        # Different work, generally a different verifier.
        others = {verifier_for(checkpoint, c, candidates)
                  for c in ids(40, b"spread")}
        self.assertGreater(len(others), 1)

    def test_NoCandidatesMeansNoVerifier(self):
        checkpoint, contribution = ids(2)
        self.assertIsNone(verifier_for(checkpoint, contribution, []))


class VerdictShapeTests(unittest.TestCase):
    """The invariant the object layer enforces, restated here because it is the
    one that keeps INDETERMINATE from becoming a hiding place."""

    def test_IndeterminateIsAnAnswerNotAFailure(self):
        self.assertEqual(int(Kind.INDETERMINATE), 3)
        self.assertNotEqual(Kind.INDETERMINATE, Kind.MISMATCH)

    def test_TheThreeVerdictsAreDistinct(self):
        self.assertEqual(len({Kind.MATCH, Kind.MISMATCH, Kind.INDETERMINATE}), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)

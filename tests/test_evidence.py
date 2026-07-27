"""Turning verdicts into evidence.

Every test here is about a way the quorum could be cheated if the counting
were done the obvious way.
"""

import unittest

from rnet.consensus.objects import SlashReason, VerificationVerdict, Verdict
from rnet.verification.evidence import Accusation, EvidenceCollector

CHALLENGE = bytes([1]) * 32
CONTRIBUTION = bytes([2]) * 32


def verdict(verifier: int, kind: Verdict = Verdict.MISMATCH,
            determinism_class: int = 0xDF7C5208) -> VerificationVerdict:
    return VerificationVerdict(
        challenge_id=CHALLENGE, verifier_id=verifier, verdict=kind,
        replay_payload_hash=(bytes(32) if kind is Verdict.INDETERMINATE
                             else bytes([verifier % 251]) * 32),
        determinism_class=determinism_class)


def collector(quorum: int = 3, shadow: bool = True) -> EvidenceCollector:
    c = EvidenceCollector(quorum=quorum, shadow_mode=shadow)
    c.expect(CHALLENGE, Accusation(contribution_id=CONTRIBUTION,
                                   target_worker_id=7, round_id=0))
    return c


class EvidenceTests(unittest.TestCase):

    def test_AQuorumProducesEvidence(self):
        c = collector(quorum=3)
        self.assertIsNone(c.add(verdict(1), bytes([11]) * 32))
        self.assertIsNone(c.add(verdict(2), bytes([12]) * 32))
        proof = c.add(verdict(3), bytes([13]) * 32)
        self.assertIsNotNone(proof)
        self.assertEqual(proof.target_worker_id, 7)
        self.assertEqual(proof.contribution_id, CONTRIBUTION)
        self.assertEqual(proof.reason, SlashReason.WRONG_UPDATE)
        self.assertEqual(len(proof.verdict_ids), 3)

    def test_OneVerifierCannotBeAQuorum(self):
        """Counting verdict objects rather than verifiers makes a quorum of
        three a quorum of one with three signatures it made itself."""
        c = collector(quorum=3)
        for i in range(10):
            self.assertIsNone(c.add(verdict(1), bytes([20 + i]) * 32))
        self.assertFalse(c.is_proven(7))

    def test_ALaterVerdictReplacesTheSameVerifiersEarlierOne(self):
        c = collector(quorum=2)
        self.assertIsNone(c.add(verdict(1, Verdict.MATCH), bytes([11]) * 32))
        self.assertIsNone(c.add(verdict(1, Verdict.MISMATCH), bytes([12]) * 32))
        self.assertIsNone(c.add(verdict(2, Verdict.MATCH), bytes([13]) * 32))
        # One mismatch, one match: no quorum.
        self.assertFalse(c.is_proven(7))
        self.assertIsNotNone(c.add(verdict(2, Verdict.MISMATCH), bytes([14]) * 32))

    def test_IndeterminateNeverCountsTowardGuilt(self):
        """It is a verifier saying it could not judge; counting it would
        convict a worker for its verifiers' hardware."""
        c = collector(quorum=2)
        self.assertIsNone(c.add(verdict(1, Verdict.INDETERMINATE), bytes([11]) * 32))
        self.assertIsNone(c.add(verdict(2, Verdict.INDETERMINATE), bytes([12]) * 32))
        self.assertIsNone(c.add(verdict(3, Verdict.INDETERMINATE), bytes([13]) * 32))
        self.assertFalse(c.is_proven(7))

    def test_MatchNeverCounts(self):
        c = collector(quorum=2)
        for i in (1, 2, 3):
            self.assertIsNone(c.add(verdict(i, Verdict.MATCH), bytes([10 + i]) * 32))
        self.assertFalse(c.is_proven(7))

    def test_VerdictsFromDifferentClassesDoNotCorroborate(self):
        """Two machines that were never going to agree, counted as though they
        had, is arithmetic about hardware rather than evidence."""
        c = collector(quorum=2)
        self.assertIsNone(c.add(verdict(1, determinism_class=0xAAAA), bytes([11]) * 32))
        self.assertIsNone(c.add(verdict(2, determinism_class=0xBBBB), bytes([12]) * 32))
        self.assertFalse(c.is_proven(7))
        # A second in the first class does reach quorum.
        self.assertIsNotNone(
            c.add(verdict(3, determinism_class=0xAAAA), bytes([13]) * 32))

    def test_EvidenceIsEmittedExactlyOnce(self):
        """Relaying the same accusation repeatedly makes one worker's guilt
        look like many workers' guilt."""
        c = collector(quorum=2)
        c.add(verdict(1), bytes([11]) * 32)
        self.assertIsNotNone(c.add(verdict(2), bytes([12]) * 32))
        self.assertIsNone(c.add(verdict(3), bytes([13]) * 32))
        self.assertIsNone(c.add(verdict(4), bytes([14]) * 32))

    def test_OneVerdictDoesNotSettleAChallenge(self):
        """Closing on the first answer makes the quorum unreachable: the second
        verifier is told there is no such challenge, so slash_quorum is
        decorative and every accusation rests on one opinion. Measured — the
        second verdict came back UNKNOWN_ASSIGNMENT."""
        c = collector(quorum=2)
        c.add(verdict(1, Verdict.MISMATCH), bytes([11]) * 32)
        self.assertFalse(c.is_settled(CHALLENGE))
        c.add(verdict(2, Verdict.MISMATCH), bytes([12]) * 32)
        self.assertTrue(c.is_settled(CHALLENGE))

    def test_AQuorumOfMatchesSettlesItToo(self):
        """A challenge answered and found honest is not the same as one nobody
        answered, and both must stop being handed out."""
        c = collector(quorum=2)
        c.add(verdict(1, Verdict.MATCH), bytes([11]) * 32)
        self.assertFalse(c.is_settled(CHALLENGE))
        c.add(verdict(2, Verdict.MATCH), bytes([12]) * 32)
        self.assertTrue(c.is_settled(CHALLENGE))
        self.assertTrue(c.cleared(CHALLENGE))
        self.assertFalse(c.is_proven(7))

    def test_IndeterminateNeverSettles(self):
        """Otherwise a verifier retires a challenge by being unable to answer
        it, which is the cheapest way to protect a friend."""
        c = collector(quorum=2)
        for i in (1, 2, 3, 4):
            c.add(verdict(i, Verdict.INDETERMINATE), bytes([10 + i]) * 32)
        self.assertFalse(c.is_settled(CHALLENGE))

    def test_ASplitDecisionStaysOpen(self):
        c = collector(quorum=2)
        c.add(verdict(1, Verdict.MATCH), bytes([11]) * 32)
        c.add(verdict(2, Verdict.MISMATCH), bytes([12]) * 32)
        self.assertFalse(c.is_settled(CHALLENGE))

    def test_VerdictsAboutAnUnknownChallengeAreIgnored(self):
        c = EvidenceCollector(quorum=2)
        self.assertIsNone(c.add(verdict(1), bytes([11]) * 32))
        self.assertIsNone(c.add(verdict(2), bytes([12]) * 32))

    def test_AQuorumOfOneIsRefused(self):
        for q in (0, 1):
            with self.assertRaises(ValueError):
                EvidenceCollector(quorum=q)

    def test_TheEvidenceObjectSurvivesTheWire(self):
        c = collector(quorum=2)
        c.add(verdict(1), bytes([11]) * 32)
        proof = c.add(verdict(2), bytes([12]) * 32)
        from rnet.consensus.objects import SlashEvidence
        self.assertEqual(SlashEvidence.from_container(proof.to_container()), proof)

    # -- what it does with the answer --------------------------------------

    def test_ShadowModeRecordsAndDoesNothing(self):
        """The right default for a rule whose first act, historically, is to be
        wrong about somebody."""
        c = collector(quorum=2, shadow=True)
        c.add(verdict(1), bytes([11]) * 32)
        c.add(verdict(2), bytes([12]) * 32)
        self.assertTrue(c.is_proven(7))
        self.assertFalse(c.should_refuse(7))
        self.assertIn("shadow mode", c.status())

    def test_EnforcingRefusesFurtherWork(self):
        c = collector(quorum=2, shadow=False)
        c.add(verdict(1), bytes([11]) * 32)
        c.add(verdict(2), bytes([12]) * 32)
        self.assertTrue(c.should_refuse(7))
        self.assertFalse(c.should_refuse(8))
        self.assertIn("ENFORCING", c.status())

    def test_TurningItOnIsAConsensusChange(self):
        """shadow_mode lives in the hashed policy, so flipping it is a fork
        rather than a flag — which is why the enforcement path is built and
        exercised while the shipped networks stay in shadow."""
        from rnet.consensus import genesis
        for network in genesis.networks():
            self.assertTrue(genesis.policy_descriptor(network).shadow_mode, network)


if __name__ == "__main__":
    unittest.main(verbosity=2)

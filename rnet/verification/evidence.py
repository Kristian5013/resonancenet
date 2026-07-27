"""Turning verdicts into evidence, and evidence into a refusal.

One verdict is never enough. A verifier can be wrong, malicious, or simply on
hardware that rounds differently, and a protocol that acted on one report is a
protocol where accusing costs less than working — which makes accusation the
cheapest attack available. So a quorum of DISTINCT verifiers must agree before
anything is written down.

THREE RULES, AND EACH CLOSES A SPECIFIC WAY TO CHEAT THE QUORUM:

  * ONLY MISMATCH COUNTS. An INDETERMINATE verdict is a verifier saying it could
    not judge, and counting it toward guilt would convict a worker for its
    verifiers' hardware.
  * DISTINCT VERIFIERS, NOT DISTINCT VERDICTS. One verifier can produce many
    well-formed verdict objects for one challenge; counting objects rather than
    verifiers means a quorum of three is a quorum of one with three signatures
    it made itself.
  * SAME DETERMINISM CLASS. Verdicts from different classes are not corroborating
    each other — they are two machines that were never going to agree being
    counted as though they had.

WHAT SLASHING MEANS HERE, PLAINLY. There is no stake, so there is nothing to
take. What evidence buys today is that a node stops spending its aggregate on a
worker proven to have submitted work it did not do. That is real and it is not
economic punishment, and calling it slashing without saying so would be the kind
of overclaim that survives until somebody depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..consensus.objects import (SlashEvidence, SlashReason,
                                 VerificationVerdict, Verdict)


@dataclass(frozen=True)
class Accusation:
    """What a set of verdicts is about, supplied by whoever holds the round."""

    contribution_id: bytes
    target_worker_id: int
    round_id: int


@dataclass
class EvidenceCollector:
    """Gathers verdicts and emits evidence exactly once per contribution."""

    quorum: int
    shadow_mode: bool = True

    # challenge_id -> {verifier_id: (verdict, verdict_object_id, class)}
    _seen: dict = field(default_factory=dict)
    _accusations: dict = field(default_factory=dict)
    _emitted: set = field(default_factory=set)
    proven: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.quorum < 2:
            # A quorum of one makes accusing cheaper than working, which the
            # policy object also refuses; checked here too because this class is
            # usable without one.
            raise ValueError(f"evidence: a quorum of {self.quorum} is not a quorum")

    def expect(self, challenge_id: bytes, accusation: Accusation) -> None:
        """Say what a challenge is about, before its verdicts arrive."""
        self._accusations[challenge_id] = accusation

    def add(self, verdict: VerificationVerdict,
            verdict_object_id: bytes) -> SlashEvidence | None:
        """Record a verdict. Returns evidence the moment a quorum is reached.

        Exactly once: a contribution already proven does not produce a second
        piece of evidence when a fourth verifier agrees, because relaying the
        same accusation repeatedly is a way to make one worker's guilt look like
        many workers' guilt.
        """
        accusation = self._accusations.get(verdict.challenge_id)
        if accusation is None:
            return None
        if verdict.challenge_id in self._emitted:
            return None

        seen = self._seen.setdefault(verdict.challenge_id, {})
        # One verifier, one opinion. A later verdict from the same verifier
        # replaces its earlier one rather than adding to the count.
        seen[verdict.verifier_id] = (verdict.verdict, verdict_object_id,
                                     verdict.determinism_class)

        agreeing = self._agreeing_mismatches(seen)
        if len(agreeing) < self.quorum:
            return None

        self._emitted.add(verdict.challenge_id)
        self.proven[accusation.target_worker_id] = accusation.contribution_id
        return SlashEvidence(
            round_id=accusation.round_id,
            target_worker_id=accusation.target_worker_id,
            contribution_id=accusation.contribution_id,
            reason=SlashReason.WRONG_UPDATE,
            verdict_ids=tuple(sorted(agreeing)))

    @staticmethod
    def _agreeing_mismatches(seen: dict) -> list:
        """Verdict ids of the largest set of mismatches sharing one class.

        Grouped by class because verdicts from different determinism classes do
        not corroborate one another: two machines that were never going to agree
        being counted as though they had is not evidence, it is arithmetic about
        hardware.
        """
        by_class: dict = {}
        for _verifier, (kind, object_id, cls) in seen.items():
            if kind is not Verdict.MISMATCH:
                continue
            by_class.setdefault(cls, []).append(object_id)
        return max(by_class.values(), default=[], key=len)

    def is_settled(self, challenge_id: bytes) -> bool:
        """Whether enough verifiers have agreed, either way, to stop asking.

        A challenge must NOT close on the first answer. Closing on one verdict
        makes the quorum unreachable by construction — the second verifier is
        told there is no such challenge — which leaves `slash_quorum`
        decorative and every accusation resting on one opinion. It closes when
        a quorum agrees on guilt, or a quorum agrees on innocence, and
        otherwise stays open until it expires. Proved by
        EvidenceTests.OneVerdictDoesNotSettleAChallenge.
        """
        seen = self._seen.get(challenge_id, {})
        if len(self._agreeing_mismatches(seen)) >= self.quorum:
            return True
        return self._agreeing(seen, Verdict.MATCH) >= self.quorum

    @staticmethod
    def _agreeing(seen: dict, kind: Verdict) -> int:
        by_class: dict = {}
        for _verifier, (verdict, _object_id, cls) in seen.items():
            if verdict is kind:
                by_class[cls] = by_class.get(cls, 0) + 1
        return max(by_class.values(), default=0)

    def cleared(self, challenge_id: bytes) -> bool:
        """A quorum agreed the work was done. Worth recording as much as guilt:
        a challenge that was answered and found honest is not the same as one
        nobody answered."""
        return (self._agreeing(self._seen.get(challenge_id, {}), Verdict.MATCH)
                >= self.quorum)

    def is_proven(self, worker_id: int) -> bool:
        return worker_id in self.proven

    def should_refuse(self, worker_id: int) -> bool:
        """Whether to stop giving this worker work.

        Shadow mode records and does nothing, which is the right default for a
        rule whose first act, historically, is to be wrong about somebody.
        """
        return not self.shadow_mode and self.is_proven(worker_id)

    def status(self) -> str:
        open_challenges = len(self._seen) - len(self._emitted)
        return (f"evidence: {len(self._emitted)} proven, {open_challenges} open"
                + (" (shadow mode)" if self.shadow_mode else " — ENFORCING"))

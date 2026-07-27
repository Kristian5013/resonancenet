"""Tests for the ten consensus objects.

Two kinds of test here. The generic ones walk every registered type and assert
properties that must hold for all of them — round-tripping, tamper detection,
enum completeness. The specific ones pin the invariants that exist to stop a
particular attack, and each names it.
"""

import unittest

from rnet.canon.container import ObjectType, wrap
from rnet.canon.stream import CanonError, Reader, Writer
from rnet.consensus.model_spec import ModelSpec
from rnet.consensus.numerics import ROUND0, ContributionFormat
from rnet.consensus.objects import (CODECS, MAX_VERDICT_REFS, OPT_ADAFACTOR,
                                    BatchSeed, ChallengeOrder, CheckpointHeader,
                                    ContributionHeader, DatasetManifest,
                                    ExpertShard, PolicyDescriptor,
                                    RoundDescriptor, SlashEvidence, SlashReason,
                                    VerificationVerdict, Verdict, decode)
from tests.test_model_spec import DENSE_400M, MOE_30B

H = [bytes([i]) * 32 for i in range(1, 12)]
PROTO = 0x0002_0000


def round_desc(**kw) -> RoundDescriptor:
    base = dict(protocol_version=PROTO, network_magic=0x524E_4D31, round_id=0,
                model=DENSE_400M, numerics=ROUND0, optimizer_id=OPT_ADAFACTOR,
                tokenizer_hash=H[0], dataset_root=H[1], dataset_chunks=6_956_933)
    return RoundDescriptor(**{**base, **kw})


def policy(**kw) -> PolicyDescriptor:
    base = dict(protocol_version=PROTO, network_magic=0x524E_4D31, policy_version=1,
                inner_steps=200, micro_batch=1, min_contributors=2,
                checkpoint_interval=1, round_deadline_ms=1_200_000,
                outer_momentum_q16=58_982, outer_lr_q16=45_875, nesterov=True,
                challenge_percent=25, challenge_deadline_steps=4,
                retained_checkpoints=16, slash_quorum=3, shadow_mode=True)
    return PolicyDescriptor(**{**base, **kw})


SAMPLES: dict[ObjectType, object] = {
    ObjectType.ROUND_DESCRIPTOR: round_desc(),
    ObjectType.DATASET_MANIFEST: DatasetManifest(H[0], H[1], 7_359_506_899_436,
                                                 1_048_576, 6_956_933),
    ObjectType.BATCH_SEED: BatchSeed(H[0], 0, 7, 42, 3),
    ObjectType.CONTRIBUTION_HEADER: ContributionHeader(
        0, 5, 7, 99, H[0], H[1], H[2], 397_728_768, -24, ContributionFormat.INT8_POW2),
    ObjectType.CHECKPOINT_HEADER: CheckpointHeader(0, 5, H[0], H[1], H[2], H[3],
                                                   7, 1_785_000_000_000),
    ObjectType.CHALLENGE_ORDER: ChallengeOrder(0, 5, 1, 2, H[0], 9, H[1]),
    ObjectType.VERIFICATION_VERDICT: VerificationVerdict(H[0], 3, Verdict.MATCH,
                                                         H[1], 0xDEAD_BEEF),
    ObjectType.SLASH_EVIDENCE: SlashEvidence(0, 7, H[0], SlashReason.WRONG_UPDATE,
                                             (H[1], H[2], H[3])),
    ObjectType.POLICY_DESCRIPTOR: policy(),
    ObjectType.EXPERT_SHARD: ExpertShard(0, 5, 2, 4, 32, 16, H[0]),
}


class ObjectTests(unittest.TestCase):

    # -- properties every object must have ---------------------------------

    def test_EveryObjectTypeHasACodec(self):
        """A type in the enum and nowhere else is not a type.

        The previous implementation shipped exactly that and found out on a
        live node, as "unknown message type 0x0060".
        """
        missing = [t.name for t in ObjectType if t not in CODECS]
        self.assertEqual(missing, [], f"ObjectType members with no codec: {missing}")
        for t, codec in CODECS.items():
            self.assertIs(codec.OBJ_TYPE, t, f"{codec.__name__} is registered under {t.name}")

    def test_EveryObjectTypeHasASample(self):
        """Otherwise the generic tests below silently cover less than they say."""
        self.assertEqual(sorted(t.name for t in ObjectType),
                         sorted(t.name for t in SAMPLES))

    def test_EveryObjectRoundTrips(self):
        for t, obj in SAMPLES.items():
            with self.subTest(t.name):
                self.assertEqual(type(obj).from_container(obj.to_container()), obj)

    def test_EveryObjectDecodesWithoutBeingToldItsType(self):
        for t, obj in SAMPLES.items():
            with self.subTest(t.name):
                self.assertEqual(decode(obj.to_container()), obj)

    def test_EveryObjectHasAStableIdentity(self):
        ids = {}
        for t, obj in SAMPLES.items():
            with self.subTest(t.name):
                self.assertEqual(obj.id, type(obj).from_container(obj.to_container()).id)
                self.assertEqual(len(obj.id), 32)
                ids[obj.id] = t.name
        self.assertEqual(len(ids), len(SAMPLES), "two objects share an identity")

    def test_EveryObjectRefusesTrailingBytes(self):
        """A parse that leaves bytes means writer and reader disagree."""
        for t, obj in SAMPLES.items():
            with self.subTest(t.name):
                with self.assertRaises(CanonError):
                    type(obj).from_container(wrap(t, obj.content() + b"\x00"))

    def test_EveryObjectRefusesTruncation(self):
        for t, obj in SAMPLES.items():
            with self.subTest(t.name):
                with self.assertRaises(CanonError):
                    type(obj).from_container(wrap(t, obj.content()[:-1]))

    def test_AnObjectWillNotParseAsAnotherType(self):
        for t, obj in SAMPLES.items():
            other = RoundDescriptor if t is not ObjectType.ROUND_DESCRIPTOR else DatasetManifest
            with self.subTest(t.name):
                with self.assertRaises(CanonError):
                    other.from_container(obj.to_container())

    # -- round descriptor ---------------------------------------------------

    def test_TheDeterminismClassComesFromTheArithmetic(self):
        """Derived, so it cannot disagree with what it describes."""
        r = round_desc()
        self.assertEqual(r.determinism_class, ROUND0.determinism_class)
        # And there is no field to disagree with: the bytes are shorter than a
        # layout that stored it would be.
        self.assertNotIn(ROUND0.determinism_class.to_bytes(4, "big"), r.content())

    def test_ACorpusIsPinnedWholeOrNotAtAll(self):
        with self.assertRaises(CanonError):
            round_desc(dataset_root=bytes(32)).from_container(
                round_desc(dataset_root=bytes(32)).to_container())
        with self.assertRaises(CanonError):
            RoundDescriptor.from_container(round_desc(dataset_chunks=0).to_container())
        # Both zero is legal and means no corpus yet.
        RoundDescriptor.from_container(
            round_desc(dataset_root=bytes(32), dataset_chunks=0).to_container())

    def test_AByteLevelTokenizerIsTheZeroHash(self):
        self.assertFalse(round_desc().byte_level_tokenizer)
        self.assertTrue(round_desc(tokenizer_hash=bytes(32)).byte_level_tokenizer)

    def test_AnUnknownOptimizerIsRefused(self):
        with self.assertRaises(CanonError):
            RoundDescriptor.from_container(round_desc(optimizer_id=7).to_container())

    def test_AMixtureRoundTripsThroughTheDescriptor(self):
        r = round_desc(model=MOE_30B)
        back = RoundDescriptor.from_container(r.to_container())
        self.assertEqual(back.model, MOE_30B)
        self.assertEqual(back.model.parameter_count(), 29_408_635_904)

    # -- contribution -------------------------------------------------------

    def test_AnEmptyPayloadIsNotAnUpdate(self):
        h = SAMPLES[ObjectType.CONTRIBUTION_HEADER]
        with self.assertRaises(CanonError):
            ContributionHeader.from_container(
                ContributionHeader(**{**h.__dict__, "payload_bytes": 0}).to_container())

    def test_AnUnrepresentableScaleIsRefused(self):
        """Beyond float range the dequantised update is infinity or zero."""
        h = SAMPLES[ObjectType.CONTRIBUTION_HEADER]
        for exp in (-400, 400):
            with self.assertRaises(CanonError):
                ContributionHeader.from_container(
                    ContributionHeader(**{**h.__dict__, "scale_exp": exp}).to_container())

    def test_ANegativeScaleSurvivesTheWire(self):
        """Scales are almost always negative; an unsigned field would wrap."""
        h = SAMPLES[ObjectType.CONTRIBUTION_HEADER]
        self.assertEqual(ContributionHeader.from_container(h.to_container()).scale_exp, -24)

    # -- checkpoint ---------------------------------------------------------

    def test_OnlyGenesisHasNoParent(self):
        c = SAMPLES[ObjectType.CHECKPOINT_HEADER]
        with self.assertRaises(CanonError):
            CheckpointHeader.from_container(
                CheckpointHeader(**{**c.__dict__, "parent": bytes(32)}).to_container())
        with self.assertRaises(CanonError):
            CheckpointHeader.from_container(
                CheckpointHeader(**{**c.__dict__, "outer_step": 0}).to_container())
        CheckpointHeader.from_container(
            CheckpointHeader(**{**c.__dict__, "outer_step": 0,
                                "parent": bytes(32)}).to_container())

    # -- verification -------------------------------------------------------

    def test_AWorkerCannotChallengeItself(self):
        c = SAMPLES[ObjectType.CHALLENGE_ORDER]
        with self.assertRaises(CanonError):
            ChallengeOrder.from_container(
                ChallengeOrder(**{**c.__dict__, "target_worker_id": 1}).to_container())

    def test_AChallengeDeadlineMustBeInTheFuture(self):
        c = SAMPLES[ObjectType.CHALLENGE_ORDER]
        for deadline in (0, 5):
            with self.assertRaises(CanonError):
                ChallengeOrder.from_container(
                    ChallengeOrder(**{**c.__dict__, "deadline_step": deadline}).to_container())

    def test_IndeterminateCarriesNoReplayAndEveryOtherVerdictDoes(self):
        """Otherwise 'could not judge' becomes a place to hide a mismatch."""
        v = SAMPLES[ObjectType.VERIFICATION_VERDICT]
        with self.assertRaises(CanonError):
            VerificationVerdict.from_container(
                VerificationVerdict(**{**v.__dict__,
                                       "verdict": Verdict.INDETERMINATE}).to_container())
        with self.assertRaises(CanonError):
            VerificationVerdict.from_container(
                VerificationVerdict(**{**v.__dict__,
                                       "replay_payload_hash": bytes(32)}).to_container())
        VerificationVerdict.from_container(
            VerificationVerdict(**{**v.__dict__, "verdict": Verdict.INDETERMINATE,
                                   "replay_payload_hash": bytes(32)}).to_container())

    def test_AnUnknownVerdictIsRefused(self):
        raw = Writer().hash(H[0]).u64(3).u16(9).hash(H[1]).u32(0).take()
        with self.assertRaises(CanonError):
            VerificationVerdict.parse(Reader(raw))

    def test_EvidenceNeedsVerdictsAndTheyMustBeDistinct(self):
        """A repeated verdict would let one verifier reach any quorum."""
        e = SAMPLES[ObjectType.SLASH_EVIDENCE]
        with self.assertRaises(CanonError):
            SlashEvidence.from_container(
                SlashEvidence(**{**e.__dict__, "verdict_ids": ()}).to_container())
        with self.assertRaises(CanonError):
            SlashEvidence.from_container(
                SlashEvidence(**{**e.__dict__,
                                 "verdict_ids": (H[1], H[1], H[2])}).to_container())

    def test_EvidenceIsBounded(self):
        e = SAMPLES[ObjectType.SLASH_EVIDENCE]
        too_many = tuple(bytes([i // 256, i % 256]) + bytes(30)
                         for i in range(MAX_VERDICT_REFS + 1))
        with self.assertRaises(CanonError):
            SlashEvidence.from_container(
                SlashEvidence(**{**e.__dict__, "verdict_ids": too_many}).to_container())

    def test_AnUnknownSlashReasonIsRefused(self):
        raw = Writer().u64(0).u64(7).hash(H[0]).u16(99).u32(1).hash(H[1]).take()
        with self.assertRaises(CanonError):
            SlashEvidence.parse(Reader(raw))

    # -- expert shard -------------------------------------------------------

    def test_AShardIndexMustBeInsideItsCount(self):
        s = SAMPLES[ObjectType.EXPERT_SHARD]
        for bad in ({"shard_index": 4}, {"shard_count": 0}, {"n_experts": 0}):
            with self.assertRaises(CanonError):
                ExpertShard.from_container(ExpertShard(**{**s.__dict__, **bad}).to_container())

    def test_EveryShardOfAMixtureIsDescribable(self):
        """Four shards of MOE_30B cover its experts exactly once."""
        per = MOE_30B.experts_per_shard
        covered = []
        for i in range(MOE_30B.expert_shard_count):
            shard = ExpertShard(0, 0, i, MOE_30B.expert_shard_count, i * per, per, H[0])
            ExpertShard.from_container(shard.to_container())
            covered += list(range(shard.first_expert, shard.first_expert + shard.n_experts))
        self.assertEqual(covered, list(range(MOE_30B.n_experts)))

    # -- policy -------------------------------------------------------------

    def test_AChallengeMustExpireWhileItsInputsAreHeld(self):
        """Otherwise 'I no longer have the weights' is indistinguishable from
        refusing to answer."""
        with self.assertRaises(CanonError):
            PolicyDescriptor.from_container(
                policy(challenge_deadline_steps=16, retained_checkpoints=16).to_container())
        with self.assertRaises(CanonError):
            PolicyDescriptor.from_container(
                policy(challenge_deadline_steps=20, retained_checkpoints=16).to_container())

    def test_AQuorumOfOneIsRefused(self):
        """Accusing must not be cheaper than working."""
        for q in (0, 1):
            with self.assertRaises(CanonError):
                PolicyDescriptor.from_container(policy(slash_quorum=q).to_container())

    def test_ZeroKnobsAreRefused(self):
        for name in ("inner_steps", "micro_batch", "min_contributors",
                     "checkpoint_interval", "round_deadline_ms", "retained_checkpoints"):
            with self.subTest(name):
                with self.assertRaises(CanonError):
                    PolicyDescriptor.from_container(policy(**{name: 0}).to_container())

    def test_AChallengeRateBeyondEveryContributionIsRefused(self):
        with self.assertRaises(CanonError):
            PolicyDescriptor.from_container(policy(challenge_percent=101).to_container())
        PolicyDescriptor.from_container(policy(challenge_percent=100).to_container())


if __name__ == "__main__":
    unittest.main(verbosity=2)

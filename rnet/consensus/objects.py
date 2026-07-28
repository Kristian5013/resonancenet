"""Every object the network hashes and agrees on.

Ten types, each a fixed field layout, each able to go both ways. That last part
is new: the implementation this replaces could author objects only in C++ and
read them only in Python, which meant the two halves had to be kept in agreement
by a cross-language test suite. One implementation has no second opinion to
disagree with.

WHAT MAKES A TYPE BELONG HERE. An object is in this module if its bytes are
hashed and that hash is something nodes must agree about. Wire messages that
merely move data are not consensus objects and live in `rnet/net/`. The
distinction matters because everything here is frozen forever once a network
pins it, and everything there can change in a release.

THE REGISTRY IS THE POINT. `CODECS` maps every ObjectType to the class that
encodes it, and ObjectTests.EveryObjectTypeHasACodec walks the enum to prove
none is missing. That test exists because the previous implementation shipped a
message type that had been added to its enum and to nothing else, and the
failure surfaced at runtime, on a live node, as "unknown message type 0x0060".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from ..canon.container import ObjectType, parse as parse_container, wrap
from ..canon.stream import CanonError, Reader, Writer
from .model_spec import ModelSpec
from .numerics import ContributionFormat, Numerics

# Optimizers a round may name. One today; the field exists so that adding one is
# a value in an artifact rather than a format break.
OPT_ADAFACTOR = 1

# A slash needs several verdicts, and the count is read off the wire before the
# reads happen. A quorum is a handful; a thousand is four orders of headroom and
# still refuses a length field that claims four billion.
MAX_VERDICT_REFS = 1000


class Object:
    """Shared behaviour: an object knows its type and how to become a container.

    Subclasses supply OBJ_TYPE, serialize() and parse(). Nothing here is
    optional — a subclass missing one fails the registry test rather than
    failing on a node.
    """

    OBJ_TYPE: ObjectType

    def serialize(self, w: Writer) -> Writer:      # pragma: no cover - interface
        raise NotImplementedError

    @classmethod
    def parse(cls, r: Reader):                     # pragma: no cover - interface
        raise NotImplementedError

    def content(self) -> bytes:
        return self.serialize(Writer()).take()

    def to_container(self) -> bytes:
        return wrap(self.OBJ_TYPE, self.content())

    @property
    def id(self) -> bytes:
        """SHA3-256 of the whole container — this object's identity.

        NOT memoised, deliberately, and it was for one commit. Caching it in
        __dict__ makes `Object(**other.__dict__)` fail with an unexpected
        keyword, which is a semantic change to every object in the protocol paid
        for a speedup in one caller. The caller was fixed instead: see
        Chain._hold_orphan, which compared ids in a linear scan and now compares
        headers.

        It IS expensive — a full re-serialisation and a SHA3 — so treat it as a
        computation, not a field, and hold the result where it is needed twice.
        """
        from ..canon.container import sha3_256
        return sha3_256(self.to_container())

    @classmethod
    def from_container(cls, raw: bytes):
        container = parse_container(raw)
        if container.obj_type is not cls.OBJ_TYPE:
            raise CanonError(
                f"canon: expected {cls.OBJ_TYPE.name}, got {container.obj_type.name}")
        r = Reader(container.content)
        obj = cls.parse(r)
        r.expect_exhausted()
        return obj


# ---------------------------------------------------------------------------
# 1. What is being trained
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RoundDescriptor(Object):
    """The identity of a training run. Hashing this gives the genesis anchor.

    Everything a node must agree on before it can contribute a single step:
    which model, in which arithmetic, over which corpus, with which tokenizer.
    A node that disagrees about any of it produces work nobody can check.

    There is no separate determinism_class field. It is derived from `numerics`,
    so it cannot disagree with the arithmetic it is supposed to describe — which
    a stored copy eventually would.
    """

    OBJ_TYPE = ObjectType.ROUND_DESCRIPTOR

    protocol_version: int
    network_magic: int
    round_id: int
    model: ModelSpec
    numerics: Numerics
    optimizer_id: int
    tokenizer_hash: bytes
    dataset_root: bytes
    dataset_chunks: int

    @property
    def determinism_class(self) -> int:
        return self.numerics.determinism_class

    @property
    def byte_level_tokenizer(self) -> bool:
        """An all-zero tokenizer hash means bytes are tokens.

        A real choice, not a missing field: a byte-level corpus needs no
        tokenizer artifact at all, and saying so with the zero hash keeps the
        layout fixed-width.
        """
        return self.tokenizer_hash == bytes(32)

    def serialize(self, w: Writer) -> Writer:
        w.u32(self.protocol_version).u32(self.network_magic).u64(self.round_id)
        self.model.serialize(w)
        self.numerics.serialize(w)
        return (w.u16(self.optimizer_id)
                 .hash(self.tokenizer_hash)
                 .hash(self.dataset_root)
                 .u64(self.dataset_chunks))

    @classmethod
    def parse(cls, r: Reader) -> "RoundDescriptor":
        from ..canon.container import PROTOCOL_VERSION
        protocol_version = r.u32()
        if protocol_version != PROTOCOL_VERSION:
            raise CanonError(f"round: protocol {protocol_version:#x} is not this build's")
        out = cls(
            protocol_version=protocol_version,
            network_magic=r.u32(),
            round_id=r.u64(),
            model=ModelSpec.parse(r),
            numerics=Numerics.parse(r),
            optimizer_id=r.u16(),
            tokenizer_hash=r.hash(),
            dataset_root=r.hash(),
            dataset_chunks=r.u64(),
        )
        if out.optimizer_id != OPT_ADAFACTOR:
            raise CanonError(f"round: unknown optimizer {out.optimizer_id}")
        # A corpus root with no chunks, or chunks with no root, is a round that
        # cannot name a batch. Both-zero is legal and means "no corpus pinned".
        if (out.dataset_root == bytes(32)) != (out.dataset_chunks == 0):
            raise CanonError(
                "round: dataset_root and dataset_chunks must both be set or both be zero")
        return out


@dataclass(frozen=True)
class DatasetManifest(Object):
    """What a corpus is: raw UTF-8, addressed in document-aligned chunks.

    No offset table. Boundaries are derived by scanning for the document
    separator at or after `target_chunk_bytes`, so the manifest stays three
    numbers whether the corpus is a megabyte or seven terabytes.
    """

    OBJ_TYPE = ObjectType.DATASET_MANIFEST

    dataset_root: bytes
    tokenizer_hash: bytes
    n_bytes: int
    target_chunk_bytes: int
    n_chunks: int

    def serialize(self, w: Writer) -> Writer:
        return (w.hash(self.dataset_root)
                 .hash(self.tokenizer_hash)
                 .u64(self.n_bytes)
                 .u32(self.target_chunk_bytes)
                 .u64(self.n_chunks))

    @classmethod
    def parse(cls, r: Reader) -> "DatasetManifest":
        m = cls(dataset_root=r.hash(), tokenizer_hash=r.hash(), n_bytes=r.u64(),
                target_chunk_bytes=r.u32(), n_chunks=r.u64())
        if m.n_bytes == 0:
            raise CanonError("dataset: empty corpus")
        if m.target_chunk_bytes == 0:
            raise CanonError("dataset: target_chunk_bytes must be non-zero")
        if m.n_chunks == 0:
            raise CanonError("dataset: n_chunks must be non-zero")
        # Each chunk runs to the first separator at or after the target, so all
        # but the last are at least that long. More chunks than that allows means
        # the boundaries do not follow the rule, and any node that scans the
        # corpus itself would disagree about what chunk seven is.
        if m.n_chunks > m.n_bytes // m.target_chunk_bytes + 1:
            raise CanonError(
                f"dataset: {m.n_chunks} chunks over {m.n_bytes} bytes at a target of "
                f"{m.target_chunk_bytes} — at least one is shorter than the target")
        return m


@dataclass(frozen=True)
class BatchSeed(Object):
    """The pre-image a worker's data draw is derived from.

    Never stored or sent. It exists as an object so that the bytes being hashed
    are produced by the same writer as everything else, and so a verifier
    rebuilds them the same way rather than by a private convention.
    """

    OBJ_TYPE = ObjectType.BATCH_SEED

    dataset_root: bytes
    round_id: int
    worker_id: int
    outer_step: int
    inner_index: int

    def serialize(self, w: Writer) -> Writer:
        return (w.hash(self.dataset_root)
                 .u64(self.round_id)
                 .u64(self.worker_id)
                 .u64(self.outer_step)
                 .u32(self.inner_index))

    @classmethod
    def parse(cls, r: Reader) -> "BatchSeed":
        return cls(dataset_root=r.hash(), round_id=r.u64(), worker_id=r.u64(),
                   outer_step=r.u64(), inner_index=r.u32())


# ---------------------------------------------------------------------------
# 2. What a worker did
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContributionHeader(Object):
    """A worker's claim: from these weights, over this assignment, I produced
    this update.

    The payload itself is not here — it is hundreds of megabytes and travels
    separately. What is here is everything needed to decide whether the payload
    is worth fetching and, later, whether the claim was true.

    `base_weights_hash` is what makes the claim checkable at all. Without it a
    contribution would be an update with no stated starting point, and a
    verifier could not tell a wrong answer from an answer to a different
    question.
    """

    OBJ_TYPE = ObjectType.CONTRIBUTION_HEADER

    round_id: int
    outer_step: int
    worker_id: int
    assignment_id: int
    base_checkpoint: bytes
    base_weights_hash: bytes
    payload_hash: bytes
    payload_bytes: int
    scale_exp: int
    contribution_format: ContributionFormat

    def serialize(self, w: Writer) -> Writer:
        return (w.u64(self.round_id)
                 .u64(self.outer_step)
                 .u64(self.worker_id)
                 .u64(self.assignment_id)
                 .hash(self.base_checkpoint)
                 .hash(self.base_weights_hash)
                 .hash(self.payload_hash)
                 .u64(self.payload_bytes)
                 .i64(self.scale_exp)
                 .u16(self.contribution_format))

    @classmethod
    def parse(cls, r: Reader) -> "ContributionHeader":
        try:
            out = cls(
                round_id=r.u64(), outer_step=r.u64(), worker_id=r.u64(),
                assignment_id=r.u64(), base_checkpoint=r.hash(),
                base_weights_hash=r.hash(), payload_hash=r.hash(),
                payload_bytes=r.u64(), scale_exp=r.i64(),
                contribution_format=ContributionFormat(r.u16()),
            )
        except ValueError as exc:
            raise CanonError(f"contribution: unknown quantiser ({exc})") from exc
        if out.payload_bytes == 0:
            raise CanonError("contribution: an empty payload is not an update")
        # The scale is an exponent applied to a float, so it must stay inside
        # what a float can represent. Beyond this the dequantised update is
        # infinity or zero, and either way it is not what the worker computed.
        if not -300 <= out.scale_exp <= 300:
            raise CanonError(f"contribution: scale exponent {out.scale_exp} is out of range")
        return out


@dataclass(frozen=True)
class CheckpointHeader(Object):
    """What the network agreed the weights became.

    `optimizer_state_hash` is a commitment, not the state: momentum for a 30B
    model is tens of gigabytes and does not belong in a header. Committing to it
    is what lets two producers be caught having applied the same contributions
    to different optimizer state.

    `contribution_root` is the Merkle root over the contribution ids that went
    in, in canonical order. It is what makes "every contribution was aggregated"
    a checkable claim rather than a promise.
    """

    OBJ_TYPE = ObjectType.CHECKPOINT_HEADER

    round_id: int
    outer_step: int
    parent: bytes
    weights_hash: bytes
    optimizer_state_hash: bytes
    contribution_root: bytes
    producer_id: int
    timestamp_ms: int

    def serialize(self, w: Writer) -> Writer:
        return (w.u64(self.round_id)
                 .u64(self.outer_step)
                 .hash(self.parent)
                 .hash(self.weights_hash)
                 .hash(self.optimizer_state_hash)
                 .hash(self.contribution_root)
                 .u64(self.producer_id)
                 .u64(self.timestamp_ms))

    @classmethod
    def parse(cls, r: Reader) -> "CheckpointHeader":
        out = cls(round_id=r.u64(), outer_step=r.u64(), parent=r.hash(),
                  weights_hash=r.hash(), optimizer_state_hash=r.hash(),
                  contribution_root=r.hash(), producer_id=r.u64(),
                  timestamp_ms=r.u64())
        # Only the genesis checkpoint has no parent, and only it is at step 0.
        # Either without the other is a chain that does not reach genesis.
        if (out.parent == bytes(32)) != (out.outer_step == 0):
            raise CanonError(
                "checkpoint: an empty parent is only legal at outer step 0")
        return out


@dataclass(frozen=True)
class ExpertShard(Object):
    """Which experts one holder carries, and what they hash to.

    A sharded mixture has no single weights blob, so a checkpoint's
    `weights_hash` covers the shared parameters and each shard commits to its
    own experts separately. That is what lets a 30B model have a checkpoint
    nobody had to assemble in one place.

    `first_expert` and `n_experts` are stated rather than derived from
    `shard_index` so that an unequal split can be described later without a
    format break, even though the model spec refuses one today.
    """

    OBJ_TYPE = ObjectType.EXPERT_SHARD

    round_id: int
    outer_step: int
    shard_index: int
    shard_count: int
    first_expert: int
    n_experts: int
    weights_hash: bytes

    def serialize(self, w: Writer) -> Writer:
        return (w.u64(self.round_id)
                 .u64(self.outer_step)
                 .u32(self.shard_index)
                 .u32(self.shard_count)
                 .u32(self.first_expert)
                 .u32(self.n_experts)
                 .hash(self.weights_hash))

    @classmethod
    def parse(cls, r: Reader) -> "ExpertShard":
        out = cls(round_id=r.u64(), outer_step=r.u64(), shard_index=r.u32(),
                  shard_count=r.u32(), first_expert=r.u32(), n_experts=r.u32(),
                  weights_hash=r.hash())
        if out.shard_count == 0:
            raise CanonError("shard: shard_count must be non-zero")
        if out.shard_index >= out.shard_count:
            raise CanonError(f"shard: index {out.shard_index} of {out.shard_count}")
        if out.n_experts == 0:
            raise CanonError("shard: a shard with no experts holds nothing")
        return out


# ---------------------------------------------------------------------------
# 3. Whether it was true
# ---------------------------------------------------------------------------

class Verdict(IntEnum):
    """What a verifier concluded after replaying a contribution.

    INDETERMINATE is the one that matters and the one a naive design omits. A
    verifier in a different determinism class cannot reproduce the work and
    must say so, because reporting MISMATCH would slash an honest worker for
    owning different hardware.
    """

    MATCH = 1
    MISMATCH = 2
    INDETERMINATE = 3


class SlashReason(IntEnum):
    WRONG_UPDATE = 1        # replay produced different bytes
    WRONG_DATA = 2          # trained on batches other than the derived ones
    WRONG_BASE = 3          # started from weights other than the named checkpoint
    UNANSWERED = 4          # never answered a challenge before its deadline


@dataclass(frozen=True)
class ChallengeOrder(Object):
    """An instruction to reproduce someone's work.

    The nonce is what stops a worker from precomputing answers: it is not known
    until the order is issued, so an answer that exists before the question is
    an answer to a different question.
    """

    OBJ_TYPE = ObjectType.CHALLENGE_ORDER

    round_id: int
    outer_step: int
    challenger_id: int
    target_worker_id: int
    contribution_id: bytes
    deadline_step: int
    nonce: bytes

    def serialize(self, w: Writer) -> Writer:
        return (w.u64(self.round_id)
                 .u64(self.outer_step)
                 .u64(self.challenger_id)
                 .u64(self.target_worker_id)
                 .hash(self.contribution_id)
                 .u64(self.deadline_step)
                 .hash(self.nonce))

    @classmethod
    def parse(cls, r: Reader) -> "ChallengeOrder":
        out = cls(round_id=r.u64(), outer_step=r.u64(), challenger_id=r.u64(),
                  target_worker_id=r.u64(), contribution_id=r.hash(),
                  deadline_step=r.u64(), nonce=r.hash())
        if out.deadline_step <= out.outer_step:
            raise CanonError(
                f"challenge: deadline {out.deadline_step} is not after step {out.outer_step}")
        if out.challenger_id == out.target_worker_id:
            raise CanonError("challenge: a worker cannot challenge itself")
        return out


@dataclass(frozen=True)
class VerificationVerdict(Object):
    """What one verifier found, signed by nothing yet and worth exactly that.

    `determinism_class` is carried because a verdict from a different class is
    not evidence. Without it, MISMATCH from an incompatible verifier looks
    identical to MISMATCH from a compatible one.
    """

    OBJ_TYPE = ObjectType.VERIFICATION_VERDICT

    challenge_id: bytes
    verifier_id: int
    verdict: Verdict
    replay_payload_hash: bytes
    determinism_class: int

    def serialize(self, w: Writer) -> Writer:
        return (w.hash(self.challenge_id)
                 .u64(self.verifier_id)
                 .u16(self.verdict)
                 .hash(self.replay_payload_hash)
                 .u32(self.determinism_class))

    @classmethod
    def parse(cls, r: Reader) -> "VerificationVerdict":
        challenge_id = r.hash()
        verifier_id = r.u64()
        try:
            verdict = Verdict(r.u16())
        except ValueError as exc:
            raise CanonError(f"verdict: unknown value ({exc})") from exc
        out = cls(challenge_id=challenge_id, verifier_id=verifier_id, verdict=verdict,
                  replay_payload_hash=r.hash(), determinism_class=r.u32())
        # A verifier that could not judge has no replay to show for it, and one
        # that did must show what it got. Otherwise INDETERMINATE becomes a
        # place to hide a real mismatch.
        if (out.verdict is Verdict.INDETERMINATE) != (out.replay_payload_hash == bytes(32)):
            raise CanonError(
                "verdict: INDETERMINATE carries no replay hash, and every other "
                "verdict must carry one")
        return out


@dataclass(frozen=True)
class SlashEvidence(Object):
    """The verdicts that together justify punishing a worker.

    A single verdict is never enough. One verifier can be wrong, malicious, or
    simply on hardware that rounds differently, and a protocol that slashes on
    one report is a protocol where accusing is cheaper than working. The quorum
    is policy; this object just carries what was gathered.
    """

    OBJ_TYPE = ObjectType.SLASH_EVIDENCE

    round_id: int
    target_worker_id: int
    contribution_id: bytes
    reason: SlashReason
    verdict_ids: tuple[bytes, ...] = field(default=())

    def serialize(self, w: Writer) -> Writer:
        w.u64(self.round_id).u64(self.target_worker_id).hash(self.contribution_id)
        w.u16(self.reason).u32(len(self.verdict_ids))
        for v in self.verdict_ids:
            w.hash(v)
        return w

    @classmethod
    def parse(cls, r: Reader) -> "SlashEvidence":
        round_id = r.u64()
        target_worker_id = r.u64()
        contribution_id = r.hash()
        try:
            reason = SlashReason(r.u16())
        except ValueError as exc:
            raise CanonError(f"slash: unknown reason ({exc})") from exc
        count = r.u32()
        if count == 0:
            raise CanonError("slash: evidence with no verdicts is an accusation")
        if count > MAX_VERDICT_REFS:
            raise CanonError(f"slash: {count} verdicts exceeds the {MAX_VERDICT_REFS} limit")
        verdicts = tuple(r.hash() for _ in range(count))
        if len(set(verdicts)) != len(verdicts):
            # Otherwise one verifier's verdict, repeated, reaches any quorum.
            raise CanonError("slash: the same verdict is cited more than once")
        return cls(round_id=round_id, target_worker_id=target_worker_id,
                   contribution_id=contribution_id, reason=reason,
                   verdict_ids=verdicts)


# ---------------------------------------------------------------------------
# 4. How the network is operated
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyDescriptor(Object):
    """Operational knobs — the second hashed artifact.

    Separate from the round because these are the things a network may want to
    tune from real data (how often to challenge, how long a round may take)
    while the model's identity stays frozen. Changing policy is a decision the
    network makes; changing the round is a new network.
    """

    OBJ_TYPE = ObjectType.POLICY_DESCRIPTOR

    protocol_version: int
    network_magic: int
    policy_version: int
    inner_steps: int
    micro_batch: int
    min_contributors: int
    checkpoint_interval: int
    round_deadline_ms: int
    outer_momentum_q16: int
    outer_lr_q16: int
    nesterov: bool
    challenge_percent: int
    challenge_deadline_steps: int
    retained_checkpoints: int
    slash_quorum: int
    shadow_mode: bool

    def serialize(self, w: Writer) -> Writer:
        return (w.u32(self.protocol_version)
                 .u32(self.network_magic)
                 .u64(self.policy_version)
                 .u32(self.inner_steps)
                 .u32(self.micro_batch)
                 .u32(self.min_contributors)
                 .u32(self.checkpoint_interval)
                 .u32(self.round_deadline_ms)
                 .u32(self.outer_momentum_q16)
                 .u32(self.outer_lr_q16)
                 .boolean(self.nesterov)
                 .u32(self.challenge_percent)
                 .u32(self.challenge_deadline_steps)
                 .u32(self.retained_checkpoints)
                 .u32(self.slash_quorum)
                 .boolean(self.shadow_mode))

    @classmethod
    def parse(cls, r: Reader) -> "PolicyDescriptor":
        from ..canon.container import PROTOCOL_VERSION
        p = cls(
            protocol_version=r.u32(), network_magic=r.u32(), policy_version=r.u64(),
            inner_steps=r.u32(), micro_batch=r.u32(), min_contributors=r.u32(),
            checkpoint_interval=r.u32(), round_deadline_ms=r.u32(),
            outer_momentum_q16=r.u32(), outer_lr_q16=r.u32(), nesterov=r.boolean(),
            challenge_percent=r.u32(), challenge_deadline_steps=r.u32(),
            retained_checkpoints=r.u32(), slash_quorum=r.u32(), shadow_mode=r.boolean(),
        )
        if p.protocol_version != PROTOCOL_VERSION:
            raise CanonError(f"policy: protocol {p.protocol_version:#x} is not this build's")
        for name in ("inner_steps", "micro_batch", "min_contributors",
                     "checkpoint_interval", "round_deadline_ms", "retained_checkpoints"):
            if getattr(p, name) == 0:
                raise CanonError(f"policy: {name} must be non-zero")
        if p.challenge_percent > 100:
            raise CanonError(f"policy: challenge_percent {p.challenge_percent} exceeds 100")
        # The invariant tying verification to storage: a challenge must expire
        # while the weights needed to answer it are still held. Otherwise the
        # honest answer to a live challenge is "I no longer have the inputs",
        # which is indistinguishable from refusing to answer.
        if p.challenge_deadline_steps >= p.retained_checkpoints:
            raise CanonError(
                f"policy: challenge_deadline_steps {p.challenge_deadline_steps} >= "
                f"retained_checkpoints {p.retained_checkpoints} — unanswerable challenges")
        # A quorum of one makes accusing cheaper than working.
        if p.slash_quorum < 2:
            raise CanonError(f"policy: slash_quorum {p.slash_quorum} must be at least 2")
        return p


# ---------------------------------------------------------------------------

CODECS: dict[ObjectType, type[Object]] = {
    ObjectType.ROUND_DESCRIPTOR: RoundDescriptor,
    ObjectType.DATASET_MANIFEST: DatasetManifest,
    ObjectType.BATCH_SEED: BatchSeed,
    ObjectType.CONTRIBUTION_HEADER: ContributionHeader,
    ObjectType.CHECKPOINT_HEADER: CheckpointHeader,
    ObjectType.CHALLENGE_ORDER: ChallengeOrder,
    ObjectType.VERIFICATION_VERDICT: VerificationVerdict,
    ObjectType.SLASH_EVIDENCE: SlashEvidence,
    ObjectType.POLICY_DESCRIPTOR: PolicyDescriptor,
    ObjectType.EXPERT_SHARD: ExpertShard,
}


def decode(raw: bytes) -> Object:
    """Parse any container into the object it holds.

    Used where the type is not known in advance — an unsolicited object off the
    wire, mostly. Where the type IS known, call that class's from_container, so
    that receiving the wrong kind is an error rather than a surprise later.
    """
    container = parse_container(raw)
    codec = CODECS.get(container.obj_type)
    if codec is None:
        raise CanonError(f"canon: {container.obj_type.name} has no codec")
    r = Reader(container.content)
    obj = codec.parse(r)
    r.expect_exhausted()
    return obj

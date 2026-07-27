"""The channel between a daemon and the worker that trains for it.

TWO PROCESSES, ONE IMPLEMENTATION. They stay separate because the daemon must
run without a GPU and must survive a trainer that crashes or is restarted mid
round. What they do NOT do is speak different dialects: both sides import these
same message classes, the same Writer, the same framing as the peer-to-peer
wire. The implementation this replaces had a hand-written CBOR profile, a second
frame codec and a Python mirror of both — 3,445 lines carrying no training and
no consensus logic, and twelve of the last forty commits touching it. None of
that is here because none of it is needed once the two sides share a module.

THE DAEMON DECIDES NOTHING ABOUT DATA AND NEITHER DOES THE WORKER. An assignment
says which step, from which weights, for how many inner steps. WHICH TEXT is
derived by the worker from the schedule, which is a pure function of protocol
state — so a worker that wanted different data would have to lie about something
a verifier recomputes.

THE SOCKET IS LOCAL AND STILL NOT TRUSTED. It lives in the datadir at 0700 and
the daemon checks the peer's uid, because "only a local process can reach it" is
an argument about the filesystem, not about the process. Every length is bounded
the same way it is on the network, and the submit payload is bounded by what the
round's model would actually produce rather than by a constant nobody revisits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from ..canon.stream import CanonError, Reader, Writer
from ..net import framing

# Not a network magic. A different constant so that pointing a worker at a
# peer-to-peer port, or the reverse, fails at the first four bytes instead of
# somewhere confusing.
IPC_MAGIC = 0x524E_5750     # "RNWP"
IPC_VERSION = 1

# A contribution for the 29-billion-parameter mixture is 29 GB at one byte per
# parameter, which does not belong in a single frame; the largest thing that
# does is the dense 400M model's 397 MB. The daemon narrows this to what the
# round's model actually produces — see `expected_payload_bytes`.
MAX_IPC_PAYLOAD = 512 << 20

MAX_REASON = 256


class Command(IntEnum):
    """Every message on this channel.

    Walked by WorkerIpcTests.EveryCommandHasACodec, for the same reason the
    network's enum is: a command added here and to no dispatch table is not a
    command, and the last implementation found that out on a live node.
    """

    HELLO = 1
    WELCOME = 2
    GET_ASSIGNMENT = 3
    ASSIGNMENT = 4
    NO_WORK = 5
    SUBMIT = 6
    ACCEPTED = 7
    REFUSED = 8
    GET_CHUNK = 9
    CHUNK = 10
    APPLY = 11
    APPLIED = 12


class IpcError(Exception):
    pass


class Refusal(IntEnum):
    UNKNOWN_ASSIGNMENT = 1
    STALE_ASSIGNMENT = 2      # the round moved on while the worker trained
    BAD_PAYLOAD = 3
    NOT_READY = 4             # the daemon has no round open
    UNAVAILABLE = 5           # asked for a chunk that is being fetched


def expected_payload_bytes(parameter_count: int, contribution_bits: int) -> int:
    """How large a legitimate submission is, exactly.

    Bounding by this rather than by a constant is the difference between a
    limit that tracks the round and one that is either too small for the next
    model or too large to be a limit.
    """
    return (parameter_count * contribution_bits + 7) // 8


class Message:
    COMMAND: Command

    def serialize(self, w: Writer) -> Writer:      # pragma: no cover - interface
        raise NotImplementedError

    @classmethod
    def parse(cls, r: Reader):                     # pragma: no cover - interface
        raise NotImplementedError

    def payload(self) -> bytes:
        return self.serialize(Writer()).take()

    def to_frame(self) -> bytes:
        try:
            return framing.frame(IPC_MAGIC, self.COMMAND, self.payload(),
                                 max_payload=MAX_IPC_PAYLOAD)
        except framing.FramingError as exc:
            raise IpcError(str(exc)) from exc


@dataclass(frozen=True)
class Hello(Message):
    """The worker states which rules it believes it is training under.

    Both anchors, checked before anything else happens. A worker on a different
    commit produces contributions nobody can use and cannot be told apart from
    one that is cheating, so the connection ends here rather than after twenty
    minutes of wasted GPU.
    """

    COMMAND = Command.HELLO
    version: int = IPC_VERSION
    genesis_hash: bytes = bytes(32)
    policy_hash: bytes = bytes(32)
    user_agent: str = ""

    def serialize(self, w: Writer) -> Writer:
        return (w.u32(self.version).hash(self.genesis_hash)
                 .hash(self.policy_hash).string(self.user_agent))

    @classmethod
    def parse(cls, r: Reader) -> "Hello":
        return cls(version=r.u32(), genesis_hash=r.hash(), policy_hash=r.hash(),
                   user_agent=r.string(limit=64))


@dataclass(frozen=True)
class Welcome(Message):
    """The daemon assigns the worker its id.

    Assigned, never chosen: the worker id is an input to the batch schedule, so
    a worker that could name itself could take another's data, or fish for an id
    whose windows it liked.
    """

    COMMAND = Command.WELCOME
    worker_id: int = 0
    network_magic: int = 0
    parameter_count: int = 0
    contribution_bits: int = 8

    def serialize(self, w: Writer) -> Writer:
        return (w.u64(self.worker_id).u32(self.network_magic)
                 .u64(self.parameter_count).u16(self.contribution_bits))

    @classmethod
    def parse(cls, r: Reader) -> "Welcome":
        return cls(worker_id=r.u64(), network_magic=r.u32(),
                   parameter_count=r.u64(), contribution_bits=r.u16())


@dataclass(frozen=True)
class GetAssignment(Message):
    COMMAND = Command.GET_ASSIGNMENT

    def serialize(self, w: Writer) -> Writer:
        return w

    @classmethod
    def parse(cls, r: Reader) -> "GetAssignment":
        return cls()


@dataclass(frozen=True)
class Assignment(Message):
    """Which step, from which weights, for how long.

    Not which data. That is derived by the worker from the schedule, and the
    daemon could not usefully send it anyway: a verifier recomputes it from the
    same protocol state, so anything the daemon said about it would be either
    redundant or a lie.
    """

    COMMAND = Command.ASSIGNMENT
    assignment_id: int = 0
    round_id: int = 0
    outer_step: int = 0
    base_checkpoint: bytes = bytes(32)
    base_weights_hash: bytes = bytes(32)
    inner_steps: int = 0
    micro_batch: int = 1
    deadline_ms: int = 0

    def serialize(self, w: Writer) -> Writer:
        return (w.u64(self.assignment_id).u64(self.round_id).u64(self.outer_step)
                 .hash(self.base_checkpoint).hash(self.base_weights_hash)
                 .u32(self.inner_steps).u32(self.micro_batch).u64(self.deadline_ms))

    @classmethod
    def parse(cls, r: Reader) -> "Assignment":
        out = cls(assignment_id=r.u64(), round_id=r.u64(), outer_step=r.u64(),
                  base_checkpoint=r.hash(), base_weights_hash=r.hash(),
                  inner_steps=r.u32(), micro_batch=r.u32(), deadline_ms=r.u64())
        if out.inner_steps == 0 or out.micro_batch == 0:
            raise CanonError("assignment: no work is not an assignment")
        return out


@dataclass(frozen=True)
class NoWork(Message):
    """There is nothing to do yet. Says why, so a worker waiting on a quorum
    looks different from one waiting on a daemon that is stuck."""

    COMMAND = Command.NO_WORK
    reason: str = ""
    retry_ms: int = 5000

    def serialize(self, w: Writer) -> Writer:
        return w.string(self.reason[:MAX_REASON]).u32(self.retry_ms)

    @classmethod
    def parse(cls, r: Reader) -> "NoWork":
        return cls(reason=r.string(limit=MAX_REASON), retry_ms=r.u32())


@dataclass(frozen=True)
class Submit(Message):
    """The quantised difference between where the worker started and ended.

    `payload` is packed at the round's contribution width, so int4 really is
    half the bytes of int8 on this channel as well as on the wire.
    """

    COMMAND = Command.SUBMIT
    assignment_id: int = 0
    scale_exp: int = 0
    value_count: int = 0
    final_loss: float = 0.0
    # Named `packed` and not `payload`: a field called payload would shadow
    # Message.payload(), and the failure is a TypeError at send time on the one
    # message that carries real data. Caught by
    # WorkerIpcTests.EveryMessageRoundTripsThroughAFrame.
    packed: bytes = b""

    def serialize(self, w: Writer) -> Writer:
        return (w.u64(self.assignment_id).i64(self.scale_exp)
                 .u64(self.value_count)
                 # The loss is carried as a fixed-point millionth rather than a
                 # float, so this channel needs no float encoding at all — one
                 # fewer thing for the two sides to agree about.
                 .i64(int(self.final_loss * 1_000_000))
                 .blob(self.packed))

    @classmethod
    def parse(cls, r: Reader) -> "Submit":
        return cls(assignment_id=r.u64(), scale_exp=r.i64(), value_count=r.u64(),
                   final_loss=r.i64() / 1_000_000.0,
                   packed=r.blob(limit=MAX_IPC_PAYLOAD))


@dataclass(frozen=True)
class Accepted(Message):
    COMMAND = Command.ACCEPTED
    contribution_id: bytes = bytes(32)

    def serialize(self, w: Writer) -> Writer:
        return w.hash(self.contribution_id)

    @classmethod
    def parse(cls, r: Reader) -> "Accepted":
        return cls(contribution_id=r.hash())


@dataclass(frozen=True)
class Refused(Message):
    COMMAND = Command.REFUSED
    code: Refusal = Refusal.BAD_PAYLOAD
    reason: str = ""

    def serialize(self, w: Writer) -> Writer:
        return w.u16(self.code).string(self.reason[:MAX_REASON])

    @classmethod
    def parse(cls, r: Reader) -> "Refused":
        try:
            code = Refusal(r.u16())
        except ValueError as exc:
            raise CanonError(f"refused: unknown code ({exc})") from exc
        return cls(code=code, reason=r.string(limit=MAX_REASON))


@dataclass(frozen=True)
class GetChunk(Message):
    """A worker asking the daemon for corpus it does not hold.

    This is what lets the corpus live on a machine with the disk for it and the
    training happen on a machine with the GPU for it.
    """

    COMMAND = Command.GET_CHUNK
    index: int = 0

    def serialize(self, w: Writer) -> Writer:
        return w.u64(self.index)

    @classmethod
    def parse(cls, r: Reader) -> "GetChunk":
        return cls(index=r.u64())


@dataclass(frozen=True)
class Chunk(Message):
    """Corpus bytes the daemon has already verified against dataset_root.

    No proof travels here, and that is not an oversight: the daemon verified it
    before storing it, and a worker that does not trust its own daemon has a
    problem this channel cannot solve.
    """

    COMMAND = Command.CHUNK
    index: int = 0
    data: bytes = b""

    def serialize(self, w: Writer) -> Writer:
        return w.u64(self.index).blob(self.data)

    @classmethod
    def parse(cls, r: Reader) -> "Chunk":
        return cls(index=r.u64(), data=r.blob(limit=MAX_IPC_PAYLOAD))


@dataclass(frozen=True)
class Apply(Message):
    """Produce: apply this aggregate and tell me what the weights became.

    The daemon holds no weights and no optimizer state — a relay node runs on a
    machine with no GPU and must not carry 3.2 GB of momentum for a model it
    cannot evaluate — so the outer step happens where the weights already are.
    The daemon supplies what consensus decided (the aggregate, the rates), the
    worker supplies the arithmetic, and the result is checkable by anyone who
    holds the same inputs.

    The aggregate travels at the round's CONTRIBUTION width, not at int64. Each
    contribution is within ±max_magnitude after alignment, a sum of n is within
    ±n·max, and the mean is within ±max again — so the average fits exactly the
    format the parts arrived in. That is 400 MB instead of 3.2 GB for the dense
    model. Proved by WorkerIpcTests.TheAggregateFitsTheContributionFormat.
    """

    COMMAND = Command.APPLY
    outer_step: int = 0
    scale_exp: int = 0
    value_count: int = 0
    momentum_q16: int = 0
    lr_q16: int = 0
    nesterov: bool = True
    packed: bytes = b""

    def serialize(self, w: Writer) -> Writer:
        return (w.u64(self.outer_step).i64(self.scale_exp).u64(self.value_count)
                 .u32(self.momentum_q16).u32(self.lr_q16).boolean(self.nesterov)
                 .blob(self.packed))

    @classmethod
    def parse(cls, r: Reader) -> "Apply":
        out = cls(outer_step=r.u64(), scale_exp=r.i64(), value_count=r.u64(),
                  momentum_q16=r.u32(), lr_q16=r.u32(), nesterov=r.boolean(),
                  packed=r.blob(limit=MAX_IPC_PAYLOAD))
        if out.value_count == 0:
            raise CanonError("apply: an empty update is not an update")
        return out


@dataclass(frozen=True)
class Applied(Message):
    """What the weights and the optimizer became.

    Both hashes, because a checkpoint commits to both: two producers who
    applied the same contributions to different momentum would otherwise agree
    about the weights and disagree about everything after.
    """

    COMMAND = Command.APPLIED
    outer_step: int = 0
    weights_hash: bytes = bytes(32)
    optimizer_state_hash: bytes = bytes(32)

    def serialize(self, w: Writer) -> Writer:
        return (w.u64(self.outer_step).hash(self.weights_hash)
                 .hash(self.optimizer_state_hash))

    @classmethod
    def parse(cls, r: Reader) -> "Applied":
        return cls(outer_step=r.u64(), weights_hash=r.hash(),
                   optimizer_state_hash=r.hash())


CODECS: dict[Command, type[Message]] = {
    Command.HELLO: Hello,
    Command.WELCOME: Welcome,
    Command.GET_ASSIGNMENT: GetAssignment,
    Command.ASSIGNMENT: Assignment,
    Command.NO_WORK: NoWork,
    Command.SUBMIT: Submit,
    Command.ACCEPTED: Accepted,
    Command.REFUSED: Refused,
    Command.GET_CHUNK: GetChunk,
    Command.CHUNK: Chunk,
    Command.APPLY: Apply,
    Command.APPLIED: Applied,
}


def decode(command: Command, payload: bytes) -> Message:
    codec = CODECS.get(command)
    if codec is None:
        raise IpcError(f"ipc: {command.name} has no codec")
    r = Reader(payload)
    try:
        message = codec.parse(r)
        r.expect_exhausted()
    except CanonError as exc:
        raise IpcError(f"ipc: {command.name}: {exc}") from exc
    return message


async def read_message(reader) -> Message:
    try:
        header, payload = await framing.read_frame(
            reader, IPC_MAGIC, Command, max_payload=MAX_IPC_PAYLOAD)
    except framing.FramingError as exc:
        raise IpcError(str(exc)) from exc
    return decode(header.command, payload)

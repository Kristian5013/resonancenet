"""The daemon's side of the worker channel.

Hands out assignments, takes contributions, and turns them into objects the
network can relay. It is the only part of the daemon that knows a worker exists,
and it is deliberately small: everything it decides comes from the round
descriptor, the policy, and the chain, so a bug here can waste a worker's time
but cannot change what the network agrees.

WHAT IT DOES NOT DO. It does not train, it does not hold weights, and it does
not produce checkpoints. A daemon runs on machines with no GPU and no
deep-learning stack, and keeping this side free of them is what makes that true
rather than aspirational — importing torch here would make every relay node
carry two gigabytes of CUDA it will never call.

THE SOCKET IS LOCAL AND STILL CHECKED. It lives in the datadir with the
directory at 0700, and the peer's uid is compared against this process's. "Only
a local process can reach it" is a claim about the filesystem, not about which
local process, and on a shared machine those are different claims.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import struct
import time
from dataclasses import dataclass, field

import numpy as np

from ..consensus.numerics import ContributionFormat
from ..consensus.objects import ContributionHeader
from ..diloco.aggregate import Contribution
from ..diloco.quantize import unpack
from ..net import protocol as P
from ..worker import ipc

SOCKET_NAME = "worker.sock"

# A daemon serves the workers on its own machine. More than a handful means
# something is wrong, and the cap is here so that something wrong is bounded.
MAX_WORKERS = 16


@dataclass
class WorkerConn:
    worker_id: int
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    user_agent: str = ""
    assignments: dict = field(default_factory=dict)
    submitted: int = 0
    connected_at: float = field(default_factory=time.monotonic)


@dataclass
class WorkerService:
    """Accepts local workers and turns their output into contributions."""

    datadir: str
    round_desc: object
    policy: object
    chain: object
    service: object = None          # the network Service, for relaying
    corpus: object = None

    workers: dict = field(default_factory=dict)
    contributions: dict = field(default_factory=dict)
    # (outer_step, contribution_id) — kept apart from the values so the root can
    # be rebuilt after the values are spent.
    _headers: list = field(default_factory=list)
    # outer_step -> {worker_id: (weights_hash, optimizer_state_hash)}
    _applied_at: dict = field(default_factory=dict)
    # outer_step -> the Apply every attached worker still owes
    _pending_apply: dict = field(default_factory=dict)

    _server: object = None
    _next_worker_id: int = 1
    _next_assignment_id: int = 1
    _log = print
    produced: int = 0

    accepted: int = 0
    refused: int = 0
    deferred: int = 0

    # -- lifetime -------------------------------------------------------------

    @property
    def socket_path(self) -> str:
        return os.path.join(self.datadir, SOCKET_NAME)

    async def start(self) -> str:
        os.makedirs(self.datadir, exist_ok=True)
        os.chmod(self.datadir, 0o700)
        path = self.socket_path
        # A socket left by a process that died is not a socket anyone is
        # listening on, and refusing to start because of one would mean a crash
        # needs a manual cleanup before the node comes back.
        if os.path.exists(path) and not self._someone_listening(path):
            os.unlink(path)
        self._server = await asyncio.start_unix_server(self._accept, path=path)
        os.chmod(path, 0o600)
        return path

    @staticmethod
    def _someone_listening(path: str) -> bool:
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.settimeout(0.2)
            probe.connect(path)
            return True
        except OSError:
            return False
        finally:
            probe.close()

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
        for conn in list(self.workers.values()):
            with contextlib.suppress(Exception):
                conn.writer.close()
        self.workers.clear()
        with contextlib.suppress(OSError):
            os.unlink(self.socket_path)

    # -- connections -----------------------------------------------------------

    async def _accept(self, reader: asyncio.StreamReader,
                      writer: asyncio.StreamWriter) -> None:
        # Say why before closing. An abrupt close reaches the worker as a reset
        # from the socket layer, which tells its operator nothing about which of
        # the several reasons applied.
        if len(self.workers) >= MAX_WORKERS:
            await self._refuse_and_close(
                writer, ipc.Refusal.NOT_READY,
                f"this daemon already serves {MAX_WORKERS} workers")
            return
        sock = writer.get_extra_info("socket")
        if sock is not None and not self._same_user(sock):
            await self._refuse_and_close(
                writer, ipc.Refusal.NOT_READY,
                "this socket serves only the user running the daemon")
            return
        conn = None
        try:
            conn = await self._handshake(reader, writer)
            while True:
                message = await ipc.read_message(reader)
                await self._handle(conn, message)
        except (ipc.IpcError, asyncio.IncompleteReadError, ConnectionError,
                OSError, asyncio.CancelledError):
            pass
        finally:
            if conn is not None:
                self.workers.pop(conn.worker_id, None)
            with contextlib.suppress(Exception):
                writer.close()

    async def _refuse_and_close(self, writer, code: ipc.Refusal,
                                reason: str) -> None:
        self.refused += 1
        with contextlib.suppress(Exception):
            writer.write(ipc.Refused(code=code, reason=reason).to_frame())
            await asyncio.wait_for(writer.drain(), timeout=2.0)
        with contextlib.suppress(Exception):
            writer.close()

    @staticmethod
    def _same_user(sock) -> bool:
        """Only this process's owner may attach a worker.

        SO_PEERCRED is the kernel's answer, not the caller's, which is the only
        kind worth asking for. Where it is unavailable the check passes — a
        platform without it is not made safer by pretending otherwise.
        """
        try:
            raw = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED,
                                  struct.calcsize("3i"))
        except (OSError, AttributeError):
            return True
        _, uid, _ = struct.unpack("3i", raw)
        return uid == os.getuid()

    async def _handshake(self, reader, writer) -> WorkerConn:
        message = await ipc.read_message(reader)
        if not isinstance(message, ipc.Hello):
            raise ipc.IpcError("ipc: expected hello")
        if message.version != ipc.IPC_VERSION:
            raise ipc.IpcError(f"ipc: worker speaks version {message.version}")

        from ..consensus import genesis as genesis_module
        network = self._network_name()
        if (message.genesis_hash.hex() != genesis_module.GENESIS_HASH[network]
                or message.policy_hash.hex() != genesis_module.POLICY_HASH[network]):
            # A worker on a different commit produces contributions nobody can
            # use and cannot be told apart from one that is cheating.
            raise ipc.IpcError("ipc: worker is on different consensus rules")

        conn = WorkerConn(worker_id=self._next_worker_id, reader=reader,
                          writer=writer, user_agent=message.user_agent)
        self._next_worker_id += 1
        self.workers[conn.worker_id] = conn
        await self._send(conn, ipc.Welcome(
            worker_id=conn.worker_id,
            network_magic=self.round_desc.network_magic,
            parameter_count=self.round_desc.model.parameter_count(),
            contribution_bits=self.round_desc.numerics.contribution_format.value))
        self._log(f"worker {conn.worker_id} attached ({conn.user_agent or '?'})")
        return conn

    def _network_name(self) -> str:
        from ..consensus.params import NETWORKS
        for name, (rd, _) in NETWORKS.items():
            if rd.network_magic == self.round_desc.network_magic:
                return name
        raise ipc.IpcError("ipc: this daemon's round belongs to no known network")

    async def _send(self, conn: WorkerConn, message: ipc.Message) -> None:
        conn.writer.write(message.to_frame())
        await conn.writer.drain()

    # -- the conversation -------------------------------------------------------

    async def _handle(self, conn: WorkerConn, message: ipc.Message) -> None:
        if isinstance(message, ipc.GetAssignment):
            await self._on_get_assignment(conn)
        elif isinstance(message, ipc.Submit):
            await self._on_submit(conn, message)
        elif isinstance(message, ipc.GetChunk):
            await self._on_get_chunk(conn, message)
        elif isinstance(message, ipc.Applied):
            await self._on_applied(conn, message)
        elif isinstance(message, ipc.Hello):
            raise ipc.IpcError("ipc: a second hello")

    async def _on_get_assignment(self, conn: WorkerConn) -> None:
        head = self.chain.head
        outer_step = head.height + 1

        # Anything already aggregated that THIS worker has not applied comes
        # first, whatever step it belongs to. A worker that skipped an outer
        # step holds weights the rest of the network has moved past, and would
        # spend the next round training from them.
        for step in sorted(self._pending_apply):
            if conn.worker_id not in self._applied_at.get(step, {}):
                await self._send(conn, self._pending_apply[step])
                return

        # Production comes before more training. A round whose quorum is met and
        # which nobody applies is a round that never closes, and every worker
        # would keep contributing to a step that can no longer accept them.
        if self.ready_to_produce(outer_step):
            await self._ask_to_produce(conn, outer_step)
            return

        # One assignment per worker per step, whether it is still training or
        # has already submitted. Two would mean the same worker's id feeding the
        # schedule twice for one step, so the second run trains on exactly the
        # data the first did and adds nothing but its own weight to the mean.
        if (outer_step, conn.worker_id) in self.contributions:
            await self._send(conn, ipc.NoWork(
                reason=f"you have already contributed to step {outer_step}",
                retry_ms=5000))
            return
        outstanding = [a for a in conn.assignments.values()
                       if a.outer_step == outer_step]
        if outstanding:
            await self._send(conn, ipc.NoWork(
                reason=f"you already hold assignment {outstanding[0].assignment_id} "
                       f"for step {outer_step}", retry_ms=5000))
            return

        assignment_id = self._next_assignment_id
        self._next_assignment_id += 1
        assignment = ipc.Assignment(
            assignment_id=assignment_id,
            round_id=self.round_desc.round_id,
            outer_step=outer_step,
            base_checkpoint=head.id,
            base_weights_hash=head.header.weights_hash,
            inner_steps=self.policy.inner_steps,
            micro_batch=self.policy.micro_batch,
            deadline_ms=int(time.time() * 1000) + self.policy.round_deadline_ms)
        conn.assignments[assignment_id] = assignment
        await self._send(conn, assignment)

    async def _ask_to_produce(self, conn: WorkerConn, outer_step: int) -> None:
        from ..diloco.aggregate import average_contributions
        from ..diloco.quantize import pack

        parts = self.contributions_at(outer_step)
        mean, exp = average_contributions(parts)
        fmt = self.round_desc.numerics.contribution_format
        if int(np.abs(mean).max()) > fmt.max_magnitude:
            # The bound that lets the aggregate ride in the contribution format
            # is arithmetic, not hope — but it is checked, because a violation
            # would silently clamp the update rather than fail.
            raise ipc.IpcError(
                f"produce: aggregate reached {int(np.abs(mean).max())}, past "
                f"int{fmt.value}'s {fmt.max_magnitude}")
        # Held, and handed out when each worker NEXT ASKS. Every worker must
        # apply — DiLoCo's outer step is not the producer's privilege, and one
        # that skipped it would start the next round from weights the network
        # has moved past — but pushing it to all of them was the obvious design
        # and it is wrong. This channel is strict request-and-response, so an
        # unsolicited message lands in the middle of whatever exchange the other
        # worker was in and desynchronises it. Measured as `expected accepted,
        # got NoWork` on a worker that had submitted perfectly correctly.
        # Proved by WorkerServiceTests.EveryWorkerIsHandedTheApplyWhenItAsks.
        self._applied_at.setdefault(outer_step, {})
        self._pending_apply[outer_step] = ipc.Apply(
            outer_step=outer_step, scale_exp=exp, value_count=int(mean.size),
            momentum_q16=self.policy.outer_momentum_q16,
            lr_q16=self.policy.outer_lr_q16, nesterov=self.policy.nesterov,
            packed=pack(mean, fmt))
        await self._send(conn, self._pending_apply[outer_step])
        self._log(f"step {outer_step}: applying {len(parts)} contribution(s)")

    async def _on_applied(self, conn: WorkerConn, message: ipc.Applied) -> None:
        from ..consensus.objects import CheckpointHeader
        from ..diloco.aggregate import contribution_root

        seen = self._applied_at.get(message.outer_step)
        if seen is None:
            await self._refuse(conn, ipc.Refusal.UNKNOWN_ASSIGNMENT,
                               f"nobody was asked to apply step {message.outer_step}")
            return
        seen[conn.worker_id] = (message.weights_hash, message.optimizer_state_hash)

        # Every applier should reach the same two hashes: same aggregate, same
        # base, integer arithmetic. A disagreement is not a peer being
        # dishonest — it is this node's own workers computing differently, which
        # is worth shouting about rather than resolving quietly.
        distinct = set(seen.values())
        if len(distinct) > 1:
            self._log(f"step {message.outer_step}: WORKERS DISAGREE — "
                      + ", ".join(f"{w}:{h[0].hex()[:12]}" for w, h in seen.items()))

        if self.chain.height >= message.outer_step:
            # Already built from somebody else's report. Still answered, because
            # every worker message gets exactly one reply.
            await self._send(conn, ipc.Accepted(
                contribution_id=self.chain.at_height(message.outer_step).id))
            return

        parts = self.contributions_at(message.outer_step)
        header = CheckpointHeader(
            round_id=self.round_desc.round_id, outer_step=message.outer_step,
            parent=self.chain.head.id,
            weights_hash=message.weights_hash,
            optimizer_state_hash=message.optimizer_state_hash,
            contribution_root=contribution_root(
                sorted(self._contribution_ids(message.outer_step))),
            producer_id=conn.worker_id, timestamp_ms=int(time.time() * 1000))

        outcome = self.chain.add(header)
        self.produced += 1
        # The contributions are spent. Keeping them would let the next round
        # aggregate work that was already applied.
        for key in [k for k in self.contributions if k[0] == message.outer_step]:
            del self.contributions[key]

        if self.service is not None:
            self.service.node.height = self.chain.height
            with contextlib.suppress(Exception):
                self.service.publish(header)
        self._log(f"step {message.outer_step}: checkpoint {header.id.hex()[:16]} "
                  f"({outcome.value}), weights {message.weights_hash.hex()[:16]}, "
                  f"{len(parts)} contribution(s)")
        self._retire_applied()
        await self._send(conn, ipc.Accepted(contribution_id=header.id))

    def _retire_applied(self) -> None:
        """Drop an Apply once every attached worker has taken it.

        Kept until then, and no longer: a worker that attaches afterwards has
        weights from wherever it started, and replaying one outer step at it
        would not put it on the chain — it needs the checkpoint, which is a
        different problem and not this one.
        """
        attached = set(self.workers)
        for step in list(self._pending_apply):
            if attached and attached <= set(self._applied_at.get(step, {})):
                del self._pending_apply[step]

    def _contribution_ids(self, outer_step: int) -> list:
        return [i for (step, i) in self._headers if step == outer_step]

    async def _on_submit(self, conn: WorkerConn, message: ipc.Submit) -> None:
        assignment = conn.assignments.get(message.assignment_id)
        if assignment is None:
            await self._refuse(conn, ipc.Refusal.UNKNOWN_ASSIGNMENT,
                               f"assignment {message.assignment_id} is not one of yours")
            return
        if assignment.base_checkpoint != self.chain.head.id:
            # The round moved while this worker trained. Its work is against a
            # checkpoint that is no longer the head, so it is not wrong — it is
            # late, and saying which matters to whoever reads the log.
            await self._refuse(conn, ipc.Refusal.STALE_ASSIGNMENT,
                               "the chain moved on while you were training")
            return

        expected = self.round_desc.model.parameter_count()
        if message.value_count != expected:
            await self._refuse(
                conn, ipc.Refusal.BAD_PAYLOAD,
                f"{message.value_count} values against {expected} parameters")
            return

        fmt = self.round_desc.numerics.contribution_format
        want = ipc.expected_payload_bytes(expected, fmt.value)
        if len(message.packed) != want:
            await self._refuse(
                conn, ipc.Refusal.BAD_PAYLOAD,
                f"{len(message.packed)} bytes against {want} for int{fmt.value}")
            return

        try:
            values = unpack(message.packed, expected, fmt)
        except Exception as exc:
            await self._refuse(conn, ipc.Refusal.BAD_PAYLOAD, str(exc))
            return

        header = ContributionHeader(
            round_id=assignment.round_id, outer_step=assignment.outer_step,
            worker_id=conn.worker_id, assignment_id=assignment.assignment_id,
            base_checkpoint=assignment.base_checkpoint,
            base_weights_hash=assignment.base_weights_hash,
            payload_hash=_payload_hash(values, message.scale_exp),
            payload_bytes=len(message.packed), scale_exp=message.scale_exp,
            contribution_format=fmt)

        self.contributions[(assignment.outer_step, conn.worker_id)] = Contribution(
            values=values, scale_exp=message.scale_exp, fmt=fmt,
            worker_id=conn.worker_id)
        del conn.assignments[message.assignment_id]
        self._headers.append((assignment.outer_step, header.id))
        conn.submitted += 1
        self.accepted += 1

        if self.service is not None:
            with contextlib.suppress(Exception):
                self.service.publish(header)

        self._log(f"worker {conn.worker_id} contributed {header.id.hex()[:16]} "
                  f"at step {assignment.outer_step} (loss {message.final_loss:.4f})")
        await self._send(conn, ipc.Accepted(contribution_id=header.id))

    async def _refuse(self, conn: WorkerConn, code: ipc.Refusal, reason: str) -> None:
        # "Ask again" is not a rejection. Counting deferrals as refusals made a
        # healthy node report two thousand rejections an hour in the
        # implementation this replaces, and made an operator's only health
        # signal read exactly backwards.
        if code is ipc.Refusal.UNAVAILABLE:
            self.deferred += 1
        else:
            self.refused += 1
            self._log(f"worker {conn.worker_id} refused ({code.name}): {reason}")
        await self._send(conn, ipc.Refused(code=code, reason=reason))

    async def _on_get_chunk(self, conn: WorkerConn, message: ipc.GetChunk) -> None:
        if self.corpus is None:
            await self._refuse(conn, ipc.Refusal.NOT_READY, "this node holds no corpus")
            return
        data = self.corpus.get(message.index)
        if data is None:
            self.corpus.request(message.index)
            await self._refuse(conn, ipc.Refusal.UNAVAILABLE,
                               f"chunk {message.index} is being fetched; ask again")
            return
        await self._send(conn, ipc.Chunk(index=message.index, data=data))

    # -- what the round has ------------------------------------------------------

    def contributions_at(self, outer_step: int) -> list[Contribution]:
        return [c for (step, _), c in sorted(self.contributions.items())
                if step == outer_step]

    def ready_to_produce(self, outer_step: int) -> bool:
        return len(self.contributions_at(outer_step)) >= self.policy.min_contributors

    def status(self) -> str:
        return (f"workers {len(self.workers)} — {self.accepted} accepted, "
                f"{self.refused} refused, {self.deferred} deferred, "
                f"{self.produced} produced — "
                f"{len(self.contributions)} contribution(s) held")


def _payload_hash(values: np.ndarray, scale_exp: int) -> bytes:
    import hashlib
    return hashlib.sha3_256(
        scale_exp.to_bytes(8, "big", signed=True)
        + np.ascontiguousarray(values, dtype=np.int64).astype(">i8").tobytes()
    ).digest()

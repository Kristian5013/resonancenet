"""The worker's side of the channel: blocking, because a worker does one thing.

A trainer's life is a straight line — ask, train for twenty minutes, submit,
repeat — and asyncio buys nothing against that shape while costing a whole
concurrency model to reason about. The daemon is the side that has to serve
many things at once, and it is the side that is asynchronous.

WHAT THIS REFUSES TO DO. It does not choose its worker id, its data, or its
schedule. All three come from the daemon, which gets them from consensus. A
worker that could choose any of them would produce work nobody could check, and
would be indistinguishable from one that was cheating.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass

from ..net import framing
from . import ipc


class WorkerError(Exception):
    pass


class AnchorMismatch(WorkerError):
    """The daemon and this worker disagree about what is being trained."""


@dataclass
class DaemonClient:
    """A connection to the local daemon."""

    socket_path: str
    genesis_hash: bytes
    policy_hash: bytes
    user_agent: str = "rnet-worker/0.2"
    timeout: float = 300.0

    _sock: socket.socket = None
    worker_id: int = 0
    parameter_count: int = 0
    contribution_bits: int = 8

    def __enter__(self) -> "DaemonClient":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def connect(self) -> None:
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(self.timeout)
        try:
            self._sock.connect(self.socket_path)
        except OSError as exc:
            raise WorkerError(
                f"worker: no daemon at {self.socket_path} ({exc}). Start one:\n"
                f"    python -m rnet daemon --network <network>") from exc

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    # -- framing ---------------------------------------------------------------

    def _send(self, message: ipc.Message) -> None:
        self._sock.sendall(message.to_frame())

    def _recv_exactly(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            try:
                chunk = self._sock.recv(min(1 << 20, n - len(out)))
            except OSError as exc:
                # A reset is what an abrupt close looks like from here. Letting
                # the raw socket exception out would make every caller handle
                # errno, and would tell the operator nothing.
                raise WorkerError(
                    f"worker: the daemon hung up ({exc})") from exc
            if not chunk:
                raise WorkerError("worker: the daemon closed the connection")
            out += chunk
        return bytes(out)

    def _recv(self) -> ipc.Message:
        head = self._recv_exactly(framing.HEADER_SIZE)
        try:
            header = framing.parse_header(head, ipc.IPC_MAGIC, ipc.Command,
                                          max_payload=ipc.MAX_IPC_PAYLOAD)
            payload = self._recv_exactly(header.length) if header.length else b""
            framing.check_payload(header, payload)
        except framing.FramingError as exc:
            raise WorkerError(str(exc)) from exc
        return ipc.decode(header.command, payload)

    # -- the conversation --------------------------------------------------------

    def hello(self) -> ipc.Welcome:
        """Introduce, and be told which worker this is.

        The anchors are checked by both sides. A mismatch ends the connection
        here rather than after twenty minutes of GPU spent on an update nobody
        can use.
        """
        self._send(ipc.Hello(version=ipc.IPC_VERSION,
                             genesis_hash=self.genesis_hash,
                             policy_hash=self.policy_hash,
                             user_agent=self.user_agent))
        reply = self._recv()
        if isinstance(reply, ipc.Refused):
            raise AnchorMismatch(reply.reason)
        if not isinstance(reply, ipc.Welcome):
            raise WorkerError(f"worker: expected welcome, got {type(reply).__name__}")
        self.worker_id = reply.worker_id
        self.parameter_count = reply.parameter_count
        self.contribution_bits = reply.contribution_bits
        return reply

    def next_work(self):
        """Ask what to do next. Four answers, in priority order.

        A Verify comes first — it is cheap against a round, it expires, and a
        node that put training first would let every challenge lapse under load,
        which is exactly when a cheat would choose to act. Then an Apply, because
        a round nobody applies never closes. Then an Assignment. Then NoWork.
        """
        self._send(ipc.GetAssignment())
        reply = self._recv()
        if isinstance(reply, (ipc.Assignment, ipc.NoWork, ipc.Apply, ipc.Verify)):
            return reply
        raise WorkerError(f"worker: expected work, got {type(reply).__name__}")

    def verdict(self, message: ipc.Verdict) -> ipc.Accepted:
        self._send(message)
        reply = self._recv()
        if isinstance(reply, ipc.Refused):
            raise WorkerError(f"worker: verdict refused ({reply.code.name}): "
                              f"{reply.reason}")
        if not isinstance(reply, ipc.Accepted):
            raise WorkerError(f"worker: expected accepted, got {type(reply).__name__}")
        return reply

    def applied(self, outer_step: int, weights_hash: bytes,
                optimizer_state_hash: bytes) -> ipc.Accepted:
        """Report what the outer step produced, and wait to be told it landed.

        Every message a worker sends gets exactly one reply — no exceptions,
        because a channel where some messages are answered and some are not is a
        channel where one unanswered message shifts every reply after it by one.
        That is not a hypothetical: it is how a correct submission came back as
        `expected accepted, got NoWork`.
        """
        self._send(ipc.Applied(outer_step=outer_step, weights_hash=weights_hash,
                               optimizer_state_hash=optimizer_state_hash))
        reply = self._recv()
        if isinstance(reply, ipc.Refused):
            raise WorkerError(f"worker: apply refused ({reply.code.name}): "
                              f"{reply.reason}")
        if not isinstance(reply, ipc.Accepted):
            raise WorkerError(f"worker: expected accepted, got {type(reply).__name__}")
        return reply

    def submit(self, assignment_id: int, payload: bytes, scale_exp: int,
               value_count: int, final_loss: float) -> ipc.Accepted:
        self._send(ipc.Submit(assignment_id=assignment_id, scale_exp=scale_exp,
                              value_count=value_count, final_loss=final_loss,
                              packed=payload))
        reply = self._recv()
        if isinstance(reply, ipc.Refused):
            raise WorkerError(f"worker: contribution refused "
                              f"({reply.code.name}): {reply.reason}")
        if not isinstance(reply, ipc.Accepted):
            raise WorkerError(f"worker: expected accepted, got {type(reply).__name__}")
        return reply

    def chunk(self, index: int) -> bytes | None:
        """One corpus chunk, or None when the daemon is still fetching it.

        None rather than an exception, because "not yet" is the ordinary answer
        while a prefetch is in flight, and a worker that treated it as an error
        would treat the mechanism working as the mechanism failing.
        """
        self._send(ipc.GetChunk(index=index))
        reply = self._recv()
        if isinstance(reply, ipc.Chunk):
            return reply.data
        if isinstance(reply, ipc.Refused) and reply.code is ipc.Refusal.UNAVAILABLE:
            return None
        if isinstance(reply, ipc.Refused):
            raise WorkerError(f"worker: chunk {index} refused: {reply.reason}")
        raise WorkerError(f"worker: expected a chunk, got {type(reply).__name__}")

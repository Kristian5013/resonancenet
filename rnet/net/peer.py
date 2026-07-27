"""One connection: its handshake, its queues, and how it ends.

A peer is a hostile input with a socket attached. Everything here is written on
that basis — every read is bounded before it happens, every queue has a
ceiling, every timer fires whether or not the far end cooperates.

THE THREE WAYS A CONNECTION ENDS, and they are different on purpose:

  * PROTOCOL ERROR — the peer said something this protocol cannot express. The
    conversation is over immediately; there is nothing to negotiate about a
    frame that does not parse.
  * MISBEHAVIOUR — the peer said something well-formed and wrong. Scored, so
    that a peer with one bad object is not treated like one sending nothing
    else, and disconnected with a ban when the score crosses.
  * IDLE — the peer stopped answering. A ping goes unanswered for long enough
    and the socket is closed, because a connection that cannot be measured is
    a slot that could have gone to a peer that answers.

THE DISCONNECT REASON IS FLUSHED BEFORE THE SOCKET CLOSES. This looks like
politeness and is not. The implementation this replaces tore down the peer
before its send queue drained, so every stated reason was silently discarded —
for months, invisibly, because the code that wrote the reason looked correct
and the code that dropped it was somewhere else. Proved by
PeerTests.TheDisconnectReasonIsSentBeforeTheSocketCloses.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum

from . import protocol as P
from .address import NetAddress, Services

# A peer that cannot keep up with this many queued messages is not going to
# catch up. Buffering further would mean one slow peer costs memory in
# proportion to how slow it is, which is a denial of service with no attacker.
MAX_SEND_QUEUE = 1000

HANDSHAKE_TIMEOUT_S = 20.0
PING_INTERVAL_S = 120.0
PING_TIMEOUT_S = 60.0

# Misbehaviour scores. A peer reaching the threshold is disconnected and its
# address is banned for a while.
BAN_THRESHOLD = 100


class Score:
    """What each kind of wrongness costs.

    Graded rather than uniform because the kinds differ in what they imply. An
    unsolicited object is probably a race with our own request; an object whose
    hash does not match what was asked for cannot be anything but deliberate.
    """

    UNSOLICITED_OBJECT = 5
    UNKNOWN_OBJECT = 10
    WRONG_HASH = 50
    BAD_PROOF = 50
    FLOOD = 20
    STALE_HANDSHAKE = 100


class State(Enum):
    CONNECTING = "connecting"
    HANDSHAKING = "handshaking"
    READY = "ready"
    CLOSED = "closed"


class Disconnect(Exception):
    """Raised to end a connection with a stated reason."""

    def __init__(self, reason: str, code: P.RejectCode = P.RejectCode.INVALID,
                 command: P.Command = P.Command.VERSION):
        super().__init__(reason)
        self.reason = reason
        self.code = code
        self.command = command


@dataclass
class Peer:
    """A connection, from the socket up to parsed messages."""

    address: NetAddress
    magic: int
    inbound: bool
    reader: asyncio.StreamReader | None = None
    writer: asyncio.StreamWriter | None = None

    state: State = State.CONNECTING
    services: Services = Services.NONE
    version: int = 0
    user_agent: str = ""
    start_height: int = 0
    nonce: int = 0

    # The handshake is two exchanges, not one, and both halves must land before
    # a peer is usable. Tracking them separately is what stops a node from
    # declaring itself ready on the strength of what IT said.
    got_version: bool = False
    got_verack: bool = False

    misbehaviour: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    messages_sent: int = 0
    messages_received: int = 0

    connected_at: float = field(default_factory=time.monotonic)
    last_received: float = field(default_factory=time.monotonic)
    last_ping_sent: float = 0.0
    ping_nonce: int = 0
    latency_ms: float | None = None

    _queue: asyncio.Queue | None = None
    _closing: bool = False

    # -- lifecycle -----------------------------------------------------------

    def attach(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
               ) -> None:
        self.reader = reader
        self.writer = writer
        self._queue = asyncio.Queue(maxsize=MAX_SEND_QUEUE)
        self.state = State.HANDSHAKING

    def send(self, message: P.Message) -> bool:
        """Queue a message. False if the peer is too far behind to take it.

        Never blocks and never grows without bound: a full queue means this peer
        gets dropped, not that everyone else waits for it.
        """
        if self._closing or self._queue is None:
            return False
        try:
            self._queue.put_nowait(message)
            return True
        except asyncio.QueueFull:
            return False

    async def writer_loop(self) -> None:
        """Drains the queue until the peer closes.

        Kept separate from the read loop so that a peer which stops reading
        cannot stall the node's handling of everyone else — the queue fills, the
        send fails, and the peer is dropped by the rule above.
        """
        assert self._queue is not None and self.writer is not None
        try:
            while True:
                message = await self._queue.get()
                if message is None:
                    break
                raw = message.to_frame(self.magic)
                self.writer.write(raw)
                await self.writer.drain()
                self.bytes_sent += len(raw)
                self.messages_sent += 1
        except (ConnectionError, asyncio.CancelledError):
            pass

    async def read_message(self) -> P.Message:
        """One message, with every bound applied before the bytes are taken."""
        assert self.reader is not None
        head = await self.reader.readexactly(P.HEADER_SIZE)
        header = P.parse_header(head, self.magic)
        payload = (await self.reader.readexactly(header.length)
                   if header.length else b"")
        P.check_payload(header, payload)
        self.bytes_received += P.HEADER_SIZE + header.length
        self.messages_received += 1
        self.last_received = time.monotonic()
        return P.decode(header.command, payload)

    async def close(self, reason: str = "", code: P.RejectCode | None = None,
                    command: P.Command = P.Command.VERSION) -> None:
        """End the connection, having actually said why.

        The reject is written directly rather than queued: the queue is drained
        by a task that is about to be cancelled, and putting the last message
        behind it is how the reason got lost in the implementation this
        replaces.
        """
        if self._closing:
            return
        self._closing = True
        if reason and self.writer is not None and not self.writer.is_closing():
            try:
                raw = P.Reject(command=command,
                               code=code or P.RejectCode.INVALID,
                               reason=reason).to_frame(self.magic)
                self.writer.write(raw)
                await asyncio.wait_for(self.writer.drain(), timeout=2.0)
            except (ConnectionError, asyncio.TimeoutError, OSError):
                pass
        if self._queue is not None:
            try:
                self._queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
        if self.writer is not None:
            try:
                self.writer.close()
            except (ConnectionError, OSError):
                pass
        self.state = State.CLOSED

    # -- behaviour ------------------------------------------------------------

    def penalise(self, points: int, why: str) -> bool:
        """Record misbehaviour. True when the peer has earned a ban."""
        self.misbehaviour += points
        return self.misbehaviour >= BAN_THRESHOLD

    @property
    def banned(self) -> bool:
        return self.misbehaviour >= BAN_THRESHOLD

    # -- liveness -------------------------------------------------------------

    def needs_ping(self, now: float) -> bool:
        return (self.state is State.READY and self.ping_nonce == 0
                and now - self.last_received > PING_INTERVAL_S)

    def is_stalled(self, now: float) -> bool:
        """Whether this connection has stopped answering.

        A handshake that never finishes is stalled too — otherwise a peer that
        connects and says nothing holds a slot forever, which is the cheapest
        denial of service there is.
        """
        if self.state is State.HANDSHAKING:
            return now - self.connected_at > HANDSHAKE_TIMEOUT_S
        if self.ping_nonce and now - self.last_ping_sent > PING_TIMEOUT_S:
            return True
        return False

    def on_pong(self, nonce: int, now: float) -> bool:
        """True if this pong answers the ping we sent.

        A pong with the wrong nonce is not an answer; accepting it would let a
        peer keep a connection alive without ever having received our ping.
        """
        if nonce != self.ping_nonce or self.ping_nonce == 0:
            return False
        self.latency_ms = (now - self.last_ping_sent) * 1000.0
        self.ping_nonce = 0
        return True

    def describe(self) -> str:
        direction = "in " if self.inbound else "out"
        latency = f"{self.latency_ms:.0f}ms" if self.latency_ms else "—"
        return (f"{direction} {self.address} {self.user_agent or '?'} "
                f"h={self.start_height} {latency} "
                f"{self.bytes_sent // 1024}K↑ {self.bytes_received // 1024}K↓"
                + (f" score={self.misbehaviour}" if self.misbehaviour else ""))

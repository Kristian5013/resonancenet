"""The node: listens on both families, dials out, and keeps the peers honest.

WHAT DUAL-STACK MEANS HERE. Two listening sockets, bound separately: 0.0.0.0
and :: with IPV6_V6ONLY set. Not one socket with V6ONLY off, which is the
shorter way and the wrong one — whether a dual-stack socket accepts IPv4 depends
on the operating system, on a sysctl, and on the phase of the moon, so a node
that relied on it would listen on one family or two depending on the host and
have no way to tell which. Two sockets means the answer is the same everywhere,
and the log says exactly what is bound.

OUTBOUND SLOTS ARE ALLOCATED BY GROUP, NOT BY ADDRESS. Eight connections to
eight addresses in one /16 is one connection as far as an eclipse is concerned.
The address manager already buckets by group; this refuses to spend two outbound
slots on one group, which is the other half of the same defence.

A NODE MUST NOTICE WHEN IT DIALS ITSELF. Every outbound connection carries a
fresh nonce, and a VERSION arriving with a nonce this node issued means the far
end is this node. It then FORGETS the address rather than merely dropping the
connection — a node that only dropped it would redial forever, which is exactly
what the implementation this replaces did until it was found by hand.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import time
from dataclasses import dataclass, field

from . import protocol as P
from .address import NetAddress, Services, TimestampedAddress
from .addrman import AddrMan
from .peer import Disconnect, Peer, Score, State

DEFAULT_PORT = 9444

MAX_OUTBOUND = 8
MAX_INBOUND = 64

# How long a banned address stays refused. Long enough that a misbehaving peer
# is not simply back in a second; short enough that a node which was wrong about
# a peer is not wrong about it forever.
BAN_SECONDS = 24 * 60 * 60

CONNECT_TIMEOUT_S = 10.0
DIAL_INTERVAL_S = 2.0
HOUSEKEEPING_INTERVAL_S = 5.0


def listening_sockets(port: int, *, v4: bool = True, v6: bool = True) -> list:
    """One socket per family, each explicit about what it accepts.

    Returns whatever could be bound: a host with no IPv6 gets an IPv4 socket and
    a note, rather than a node that refuses to start.
    """
    made = []
    if v6:
        try:
            s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # The whole point: this socket is IPv6 and nothing else, so the IPv4
            # socket below can bind the same port without a conflict and the
            # behaviour does not depend on the host's default.
            s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            s.bind(("::", port))
            s.setblocking(False)
            made.append(s)
        except OSError:
            pass
    if v4:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", port))
            s.setblocking(False)
            made.append(s)
        except OSError:
            if not made:
                raise
    return made


@dataclass
class NodeConfig:
    magic: int
    genesis_hash: bytes
    policy_hash: bytes
    port: int = DEFAULT_PORT
    services: Services = Services.CHAIN
    user_agent: str = "rnet/0.2"
    max_outbound: int = MAX_OUTBOUND
    max_inbound: int = MAX_INBOUND
    listen_v4: bool = True
    listen_v6: bool = True


@dataclass
class Node:
    """The event loop. Owns the peers, the address table, and the listeners."""

    config: NodeConfig
    addrman: AddrMan = field(default_factory=AddrMan)
    peers: dict[int, Peer] = field(default_factory=dict)
    banned: dict[bytes, float] = field(default_factory=dict)

    height: int = 0
    on_message = None

    _next_id: int = 1
    # nonce -> the address it was dialled towards, or None when we issued it on
    # an inbound connection. Kept as a map rather than a set because the side
    # that DETECTS a self-connection is not the side that knows which address
    # caused it: the accepted peer sees the nonce, the dialling peer holds the
    # address, and the node is the only thing that can see both.
    _nonces: dict = field(default_factory=dict)
    _servers: list = field(default_factory=list)
    _tasks: set = field(default_factory=set)
    _running: bool = False
    _dialling: set = field(default_factory=set)

    # -- accounting ----------------------------------------------------------

    @property
    def outbound(self) -> list[Peer]:
        return [p for p in self.peers.values() if not p.inbound]

    @property
    def inbound(self) -> list[Peer]:
        return [p for p in self.peers.values() if p.inbound]

    @property
    def ready(self) -> list[Peer]:
        return [p for p in self.peers.values() if p.state is State.READY]

    def outbound_groups(self) -> set[bytes]:
        return {p.address.group for p in self.outbound} | {
            a.group for a in self._dialling}

    def is_banned(self, address: NetAddress, now: float) -> bool:
        until = self.banned.get(address.group)
        if until is None:
            return False
        if until < now:
            del self.banned[address.group]
            return False
        return True

    # -- running -------------------------------------------------------------

    async def start(self) -> list[str]:
        """Bind and begin. Returns what is actually listening, by name."""
        self._running = True
        bound = []
        for sock in listening_sockets(self.config.port, v4=self.config.listen_v4,
                                      v6=self.config.listen_v6):
            server = await asyncio.start_server(self._accept, sock=sock)
            self._servers.append(server)
            for s in server.sockets:
                bound.append(str(NetAddress.from_sockaddr(s.getsockname(), s.family)))
        self._spawn(self._housekeeping())
        self._spawn(self._dial_loop())
        return bound

    async def stop(self) -> None:
        self._running = False
        for server in self._servers:
            server.close()
        for peer in list(self.peers.values()):
            await peer.close()
        for task in list(self._tasks):
            task.cancel()
        for task in list(self._tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    # -- connections ----------------------------------------------------------

    async def _accept(self, reader: asyncio.StreamReader,
                      writer: asyncio.StreamWriter) -> None:
        sock = writer.get_extra_info("socket")
        address = NetAddress.from_sockaddr(writer.get_extra_info("peername"),
                                           sock.family if sock else socket.AF_INET6)
        now = time.monotonic()
        if len(self.inbound) >= self.config.max_inbound or self.is_banned(address, now):
            writer.close()
            return
        peer = Peer(address=address, magic=self.config.magic, inbound=True)
        peer.attach(reader, writer)
        self._register(peer)
        self._spawn(self._serve(peer))

    async def connect(self, address: NetAddress) -> Peer | None:
        """Dial one address. Returns the peer once its handshake completes.

        Refuses to open a second connection to somewhere already connected or
        being connected to. The dial loop and an explicit --connect can both
        reach for the same address in the same tick — the loop runs once before
        its first sleep, while the explicit dial is still only a scheduled
        future — and two sockets to one peer waste two outbound slots while
        looking like two peers in every count. Proved by
        NodeTests.ThereIsOnlyEverOneConnectionToAnAddress.
        """
        now = time.monotonic()
        if self.is_banned(address, now):
            return None
        if address in self._dialling:
            return None
        if any(p.address == address and not p.inbound for p in self.peers.values()):
            return None
        self._dialling.add(address)
        try:
            self.addrman.attempted(address, int(time.time() * 1000))
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(address.host, address.port,
                                        family=address.family),
                timeout=CONNECT_TIMEOUT_S)
        except (OSError, asyncio.TimeoutError):
            self._dialling.discard(address)
            return None
        peer = Peer(address=address, magic=self.config.magic, inbound=False)
        peer.attach(reader, writer)
        self._register(peer)
        self._spawn(self._serve(peer, initiate=True))
        self._dialling.discard(address)
        return peer

    def _register(self, peer: Peer) -> None:
        peer.id = self._next_id
        self.peers[self._next_id] = peer
        self._next_id += 1
        self._spawn(peer.writer_loop())

    async def _drop(self, peer: Peer, reason: str = "",
                    code: P.RejectCode | None = None,
                    command: P.Command = P.Command.VERSION) -> None:
        await peer.close(reason, code, command)
        self.peers.pop(getattr(peer, "id", -1), None)
        # A nonce outlives its peer only long enough to be useless. Dropping it
        # here is what keeps the map bounded by live connections rather than by
        # how long the node has been up.
        self._nonces.pop(peer.nonce, None)
        if peer.banned:
            self.banned[peer.address.group] = time.monotonic() + BAN_SECONDS

    # -- the conversation ------------------------------------------------------

    def _version_for(self, peer: Peer) -> P.Version:
        nonce = int.from_bytes(os.urandom(8), "big")
        # Only an outbound nonce names an address worth forgetting; an inbound
        # peer's address is the ephemeral port it dialled from, which the table
        # never held.
        self._nonces[nonce] = None if peer.inbound else peer.address
        peer.nonce = nonce
        return P.Version(
            version=P.PROTOCOL_VERSION, services=self.config.services,
            timestamp_ms=int(time.time() * 1000), receiver=peer.address,
            sender=NetAddress(bytes(16), self.config.port), nonce=nonce,
            user_agent=self.config.user_agent, start_height=self.height,
            genesis_hash=self.config.genesis_hash,
            policy_hash=self.config.policy_hash)

    async def _serve(self, peer: Peer, initiate: bool = False) -> None:
        try:
            if initiate:
                peer.send(self._version_for(peer))
            while self._running and peer.state is not State.CLOSED:
                message = await peer.read_message()
                await self._handle(peer, message)
        except Disconnect as exc:
            await self._drop(peer, exc.reason, exc.code, exc.command)
        except (P.ProtocolError,) as exc:
            await self._drop(peer, str(exc), P.RejectCode.MALFORMED)
        except (asyncio.IncompleteReadError, ConnectionError, OSError,
                asyncio.CancelledError):
            await self._drop(peer)

    async def _handle(self, peer: Peer, message: P.Message) -> None:
        if peer.state is State.HANDSHAKING:
            await self._handshake(peer, message)
            return

        if isinstance(message, P.Ping):
            peer.send(P.Pong(nonce=message.nonce))
        elif isinstance(message, P.Pong):
            peer.on_pong(message.nonce, time.monotonic())
        elif isinstance(message, P.GetAddr):
            addresses = self.addrman.to_gossip(int(time.time() * 1000), P.MAX_ADDR)
            if addresses:
                peer.send(P.Addr(tuple(addresses)))
        elif isinstance(message, P.Addr):
            now_ms = int(time.time() * 1000)
            for entry in message.addresses:
                self.addrman.add(entry, peer.address, now_ms)
        elif isinstance(message, P.Version):
            raise Disconnect("a second version message",
                             P.RejectCode.DUPLICATE, P.Command.VERSION)
        elif self.on_message is not None:
            await self.on_message(peer, message)

    async def _handshake(self, peer: Peer, message: P.Message) -> None:
        """Both halves must land before a peer is ready.

        Getting this wrong is subtle and quiet: marking a peer ready when we
        SEND our verack means the peer's verack arrives after the handshake is
        considered over, lands in the ordinary dispatch, matches nothing, and is
        dropped. Nothing errors. What breaks is whatever was supposed to happen
        on completion — here, address gossip simply never started.
        """
        if isinstance(message, P.Verack):
            if not peer.got_version:
                raise Disconnect("verack before version",
                                 P.RejectCode.MALFORMED, P.Command.VERACK)
            peer.got_verack = True
            self._finish_handshake(peer)
            return

        if not isinstance(message, P.Version):
            raise Disconnect("expected a version message first",
                             P.RejectCode.MALFORMED, P.Command.VERSION)

        # Ourselves. Dropping alone is not enough: the address is real and
        # answers, so a node that only dropped would redial it forever.
        if message.nonce in self._nonces:
            # A self-connection makes two peers and the detection lands on the
            # WRONG one: the accepted peer receives the nonce, but its address
            # is the ephemeral port the dial went out from — an address the
            # table never held. The address that has to be forgotten is the one
            # the dial was aimed at, which only the nonce map remembers. Get
            # this wrong and the node redials itself forever having apparently
            # handled it.
            dialled = self._nonces.get(message.nonce)
            if dialled is not None:
                self.addrman.forget(dialled)
            raise Disconnect("connected to self", P.RejectCode.DUPLICATE,
                             P.Command.VERSION)

        if message.genesis_hash != self.config.genesis_hash:
            raise Disconnect(
                f"genesis {message.genesis_hash.hex()[:16]} is not "
                f"{self.config.genesis_hash.hex()[:16]}",
                P.RejectCode.OBSOLETE, P.Command.VERSION)
        if message.policy_hash != self.config.policy_hash:
            raise Disconnect(
                f"policy {message.policy_hash.hex()[:16]} is not "
                f"{self.config.policy_hash.hex()[:16]}",
                P.RejectCode.OBSOLETE, P.Command.VERSION)

        peer.version = message.version
        peer.services = message.services
        peer.user_agent = message.user_agent
        peer.start_height = message.start_height

        peer.got_version = True
        if peer.inbound:
            # The receiver answers with its own version before acknowledging,
            # so the initiator learns who it reached even when the answer is a
            # refusal. Without this, a rejected dialler cannot tell a wrong
            # network from a dead socket.
            peer.send(self._version_for(peer))
        peer.send(P.Verack())
        self._finish_handshake(peer)

    def _finish_handshake(self, peer: Peer) -> None:
        if not (peer.got_version and peer.got_verack):
            return
        peer.state = State.READY
        if not peer.inbound:
            # Only an outbound connection proves the address is dialable, which
            # is the whole distinction between the two address tables. An
            # inbound peer's port is whatever ephemeral one it dialled from.
            self.addrman.connected(peer.address, int(time.time() * 1000))
        peer.send(P.GetAddr())

    # -- upkeep ----------------------------------------------------------------

    async def _dial_loop(self) -> None:
        while self._running:
            try:
                await self._dial_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            await asyncio.sleep(DIAL_INTERVAL_S)

    async def _dial_once(self) -> None:
        if len(self.outbound) + len(self._dialling) >= self.config.max_outbound:
            return
        now_ms = int(time.time() * 1000)
        taken = self.outbound_groups()
        exclude = {(p.address.ip, p.address.port) for p in self.peers.values()}
        for _ in range(8):
            address = self.addrman.select(now_ms, exclude=exclude)
            if address is None:
                return
            # One outbound slot per group: eight connections into one /16 is one
            # connection as far as an eclipse is concerned.
            if address.group in taken or self.is_banned(address, time.monotonic()):
                exclude.add((address.ip, address.port))
                continue
            await self.connect(address)
            return

    async def _housekeeping(self) -> None:
        while self._running:
            now = time.monotonic()
            for peer in list(self.peers.values()):
                if peer.is_stalled(now):
                    await self._drop(peer, "no answer", P.RejectCode.UNAVAILABLE,
                                     P.Command.PING)
                elif peer.needs_ping(now):
                    peer.ping_nonce = int.from_bytes(os.urandom(8), "big") or 1
                    peer.last_ping_sent = now
                    peer.send(P.Ping(nonce=peer.ping_nonce))
            await asyncio.sleep(HOUSEKEEPING_INTERVAL_S)

    # -- relay -----------------------------------------------------------------

    def broadcast(self, message: P.Message, *, skip: Peer | None = None) -> int:
        """Send to every ready peer. Returns how many took it.

        A peer whose queue is full does not take it and is not waited for —
        relay is best-effort by construction, because the alternative is one
        slow peer setting the pace for the network.
        """
        sent = 0
        for peer in self.ready:
            if peer is skip:
                continue
            if peer.send(message):
                sent += 1
        return sent

    def penalise(self, peer: Peer, points: int, why: str) -> None:
        if peer.penalise(points, why):
            self._spawn(self._drop(peer, f"misbehaving: {why}",
                                   P.RejectCode.INVALID))

    def status(self) -> str:
        ready = len(self.ready)
        return (f"peers {len(self.peers)} ({ready} ready, {len(self.inbound)} in, "
                f"{len(self.outbound)} out) — addresses {len(self.addrman)} "
                f"({self.addrman.tried_count} tried, {len(self.addrman.groups)} "
                f"groups) — banned {len(self.banned)}")

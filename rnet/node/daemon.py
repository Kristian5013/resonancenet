"""The daemon: a node you can actually leave running.

Everything the other modules build, assembled and given a directory to live in.
What it adds over the pieces is the unglamorous half — where state goes, what
survives a restart, what happens on a signal, and what an operator sees when
they look at it.

WHAT PERSISTS AND WHAT DOES NOT. The address table persists, because a node
that forgot its peers on every restart would rejoin the network through the
seeds every time and put the whole network's reachability on whoever runs them.
The bucket key persists with it, or the table would be re-bucketed on load and
the eclipse resistance it was built with would be thrown away. The chain does
not persist yet; that wants a store, and a half-written one is worse than none.

THE STATUS LINE IS THE INTERFACE. A daemon that runs silently is a daemon
nobody can tell is broken. Once a minute it says how many peers, how far along
the chain, and how much it has moved — enough that "it stopped doing anything"
is visible without attaching a debugger.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import time
from dataclasses import dataclass, field

from ..canon.stream import CanonError, Reader, Writer
from ..consensus import genesis as genesis_module
from ..consensus.init import hash_of_values
from ..consensus.objects import CheckpointHeader
from ..diloco.chain import Chain
from ..net.address import NetAddress, Services, TimestampedAddress
from ..net.addrman import AddrMan
from ..net.node import DEFAULT_PORT, Node, NodeConfig
from ..net.peer import State
from .objects import ObjectStore
from .service import Service

PEERS_FILE = "peers.dat"
PEERS_MAGIC = b"RNPR"
PEERS_VERSION = 1
MAX_PERSISTED = 20_000

STATUS_INTERVAL_S = 60.0


@dataclass
class DaemonConfig:
    network: str = "regtest"
    datadir: str = ""
    port: int = DEFAULT_PORT
    connect: tuple[str, ...] = ()
    listen_v4: bool = True
    listen_v6: bool = True
    max_outbound: int = 8
    max_inbound: int = 64
    status_interval_s: float = STATUS_INTERVAL_S

    def resolved_datadir(self) -> str:
        return self.datadir or os.path.join(
            os.path.expanduser("~"), ".rnet", self.network)


def save_peers(path: str, addrman: AddrMan, now_ms: int) -> int:
    """Write the table, key and all, atomically.

    The key goes in the file. Regenerating it on every start would re-bucket
    every address, which discards exactly the property the buckets exist for:
    an attacker's addresses would land in fresh buckets each time and the
    diversity built up over a long run would be thrown away with them.
    """
    entries = addrman.to_gossip(now_ms, MAX_PERSISTED)
    w = (Writer().raw(PEERS_MAGIC).u16(PEERS_VERSION)
         .raw(addrman.key).u32(len(entries)))
    for entry in entries:
        entry.serialize(w)
    raw = w.take()
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return len(entries)


def load_peers(path: str, now_ms: int) -> AddrMan:
    """Read the table back, or start a fresh one if there is nothing to read.

    A corrupt file is not an error worth stopping for: the address table is a
    cache of hearsay, and starting empty costs one round of seeding.
    """
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError:
        return AddrMan()
    try:
        r = Reader(raw)
        if r._take(4) != PEERS_MAGIC:
            return AddrMan()
        if r.u16() != PEERS_VERSION:
            return AddrMan()
        addrman = AddrMan(key=r._take(32))
        count = r.u32()
        if count > MAX_PERSISTED:
            return AddrMan()
        for _ in range(count):
            addrman.add(TimestampedAddress.parse_wire(r), None, now_ms)
        return addrman
    except (CanonError, ValueError):
        return AddrMan()


@dataclass
class Daemon:
    """One running node, with its directory and its lifetime."""

    config: DaemonConfig
    node: Node = None
    service: Service = None
    started_at: float = field(default_factory=time.monotonic)
    _stopping: asyncio.Event = None
    _log = print

    # -- setup ----------------------------------------------------------------

    def prepare(self) -> str:
        """Make the directory, emit the artifacts, and check them.

        Emitting on start rather than expecting them to be there means a fresh
        datadir works with no ceremony, and verifying them afterwards means a
        datadir carried over from a different build fails here rather than at
        the first handshake.
        """
        datadir = self.config.resolved_datadir()
        os.makedirs(datadir, exist_ok=True)
        genesis_module.verify_build(self.config.network)
        genesis_module.emit(self.config.network, datadir)
        return datadir

    def build(self) -> None:
        network = self.config.network
        datadir = self.prepare()
        now_ms = int(time.time() * 1000)

        round_desc = genesis_module.round_descriptor(network)
        policy = genesis_module.policy_descriptor(network)

        addrman = load_peers(os.path.join(datadir, PEERS_FILE), now_ms)
        self.node = Node(
            config=NodeConfig(
                magic=round_desc.network_magic,
                genesis_hash=bytes.fromhex(genesis_module.GENESIS_HASH[network]),
                policy_hash=bytes.fromhex(genesis_module.POLICY_HASH[network]),
                port=self.config.port,
                services=Services.CHAIN,
                user_agent=f"rnet/0.2/{network}",
                max_outbound=self.config.max_outbound,
                max_inbound=self.config.max_inbound,
                listen_v4=self.config.listen_v4,
                listen_v6=self.config.listen_v6),
            addrman=addrman)

        genesis_checkpoint = CheckpointHeader(
            round_id=round_desc.round_id, outer_step=0, parent=bytes(32),
            weights_hash=bytes.fromhex(genesis_module.WEIGHTS_HASH[network]),
            optimizer_state_hash=bytes(32), contribution_root=bytes(32),
            producer_id=0, timestamp_ms=0)
        chain = Chain(genesis_checkpoint, retained=policy.retained_checkpoints)

        self.service = Service(
            node=self.node, chain=chain, objects=ObjectStore(),
            dataset_root=round_desc.dataset_root,
            dataset_chunks=round_desc.dataset_chunks)
        self.service.install()
        self.node.height = chain.height

    # -- running ---------------------------------------------------------------

    async def run(self) -> int:
        self.build()
        datadir = self.config.resolved_datadir()
        network = self.config.network
        self._stopping = asyncio.Event()

        bound = await self.node.start()
        self._log(f"rnet {network}: listening on {', '.join(bound) or 'nothing'}")
        self._log(f"  datadir  {datadir}")
        self._log(f"  genesis  {genesis_module.GENESIS_HASH[network][:32]}…")
        self._log(f"  weights  {genesis_module.WEIGHTS_HASH[network][:32]}…")
        self._log(f"  peers    {len(self.node.addrman)} known")

        for target in self.config.connect:
            try:
                address = NetAddress.parse(target, default_port=DEFAULT_PORT)
            except Exception as exc:
                self._log(f"  connect  {target}: {exc}")
                continue
            # An explicit --connect is an instruction, not hearsay: it is often
            # a LAN address or one behind a tunnel, so it bypasses the
            # routability filter. It is never gossiped onward.
            self.node.addrman.add(
                TimestampedAddress(address, Services.CHAIN, int(time.time() * 1000)),
                None, int(time.time() * 1000), explicit=True)
            asyncio.ensure_future(self.node.connect(address))
            self._log(f"  connect  {address}")

        self._install_signals()
        status = asyncio.ensure_future(self._status_loop())
        try:
            await self._stopping.wait()
        finally:
            status.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await status
            await self.shutdown()
        return 0

    def _install_signals(self) -> None:
        loop = asyncio.get_event_loop()
        for name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, self._stopping.set)

    async def shutdown(self) -> None:
        """Save what is worth saving, then close.

        Peers first: if the save failed the node still stops, but a node that
        stopped and lost its table is a node that comes back needing the seeds.
        """
        datadir = self.config.resolved_datadir()
        try:
            saved = save_peers(os.path.join(datadir, PEERS_FILE),
                               self.node.addrman, int(time.time() * 1000))
            self._log(f"rnet: saved {saved} addresses")
        except OSError as exc:
            self._log(f"rnet: could not save addresses: {exc}")
        await self.node.stop()
        self._log("rnet: stopped")

    async def _status_loop(self) -> None:
        while True:
            await asyncio.sleep(self.config.status_interval_s)
            self._log(self.status())

    def status(self) -> str:
        uptime = time.monotonic() - self.started_at
        return (f"[{uptime / 60:6.1f}m] {self.node.status()} | "
                f"{self.service.status()}")

    def peer_lines(self) -> list[str]:
        return [f"  {p.describe()}" for p in self.node.peers.values()]

    def stop(self) -> None:
        if self._stopping is not None:
            self._stopping.set()

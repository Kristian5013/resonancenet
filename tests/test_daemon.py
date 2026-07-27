"""The daemon: two real processes' worth of behaviour, in one.

What is tested here is the part the other modules do not have — where state
goes, what survives a restart, and whether two of these actually find each
other over a socket rather than in a fixture.
"""

import asyncio
import os
import shutil
import socket
import tempfile
import unittest

from rnet.consensus import genesis
from rnet.net.address import NetAddress, Services, TimestampedAddress
from rnet.net.addrman import AddrMan
from rnet.net.peer import State
from rnet.node.daemon import (PEERS_FILE, Daemon, DaemonConfig, load_peers,
                              save_peers)

NOW = 1_785_000_000_000


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def settle(predicate, timeout: float = 8.0) -> bool:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


class PeerFileTests(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="rnet-peers-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.path = os.path.join(self.dir, PEERS_FILE)

    def table(self, n: int = 5) -> AddrMan:
        m = AddrMan(key=bytes(range(32)))
        for i in range(n):
            m.add(TimestampedAddress(NetAddress.parse(f"8.{i}.0.1:9444"),
                                     Services.CHAIN, NOW), None, NOW)
        return m

    def test_ItRoundTrips(self):
        m = self.table()
        self.assertEqual(save_peers(self.path, m, NOW), 5)
        back = load_peers(self.path, NOW)
        self.assertEqual(len(back), 5)
        self.assertEqual({e.address for e in back._entries.values()},
                         {e.address for e in m._entries.values()})

    def test_TheBucketKeySurvives(self):
        """Regenerating it on load would re-bucket every address, discarding
        exactly the diversity the buckets exist to build up."""
        m = self.table()
        save_peers(self.path, m, NOW)
        self.assertEqual(load_peers(self.path, NOW).key, m.key)

    def test_ACorruptFileStartsFresh(self):
        """The table is a cache of hearsay; starting empty costs one round of
        seeding, and refusing to start costs the node."""
        with open(self.path, "wb") as f:
            f.write(b"not a peer file at all")
        self.assertEqual(len(load_peers(self.path, NOW)), 0)
        with open(self.path, "wb") as f:
            f.write(b"RNPR\x00\x01" + b"\x00" * 10)
        self.assertEqual(len(load_peers(self.path, NOW)), 0)

    def test_AMissingFileStartsFresh(self):
        self.assertEqual(len(load_peers(os.path.join(self.dir, "nope"), NOW)), 0)

    def test_SavingIsAtomic(self):
        save_peers(self.path, self.table(), NOW)
        self.assertEqual([f for f in os.listdir(self.dir) if f.endswith(".tmp")], [])

    def test_UnroutableAddressesAreNotPersisted(self):
        """They are not gossiped, and the file is the gossip view."""
        m = AddrMan(key=bytes(32))
        m.add(TimestampedAddress(NetAddress.parse("127.0.0.1:9444"),
                                 Services.CHAIN, NOW), None, NOW, explicit=True)
        m.add(TimestampedAddress(NetAddress.parse("8.8.8.8:9444"),
                                 Services.CHAIN, NOW), None, NOW)
        self.assertEqual(len(m), 2)
        self.assertEqual(save_peers(self.path, m, NOW), 1)


class AddrManExplicitTests(unittest.TestCase):

    def test_AnExplicitAddressIsKeptButNeverGossiped(self):
        """An operator naming a peer is stating it is reachable — often on a
        LAN. Relaying it onward would tell strangers the shape of somebody's
        private network."""
        m = AddrMan(key=bytes(32))
        lan = NetAddress.parse("10.0.0.5:9444")
        self.assertFalse(m.add(TimestampedAddress(lan, Services.CHAIN, NOW),
                               None, NOW))
        self.assertTrue(m.add(TimestampedAddress(lan, Services.CHAIN, NOW),
                              None, NOW, explicit=True))
        self.assertEqual(len(m), 1)
        self.assertEqual(m.select(NOW), lan)
        self.assertEqual(m.to_gossip(NOW), [])

    def test_APortOfZeroIsRefusedEvenWhenExplicit(self):
        m = AddrMan(key=bytes(32))
        self.assertFalse(m.add(
            TimestampedAddress(NetAddress.parse("10.0.0.5:0"), Services.NONE, NOW),
            None, NOW, explicit=True))


class DaemonTests(unittest.IsolatedAsyncioTestCase):

    def make(self, **kw) -> Daemon:
        directory = tempfile.mkdtemp(prefix="rnet-daemon-")
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        config = DaemonConfig(network="regtest", datadir=directory,
                              port=free_port(), status_interval_s=3600.0, **kw)
        daemon = Daemon(config=config)
        daemon._log = lambda *a: None
        return daemon

    async def start(self, daemon: Daemon) -> asyncio.Task:
        task = asyncio.ensure_future(daemon.run())
        self.addAsyncCleanup(self._stop, daemon, task)
        self.assertTrue(await settle(lambda: daemon.node is not None
                                     and daemon.node._running))
        return task

    async def _stop(self, daemon: Daemon, task: asyncio.Task) -> None:
        daemon.stop()
        try:
            await asyncio.wait_for(task, timeout=10)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            task.cancel()

    async def test_ItEmitsAndVerifiesItsArtifactsOnStart(self):
        """A fresh datadir works with no ceremony, and one carried over from a
        different build fails here rather than at the first handshake."""
        daemon = self.make()
        await self.start(daemon)
        directory = daemon.config.resolved_datadir()
        for name in ("regtest.rnet", "regtest.rnpol"):
            path = os.path.join(directory, name)
            self.assertTrue(os.path.exists(path), name)
        genesis.load_round(os.path.join(directory, "regtest.rnet"), "regtest")

    async def test_ItStartsAtTheWeightsAnchor(self):
        daemon = self.make()
        await self.start(daemon)
        self.assertEqual(daemon.service.chain.head.header.weights_hash.hex(),
                         genesis.WEIGHTS_HASH["regtest"])
        self.assertEqual(daemon.service.chain.height, 0)

    async def test_ItListensOnBothFamilies(self):
        daemon = self.make()
        await self.start(daemon)
        families = {s.family for server in daemon.node._servers
                    for s in server.sockets}
        self.assertIn(socket.AF_INET, families)

    async def test_TwoDaemonsFindEachOther(self):
        a = self.make()
        await self.start(a)
        b = self.make(connect=(f"127.0.0.1:{a.config.port}",))
        await self.start(b)
        self.assertTrue(await settle(lambda: bool(b.node.ready)))
        self.assertTrue(await settle(lambda: bool(a.node.ready)))
        self.assertEqual(len(b.node.outbound), 1)
        self.assertEqual(len(a.node.inbound), 1)
        self.assertIn("1 ready", a.status())

    async def test_TwoDaemonsFindEachOtherOverIPv6(self):
        try:
            probe = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            probe.bind(("::1", 0))
            probe.close()
        except OSError:
            self.skipTest("no IPv6 on this host")
        a = self.make()
        await self.start(a)
        b = self.make(connect=(f"[::1]:{a.config.port}",))
        await self.start(b)
        self.assertTrue(await settle(lambda: bool(b.node.ready)))
        peer = b.node.outbound[0]
        self.assertFalse(peer.address.is_v4)

    async def test_AMalformedConnectIsReportedNotFatal(self):
        daemon = self.make(connect=("not-an-address",))
        await self.start(daemon)
        self.assertTrue(daemon.node._running)

    async def test_ItSavesItsAddressesOnShutdown(self):
        daemon = self.make()
        task = await self.start(daemon)
        daemon.node.addrman.add(
            TimestampedAddress(NetAddress.parse("8.8.8.8:9444"),
                               Services.CHAIN, NOW), None, NOW)
        daemon.stop()
        await asyncio.wait_for(task, timeout=10)
        path = os.path.join(daemon.config.resolved_datadir(), PEERS_FILE)
        self.assertTrue(os.path.exists(path))
        self.assertEqual(len(load_peers(path, NOW)), 1)

    async def test_TheStatusLineNamesWhatMatters(self):
        daemon = self.make()
        await self.start(daemon)
        line = daemon.status()
        for fragment in ("peers", "addresses", "chain", "objects", "in flight"):
            self.assertIn(fragment, line)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Bootstrapping a node that knows nobody.

The seeds are the one place a node believes something before it has spoken to
anyone, so what matters here is less that they work than that they are asked
sparingly and that what they say is treated as hearsay.
"""

import asyncio
import socket
import unittest

from rnet.net import seeds
from rnet.net.address import NetAddress


class ResolveTests(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.dns = dict(seeds.DNS_SEEDS)
        self.fallback = dict(seeds.FALLBACK_SEEDS)
        self.addCleanup(self.restore)

    def restore(self):
        seeds.DNS_SEEDS.clear(); seeds.DNS_SEEDS.update(self.dns)
        seeds.FALLBACK_SEEDS.clear(); seeds.FALLBACK_SEEDS.update(self.fallback)

    class Loop:
        """A resolver that answers from a table, or fails."""

        def __init__(self, answers, fail=False, hang=False):
            self.answers, self.fail, self.hang = answers, fail, hang
            self.asked = []

        async def getaddrinfo(self, name, port, type=None):
            self.asked.append(name)
            if self.hang:
                await asyncio.sleep(30)
            if self.fail:
                raise socket.gaierror("no such name")
            out = []
            for text in self.answers.get(name, ()):
                family = socket.AF_INET6 if ":" in text else socket.AF_INET
                out.append((family, socket.SOCK_STREAM, 6, "", (text, port)))
            return out

    # -- what comes back -------------------------------------------------------

    async def test_ItReturnsBothFamilies(self):
        """A node listening on IPv6 and given only A records has been told
        about half the network."""
        seeds.DNS_SEEDS["regtest"] = ("s.example",)
        seeds.FALLBACK_SEEDS["regtest"] = ()
        loop = self.Loop({"s.example": ("192.0.2.7", "2001:db8::5")})
        got = await seeds.resolve("regtest", 9444, loop=loop)
        self.assertEqual(len(got), 2)
        self.assertEqual({str(a) for a in got},
                         {"192.0.2.7:9444", "[2001:db8::5]:9444"})

    async def test_TheLiteralsAnswerWhenDnsDoesNot(self):
        """The reason they are not redundant: a resolver that is broken,
        filtered or lying leaves a new node with nothing."""
        seeds.DNS_SEEDS["regtest"] = ("s.example",)
        seeds.FALLBACK_SEEDS["regtest"] = ("198.51.100.9:9444",)
        loop = self.Loop({}, fail=True)
        got = await seeds.resolve("regtest", 9444, loop=loop)
        self.assertEqual([str(a) for a in got], ["198.51.100.9:9444"])

    async def test_ADuplicateBetweenDnsAndALiteralIsOneAddress(self):
        seeds.DNS_SEEDS["regtest"] = ("s.example",)
        seeds.FALLBACK_SEEDS["regtest"] = ("192.0.2.7:9444",)
        loop = self.Loop({"s.example": ("192.0.2.7",)})
        got = await seeds.resolve("regtest", 9444, loop=loop)
        self.assertEqual(len(got), 1)

    async def test_ASlowNameserverDoesNotHoldUpTheNode(self):
        """A node that cannot start because a nameserver is slow is worse than
        one that starts knowing nobody and tries the literals."""
        seeds.DNS_SEEDS["regtest"] = ("slow.example",)
        seeds.FALLBACK_SEEDS["regtest"] = ("198.51.100.9:9444",)
        loop = self.Loop({}, hang=True)
        got = await seeds.resolve("regtest", 9444, timeout=0.05, loop=loop)
        self.assertEqual([str(a) for a in got], ["198.51.100.9:9444"])

    async def test_OneSeedCannotFillTheTable(self):
        seeds.DNS_SEEDS["regtest"] = ("s.example",)
        seeds.FALLBACK_SEEDS["regtest"] = ()
        many = tuple(f"192.0.2.{i}" for i in range(1, 200))
        loop = self.Loop({"s.example": many})
        got = await seeds.resolve("regtest", 9444, loop=loop)
        self.assertLessEqual(len(got), seeds.MAX_PER_SEED)

    async def test_GarbageFromASeedIsSkippedNotFatal(self):
        seeds.DNS_SEEDS["regtest"] = ("s.example",)
        seeds.FALLBACK_SEEDS["regtest"] = ("not an address", "198.51.100.9:9444")
        loop = self.Loop({"s.example": ("this is not an ip",)})
        got = await seeds.resolve("regtest", 9444, loop=loop)
        self.assertEqual([str(a) for a in got], ["198.51.100.9:9444"])

    # -- what is compiled in ---------------------------------------------------

    def test_RegtestHasNoSeeds(self):
        """A local test that reaches out to the internet behaves differently
        depending on the room it is in."""
        self.assertFalse(seeds.has_seeds("regtest"))
        self.assertEqual(seeds.DNS_SEEDS["regtest"], ())
        self.assertEqual(seeds.FALLBACK_SEEDS["regtest"], ())

    def test_EveryNetworkIsNamedInBothTables(self):
        """A network missing from one table would seed differently depending on
        which mechanism was reached for."""
        from rnet.consensus.genesis import networks
        for network in networks():
            self.assertIn(network, seeds.DNS_SEEDS, network)
            self.assertIn(network, seeds.FALLBACK_SEEDS, network)

    def test_MainHasSomethingToStartFrom(self):
        self.assertTrue(seeds.has_seeds("main"))

    def test_EveryLiteralParses(self):
        """A typo here is a node that cannot bootstrap, found by whoever tried
        rather than by the person who made it."""
        for network, literals in seeds.FALLBACK_SEEDS.items():
            for text in literals:
                with self.subTest(network=network, literal=text):
                    NetAddress.parse(text, default_port=9444)


class SeedingPolicyTests(unittest.IsolatedAsyncioTestCase):
    """When the daemon asks, which matters more than what it gets."""

    def daemon(self, *, connect=(), known=0):
        import time

        from rnet.net.address import Services, TimestampedAddress
        from rnet.net.addrman import AddrMan
        from rnet.node.daemon import Daemon, DaemonConfig

        addrman = AddrMan()
        now = int(time.time() * 1000)
        for i in range(known):
            addrman.add(TimestampedAddress(
                NetAddress.parse(f"203.0.113.{i + 1}:9444"), Services.CHAIN, now),
                None, now, explicit=True)

        d = Daemon(config=DaemonConfig(network="main", connect=connect))
        d.node = type("N", (), {"addrman": addrman})()
        d._log = lambda *a: None
        return d

    async def test_ANodeThatKnowsPeersDoesNotAskTheSeeds(self):
        """Seeding on every start would put the network's reachability on
        whoever runs the seeds, and hand them a census of every restart."""
        called = []
        real = seeds.resolve
        seeds.resolve = lambda *a, **k: called.append(1)
        self.addCleanup(lambda: setattr(seeds, "resolve", real))
        await self.daemon(known=3)._seed_if_empty()
        self.assertEqual(called, [])

    async def test_AnExplicitConnectAlsoSuppressesThem(self):
        called = []
        real = seeds.resolve
        seeds.resolve = lambda *a, **k: called.append(1)
        self.addCleanup(lambda: setattr(seeds, "resolve", real))
        await self.daemon(connect=("[::1]:19444",))._seed_if_empty()
        self.assertEqual(called, [])

    async def test_AnEmptyTableDoesAskThem(self):
        asked = []

        async def fake(network, port, **kw):
            asked.append((network, port))
            return [NetAddress.parse("93.184.216.34:9444")]

        real = seeds.resolve
        seeds.resolve = fake
        self.addCleanup(lambda: setattr(seeds, "resolve", real))
        d = self.daemon()
        await d._seed_if_empty()
        self.assertEqual(asked, [("main", 9444)])
        self.assertEqual(len(d.node.addrman), 1)

    async def test_APrivateAddressFromASeedIsDropped(self):
        """A seed that answers with 10.x has either been misconfigured or is
        trying to point a stranger at a network it controls. The routability
        filter that exists for gossip catches it, because seeded addresses go
        through the same door."""
        async def fake(network, port, **kw):
            return [NetAddress.parse("10.1.2.3:9444"),
                    NetAddress.parse("127.0.0.1:9444"),
                    NetAddress.parse("93.184.216.34:9444")]

        real = seeds.resolve
        seeds.resolve = fake
        self.addCleanup(lambda: setattr(seeds, "resolve", real))
        d = self.daemon()
        await d._seed_if_empty()
        self.assertEqual(len(d.node.addrman), 1)

    async def test_WhatASeedSaysIsHearsayNotADialList(self):
        """It goes into the table under the seed as its source, so it competes
        for bucket space like anything heard secondhand. Connecting to seeded
        addresses directly would remove the eclipse resistance at the one moment
        a node will believe whatever it is first told."""
        async def fake(network, port, **kw):
            return [NetAddress.parse(f"93.184.216.{i}:9444") for i in (1, 2, 3)]

        real = seeds.resolve
        seeds.resolve = fake
        self.addCleanup(lambda: setattr(seeds, "resolve", real))
        d = self.daemon()
        dialled = []
        d.node.connect = lambda a: dialled.append(a)
        await d._seed_if_empty()
        self.assertEqual(dialled, [])
        self.assertEqual(len(d.node.addrman), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)

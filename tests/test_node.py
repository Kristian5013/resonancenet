"""Two nodes on a real loopback socket: handshake, gossip, and the ways it ends.

These bind real ports on 127.0.0.1 and ::1. That is deliberate — the failures
this layer has are in socket setup, address families and teardown ordering, and
none of them appear against a mock.
"""

import asyncio
import socket
import unittest

from rnet.net import protocol as P
from rnet.net.address import NetAddress, Services, TimestampedAddress
from rnet.net.addrman import AddrMan
from rnet.net.node import Node, NodeConfig, listening_sockets
from rnet.net.peer import Peer, State

MAGIC = 0x524E_5231
GEN = bytes([1]) * 32
POL = bytes([2]) * 32


def has_v6() -> bool:
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        s.bind(("::1", 0))
        s.close()
        return True
    except OSError:
        return False


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def config(port: int, **kw) -> NodeConfig:
    base = dict(magic=MAGIC, genesis_hash=GEN, policy_hash=POL, port=port,
                user_agent="rnet/test")
    return NodeConfig(**{**base, **kw})


async def settle(predicate, timeout: float = 5.0) -> bool:
    """Wait for a condition rather than for a duration.

    Sleeping a fixed time is how a network test becomes flaky on a loaded
    machine and slow on an idle one.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


class ListenerTests(unittest.TestCase):

    def test_ItBindsBothFamiliesOnOnePort(self):
        """Two sockets, not one dual-stack socket: whether a V6ONLY-off socket
        accepts IPv4 depends on the host, so a node relying on it would listen
        on one family or two with no way to tell which."""
        port = free_port()
        socks = listening_sockets(port)
        try:
            families = {s.family for s in socks}
            self.assertIn(socket.AF_INET, families)
            if has_v6():
                self.assertIn(socket.AF_INET6, families)
                v6 = next(s for s in socks if s.family == socket.AF_INET6)
                self.assertEqual(
                    v6.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY), 1)
        finally:
            for s in socks:
                s.close()

    def test_AHostWithoutIPv6StillStarts(self):
        port = free_port()
        socks = listening_sockets(port, v6=False)
        try:
            self.assertEqual([s.family for s in socks], [socket.AF_INET])
        finally:
            for s in socks:
                s.close()


class NodeTests(unittest.IsolatedAsyncioTestCase):

    async def make(self, **kw) -> Node:
        port = free_port()
        node = Node(config=config(port, **kw), addrman=AddrMan(key=bytes(32)))
        await node.start()
        self.addAsyncCleanup(node.stop)
        return node

    async def test_TwoNodesShakeHands(self):
        a, b = await self.make(), await self.make()
        peer = await a.connect(NetAddress.parse(f"127.0.0.1:{b.config.port}"))
        self.assertIsNotNone(peer)
        self.assertTrue(await settle(lambda: peer.state is State.READY))
        self.assertTrue(await settle(lambda: len(b.ready) == 1))
        self.assertEqual(peer.user_agent, "rnet/test")
        self.assertEqual(len(a.outbound), 1)
        self.assertEqual(len(b.inbound), 1)

    @unittest.skipUnless(has_v6(), "no IPv6 on this host")
    async def test_TwoNodesShakeHandsOverIPv6(self):
        a, b = await self.make(), await self.make()
        peer = await a.connect(NetAddress.parse(f"[::1]:{b.config.port}"))
        self.assertIsNotNone(peer)
        self.assertTrue(await settle(lambda: peer.state is State.READY))
        self.assertFalse(peer.address.is_v4)
        self.assertTrue(await settle(lambda: len(b.ready) == 1))

    async def test_ANodeThatDialsItselfForgetsTheAddress(self):
        """Dropping alone is not enough: the address is real and answers, so a
        node that only dropped would redial it forever.

        Loopback cannot be planted in the table — addrman refuses unroutable
        addresses, correctly — so the forget call is observed directly, which
        is the behaviour that matters anyway.
        """
        a = await self.make()
        forgotten = []
        original = a.addrman.forget
        a.addrman.forget = lambda address: (forgotten.append(address),
                                            original(address))[1]

        own = NetAddress.parse(f"127.0.0.1:{a.config.port}")
        peer = await a.connect(own)
        self.assertTrue(await settle(lambda: peer.state is State.CLOSED))
        self.assertTrue(await settle(lambda: bool(forgotten)))
        # The address we DIALLED, not the ephemeral port the dial went out
        # from: a self-connection makes two peers, and forgetting the accepted
        # one removes an address the table never held while leaving the real
        # culprit in place.
        self.assertEqual([f.port for f in forgotten], [a.config.port])
        self.assertEqual(a.ready, [])

    async def test_AForeignGenesisIsRefusedAtTheHandshake(self):
        a = await self.make()
        b = await self.make(genesis_hash=bytes([9]) * 32)
        peer = await a.connect(NetAddress.parse(f"127.0.0.1:{b.config.port}"))
        self.assertTrue(await settle(lambda: peer.state is State.CLOSED))
        self.assertEqual(a.ready, [])

    async def test_AForeignPolicyIsRefusedAtTheHandshake(self):
        a = await self.make()
        b = await self.make(policy_hash=bytes([9]) * 32)
        peer = await a.connect(NetAddress.parse(f"127.0.0.1:{b.config.port}"))
        self.assertTrue(await settle(lambda: peer.state is State.CLOSED))

    async def test_TheDisconnectReasonIsSentBeforeTheSocketCloses(self):
        """The bug this exists for: the implementation this replaces tore the
        peer down before its queue drained, so every stated reason was silently
        discarded — for months, because the code that wrote it looked correct
        and the code that dropped it was somewhere else."""
        b = await self.make(genesis_hash=bytes([9]) * 32)

        reader, writer = await asyncio.open_connection("127.0.0.1", b.config.port)
        version = P.Version(
            version=P.PROTOCOL_VERSION, services=Services.NONE, timestamp_ms=1,
            receiver=NetAddress(bytes(16), 0), sender=NetAddress(bytes(16), 0),
            nonce=7, user_agent="probe", start_height=0,
            genesis_hash=GEN, policy_hash=POL)
        writer.write(version.to_frame(MAGIC))
        await writer.drain()

        # The far end must say why before it hangs up, and this reads it off the
        # wire rather than off an internal field.
        head = await asyncio.wait_for(reader.readexactly(P.HEADER_SIZE), 5)
        header = P.parse_header(head, MAGIC)
        payload = await reader.readexactly(header.length)
        P.check_payload(header, payload)
        message = P.decode(header.command, payload)
        # It answers with its own version first — so a rejected dialler learns
        # who it reached — and the reject follows.
        if header.command is P.Command.VERSION:
            head = await asyncio.wait_for(reader.readexactly(P.HEADER_SIZE), 5)
            header = P.parse_header(head, MAGIC)
            payload = await reader.readexactly(header.length)
            message = P.decode(header.command, payload)
        self.assertIs(header.command, P.Command.REJECT)
        self.assertIn("genesis", message.reason)
        writer.close()

    async def test_AddressesGossip(self):
        a, b = await self.make(), await self.make()
        known = NetAddress.parse("8.8.8.8:9444")
        b.addrman.add(TimestampedAddress(known, Services.CHAIN,
                                         int(asyncio.get_event_loop().time() * 1000)),
                      None, int(asyncio.get_event_loop().time() * 1000))
        import time as _t
        b.addrman.add(TimestampedAddress(known, Services.CHAIN,
                                         int(_t.time() * 1000)), None,
                      int(_t.time() * 1000))
        peer = await a.connect(NetAddress.parse(f"127.0.0.1:{b.config.port}"))
        self.assertTrue(await settle(lambda: peer.state is State.READY))
        self.assertTrue(await settle(lambda: known in
                                     {e.address for e in a.addrman._entries.values()}))

    async def test_PingIsAnsweredWithItsOwnNonce(self):
        a, b = await self.make(), await self.make()
        peer = await a.connect(NetAddress.parse(f"127.0.0.1:{b.config.port}"))
        self.assertTrue(await settle(lambda: peer.state is State.READY))
        peer.ping_nonce = 4242
        peer.last_ping_sent = asyncio.get_event_loop().time()
        peer.send(P.Ping(nonce=4242))
        self.assertTrue(await settle(lambda: peer.ping_nonce == 0))
        self.assertIsNotNone(peer.latency_ms)

    async def test_AMalformedFrameEndsTheConversation(self):
        b = await self.make()
        reader, writer = await asyncio.open_connection("127.0.0.1", b.config.port)
        writer.write(b"\x00\x00\x00\x00" + b"\x00" * 10)   # wrong magic
        await writer.drain()
        self.assertTrue(await settle(lambda: not b.peers))
        writer.close()

    async def test_InboundConnectionsAreCapped(self):
        b = await self.make(max_inbound=1)
        first = await asyncio.open_connection("127.0.0.1", b.config.port)
        self.assertTrue(await settle(lambda: len(b.peers) == 1))
        second_reader, second_writer = await asyncio.open_connection(
            "127.0.0.1", b.config.port)
        # The second is closed without a handshake rather than queued.
        self.assertTrue(await settle(lambda: len(b.peers) == 1))
        for r, w in (first, (second_reader, second_writer)):
            w.close()

    async def test_OneOutboundSlotPerGroup(self):
        """Eight connections into one /16 is one connection as far as an
        eclipse is concerned."""
        a = await self.make()
        b, c = await self.make(), await self.make()
        for node in (b, c):
            addr = NetAddress.parse(f"127.0.0.1:{node.config.port}")
            a.addrman.add(TimestampedAddress(addr, Services.CHAIN,
                                             int(__import__("time").time() * 1000)),
                          None, int(__import__("time").time() * 1000))
        # Both are 127.0.0.0/16, so exactly one may be dialled.
        await a._dial_once()
        await a._dial_once()
        self.assertLessEqual(len(a.outbound) + len(a._dialling), 1)

    async def test_ThereIsOnlyEverOneConnectionToAnAddress(self):
        """Two sockets to one peer waste two outbound slots while looking like
        two peers in every count."""
        a, b = await self.make(), await self.make()
        target = NetAddress.parse(f"127.0.0.1:{b.config.port}")
        first, second = await asyncio.gather(a.connect(target), a.connect(target))
        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(len(a.outbound), 1)
        # And again once the first is established, not merely in flight.
        self.assertTrue(await settle(lambda: first.state is State.READY))
        self.assertIsNone(await a.connect(target))
        self.assertEqual(len(a.outbound), 1)

    async def test_ABannedGroupIsNotDialled(self):
        import time as _t
        a = await self.make()
        target = NetAddress.parse("8.8.8.8:9444")
        a.banned[target.group] = _t.monotonic() + 60
        self.assertTrue(a.is_banned(target, _t.monotonic()))
        self.assertIsNone(await a.connect(target))

    async def test_ABanExpires(self):
        import time as _t
        a = await self.make()
        target = NetAddress.parse("8.8.8.8:9444")
        a.banned[target.group] = _t.monotonic() - 1
        self.assertFalse(a.is_banned(target, _t.monotonic()))
        self.assertNotIn(target.group, a.banned)

    async def test_TheStatusLineNamesWhatIsThere(self):
        a, b = await self.make(), await self.make()
        peer = await a.connect(NetAddress.parse(f"127.0.0.1:{b.config.port}"))
        self.assertTrue(await settle(lambda: peer.state is State.READY))
        line = a.status()
        self.assertIn("1 ready", line)
        self.assertIn("0 in, 1 out", line)


class SendQueueTests(unittest.TestCase):

    def test_AFullQueueRefusesRatherThanGrowing(self):
        """One slow peer must not cost memory in proportion to how slow it is."""
        async def run():
            peer = Peer(address=NetAddress.parse("1.2.3.4:1"), magic=MAGIC,
                        inbound=False)
            reader = asyncio.StreamReader()
            writer = unittest.mock.MagicMock()
            peer.attach(reader, writer)
            accepted = sum(1 for _ in range(2000) if peer.send(P.Ping(nonce=1)))
            return accepted
        import unittest.mock
        from rnet.net.peer import MAX_SEND_QUEUE
        self.assertEqual(asyncio.run(run()), MAX_SEND_QUEUE)


if __name__ == "__main__":
    unittest.main(verbosity=2)

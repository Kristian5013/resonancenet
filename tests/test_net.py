"""Tests for the network layer: addresses, the wire, and the address table.

Standard library only — none of this needs torch, which is the point: a node
that relays and verifies should be runnable by someone with no interest in
training anything.
"""

import unittest

from rnet.canon.stream import CanonError, Reader, Writer
from rnet.net import protocol as P
from rnet.net.address import (V4_MAPPED_PREFIX, AddressError, NetAddress,
                              Services, TimestampedAddress)
from rnet.net.addrman import BUCKET_SIZE, AddrMan

MAGIC = 0x524E_5231
KEY = bytes(range(32))
NOW = 1_785_000_000_000


def addr(text: str, port: int = 9444) -> NetAddress:
    return NetAddress.parse(text, default_port=port)


class AddressTests(unittest.TestCase):

    def test_EveryAddressIsSixteenBytes(self):
        for text in ("1.2.3.4", "2001:db8::1", "::1", "127.0.0.1"):
            self.assertEqual(len(addr(text).ip), 16, text)

    def test_TheTwoSpellingsOfAnIPv4AddressFold(self):
        """Otherwise the address table holds both and counts one peer as two."""
        a = addr("1.2.3.4:9444")
        b = addr("[::ffff:1.2.3.4]:9444")
        self.assertEqual(a, b)
        self.assertEqual(a.ip, V4_MAPPED_PREFIX + bytes([1, 2, 3, 4]))
        self.assertTrue(a.is_v4)
        self.assertTrue(b.is_v4)

    def test_ItParsesBothFamiliesWithAndWithoutPorts(self):
        self.assertEqual(addr("1.2.3.4:1234").port, 1234)
        self.assertEqual(addr("1.2.3.4").port, 9444)
        self.assertEqual(addr("[2001:db8::1]:1234").port, 1234)
        self.assertEqual(addr("2001:db8::1").port, 9444)
        self.assertEqual(addr("[::1]:5").host, "::1")

    def test_ABareIPv6WithAPortIsRefused(self):
        """2001:db8::1:9444 is ambiguous, and guessing would resolve it
        differently on different days."""
        with self.assertRaises(AddressError):
            NetAddress.parse("[2001:db8::1")
        with self.assertRaises(AddressError):
            NetAddress.parse("[2001:db8::1]junk")
        with self.assertRaises(AddressError):
            NetAddress.parse("not an address")
        with self.assertRaises(AddressError):
            NetAddress.parse("")

    def test_ItPrintsInTheFormItCanReadBack(self):
        for text in ("1.2.3.4:9444", "[2001:db8::1]:9444", "[::1]:1"):
            self.assertEqual(str(NetAddress.parse(text)), text)

    def test_UnroutableAddressesAreRecognised(self):
        for text in ("127.0.0.1", "10.0.0.1", "192.168.1.1", "172.16.0.1",
                     "169.254.1.1", "0.0.0.0", "::1", "fe80::1", "fc00::1",
                     "ff02::1"):
            self.assertFalse(addr(text).is_routable, text)
        # 2001:db8::/32 is the documentation prefix from RFC 3849 and is
        # correctly NOT routable — it is used elsewhere in this file only as a
        # wire sample, where routability is not consulted.
        self.assertFalse(addr("2001:db8::1").is_routable)
        for text in ("1.2.3.4", "8.8.8.8", "2606:4700:4700::1111", "2a00:1450::1"):
            self.assertTrue(addr(text).is_routable, text)

    def test_GroupsAreSixteenBitsOfIPv4AndThirtyTwoOfIPv6(self):
        self.assertEqual(addr("1.2.3.4").group, addr("1.2.9.9").group)
        self.assertNotEqual(addr("1.2.3.4").group, addr("1.3.3.4").group)
        self.assertEqual(addr("2001:db8:1::1").group, addr("2001:db8:9::9").group)
        self.assertNotEqual(addr("2001:db8::1").group, addr("2001:db9::1").group)

    def test_AnIPv4GroupIsNeverAnIPv6Group(self):
        """The family tag stops a crafted IPv6 prefix from colliding with a
        real IPv4 netblock and sharing its bucket."""
        self.assertTrue(addr("1.2.3.4").group.startswith(b"\x04"))
        self.assertTrue(addr("2001:db8::1").group.startswith(b"\x06"))

    def test_TunnelsAreGroupedByTheIPv4TheyRideOn(self):
        """Otherwise a 6to4 or Teredo range is a free identity factory."""
        six_to_four = addr("[2002:0102:0304::1]:9444")     # 1.2.3.4
        self.assertEqual(six_to_four.group, addr("1.2.3.4").group)
        # Teredo carries the client IPv4 inverted in the last four bytes.
        inverted = bytes(b ^ 0xFF for b in bytes([1, 2, 3, 4]))
        teredo = NetAddress(bytes.fromhex("20010000") + bytes(8) + inverted, 9444)
        self.assertEqual(teredo.group, addr("1.2.3.4").group)

    def test_AddressesRoundTripOnTheWire(self):
        for text in ("1.2.3.4:9444", "[2001:db8::1]:1"):
            a = NetAddress.parse(text)
            r = Reader(a.serialize(Writer()).take())
            self.assertEqual(NetAddress.parse_wire(r), a)
            r.expect_exhausted()

    def test_AMalformedAddressOnTheWireIsRefused(self):
        with self.assertRaises(AddressError):
            NetAddress(b"short", 1)
        with self.assertRaises(AddressError):
            NetAddress(bytes(16), 70000)


class WireTests(unittest.TestCase):

    def samples(self) -> dict:
        a = addr("1.2.3.4:9444")
        v6 = addr("[2001:db8::1]:9444")
        h = [bytes([i]) * 32 for i in range(4)]
        return {
            P.Command.VERSION: P.Version(
                version=P.PROTOCOL_VERSION, services=Services.CHAIN | Services.CORPUS,
                timestamp_ms=NOW, receiver=a, sender=v6, nonce=12345,
                user_agent="rnet/0.2", start_height=7,
                genesis_hash=h[0], policy_hash=h[1]),
            P.Command.VERACK: P.Verack(),
            P.Command.PING: P.Ping(nonce=99),
            P.Command.PONG: P.Pong(nonce=99),
            P.Command.GETADDR: P.GetAddr(),
            P.Command.ADDR: P.Addr((TimestampedAddress(a, Services.CHAIN, NOW),
                                    TimestampedAddress(v6, Services.NONE, NOW))),
            P.Command.INV: P.Inv((P.InvEntry(P.InvType.CHECKPOINT, h[0]),)),
            P.Command.GETDATA: P.GetData((P.InvEntry(P.InvType.CONTRIBUTION, h[1]),)),
            P.Command.OBJECT: P.Object(b"a canonical container"),
            P.Command.NOTFOUND: P.NotFound((P.InvEntry(P.InvType.VERDICT, h[2]),)),
            P.Command.GETHEADERS: P.GetHeaders((h[0], h[1]), h[2]),
            P.Command.HEADERS: P.Headers((b"header one", b"header two")),
            P.Command.GETCHUNK: P.GetChunk(h[0], 6_956_932),
            P.Command.CHUNK: P.Chunk(h[0], 5, 1024, (h[1], h[2]), b"text bytes"),
            P.Command.REJECT: P.Reject(P.Command.OBJECT, P.RejectCode.INVALID, "no"),
        }

    def test_EveryCommandHasACodec(self):
        """A command in the enum and nowhere else is not a command."""
        missing = [c.name for c in P.Command if c not in P.CODECS]
        self.assertEqual(missing, [])
        for command, codec in P.CODECS.items():
            self.assertIs(codec.COMMAND, command)

    def test_EveryCommandHasASample(self):
        self.assertEqual(sorted(c.name for c in P.Command),
                         sorted(c.name for c in self.samples()))

    def test_EveryMessageRoundTripsThroughAFrame(self):
        for command, message in self.samples().items():
            with self.subTest(command.name):
                raw = message.to_frame(MAGIC)
                header = P.parse_header(raw[:P.HEADER_SIZE], MAGIC)
                body = raw[P.HEADER_SIZE:]
                P.check_payload(header, body)
                self.assertEqual(header.command, command)
                self.assertEqual(P.decode(header.command, body), message)

    def test_AForeignMagicIsRefusedAtTheFirstFourBytes(self):
        raw = P.Ping(1).to_frame(MAGIC)
        with self.assertRaises(P.ProtocolError):
            P.parse_header(raw[:P.HEADER_SIZE], MAGIC + 1)

    def test_AnOversizedLengthIsRefusedBeforeTheBody(self):
        """The reason the length lives in a fixed-width header: a peer claiming
        four gigabytes costs fourteen bytes and a disconnection."""
        raw = (Writer().u32(MAGIC).u16(P.Command.OBJECT)
               .u32(P.MAX_PAYLOAD + 1).u32(0).take())
        with self.assertRaises(P.ProtocolError) as ctx:
            P.parse_header(raw, MAGIC)
        self.assertIn("limit", str(ctx.exception))

    def test_ACorruptPayloadFailsItsChecksum(self):
        raw = bytearray(P.Object(b"payload").to_frame(MAGIC))
        header = P.parse_header(bytes(raw[:P.HEADER_SIZE]), MAGIC)
        raw[-1] ^= 0x01
        with self.assertRaises(P.ProtocolError):
            P.check_payload(header, bytes(raw[P.HEADER_SIZE:]))

    def test_AShortPayloadIsRefused(self):
        raw = P.Object(b"payload").to_frame(MAGIC)
        header = P.parse_header(raw[:P.HEADER_SIZE], MAGIC)
        with self.assertRaises(P.ProtocolError):
            P.check_payload(header, raw[P.HEADER_SIZE:-1])

    def test_AnUnknownCommandIsRefused(self):
        raw = Writer().u32(MAGIC).u16(0xFFFE).u32(0).u32(P.checksum(b"")).take()
        with self.assertRaises(P.ProtocolError):
            P.parse_header(raw, MAGIC)

    def test_TrailingBytesInAPayloadAreRefused(self):
        with self.assertRaises(P.ProtocolError):
            P.decode(P.Command.PING, P.Ping(1).payload() + b"\x00")

    def test_EveryListMessageIsBounded(self):
        """A count read off a socket is an attacker-chosen allocation."""
        with self.assertRaises(P.ProtocolError):
            P.decode(P.Command.ADDR, Writer().u32(P.MAX_ADDR + 1).take())
        with self.assertRaises(P.ProtocolError):
            P.decode(P.Command.INV, Writer().u32(P.MAX_INV + 1).take())
        with self.assertRaises(P.ProtocolError):
            P.decode(P.Command.GETHEADERS,
                     Writer().u32(P.MAX_LOCATORS + 1).take())
        with self.assertRaises(P.ProtocolError):
            P.decode(P.Command.HEADERS, Writer().u32(P.MAX_HEADERS + 1).take())

    def test_AnAbsurdMerkleProofIsRefused(self):
        payload = (Writer().hash(bytes(32)).u64(0).u64(0).u32(1_000_000).take())
        with self.assertRaises(P.ProtocolError):
            P.decode(P.Command.CHUNK, payload)

    def test_TheHandshakeCarriesBothAnchors(self):
        """A peer on another network is worth finding out about immediately."""
        version = self.samples()[P.Command.VERSION]
        payload = version.payload()
        self.assertIn(version.genesis_hash, payload)
        self.assertIn(version.policy_hash, payload)

    def test_VersionCarriesBothFamilies(self):
        version = self.samples()[P.Command.VERSION]
        self.assertTrue(version.receiver.is_v4)
        self.assertFalse(version.sender.is_v4)
        back = P.decode(P.Command.VERSION, version.payload())
        self.assertEqual(back.receiver, version.receiver)
        self.assertEqual(back.sender, version.sender)


class AddrManTests(unittest.TestCase):

    def man(self) -> AddrMan:
        return AddrMan(key=KEY)

    def stamp(self, text: str, services=Services.CHAIN) -> TimestampedAddress:
        return TimestampedAddress(addr(text), services, NOW)

    def test_ItStoresRoutableAddresses(self):
        m = self.man()
        self.assertTrue(m.add(self.stamp("1.2.3.4"), None, NOW))
        self.assertEqual(len(m), 1)
        self.assertFalse(m.add(self.stamp("1.2.3.4"), None, NOW))

    def test_UnroutableAddressesAreRefusedOutright(self):
        """Keeping them would spend our table on peers that cannot be dialled."""
        m = self.man()
        for text in ("127.0.0.1", "10.0.0.1", "::1", "fe80::1"):
            self.assertFalse(m.add(self.stamp(text), None, NOW), text)
        self.assertEqual(len(m), 0)
        self.assertFalse(m.add(TimestampedAddress(addr("1.2.3.4:0"), Services.NONE, NOW),
                               None, NOW))

    def test_AFutureTimestampIsClamped(self):
        """An address stamped ahead would sort to the front of every table
        forever, which is a free way to monopolise outbound attempts."""
        m = self.man()
        far = TimestampedAddress(addr("1.2.3.4"), Services.NONE, NOW + 10**9)
        m.add(far, None, NOW)
        entry = next(iter(m._entries.values()))
        self.assertLessEqual(entry.last_seen_ms, NOW + 10 * 60 * 1000)

    def test_AnAncientAddressIsRefused(self):
        m = self.man()
        old = TimestampedAddress(addr("1.2.3.4"), Services.NONE, NOW - 10**11)
        self.assertFalse(m.add(old, None, NOW))

    def test_OnlyAConnectionReachesTried(self):
        """Gossip fills NEW for free and can never put anything in TRIED."""
        m = self.man()
        m.add(self.stamp("1.2.3.4"), None, NOW)
        self.assertEqual(m.tried_count, 0)
        self.assertEqual(m.new_count, 1)
        m.connected(addr("1.2.3.4"), NOW)
        self.assertEqual(m.tried_count, 1)
        self.assertEqual(m.new_count, 0)

    def test_ANetblockCannotFillTheTable(self):
        """The attack this exists for: many addresses, one group, few buckets."""
        m = self.man()
        source = addr("9.9.9.9")
        for i in range(2000):
            m.add(self.stamp(f"1.2.{i // 256}.{i % 256}"), source, NOW)
        # A /16 is one group, and one source spreads over a bounded number of
        # buckets, so the whole flood is confined.
        occupied = sum(1 for slots in m._new.values() if slots)
        self.assertLessEqual(occupied, 64)
        self.assertLessEqual(len(m), 64 * BUCKET_SIZE)

    def test_TheBucketKeyIsSecretAndChangesPlacement(self):
        """Without a per-node key an attacker computes the colliding set
        offline."""
        a, b = AddrMan(key=bytes(32)), AddrMan(key=bytes([1]) * 32)
        target = addr("1.2.3.4")
        source = addr("9.9.9.9")
        self.assertNotEqual(a._new_bucket(target, source.group),
                            b._new_bucket(target, source.group))
        self.assertNotEqual(a._tried_bucket(target), b._tried_bucket(target))

    def test_TheSameAddressFromTwoSourcesIsOneEntry(self):
        m = self.man()
        self.assertTrue(m.add(self.stamp("1.2.3.4"), addr("9.9.9.9"), NOW))
        self.assertFalse(m.add(self.stamp("1.2.3.4"), addr("8.8.8.8"), NOW))
        self.assertEqual(len(m), 1)

    def test_ServicesAccumulate(self):
        m = self.man()
        m.add(TimestampedAddress(addr("1.2.3.4"), Services.CHAIN, NOW), None, NOW)
        m.add(TimestampedAddress(addr("1.2.3.4"), Services.CORPUS, NOW), None, NOW)
        entry = next(iter(m._entries.values()))
        self.assertEqual(entry.services, Services.CHAIN | Services.CORPUS)

    def test_SelectionPrefersTriedButNotOnly(self):
        m = self.man()
        for i in range(10):
            m.add(self.stamp(f"1.{i}.0.1"), None, NOW)
        m.connected(addr("1.5.0.1"), NOW)
        self.assertEqual(m.select(NOW), addr("1.5.0.1"))
        # Excluded, it still finds something from NEW rather than giving up.
        picked = m.select(NOW, exclude={(addr("1.5.0.1").ip, 9444)})
        self.assertIsNotNone(picked)
        self.assertNotEqual(picked, addr("1.5.0.1"))

    def test_RepeatedFailureSinksAnAddressFast(self):
        """An address that refused ten times is not ten times worse than one
        that refused once — it is almost certainly gone."""
        m = self.man()
        m.add(self.stamp("1.1.0.1"), None, NOW)
        m.add(self.stamp("1.2.0.1"), None, NOW)
        for _ in range(5):
            m.attempted(addr("1.1.0.1"), NOW)
        self.assertEqual(m.select(NOW), addr("1.2.0.1"))

    def test_ForgettingIsComplete(self):
        """Used when a node finds it dialled itself: real, answers, must never
        be tried again."""
        m = self.man()
        m.add(self.stamp("1.2.3.4"), None, NOW)
        m.connected(addr("1.2.3.4"), NOW)
        m.forget(addr("1.2.3.4"))
        self.assertEqual(len(m), 0)
        self.assertIsNone(m.select(NOW))
        self.assertFalse(m._in_any_table((addr("1.2.3.4").ip, 9444)))

    def test_GossipOffersOnlyFreshAddresses(self):
        m = self.man()
        m.add(self.stamp("1.2.3.4"), None, NOW)
        m.add(TimestampedAddress(addr("5.6.7.8"), Services.NONE, NOW - 10**9),
              None, NOW)
        offered = {t.address for t in m.to_gossip(NOW)}
        self.assertIn(addr("1.2.3.4"), offered)
        m2 = self.man()
        m2.add(self.stamp("1.2.3.4"), None, NOW)
        self.assertEqual(len(m2.to_gossip(NOW, limit=0)), 0)

    def test_BothFamiliesLiveInOneTable(self):
        m = self.man()
        m.add(self.stamp("1.2.3.4"), None, NOW)
        m.add(self.stamp("[2606:4700:4700::1111]"), None, NOW)
        self.assertEqual(len(m), 2)
        self.assertEqual(len(m.groups), 2)
        self.assertIsNotNone(m.select(NOW))

    def test_SelectionIsDeterministicGivenTheKey(self):
        picks = []
        for _ in range(3):
            m = AddrMan(key=KEY)
            for i in range(20):
                m.add(self.stamp(f"1.{i}.0.1"), None, NOW)
            picks.append(m.select(NOW))
        self.assertEqual(len(set(picks)), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

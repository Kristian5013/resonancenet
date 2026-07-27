"""Relay, chain sync and corpus serving, between two real nodes.

The interesting cases are the ones where a peer is wrong rather than absent:
an object nobody asked for, a header whose parent is unknown, a chunk whose
proof is for a different tree.
"""

import asyncio
import socket
import unittest

from rnet.canon.container import ObjectType
from rnet.consensus.objects import CheckpointHeader
from rnet.crypto import merkle
from rnet.diloco.chain import Chain, Outcome
from rnet.net import protocol as P
from rnet.net.address import NetAddress, Services
from rnet.net.addrman import AddrMan
from rnet.net.node import Node, NodeConfig
from rnet.net.peer import BAN_THRESHOLD, State
from rnet.node.objects import ObjectStore, ObjectStoreError
from rnet.node.service import Service

MAGIC = 0x524E_5231
GEN = bytes([1]) * 32
POL = bytes([2]) * 32
ZERO = bytes(32)


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def header(step: int, parent: bytes, weights: int = 0) -> CheckpointHeader:
    return CheckpointHeader(
        round_id=0, outer_step=step, parent=parent,
        weights_hash=bytes([weights % 251]) * 32,
        optimizer_state_hash=bytes([weights % 241]) * 32,
        contribution_root=bytes([weights % 239]) * 32,
        producer_id=1, timestamp_ms=0)


def genesis_header() -> CheckpointHeader:
    return header(0, ZERO)


def build_chain(n: int) -> tuple[Chain, list[CheckpointHeader]]:
    g = genesis_header()
    chain = Chain(g, retained=256)
    made, parent = [g], g.id
    for step in range(1, n + 1):
        h = header(step, parent, weights=step)
        chain.add(h)
        made.append(h)
        parent = h.id
    return chain, made


async def settle(predicate, timeout: float = 5.0) -> bool:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


class FakeCorpus:
    """A corpus small enough to hold, with the same surface as a real one."""

    def __init__(self, chunks: list[bytes]):
        self.chunks = chunks
        self.leaves = [merkle.leaf_hash(c) for c in chunks]
        self.root = merkle.root(self.leaves)
        self.stored: dict[int, bytes] = {}

    def chunk_with_proof(self, index: int):
        return self.chunks[index], merkle.build_proof(self.leaves, index)

    def store(self, index: int, data: bytes) -> None:
        self.stored[index] = data


class ObjectStoreTests(unittest.TestCase):

    def test_ItStoresWhatParses(self):
        store = ObjectStore()
        h = genesis_header()
        held = store.put(h.to_container())
        self.assertEqual(held.obj_type, ObjectType.CHECKPOINT_HEADER)
        self.assertIn(h.id, store)
        self.assertEqual(store.get(h.id).parsed, h)

    def test_ItRefusesWhatDoesNotParse(self):
        with self.assertRaises(ObjectStoreError):
            ObjectStore().put(b"not a container")

    def test_AnObjectMustBeWhatItWasAskedFor(self):
        """"An object is what it says it is" is a property of the store, not a
        discipline every caller has to remember."""
        store = ObjectStore()
        h = genesis_header()
        with self.assertRaises(ObjectStoreError):
            store.put(h.to_container(), expect_id=bytes([9]) * 32)
        store.put(h.to_container(), expect_id=h.id)

    def test_ItIsBoundedByCountAndByBytes(self):
        """Either cap alone is a way in: count alone admits few enormous
        objects, bytes alone admits many tiny ones."""
        store = ObjectStore(max_objects=4, max_bytes=1 << 30)
        for i in range(10):
            # Parents are non-zero: only genesis may have an empty one, and
            # CheckpointHeader refuses the combination outright.
            store.put(header(1, bytes([i + 1]) * 32, weights=i).to_container())
        self.assertLessEqual(len(store), 4)

        tight = ObjectStore(max_objects=1000, max_bytes=400)
        for i in range(10):
            tight.put(header(1, bytes([i + 1]) * 32, weights=i).to_container())
        self.assertLessEqual(tight.total_bytes, 400)

    def test_EvictionIsOldestFirst(self):
        """Not least-recently-used: retention an attacker can steer by asking
        for things is retention the attacker controls."""
        store = ObjectStore(max_objects=2, max_bytes=1 << 30)
        first = header(1, bytes([1]) * 32, weights=1)
        store.put(first.to_container())
        store.put(header(1, bytes([2]) * 32, weights=2).to_container())
        store.get(first.id)                      # "use" it
        store.put(header(1, bytes([3]) * 32, weights=3).to_container())
        self.assertNotIn(first.id, store)


class RelayTests(unittest.IsolatedAsyncioTestCase):

    async def pair(self, **kw):
        nodes, services = [], []
        for _ in range(2):
            node = Node(config=NodeConfig(magic=MAGIC, genesis_hash=GEN,
                                          policy_hash=POL, port=free_port()),
                        addrman=AddrMan(key=bytes(32)))
            chain, made = build_chain(kw.get("height", 0))
            service = Service(node=node, chain=chain, **{
                k: v for k, v in kw.items() if k != "height"})
            service.install()
            await node.start()
            self.addAsyncCleanup(node.stop)
            nodes.append(node)
            services.append(service)
        a, b = nodes
        peer = await a.connect(NetAddress.parse(f"127.0.0.1:{b.config.port}"))
        self.assertTrue(await settle(lambda: peer.state is State.READY))
        self.assertTrue(await settle(lambda: bool(b.ready)))
        return (a, services[0], peer), (b, services[1])

    async def test_AnObjectTravelsFromInvToStore(self):
        (a, sa, peer), (b, sb) = await self.pair()
        h = header(1, bytes([7]) * 32, weights=5)
        sb.objects.put(h.to_container())
        b.broadcast(P.Inv((P.InvEntry(P.InvType.CHECKPOINT, h.id),)))
        self.assertTrue(await settle(lambda: h.id in sa.objects))
        self.assertEqual(sa.objects.get(h.id).parsed, h)

    async def test_AnObjectNobodyAskedForIsScored(self):
        """A peer that pushes objects is using this node's memory as its own."""
        (a, sa, peer), (b, sb) = await self.pair()
        h = header(1, bytes([8]) * 32, weights=6)
        b.broadcast(P.Object(container=h.to_container()))
        self.assertTrue(await settle(lambda: peer.misbehaviour > 0))
        self.assertNotIn(h.id, sa.objects)

    async def test_NotFoundReleasesTheRequest(self):
        """Silence would leave the requester unable to tell a peer that went
        quiet from one that never had it."""
        (a, sa, peer), (b, sb) = await self.pair()
        missing = bytes([9]) * 32
        b.broadcast(P.Inv((P.InvEntry(P.InvType.CHECKPOINT, missing),)))
        self.assertTrue(await settle(lambda: missing not in sa._in_flight
                                     and len(sa._in_flight) == 0))

    async def test_AnAlreadyHeldObjectIsNotRequestedAgain(self):
        (a, sa, peer), (b, sb) = await self.pair()
        h = header(1, bytes([7]) * 32, weights=5)
        sa.objects.put(h.to_container())
        b.broadcast(P.Inv((P.InvEntry(P.InvType.CHECKPOINT, h.id),)))
        await asyncio.sleep(0.2)
        self.assertEqual(len(sa._in_flight), 0)

    async def test_StaleRequestsExpire(self):
        """Otherwise one peer stalls a sync forever by announcing everything and
        delivering nothing."""
        (a, sa, peer), (b, sb) = await self.pair()
        import time as _t
        sa._in_flight[bytes([3]) * 32] = type(
            "F", (), {"peer_id": 1, "requested_at": _t.monotonic() - 10_000,
                      "inv_type": P.InvType.CHECKPOINT})()
        self.assertEqual(sa.expire_requests(), 1)
        self.assertEqual(len(sa._in_flight), 0)


class SyncTests(unittest.IsolatedAsyncioTestCase):

    async def two(self, a_height: int, b_height: int):
        made = []
        nodes, services = [], []
        for height in (a_height, b_height):
            node = Node(config=NodeConfig(magic=MAGIC, genesis_hash=GEN,
                                          policy_hash=POL, port=free_port()),
                        addrman=AddrMan(key=bytes(32)))
            chain, built = build_chain(height)
            node.height = chain.height
            service = Service(node=node, chain=chain)
            service.install()
            await node.start()
            self.addAsyncCleanup(node.stop)
            nodes.append(node)
            services.append(service)
            made.append(built)
        return nodes, services, made

    async def test_ALocatorGoesBackExponentially(self):
        nodes, services, _ = await self.two(0, 40)
        locator = services[1].locator()
        self.assertEqual(locator[0], services[1].chain.head.id)
        self.assertIn(services[1].chain.at_height(0).id, locator)
        self.assertLess(len(locator), 40)
        self.assertLessEqual(len(locator), P.MAX_LOCATORS)

    async def test_ABehindNodeCatchesUp(self):
        """The thing whose absence made the implementation this replaces unable
        to let anyone join late."""
        nodes, services, made = await self.two(0, 12)
        a, b = nodes
        sa, sb = services
        peer = await a.connect(NetAddress.parse(f"127.0.0.1:{b.config.port}"))
        self.assertTrue(await settle(lambda: peer.state is State.READY))
        peer.send(P.GetHeaders(locators=tuple(sa.locator()), stop=ZERO))
        self.assertTrue(await settle(lambda: sa.chain.height == 12))
        self.assertEqual(sa.chain.head.id, sb.chain.head.id)
        self.assertEqual(a.height, 12)

    async def test_AnOrphanedCheckpointTriggersASync(self):
        """A checkpoint whose parent is unknown asks for the chain, not for the
        one parent — a node a hundred steps behind would need a hundred round
        trips otherwise."""
        nodes, services, made = await self.two(0, 8)
        a, b = nodes
        sa, sb = services
        peer = await a.connect(NetAddress.parse(f"127.0.0.1:{b.config.port}"))
        self.assertTrue(await settle(lambda: peer.state is State.READY))
        tip = sb.chain.head.header
        sb.objects.put(tip.to_container())
        b.broadcast(P.Inv((P.InvEntry(P.InvType.CHECKPOINT, tip.id),)))
        self.assertTrue(await settle(lambda: sa.chain.height == 8))

    async def test_HeadersAnswerFromTheFirstSharedCheckpoint(self):
        nodes, services, made = await self.two(5, 12)
        sa, sb = services
        # a's chain is a prefix of b's, built the same way, so they share it.
        self.assertEqual(sa.chain.head.id, sb.chain.at_height(5).id)
        asked = []
        peer = type("P", (), {"send": lambda self, m: asked.append(m)})()
        sb._on_getheaders(peer, P.GetHeaders(tuple(sa.locator()), ZERO))
        self.assertEqual(len(asked), 1)
        self.assertEqual(len(asked[0].headers), 7)

    async def test_ABadHeaderIsScored(self):
        nodes, services, _ = await self.two(0, 0)
        a, _ = nodes
        sa = services[0]
        peer = type("P", (), {"send": lambda self, m: None,
                              "misbehaviour": 0,
                              "penalise": lambda self, p, w: False})()
        scored = []
        a.penalise = lambda p, points, why: scored.append(points)
        sa._on_headers(peer, P.Headers((b"garbage",)))
        self.assertTrue(scored)


class CorpusTests(unittest.IsolatedAsyncioTestCase):

    def corpus(self, n: int = 16) -> FakeCorpus:
        return FakeCorpus([f"document {i}\n\n".encode() for i in range(n)])

    async def one(self, corpus):
        node = Node(config=NodeConfig(magic=MAGIC, genesis_hash=GEN,
                                      policy_hash=POL, port=free_port()),
                    addrman=AddrMan(key=bytes(32)))
        chain, _ = build_chain(0)
        service = Service(node=node, chain=chain, corpus=corpus,
                          dataset_root=corpus.root,
                          dataset_chunks=len(corpus.chunks))
        service.install()
        await node.start()
        self.addAsyncCleanup(node.stop)
        return node, service

    async def test_AChunkTravelsWithItsProof(self):
        corpus = self.corpus()
        server, ss = await self.one(corpus)
        client, sc = await self.one(FakeCorpus(corpus.chunks))
        sc.corpus.stored.clear()

        peer = await client.connect(NetAddress.parse(f"127.0.0.1:{server.config.port}"))
        self.assertTrue(await settle(lambda: peer.state is State.READY))
        peer.send(P.GetChunk(dataset_root=corpus.root, index=5))
        self.assertTrue(await settle(lambda: 5 in sc.corpus.stored))
        self.assertEqual(sc.corpus.stored[5], corpus.chunks[5])

    async def test_AChunkForAnotherCorpusIsScored(self):
        corpus = self.corpus()
        node, service = await self.one(corpus)
        scored = []
        node.penalise = lambda p, points, why: scored.append(why)
        peer = type("P", (), {"send": lambda self, m: None})()
        self.assertFalse(service._on_chunk(
            peer, P.Chunk(dataset_root=bytes([9]) * 32, index=0, leaf_count=16,
                          proof=(), data=b"x")))
        self.assertTrue(scored)

    async def test_TheWidthComesFromTheManifestNotTheMessage(self):
        """A proof does not pin the width it was built at, so taking the width
        from the peer would let it choose which tree its proof is against."""
        corpus = self.corpus()
        node, service = await self.one(corpus)
        scored = []
        node.penalise = lambda p, points, why: scored.append(why)
        peer = type("P", (), {"send": lambda self, m: None})()
        data, proof = corpus.chunk_with_proof(3)
        self.assertFalse(service._on_chunk(
            peer, P.Chunk(dataset_root=corpus.root, index=3, leaf_count=99,
                          proof=proof.path, data=data)))
        self.assertIn("manifest", scored[0])

    async def test_ATamperedChunkDoesNotProve(self):
        corpus = self.corpus()
        node, service = await self.one(corpus)
        scored = []
        node.penalise = lambda p, points, why: scored.append(why)
        peer = type("P", (), {"send": lambda self, m: None})()
        data, proof = corpus.chunk_with_proof(3)
        self.assertFalse(service._on_chunk(
            peer, P.Chunk(dataset_root=corpus.root, index=3,
                          leaf_count=len(corpus.chunks), proof=proof.path,
                          data=data + b"!")))
        self.assertTrue(scored)

    async def test_AGoodChunkIsAccepted(self):
        corpus = self.corpus()
        node, service = await self.one(corpus)
        service.corpus.stored.clear()
        peer = type("P", (), {"send": lambda self, m: None})()
        data, proof = corpus.chunk_with_proof(7)
        self.assertTrue(service._on_chunk(
            peer, P.Chunk(dataset_root=corpus.root, index=7,
                          leaf_count=len(corpus.chunks), proof=proof.path,
                          data=data)))
        self.assertEqual(service.corpus.stored[7], data)


if __name__ == "__main__":
    unittest.main(verbosity=2)

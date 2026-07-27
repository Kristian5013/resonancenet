"""The worker channel: messages, and a daemon that talks to a worker over it.

Runs against a real Unix socket, because what this layer gets wrong is
handshakes, framing and refusals, and none of those appear against a mock.
"""

import asyncio
import os
import shutil
import tempfile
import unittest

import numpy as np

from rnet.canon.stream import CanonError, Reader, Writer
from rnet.consensus import genesis
from rnet.consensus.numerics import ContributionFormat
from rnet.consensus.objects import CheckpointHeader
from rnet.diloco.chain import Chain
from rnet.diloco.quantize import pack, quantize_update
from rnet.net import framing
from rnet.node.workerservice import MAX_WORKERS, WorkerService
from rnet.worker import ipc
from rnet.worker.client import AnchorMismatch, DaemonClient, WorkerError

NETWORK = "regtest"
GEN = bytes.fromhex(genesis.GENESIS_HASH[NETWORK])
POL = bytes.fromhex(genesis.POLICY_HASH[NETWORK])


def genesis_checkpoint(round_desc) -> CheckpointHeader:
    return CheckpointHeader(
        round_id=round_desc.round_id, outer_step=0, parent=bytes(32),
        weights_hash=bytes.fromhex(genesis.WEIGHTS_HASH[NETWORK]),
        optimizer_state_hash=bytes(32), contribution_root=bytes(32),
        producer_id=0, timestamp_ms=0)


class WorkerIpcTests(unittest.TestCase):

    def samples(self) -> dict:
        h = [bytes([i]) * 32 for i in range(3)]
        return {
            ipc.Command.HELLO: ipc.Hello(genesis_hash=h[0], policy_hash=h[1],
                                         user_agent="rnet-worker/0.2"),
            ipc.Command.WELCOME: ipc.Welcome(worker_id=3, network_magic=0x524E5231,
                                             parameter_count=4_000_512,
                                             contribution_bits=8),
            ipc.Command.GET_ASSIGNMENT: ipc.GetAssignment(),
            ipc.Command.ASSIGNMENT: ipc.Assignment(
                assignment_id=1, round_id=0, outer_step=1, base_checkpoint=h[0],
                base_weights_hash=h[1], inner_steps=4, micro_batch=1,
                deadline_ms=1_785_000_000_000),
            ipc.Command.NO_WORK: ipc.NoWork(reason="waiting for a quorum",
                                            retry_ms=5000),
            ipc.Command.SUBMIT: ipc.Submit(assignment_id=1, scale_exp=-17,
                                           value_count=8, final_loss=5.6843,
                                           packed=b"\x01" * 8),
            ipc.Command.ACCEPTED: ipc.Accepted(contribution_id=h[2]),
            ipc.Command.REFUSED: ipc.Refused(code=ipc.Refusal.STALE_ASSIGNMENT,
                                             reason="the chain moved on"),
            ipc.Command.GET_CHUNK: ipc.GetChunk(index=6_956_932),
            ipc.Command.CHUNK: ipc.Chunk(index=5, data=b"document text\n\n"),
        }

    def test_EveryCommandHasACodec(self):
        missing = [c.name for c in ipc.Command if c not in ipc.CODECS]
        self.assertEqual(missing, [])
        for command, codec in ipc.CODECS.items():
            self.assertIs(codec.COMMAND, command)

    def test_EveryCommandHasASample(self):
        self.assertEqual(sorted(c.name for c in ipc.Command),
                         sorted(c.name for c in self.samples()))

    def test_EveryMessageRoundTripsThroughAFrame(self):
        """This is the test that would have caught a field named `payload`
        shadowing Message.payload() — a TypeError at send time, on the one
        message that carries real data."""
        for command, message in self.samples().items():
            with self.subTest(command.name):
                raw = message.to_frame()
                header = framing.parse_header(raw[:framing.HEADER_SIZE],
                                              ipc.IPC_MAGIC, ipc.Command,
                                              max_payload=ipc.MAX_IPC_PAYLOAD)
                body = raw[framing.HEADER_SIZE:]
                framing.check_payload(header, body)
                self.assertEqual(header.command, command)
                self.assertEqual(ipc.decode(header.command, body), message)

    def test_TheIpcMagicIsNotTheNetworkMagic(self):
        """Pointing a worker at a peer-to-peer port fails at the first four
        bytes instead of somewhere confusing."""
        from rnet.consensus.params import NETWORKS
        magics = {rd.network_magic for rd, _ in NETWORKS.values()}
        self.assertNotIn(ipc.IPC_MAGIC, magics)

    def test_TheLossSurvivesAsFixedPoint(self):
        """Carried as millionths, so this channel needs no float encoding —
        one fewer thing for the two sides to agree about."""
        for value in (0.0, 5.6843, -1.5, 12345.678901):
            message = ipc.Submit(final_loss=value, packed=b"\x00")
            back = ipc.decode(ipc.Command.SUBMIT, message.payload())
            self.assertAlmostEqual(back.final_loss, value, places=5)

    def test_AnAssignmentWithNoWorkIsRefused(self):
        raw = (Writer().u64(1).u64(0).u64(1).hash(bytes(32)).hash(bytes(32))
               .u32(0).u32(1).u64(0).take())
        with self.assertRaises(ipc.IpcError):
            ipc.decode(ipc.Command.ASSIGNMENT, raw)

    def test_AnUnknownRefusalCodeIsRefused(self):
        raw = Writer().u16(99).string("why").take()
        with self.assertRaises(ipc.IpcError):
            ipc.decode(ipc.Command.REFUSED, raw)

    def test_TrailingBytesAreRefused(self):
        with self.assertRaises(ipc.IpcError):
            ipc.decode(ipc.Command.GET_CHUNK, ipc.GetChunk(1).payload() + b"\x00")

    def test_ThePayloadBoundTracksTheModel(self):
        """A limit that tracks the round, not a constant that is either too
        small for the next model or too large to be a limit."""
        self.assertEqual(ipc.expected_payload_bytes(4_000_512, 8), 4_000_512)
        self.assertEqual(ipc.expected_payload_bytes(4_000_512, 4), 2_000_256)
        self.assertEqual(ipc.expected_payload_bytes(397_728_768, 8), 397_728_768)


class WorkerServiceTests(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.dir = tempfile.mkdtemp(prefix="rnet-ws-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.round_desc = genesis.round_descriptor(NETWORK)
        self.policy = genesis.policy_descriptor(NETWORK)
        self.chain = Chain(genesis_checkpoint(self.round_desc),
                           retained=self.policy.retained_checkpoints)
        self.service = WorkerService(datadir=self.dir, round_desc=self.round_desc,
                                     policy=self.policy, chain=self.chain)
        self.service._log = lambda *a: None
        self.path = await self.service.start()
        self.addAsyncCleanup(self.service.stop)

    def client(self, **kw) -> DaemonClient:
        base = dict(socket_path=self.path, genesis_hash=GEN, policy_hash=POL,
                    timeout=10.0)
        client = DaemonClient(**{**base, **kw})
        client.connect()
        self.addCleanup(client.close)
        return client

    async def in_thread(self, fn, *args):
        return await asyncio.to_thread(fn, *args)

    # -- handshake -----------------------------------------------------------

    async def test_AWorkerIsAssignedItsId(self):
        """Assigned, never chosen: the id feeds the batch schedule."""
        a = self.client()
        welcome = await self.in_thread(a.hello)
        self.assertEqual(welcome.worker_id, 1)
        self.assertEqual(welcome.parameter_count,
                         self.round_desc.model.parameter_count())
        b = self.client()
        self.assertEqual((await self.in_thread(b.hello)).worker_id, 2)

    async def test_AWorkerOnDifferentRulesIsRefused(self):
        """Ends here rather than after twenty minutes of wasted GPU."""
        bad = self.client(genesis_hash=bytes([9]) * 32)
        with self.assertRaises(WorkerError):
            await self.in_thread(bad.hello)

    async def test_TheSocketIsPrivate(self):
        self.assertEqual(os.stat(self.dir).st_mode & 0o777, 0o700)

    async def test_AStaleSocketDoesNotBlockAStart(self):
        """A crash must not need a manual cleanup before the node comes back."""
        await self.service.stop()
        with open(self.path, "w"):
            pass
        again = WorkerService(datadir=self.dir, round_desc=self.round_desc,
                              policy=self.policy, chain=self.chain)
        again._log = lambda *a: None
        self.assertEqual(await again.start(), self.path)
        await again.stop()

    # -- assignments ----------------------------------------------------------

    async def test_AnAssignmentNamesTheHead(self):
        client = self.client()
        await self.in_thread(client.hello)
        assignment = await self.in_thread(client.get_assignment)
        self.assertIsInstance(assignment, ipc.Assignment)
        self.assertEqual(assignment.outer_step, 1)
        self.assertEqual(assignment.base_checkpoint, self.chain.head.id)
        self.assertEqual(assignment.base_weights_hash,
                         self.chain.head.header.weights_hash)
        self.assertEqual(assignment.inner_steps, self.policy.inner_steps)

    async def test_OneAssignmentPerWorkerPerStep(self):
        """Two would mean the same worker id feeding the schedule twice for one
        step, so the second run trains on exactly the data the first did."""
        client = self.client()
        await self.in_thread(client.hello)
        first = await self.in_thread(client.get_assignment)
        self.assertIsInstance(first, ipc.Assignment)
        second = await self.in_thread(client.get_assignment)
        self.assertIsInstance(second, ipc.NoWork)
        self.assertIn("already hold", second.reason)

    async def test_TwoWorkersGetDifferentAssignments(self):
        a, b = self.client(), self.client()
        await self.in_thread(a.hello)
        await self.in_thread(b.hello)
        first = await self.in_thread(a.get_assignment)
        second = await self.in_thread(b.get_assignment)
        self.assertNotEqual(first.assignment_id, second.assignment_id)
        self.assertEqual(first.outer_step, second.outer_step)

    # -- submissions -----------------------------------------------------------

    def contribution(self, count: int, seed: int = 0):
        values = np.random.default_rng(seed).normal(0, 1e-3, count)
        q, exp = quantize_update(values, ContributionFormat.INT8_POW2)
        return pack(q, ContributionFormat.INT8_POW2), exp

    async def test_AGoodSubmissionIsAccepted(self):
        client = self.client()
        await self.in_thread(client.hello)
        assignment = await self.in_thread(client.get_assignment)
        count = self.round_desc.model.parameter_count()
        packed, exp = self.contribution(count)
        accepted = await self.in_thread(
            client.submit, assignment.assignment_id, packed, exp, count, 5.68)
        self.assertEqual(len(accepted.contribution_id), 32)
        self.assertEqual(self.service.accepted, 1)
        self.assertEqual(len(self.service.contributions_at(1)), 1)

    async def test_AWorkerCannotContributeTwiceToOneStep(self):
        client = self.client()
        await self.in_thread(client.hello)
        assignment = await self.in_thread(client.get_assignment)
        count = self.round_desc.model.parameter_count()
        packed, exp = self.contribution(count)
        await self.in_thread(client.submit, assignment.assignment_id, packed,
                             exp, count, 5.68)
        again = await self.in_thread(client.get_assignment)
        self.assertIsInstance(again, ipc.NoWork)
        self.assertIn("already contributed", again.reason)

    async def test_AnUnknownAssignmentIsRefused(self):
        client = self.client()
        await self.in_thread(client.hello)
        count = self.round_desc.model.parameter_count()
        packed, exp = self.contribution(count)
        with self.assertRaises(WorkerError) as ctx:
            await self.in_thread(client.submit, 999, packed, exp, count, 5.0)
        self.assertIn("UNKNOWN_ASSIGNMENT", str(ctx.exception))

    async def test_AWrongValueCountIsRefused(self):
        client = self.client()
        await self.in_thread(client.hello)
        assignment = await self.in_thread(client.get_assignment)
        packed, exp = self.contribution(64)
        with self.assertRaises(WorkerError) as ctx:
            await self.in_thread(client.submit, assignment.assignment_id,
                                 packed, exp, 64, 5.0)
        self.assertIn("BAD_PAYLOAD", str(ctx.exception))

    async def test_AWrongByteCountIsRefused(self):
        client = self.client()
        await self.in_thread(client.hello)
        assignment = await self.in_thread(client.get_assignment)
        count = self.round_desc.model.parameter_count()
        with self.assertRaises(WorkerError) as ctx:
            await self.in_thread(client.submit, assignment.assignment_id,
                                 b"\x00" * (count - 1), 0, count, 5.0)
        self.assertIn("BAD_PAYLOAD", str(ctx.exception))

    async def test_AStaleAssignmentIsRefusedAsLateNotWrong(self):
        """The worker is not wrong, it is late, and saying which matters to
        whoever reads the log."""
        client = self.client()
        await self.in_thread(client.hello)
        assignment = await self.in_thread(client.get_assignment)
        moved = CheckpointHeader(
            round_id=0, outer_step=1, parent=self.chain.head.id,
            weights_hash=bytes([7]) * 32, optimizer_state_hash=bytes([8]) * 32,
            contribution_root=bytes([9]) * 32, producer_id=1, timestamp_ms=0)
        self.chain.add(moved)
        count = self.round_desc.model.parameter_count()
        packed, exp = self.contribution(count)
        with self.assertRaises(WorkerError) as ctx:
            await self.in_thread(client.submit, assignment.assignment_id,
                                 packed, exp, count, 5.0)
        self.assertIn("STALE_ASSIGNMENT", str(ctx.exception))

    # -- accounting ------------------------------------------------------------

    async def test_DeferralsAreNotCountedAsRefusals(self):
        """Counting them together made a healthy node report two thousand
        rejections an hour and read exactly backwards."""
        client = self.client()
        await self.in_thread(client.hello)
        # No corpus at all is a refusal, not a deferral: asking again will not
        # help, and a worker told "ask again" would poll forever.
        with self.assertRaises(WorkerError):
            await self.in_thread(client.chunk, 0)
        self.assertEqual(self.service.refused, 1)

        class Corpus:
            def get(self, index):
                return None
            def request(self, index):
                pass

        self.service.corpus = Corpus()
        self.assertIsNone(await self.in_thread(client.chunk, 0))
        self.assertEqual(self.service.deferred, 1)
        self.assertEqual(self.service.refused, 1)

    async def test_ACorpusChunkIsServed(self):
        class Corpus:
            def get(self, index):
                return b"document %d\n\n" % index
            def request(self, index):
                pass

        self.service.corpus = Corpus()
        client = self.client()
        await self.in_thread(client.hello)
        self.assertEqual(await self.in_thread(client.chunk, 3), b"document 3\n\n")

    async def test_TooManyWorkersAreRefused(self):
        clients = []
        for _ in range(MAX_WORKERS):
            c = self.client()
            await self.in_thread(c.hello)
            clients.append(c)
        self.assertEqual(len(self.service.workers), MAX_WORKERS)
        # And it says why rather than closing abruptly: a reset from the socket
        # layer tells the operator nothing about which reason applied.
        extra = self.client()
        with self.assertRaises(WorkerError) as ctx:
            await self.in_thread(extra.hello)
        self.assertIn("already serves", str(ctx.exception))

    async def test_TheStatusLineNamesWhatMatters(self):
        client = self.client()
        await self.in_thread(client.hello)
        line = self.service.status()
        for fragment in ("workers", "accepted", "refused", "deferred",
                         "contribution"):
            self.assertIn(fragment, line)


if __name__ == "__main__":
    unittest.main(verbosity=2)

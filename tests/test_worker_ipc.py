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
            ipc.Command.APPLY: ipc.Apply(
                outer_step=1, scale_exp=-17, value_count=8, momentum_q16=58982,
                lr_q16=45875, nesterov=True, packed=b"\x01" * 8),
            ipc.Command.APPLIED: ipc.Applied(
                outer_step=1, weights_hash=h[0], optimizer_state_hash=h[1]),
            ipc.Command.VERIFY: ipc.Verify(
                challenge_id=h[0], target_worker_id=2, round_id=0, outer_step=1,
                inner_steps=4, micro_batch=1, base_weights_hash=h[1],
                claimed_payload_hash=h[2], determinism_class=0xDF7C5208),
            ipc.Command.VERDICT: ipc.Verdict(
                challenge_id=h[0], verdict=1, replay_payload_hash=h[1],
                determinism_class=0xDF7C5208, note=""),
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
        assignment = await self.in_thread(client.next_work)
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
        first = await self.in_thread(client.next_work)
        self.assertIsInstance(first, ipc.Assignment)
        second = await self.in_thread(client.next_work)
        self.assertIsInstance(second, ipc.NoWork)
        self.assertIn("already hold", second.reason)

    async def test_TwoWorkersGetDifferentAssignments(self):
        a, b = self.client(), self.client()
        await self.in_thread(a.hello)
        await self.in_thread(b.hello)
        first = await self.in_thread(a.next_work)
        second = await self.in_thread(b.next_work)
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
        assignment = await self.in_thread(client.next_work)
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
        assignment = await self.in_thread(client.next_work)
        count = self.round_desc.model.parameter_count()
        packed, exp = self.contribution(count)
        await self.in_thread(client.submit, assignment.assignment_id, packed,
                             exp, count, 5.68)
        again = await self.in_thread(client.next_work)
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
        assignment = await self.in_thread(client.next_work)
        packed, exp = self.contribution(64)
        with self.assertRaises(WorkerError) as ctx:
            await self.in_thread(client.submit, assignment.assignment_id,
                                 packed, exp, 64, 5.0)
        self.assertIn("BAD_PAYLOAD", str(ctx.exception))

    async def test_AWrongByteCountIsRefused(self):
        client = self.client()
        await self.in_thread(client.hello)
        assignment = await self.in_thread(client.next_work)
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
        assignment = await self.in_thread(client.next_work)
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

    # -- production ------------------------------------------------------------

    async def two_contributions(self):
        """Two workers, both contributed to step 1, quorum met."""
        count = self.round_desc.model.parameter_count()
        clients = []
        for seed in (0, 1):
            client = self.client()
            await self.in_thread(client.hello)
            assignment = await self.in_thread(client.next_work)
            packed, exp = self.contribution(count, seed)
            await self.in_thread(client.submit, assignment.assignment_id,
                                 packed, exp, count, 5.7)
            clients.append(client)
        return clients

    async def test_TheQuorumTurnsIntoAnApply(self):
        clients = await self.two_contributions()
        self.assertTrue(self.service.ready_to_produce(1))
        work = await self.in_thread(clients[0].next_work)
        self.assertIsInstance(work, ipc.Apply)
        self.assertEqual(work.outer_step, 1)
        self.assertEqual(work.value_count,
                         self.round_desc.model.parameter_count())
        self.assertEqual(work.momentum_q16, self.policy.outer_momentum_q16)

    def test_TheAggregateFitsTheContributionFormat(self):
        """Why the aggregate can ride back at int8 instead of int64: each part
        is within the format after alignment, a sum of n is within n times it,
        and the mean is within it again. 400 MB instead of 3.2 GB."""
        from rnet.diloco.aggregate import Contribution, average_contributions
        fmt = ContributionFormat.INT8_POW2
        from rnet.diloco.aggregate import MAX_ALIGN_SHIFT
        rng = np.random.default_rng(0)
        for n in (2, 3, 8, 50):
            # Spreads within the alignment rule; wider than MAX_ALIGN_SHIFT is
            # refused outright as a different quantity, not aggregated.
            spread = [(i * MAX_ALIGN_SHIFT) // max(1, n - 1) for i in range(n)]
            for exponents in ([0] * n, spread, [-e for e in spread]):
                parts = [
                    Contribution(
                        values=rng.integers(-fmt.max_magnitude,
                                            fmt.max_magnitude + 1, 64),
                        scale_exp=e, fmt=fmt, worker_id=i)
                    for i, e in enumerate(exponents)]
                mean, _ = average_contributions(parts)
                self.assertLessEqual(int(np.abs(mean).max()), fmt.max_magnitude,
                                     f"{n} parts at {exponents[:3]}…")

    async def test_EveryWorkerIsHandedTheApplyWhenItAsks(self):
        """Not pushed. This channel is strict request-and-response, so an
        unsolicited message lands in the middle of another exchange and
        desynchronises it — measured as `expected accepted, got NoWork` on a
        worker that had submitted perfectly correctly."""
        clients = await self.two_contributions()
        for client in clients:
            work = await self.in_thread(client.next_work)
            self.assertIsInstance(work, ipc.Apply)
            self.assertEqual(work.outer_step, 1)

    async def test_AnAppliedReportBuildsTheCheckpoint(self):
        clients = await self.two_contributions()
        work = await self.in_thread(clients[0].next_work)
        weights = bytes([5]) * 32
        optimizer = bytes([6]) * 32
        accepted = await self.in_thread(clients[0].applied, work.outer_step,
                                        weights, optimizer)
        self.assertEqual(len(accepted.contribution_id), 32)
        self.assertEqual(self.chain.height, 1)
        head = self.chain.head.header
        self.assertEqual(head.weights_hash, weights)
        self.assertEqual(head.optimizer_state_hash, optimizer)
        self.assertEqual(head.outer_step, 1)
        self.assertEqual(self.service.produced, 1)

    async def test_TheContributionsAreSpent(self):
        """Keeping them would let the next round aggregate work already
        applied."""
        clients = await self.two_contributions()
        work = await self.in_thread(clients[0].next_work)
        await self.in_thread(clients[0].applied, work.outer_step,
                             bytes([5]) * 32, bytes([6]) * 32)
        self.assertEqual(self.service.contributions_at(1), [])

    async def test_TheSecondApplierDoesNotBuildASecondCheckpoint(self):
        clients = await self.two_contributions()
        for client in clients:
            work = await self.in_thread(client.next_work)
            self.assertIsInstance(work, ipc.Apply)
            await self.in_thread(client.applied, work.outer_step,
                                 bytes([5]) * 32, bytes([6]) * 32)
        self.assertEqual(self.chain.height, 1)
        self.assertEqual(self.service.produced, 1)

    async def test_DisagreeingWorkersAreShoutedAbout(self):
        """Same aggregate, same base, integer arithmetic — a disagreement is
        this node's own workers computing differently, which is worth saying
        loudly rather than resolving quietly."""
        said = []
        self.service._log = lambda *a: said.append(" ".join(str(x) for x in a))
        clients = await self.two_contributions()
        for i, client in enumerate(clients):
            work = await self.in_thread(client.next_work)
            await self.in_thread(client.applied, work.outer_step,
                                 bytes([5 + i]) * 32, bytes([6]) * 32)
        self.assertTrue(any("DISAGREE" in line for line in said), said)

    async def test_AnUnaskedAppliedIsRefused(self):
        client = self.client()
        await self.in_thread(client.hello)
        with self.assertRaises(WorkerError) as ctx:
            await self.in_thread(client.applied, 7, bytes(32), bytes(32))
        self.assertIn("UNKNOWN_ASSIGNMENT", str(ctx.exception))
        # And the channel is still usable: one message, one reply, no drift.
        self.assertIsInstance(await self.in_thread(client.next_work),
                              ipc.Assignment)

    # -- verification -----------------------------------------------------------

    async def close_a_round(self, clients):
        """Drive step 1 to a checkpoint so its challenges get issued."""
        for client in clients:
            work = await self.in_thread(client.next_work)
            self.assertIsInstance(work, ipc.Apply)
            await self.in_thread(client.applied, work.outer_step,
                                 bytes([5]) * 32, bytes([6]) * 32)

    async def test_ApplyingComesBeforeAnyChallenge(self):
        """A verifier replays against the weights the challenged round STARTED
        from, and only holds those once it has applied that round's outer step.
        With the checks the other way round, whether a challenge could be
        answered at all depended on which worker happened to report first —
        measured, as a run of INDETERMINATE verdicts."""
        clients = await self.two_contributions()
        # Force every contribution to be challenged, so the ordering is what is
        # under test rather than the draw.
        self.service.policy = type(self.policy)(
            **{**self.policy.__dict__, "challenge_percent": 100})

        first = await self.in_thread(clients[0].next_work)
        self.assertIsInstance(first, ipc.Apply)
        await self.in_thread(clients[0].applied, first.outer_step,
                             bytes([5]) * 32, bytes([6]) * 32)
        self.assertGreater(self.service.challenged, 0)

        # The other worker has not applied yet, so it gets the Apply, not a
        # challenge it could only answer INDETERMINATE.
        second = await self.in_thread(clients[1].next_work)
        self.assertIsInstance(second, ipc.Apply)
        await self.in_thread(clients[1].applied, second.outer_step,
                             bytes([5]) * 32, bytes([6]) * 32)
        third = await self.in_thread(clients[1].next_work)
        self.assertIsInstance(third, ipc.Verify)

    async def test_ChallengesAreIssuedAtTheCheckpoint(self):
        clients = await self.two_contributions()
        self.service.policy = type(self.policy)(
            **{**self.policy.__dict__, "challenge_percent": 100})
        await self.close_a_round(clients)
        self.assertEqual(self.service.challenged, 2)
        self.assertEqual(len(self.service._pending_verify), 2)

    async def test_NobodyAuditsTheirOwnWork(self):
        clients = await self.two_contributions()
        self.service.policy = type(self.policy)(
            **{**self.policy.__dict__, "challenge_percent": 100})
        await self.close_a_round(clients)
        for i, client in enumerate(clients):
            work = await self.in_thread(client.next_work)
            if isinstance(work, ipc.Verify):
                self.assertNotEqual(work.target_worker_id, client.worker_id)

    async def test_AVerdictBecomesAPublishedObject(self):
        published = []

        class FakeService:
            node = type("N", (), {"height": 0})()

            def publish(self, obj):
                published.append(obj)
                return 1

        self.service.service = FakeService()
        clients = await self.two_contributions()
        self.service.policy = type(self.policy)(
            **{**self.policy.__dict__, "challenge_percent": 100})
        await self.close_a_round(clients)

        work = await self.in_thread(clients[0].next_work)
        self.assertIsInstance(work, ipc.Verify)
        await self.in_thread(clients[0].verdict, ipc.Verdict(
            challenge_id=work.challenge_id, verdict=1,
            replay_payload_hash=bytes([7]) * 32,
            determinism_class=work.determinism_class))
        self.assertEqual(self.service.verdicts, 1)
        self.assertEqual(self.service.mismatches, 0)
        self.assertTrue(published)

    async def test_AMismatchIsCounted(self):
        clients = await self.two_contributions()
        self.service.policy = type(self.policy)(
            **{**self.policy.__dict__, "challenge_percent": 100})
        await self.close_a_round(clients)
        work = await self.in_thread(clients[0].next_work)
        await self.in_thread(clients[0].verdict, ipc.Verdict(
            challenge_id=work.challenge_id, verdict=2,
            replay_payload_hash=bytes([7]) * 32,
            determinism_class=work.determinism_class))
        self.assertEqual(self.service.mismatches, 1)
        # Still open: one verdict is not a quorum, and closing here is what made
        # slash_quorum decorative.
        self.assertIn(work.challenge_id, self.service._pending_verify)

    async def test_IndeterminateLeavesTheChallengeOpen(self):
        """Otherwise a verifier retires a challenge by being unable to answer
        it, which is the cheapest way to protect a friend."""
        clients = await self.two_contributions()
        self.service.policy = type(self.policy)(
            **{**self.policy.__dict__, "challenge_percent": 100})
        await self.close_a_round(clients)
        work = await self.in_thread(clients[0].next_work)
        await self.in_thread(clients[0].verdict, ipc.Verdict(
            challenge_id=work.challenge_id, verdict=3,
            replay_payload_hash=bytes(32),
            determinism_class=work.determinism_class,
            note="different class"))
        self.assertIn(work.challenge_id, self.service._pending_verify)

    async def test_AnUnknownChallengeIsRefused(self):
        client = self.client()
        await self.in_thread(client.hello)
        with self.assertRaises(WorkerError) as ctx:
            await self.in_thread(client.verdict, ipc.Verdict(
                challenge_id=bytes([9]) * 32, verdict=1,
                replay_payload_hash=bytes([7]) * 32, determinism_class=0))
        self.assertIn("UNKNOWN_ASSIGNMENT", str(ctx.exception))

    async def test_NobodyJudgesTheirOwnWork(self):
        """Refusing to HAND a worker its own challenge is not enough: a verdict
        is a message it can send unprompted, so without this it clears itself by
        answering a question it was never asked."""
        clients = await self.two_contributions()
        self.service.policy = type(self.policy)(
            **{**self.policy.__dict__, "challenge_percent": 100})
        await self.close_a_round(clients)
        challenge_id, (verify, _) = next(iter(self.service._pending_verify.items()))
        accused = next(c for c in clients if c.worker_id == verify.target_worker_id)
        with self.assertRaises(WorkerError) as ctx:
            await self.in_thread(accused.verdict, ipc.Verdict(
                challenge_id=challenge_id, verdict=1,
                replay_payload_hash=bytes([7]) * 32,
                determinism_class=verify.determinism_class))
        self.assertIn("own work", str(ctx.exception))

    async def test_ChallengesExpire(self):
        """The policy guarantees the deadline falls while the weights needed to
        answer are still retained, so one that expires is one nobody able to
        judge it chose to."""
        clients = await self.two_contributions()
        self.service.policy = type(self.policy)(
            **{**self.policy.__dict__, "challenge_percent": 100})
        await self.close_a_round(clients)
        self.assertEqual(self.service.expire_challenges(), 0)
        for challenge_id in list(self.service._pending_verify):
            verify, _ = self.service._pending_verify[challenge_id]
            self.service._pending_verify[challenge_id] = (verify, -1)
        self.assertEqual(self.service.expire_challenges(), 2)
        self.assertEqual(self.service._pending_verify, {})

    # -- from verdicts to a refusal ---------------------------------------------

    async def three_workers_one_challenged(self):
        """Three contributors, all challenged.

        Three because the quorum is two and nobody judges their own work: with
        two workers, every challenge has only one eligible verifier and no
        accusation can ever reach a quorum. That is a property of the network,
        not of the test — a two-worker network cannot prove anything.
        """
        published = []

        class FakeService:
            node = type("N", (), {"height": 0})()

            def publish(self, obj):
                published.append(obj)
                return 1

        self.service.service = FakeService()
        count = self.round_desc.model.parameter_count()
        clients, assignments = [], []
        # Every assignment first, then every submission. Interleaving them means
        # the third worker asks after the quorum is already met and is handed an
        # Apply instead — correct behaviour, wrong shape for this fixture.
        for _ in range(3):
            client = self.client()
            await self.in_thread(client.hello)
            clients.append(client)
            assignments.append(await self.in_thread(client.next_work))
        for seed, (client, assignment) in enumerate(zip(clients, assignments)):
            packed, exp = self.contribution(count, seed)
            await self.in_thread(client.submit, assignment.assignment_id,
                                 packed, exp, count, 5.7)
        self.service.policy = type(self.policy)(
            **{**self.policy.__dict__, "challenge_percent": 100})
        for client in clients:
            work = await self.in_thread(client.next_work)
            await self.in_thread(client.applied, work.outer_step,
                                 bytes([5]) * 32, bytes([6]) * 32)
        return clients, published

    async def test_AQuorumOfMismatchesBecomesEvidence(self):
        clients, published = await self.three_workers_one_challenged()
        challenge_id = next(iter(self.service._pending_verify))
        verify, _ = self.service._pending_verify[challenge_id]

        # regtest's quorum is two, so two distinct verifiers settle it.
        self.assertEqual(self.policy.slash_quorum, 2)
        for client in clients:
            if client.worker_id == verify.target_worker_id:
                continue                # nobody judges their own work
            await self.in_thread(client.verdict, ipc.Verdict(
                challenge_id=challenge_id, verdict=2,
                replay_payload_hash=bytes([client.worker_id]) * 32,
                determinism_class=verify.determinism_class))

        from rnet.consensus.objects import SlashEvidence
        proofs = [o for o in published if isinstance(o, SlashEvidence)]
        self.assertEqual(len(proofs), 1)
        self.assertEqual(proofs[0].target_worker_id, verify.target_worker_id)
        self.assertEqual(len(proofs[0].verdict_ids), 2)
        self.assertTrue(self.service.evidence.is_proven(verify.target_worker_id))

    async def test_ShadowModeProvesAndStillGivesWork(self):
        """Recorded, not acted on — which is the shipped setting."""
        clients, _ = await self.three_workers_one_challenged()
        challenge_id = next(iter(self.service._pending_verify))
        verify, _ = self.service._pending_verify[challenge_id]
        for client in clients:
            if client.worker_id == verify.target_worker_id:
                continue
            await self.in_thread(client.verdict, ipc.Verdict(
                challenge_id=challenge_id, verdict=2,
                replay_payload_hash=bytes([client.worker_id]) * 32,
                determinism_class=verify.determinism_class))

        accused = next(c for c in clients
                       if c.worker_id == verify.target_worker_id)
        self.assertTrue(self.service.evidence.is_proven(accused.worker_id))
        self.assertFalse(self.service.evidence.should_refuse(accused.worker_id))
        work = await self.in_thread(accused.next_work)
        self.assertIsInstance(work, (ipc.Assignment, ipc.Verify, ipc.NoWork))

    async def test_EnforcingStopsGivingTheProvenWorkerWork(self):
        clients, _ = await self.three_workers_one_challenged()
        challenge_id = next(iter(self.service._pending_verify))
        verify, _ = self.service._pending_verify[challenge_id]
        for client in clients:
            if client.worker_id == verify.target_worker_id:
                continue
            await self.in_thread(client.verdict, ipc.Verdict(
                challenge_id=challenge_id, verdict=2,
                replay_payload_hash=bytes([client.worker_id]) * 32,
                determinism_class=verify.determinism_class))

        # Flip enforcement on, as a network that had watched shadow mode long
        # enough would do by re-pinning its policy.
        self.service.evidence.shadow_mode = False
        accused = next(c for c in clients
                       if c.worker_id == verify.target_worker_id)
        innocent = next(c for c in clients
                        if c.worker_id != verify.target_worker_id)

        refused = await self.in_thread(accused.next_work)
        self.assertIsInstance(refused, ipc.NoWork)
        self.assertIn("did not replay", refused.reason)
        # And the other one is unaffected: guilt is per worker, not per node.
        self.assertNotIsInstance(await self.in_thread(innocent.next_work),
                                 ipc.NoWork)

    # -- being behind -----------------------------------------------------------

    async def test_AWorkerOutOfSyncIsDetectedByTheCommitmentAlreadyInTheHeader(self):
        """The only way a node can tell, and it needs no new mechanism.

        Momentum cannot be re-derived from the chain — under a constant
        aggregate the recurrence has a plateau of fixed points, so two differing
        histories stay different forever. But every checkpoint header already
        commits to optimizer_state_hash, so a worker's own hashes against the
        settled checkpoint answer the question exactly.
        """
        clients = await self.two_contributions()
        good = await self.in_thread(clients[0].next_work)
        await self.in_thread(clients[0].applied, good.outer_step,
                             bytes([5]) * 32, bytes([6]) * 32)
        self.assertEqual(self.chain.height, 1)

        # The second worker applies and reports something else: it is behind.
        work = await self.in_thread(clients[1].next_work)
        self.assertIsInstance(work, ipc.Apply)
        await self.in_thread(clients[1].applied, work.outer_step,
                             bytes([9]) * 32, bytes([9]) * 32)
        self.assertIs(self.service._in_sync[clients[1].worker_id], False)
        self.assertIs(self.service._in_sync.get(clients[0].worker_id), None)
        self.assertIn("behind", self.service.status())

    async def test_AWorkerThatAgreesIsMarkedInSync(self):
        clients = await self.two_contributions()
        good = await self.in_thread(clients[0].next_work)
        weights, optimizer = bytes([5]) * 32, bytes([6]) * 32
        await self.in_thread(clients[0].applied, good.outer_step, weights, optimizer)

        work = await self.in_thread(clients[1].next_work)
        await self.in_thread(clients[1].applied, work.outer_step, weights, optimizer)
        self.assertIs(self.service._in_sync[clients[1].worker_id], True)
        self.assertNotIn("behind", self.service.status())

    async def test_AWorkerKnownBehindDoesNotDefineACheckpoint(self):
        """Its arithmetic is not wrong — it is arithmetic over a momentum the
        rest of the network does not share, and a checkpoint built from it would
        be a fork with one participant."""
        clients = await self.two_contributions()
        first = await self.in_thread(clients[0].next_work)
        await self.in_thread(clients[0].applied, first.outer_step,
                             bytes([5]) * 32, bytes([6]) * 32)
        work = await self.in_thread(clients[1].next_work)
        await self.in_thread(clients[1].applied, work.outer_step,
                             bytes([9]) * 32, bytes([9]) * 32)
        self.assertIs(self.service._in_sync[clients[1].worker_id], False)

        # Next round: the behind worker must not be the one whose report becomes
        # the checkpoint.
        count = self.round_desc.model.parameter_count()
        for seed, client in enumerate(clients):
            assignment = await self.in_thread(client.next_work)
            if not isinstance(assignment, ipc.Assignment):
                continue
            packed, exp = self.contribution(count, seed + 5)
            await self.in_thread(client.submit, assignment.assignment_id,
                                 packed, exp, count, 5.7)
        behind = clients[1]
        work = await self.in_thread(behind.next_work)
        if isinstance(work, ipc.Apply):
            await self.in_thread(behind.applied, work.outer_step,
                                 bytes([9]) * 32, bytes([9]) * 32)
            self.assertEqual(self.chain.height, 1, "a behind worker produced")

    async def test_TheStatusLineNamesWhatMatters(self):
        client = self.client()
        await self.in_thread(client.hello)
        line = self.service.status()
        for fragment in ("workers", "accepted", "refused", "deferred",
                         "contribution"):
            self.assertIn(fragment, line)


if __name__ == "__main__":
    unittest.main(verbosity=2)

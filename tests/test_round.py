"""The end-to-end test: genesis to a checkpoint the chain accepts.

WHERE THIS RUNS. On CUDA when there is a card and CUBLAS_WORKSPACE_CONFIG was
set before python started; on CPU otherwise, which is correct but slow for a
reason worth writing down.

Training this model on CPU in bfloat16 is about forty times slower than the
arithmetic warrants, and not because of the dtype. `nn.Linear` computes x @ W.T,
so the backward pass hands torch a transposed operand, and torch's CPU bf16
kernel falls off its fast path when it gets one. Measured, same FLOPs each:

    [256,1024] @ [1024,256]     0.19 ms      contiguous
    [256, 256] @ [256,1024]    39.30 ms      transposed
    [1024,256] @ [256, 256]   155.65 ms      transposed

Nothing here can fix that, so the round fixtures are computed ONCE per class and
shared. The tests that follow are about consensus arithmetic, not about training,
and re-running the training for each of them would buy nothing.
"""

import os
import unittest

import numpy as np
import torch

from rnet.consensus import genesis
from rnet.consensus.init import hash_of_values
from rnet.diloco import inner
from rnet.diloco.aggregate import average_contributions, contribution_root
from rnet.diloco.chain import Outcome
from rnet.node.simulation import Simulation

# CUDA needs the cuBLAS workspace pinned before torch initialises it, which is
# at import — so this is a property of how the suite was launched, not something
# a test can arrange. ci/run_tests.sh sets it.
DEVICE = ("cuda" if torch.cuda.is_available()
          and os.environ.get("CUBLAS_WORKSPACE_CONFIG") else "cpu")


def sim(**kw) -> Simulation:
    inner.enable_determinism(0, device=DEVICE)
    inner.tune_cpu_threads()
    return Simulation("regtest", device=DEVICE, **kw)


class ThreadingTests(unittest.TestCase):
    """Thread count is a speed knob, not a consensus one — which had to be
    established rather than assumed, because a reduction whose order varied with
    it would put every CPU worker in its own determinism class."""

    def test_ThreadCountDoesNotChangeTheBytes(self):
        import hashlib

        from rnet.consensus.params import NETWORKS
        from rnet.model import weights as W

        rd, _ = NETWORKS["regtest"]
        g = bytes.fromhex(genesis.GENESIS_HASH["regtest"])
        toks = (torch.arange(rd.model.seq_len + 1).unsqueeze(0)
                % rd.model.vocab_size)

        seen = set()
        original = torch.get_num_threads()
        try:
            for threads in (1, 2, 4, 8):
                torch.set_num_threads(threads)
                torch.manual_seed(0)
                model = W.build(rd.model, rd.numerics, g, device="cpu",
                                grad_checkpointing=False)
                loss = model.loss(toks)
                loss.backward()
                grads = np.concatenate([
                    p.grad.flatten().float().numpy()
                    for _, p in sorted(model.named_parameters())])
                seen.add((round(float(loss), 10),
                          hashlib.sha3_256(grads.tobytes()).hexdigest()))
        finally:
            torch.set_num_threads(original)
        self.assertEqual(len(seen), 1, f"thread count changed the result: {seen}")

    def test_TuningReportsWhatItChose(self):
        original = torch.get_num_threads()
        try:
            self.assertEqual(inner.tune_cpu_threads(4), min(4, os.cpu_count() or 1))
            self.assertEqual(torch.get_num_threads(), min(4, os.cpu_count() or 1))
            self.assertEqual(inner.tune_cpu_threads(1), 1)
        finally:
            torch.set_num_threads(original)


class RoundTests(unittest.TestCase):
    """One round's training, done once, examined many ways."""

    @classmethod
    def setUpClass(cls):
        cls.s = sim()
        cls.genesis_head = cls.s.chain.head
        cls.results = [cls.s.train_one(w, 1) for w in (1, 2, 3)]

    def fresh(self) -> Simulation:
        """A second node, at genesis, for checking two producers agree."""
        return sim()

    # -- the round closes ---------------------------------------------------

    def test_TheChainStartsAtTheWeightsAnchor(self):
        self.assertEqual(self.genesis_head.header.weights_hash.hex(),
                         genesis.WEIGHTS_HASH["regtest"])

    def test_ARoundReachesACheckpointTheChainAccepts(self):
        node = self.fresh()
        checkpoint = node.produce(self.results, 1, producer_id=1)
        self.assertEqual(node.chain.add(checkpoint), Outcome.EXTENDED)
        self.assertEqual(node.chain.height, 1)
        self.assertEqual(checkpoint.parent, self.genesis_head.id)

    def test_TheCheckpointCommitsToTheWeightsThatCameOut(self):
        node = self.fresh()
        checkpoint = node.produce(self.results, 1, producer_id=1)
        self.assertEqual(checkpoint.weights_hash,
                         hash_of_values(node.spec, node.weights))
        # And they moved: a round that changed nothing would report the anchor.
        self.assertNotEqual(checkpoint.weights_hash.hex(),
                            genesis.WEIGHTS_HASH["regtest"])

    # -- the properties that make producing checkable ----------------------

    def test_TwoProducersWithTheSameContributionsAgree(self):
        a, b = self.fresh(), self.fresh()
        first = a.produce(self.results, 1, producer_id=1)
        second = b.produce(self.results, 1, producer_id=9)
        self.assertEqual(first.weights_hash, second.weights_hash)
        self.assertEqual(first.optimizer_state_hash, second.optimizer_state_hash)
        self.assertEqual(first.contribution_root, second.contribution_root)
        # The producer id is in the header, so the ids differ — which is exactly
        # what the fork-choice rule then settles.
        self.assertNotEqual(first.id, second.id)

    def test_ArrivalOrderDoesNotChangeTheCheckpoint(self):
        a, b = self.fresh(), self.fresh()
        forward = a.produce(self.results, 1, producer_id=1)
        backward = b.produce(list(reversed(self.results)), 1, producer_id=1)
        self.assertEqual(forward.weights_hash, backward.weights_hash)
        self.assertEqual(forward.id, backward.id)

    def test_EveryContributionIsInTheRoot(self):
        node = self.fresh()
        checkpoint = node.produce(self.results, 1, producer_id=1)
        self.assertEqual(checkpoint.contribution_root,
                         contribution_root([r.header.id for r in self.results]))
        # Dropping one changes the root, so a producer cannot quietly omit work.
        self.assertNotEqual(
            contribution_root([r.header.id for r in self.results[:-1]]),
            checkpoint.contribution_root)

    def test_TheAggregateIsTheMeanNotTheFirst(self):
        mean, _ = average_contributions([r.contribution for r in self.results])
        alone, _ = average_contributions([self.results[0].contribution])
        self.assertFalse(np.array_equal(mean, alone))

    # -- what a worker produces --------------------------------------------

    def test_WorkersTrainOnDifferentData(self):
        """Two workers seeing the same windows would make the second pointless."""
        a, b, c = (r.contribution.values for r in self.results)
        self.assertFalse(np.array_equal(a, b))
        self.assertFalse(np.array_equal(b, c))

    def test_AWorkerIsReplayable(self):
        """What a verifier does. Without it there is no telling a wrong answer
        from a different machine."""
        again = self.s.train_one(1, 1)
        first = self.results[0]
        self.assertTrue(np.array_equal(again.contribution.values,
                                       first.contribution.values))
        self.assertEqual(again.contribution.scale_exp, first.contribution.scale_exp)
        self.assertEqual(again.header.payload_hash, first.header.payload_hash)

    def test_ContributionsCarryTheWeightsTheyStartedFrom(self):
        """Without it a contribution is an update with no stated question."""
        for result in self.results:
            self.assertEqual(result.header.base_checkpoint, self.genesis_head.id)
            self.assertEqual(result.header.base_weights_hash,
                             self.genesis_head.header.weights_hash)

    def test_AContributionIsNotAllZero(self):
        """A round that moved nothing would quantise to zeros and still hash."""
        for result in self.results:
            self.assertGreater(int((result.contribution.values != 0).sum()), 0)
            self.assertLessEqual(int(np.abs(result.contribution.values).max()),
                                 result.contribution.fmt.max_magnitude)

    # -- several rounds ------------------------------------------------------

    def test_TooFewContributorsCannotCloseARound(self):
        """min_contributors is why a network of one never advances, and it says
        so rather than producing a checkpoint from one opinion."""
        node = self.fresh()
        with self.assertRaises(RuntimeError):
            node.run_round(1)
        self.assertEqual(node.chain.height, 0)

    def test_TheChainAndTheOptimizerBothAdvance(self):
        node = self.fresh()
        weights_seen = {node.chain.head.header.weights_hash}
        optimizer_seen = {node.optimizer.state_hash()}
        for step in (1, 2):
            report = node.run_round(2)
            self.assertEqual(report.outer_step, step)
            self.assertEqual(node.chain.height, step)
            self.assertNotIn(report.weights_hash, weights_seen)
            self.assertNotIn(report.optimizer_state_hash, optimizer_seen)
            weights_seen.add(report.weights_hash)
            optimizer_seen.add(report.optimizer_state_hash)
        self.assertEqual(node.optimizer.steps, 2)

    def test_TwoNodesFromGenesisAgreeCompletely(self):
        """Same code, same genesis, same rounds: the same chain head."""
        a, b = self.fresh(), self.fresh()
        for _ in range(2):
            a.run_round(2)
            b.run_round(2)
        self.assertEqual(a.chain.head.id, b.chain.head.id)
        self.assertEqual(a.optimizer.state_hash(), b.optimizer.state_hash())


if __name__ == "__main__":
    unittest.main(verbosity=2)

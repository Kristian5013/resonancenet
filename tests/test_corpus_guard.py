"""A round that pins a corpus and cannot read it must fail, not invent.

The rule was written into rnet/diloco/inner.py's docstring from the start and
the only production path violated it. An audit ran `rnet train --network test`
with no corpus anywhere on disk: it started without a word, trained fifty inner
steps on the 397-million-parameter model, and submitted a contribution at loss
10.53 — flat, pinned at ln(32000) = 10.37, the entropy of uniform random tokens.

The daemon accepted it. The chain advanced on it. And the verifier, replaying
from the same protocol state with the same absent corpus, reproduced the
identical noise and returned MATCH — so the anti-cheating machinery certified
work that could not possibly have learned anything.

Nothing in the 429 tests caught it, because every test injected a fake corpus
that production never constructs.
"""

import unittest

import numpy as np

from rnet.consensus import genesis
from rnet.consensus.numerics import ROUND0
from rnet.consensus.objects import Verdict as VerdictKind
from rnet.consensus.params import DENSE_400M, FINEWEB_EDU_ROOT, TINY_3M
from rnet.diloco import inner
from rnet.diloco.inner import InnerError
from rnet.worker import ipc
from rnet.worker.verify import replay


class CorpusGuardTests(unittest.TestCase):

    def test_APinnedCorpusWithNoReaderRefusesToTrain(self):
        with self.assertRaises(InnerError) as ctx:
            inner.derive_batch(DENSE_400M, FINEWEB_EDU_ROOT, round_id=0,
                               worker_id=1, outer_step=1, inner_index=0,
                               micro_batch=1, corpus=None)
        message = str(ctx.exception)
        self.assertIn("pins a corpus", message)
        self.assertIn(FINEWEB_EDU_ROOT.hex()[:16], message)

    def test_ANetworkWithNoCorpusPinnedStillTrains(self):
        """regtest is synthetic by design, and that is a real answer rather than
        a fallback — the difference the guard exists to make."""
        tokens = inner.derive_batch(TINY_3M, bytes(32), round_id=0, worker_id=1,
                                    outer_step=1, inner_index=0, micro_batch=1,
                                    corpus=None)
        self.assertEqual(tuple(tokens.shape), (1, TINY_3M.seq_len + 1))
        self.assertLess(int(tokens.max()), TINY_3M.vocab_size)

    def test_EveryShippedCorpusPinnedNetworkIsGuarded(self):
        for network in genesis.networks():
            round_desc = genesis.round_descriptor(network)
            if round_desc.dataset_root == bytes(32):
                continue
            with self.subTest(network):
                with self.assertRaises(InnerError):
                    inner.derive_batch(round_desc.model, round_desc.dataset_root,
                                       round_id=0, worker_id=1, outer_step=1,
                                       inner_index=0, micro_batch=1, corpus=None)

    def test_AVerifierWithoutTheCorpusSaysSoInsteadOfAccusing(self):
        """Replaying against synthetic tokens would reproduce noise, compare it
        to a real contribution's hash, and report MISMATCH — accusing an honest
        worker of exactly the thing the verifier cannot check."""
        message = ipc.Verify(
            challenge_id=bytes([1]) * 32, target_worker_id=2, round_id=0,
            outer_step=1, inner_steps=4, micro_batch=1,
            base_weights_hash=bytes([2]) * 32,
            claimed_payload_hash=bytes([3]) * 32,
            determinism_class=ROUND0.determinism_class)
        answer = replay(model=None, spec=DENSE_400M, numerics=ROUND0,
                        message=message, dataset_root=FINEWEB_EDU_ROOT,
                        base_weights=None, device="cpu", lr=1e-3, corpus=None)
        self.assertEqual(answer.verdict, int(VerdictKind.INDETERMINATE))
        self.assertIn("corpus", answer.note)
        self.assertEqual(answer.replay_payload_hash, bytes(32))

    def test_TheGuardComesBeforeAnythingExpensive(self):
        """It must refuse before a model is built or a step is taken, or the
        cost of the mistake is twenty minutes of GPU rather than nothing."""
        with self.assertRaises(InnerError):
            inner.derive_batch(DENSE_400M, FINEWEB_EDU_ROOT, round_id=0,
                               worker_id=1, outer_step=1, inner_index=0,
                               micro_batch=1, corpus=None)


if __name__ == "__main__":
    unittest.main(verbosity=2)

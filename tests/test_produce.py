"""Applying a round's aggregate, which is what decides a checkpoint.

The number this produces is `weights_hash`, and every node that applied the same
contributions to the same base must produce the same one — a disagreement here
is not a rounding difference, it is a fork. There were no tests for this class.
"""

import dataclasses
import unittest

import numpy as np

from rnet.consensus import genesis
from rnet.consensus.init import float32_to_bf16, hash_of_values
from rnet.diloco import outer as O
from rnet.diloco.quantize import pack, quantize_update
from rnet.model import weights as W
from rnet.model.layout import layout
from rnet.worker import ipc
from rnet.worker.produce import ProduceError, Producer

NETWORK = "regtest"


def _to_float(words: np.ndarray) -> np.ndarray:
    from rnet.consensus.init import bf16_to_float32
    return bf16_to_float32(words).astype(np.float64)


class ProducerTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.round_desc = genesis.round_descriptor(NETWORK)
        cls.policy = genesis.policy_descriptor(NETWORK)
        cls.spec = cls.round_desc.model
        cls.numerics = cls.round_desc.numerics
        cls.genesis_hash = bytes.fromhex(genesis.GENESIS_HASH[NETWORK])

    def model(self):
        return W.build(self.spec, self.numerics, self.genesis_hash, device="cpu")

    def message(self, seed: int = 0, *, nesterov: bool = True) -> ipc.Apply:
        count = self.spec.parameter_count()
        rng = np.random.default_rng(seed)
        values, exp = quantize_update(rng.normal(0, 1e-3, count),
                                      self.numerics.contribution_format)
        return ipc.Apply(
            outer_step=1, scale_exp=exp, value_count=count,
            momentum_q16=self.policy.outer_momentum_q16,
            lr_q16=self.policy.outer_lr_q16, nesterov=nesterov,
            packed=pack(values, self.numerics.contribution_format))

    # -- the change that mattered ---------------------------------------------

    def test_ApplyingATensorAtATimeIsTheSameWeights(self):
        """The whole-array version built two float64 arrays of every parameter
        and sliced the second apart. At 397,728,768 parameters that is 3.18 GB
        each on top of the momentum and the aggregate, and a worker peaked at
        27.8 GB and was OOM-killed. The arithmetic is elementwise, so a tensor
        at a time is the same arithmetic — and a checkpoint that differed by one
        bit from what another node produced would be a fork.
        """
        message = self.message()
        base = W.save_weights(self.model())

        # What the previous implementation did, kept here as the definition.
        from rnet.diloco.quantize import unpack
        update = unpack(message.packed, message.value_count,
                        self.numerics.contribution_format)
        optimizer = O.OuterOptimizer(momentum_q16=message.momentum_q16,
                                     lr_q16=message.lr_q16,
                                     nesterov=message.nesterov)
        step, exp = optimizer.step(update, message.scale_exp)
        order = [t.name for t in layout(self.spec)]
        flat = np.concatenate([_to_float(base[name]) for name in order])
        moved = O.apply_update(flat, step, exp)
        want = dict(base)
        at = 0
        for tensor in layout(self.spec):
            want[tensor.name] = float32_to_bf16(
                moved[at:at + tensor.numel].astype(np.float32))
            at += tensor.numel

        model = self.model()
        applied = Producer(self.spec).apply(
            model, message, self.numerics.contribution_format, base)

        self.assertEqual(applied.weights_hash, hash_of_values(self.spec, want))
        got = W.save_weights(model)
        for name in want:
            np.testing.assert_array_equal(got[name], want[name], name)

    # -- what it promises ------------------------------------------------------

    def test_TwoProducersOnTheSameInputsAgree(self):
        """The property a checkpoint rests on. Integer aggregation and a
        power-of-two scale exist so that this is equality rather than
        similarity."""
        message = self.message(3)
        base = W.save_weights(self.model())
        first = Producer(self.spec).apply(self.model(), message,
                                          self.numerics.contribution_format, base)
        second = Producer(self.spec).apply(self.model(), message,
                                           self.numerics.contribution_format, base)
        self.assertEqual(first.weights_hash, second.weights_hash)
        self.assertEqual(first.optimizer_state_hash, second.optimizer_state_hash)

    def test_TheWeightsActuallyMove(self):
        message = self.message(5)
        base = W.save_weights(self.model())
        model = self.model()
        applied = Producer(self.spec).apply(
            model, message, self.numerics.contribution_format, base)
        self.assertNotEqual(applied.weights_hash, hash_of_values(self.spec, base))

    def test_TheModelIsLeftHoldingWhatWasReported(self):
        """Not merely returned: the next round trains from this model, so a
        report that did not match what was loaded would be a worker training
        from weights it told the network it was not at."""
        message = self.message(7)
        base = W.save_weights(self.model())
        model = self.model()
        applied = Producer(self.spec).apply(
            model, message, self.numerics.contribution_format, base)
        self.assertEqual(hash_of_values(self.spec, W.save_weights(model)),
                         applied.weights_hash)

    def test_ADifferentAggregateIsADifferentCheckpoint(self):
        base = W.save_weights(self.model())
        one = Producer(self.spec).apply(self.model(), self.message(1),
                                        self.numerics.contribution_format, base)
        two = Producer(self.spec).apply(self.model(), self.message(2),
                                        self.numerics.contribution_format, base)
        self.assertNotEqual(one.weights_hash, two.weights_hash)

    # -- what it refuses -------------------------------------------------------

    def test_AWrongValueCountIsRefused(self):
        message = dataclasses.replace(self.message(),
                                      value_count=self.spec.parameter_count() + 1)
        with self.assertRaises(ProduceError) as caught:
            Producer(self.spec).apply(self.model(), message,
                                      self.numerics.contribution_format,
                                      W.save_weights(self.model()))
        self.assertIn("parameters", str(caught.exception))

    def test_RatesChangingUnderARunningOptimizerAreRefused(self):
        """The momentum carried forward was accumulated under the old rates, so
        continuing would apply a recurrence nobody agreed to."""
        producer = Producer(self.spec)
        base = W.save_weights(self.model())
        producer.apply(self.model(), self.message(1),
                       self.numerics.contribution_format, base)
        changed = dataclasses.replace(self.message(2),
                                      lr_q16=self.policy.outer_lr_q16 + 1)
        with self.assertRaises(ProduceError) as caught:
            producer.apply(self.model(), changed,
                           self.numerics.contribution_format, base)
        self.assertIn("momentum", str(caught.exception))

    def test_MomentumCarriesBetweenRounds(self):
        """Two rounds through one producer differ from the same two through a
        fresh one, or the outer optimizer would have no state at all."""
        base = W.save_weights(self.model())
        producer = Producer(self.spec)
        producer.apply(self.model(), self.message(1),
                       self.numerics.contribution_format, base)
        carried = producer.apply(self.model(), self.message(2),
                                 self.numerics.contribution_format, base)
        fresh = Producer(self.spec).apply(self.model(), self.message(2),
                                          self.numerics.contribution_format, base)
        self.assertNotEqual(carried.weights_hash, fresh.weights_hash)


if __name__ == "__main__":
    unittest.main(verbosity=2)

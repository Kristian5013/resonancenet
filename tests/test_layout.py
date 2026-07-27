"""Tests for the canonical tensor layout.

This is where the debt from model_spec.py is paid: the parameter formula and the
tensor list are two independent descriptions of one model, and here they are
compared.
"""

import unittest

from rnet.consensus.model_spec import ModelSpec
from rnet.consensus.params import DENSE_400M, MOE_29B, TINY_3M
from rnet.model.layout import (Kind, active_parameter_count, layout,
                               parameter_count, shard_layout,
                               shard_parameter_count, shard_of)

MODELS = {"dense-400m": DENSE_400M, "moe-29b": MOE_29B, "tiny-3m": TINY_3M}


class LayoutTests(unittest.TestCase):

    def test_TheFormulaMatchesTheTensors(self):
        """The debt model_spec.py records. Two descriptions, one model."""
        for name, spec in MODELS.items():
            with self.subTest(name):
                self.assertEqual(parameter_count(spec), spec.parameter_count())
                self.assertEqual(active_parameter_count(spec),
                                 spec.active_parameter_count())
                self.assertEqual(shard_parameter_count(spec, 0),
                                 spec.shard_parameter_count())

    def test_TheFormulaMatchesTheTensorsAcrossShapes(self):
        """Not just the three shipped shapes — a grid, so an edit to either
        description that happens to agree on those three still fails."""
        for layers in (1, 2, 5):
            for qk in (True, False):
                for tie in (True, False):
                    for experts, active, shared, first, stride in (
                            (0, 0, 0, 0, 1), (4, 2, 0, 0, 1),
                            (8, 1, 1, 1, 2), (4, 4, 2, 0, 1)):
                        spec = ModelSpec(
                            d_model=64, n_layers=layers, n_heads=4, n_kv_heads=2,
                            d_ff=128, vocab_size=99, seq_len=32, rope_theta=10000,
                            tie_embeddings=tie, qk_norm=qk,
                            n_experts=experts, n_experts_active=active,
                            n_shared_experts=shared,
                            d_ff_expert=48 if experts else 0,
                            moe_first_layer=first if experts else 0,
                            moe_layer_stride=stride if experts else 1,
                            expert_shard_count=2 if experts else 1)
                        if experts and first >= layers:
                            continue
                        spec.validate()
                        with self.subTest(f"{layers}L qk={qk} tie={tie} e={experts}"):
                            self.assertEqual(parameter_count(spec), spec.parameter_count())
                            self.assertEqual(active_parameter_count(spec),
                                             spec.active_parameter_count())

    def test_EveryNameIsUnique(self):
        """Two tensors sharing a name would collide in the keyed init stream."""
        for name, spec in MODELS.items():
            with self.subTest(name):
                names = [t.name for t in layout(spec)]
                self.assertEqual(len(set(names)), len(names))

    def test_TheOrderIsStable(self):
        for spec in MODELS.values():
            self.assertEqual([t.name for t in layout(spec)],
                             [t.name for t in layout(spec)])

    def test_EmbeddingsComeFirstAndTheNormLast(self):
        for name, spec in MODELS.items():
            with self.subTest(name):
                tensors = layout(spec)
                self.assertEqual(tensors[0].name, "embed.weight")
                last = "norm.weight" if spec.tie_embeddings else "lm_head.weight"
                self.assertEqual(tensors[-1].name, last)

    def test_TyingRemovesTheHead(self):
        untied = ModelSpec(**{**DENSE_400M.__dict__, "tie_embeddings": False})
        names = [t.name for t in layout(untied)]
        self.assertIn("lm_head.weight", names)
        self.assertNotIn("lm_head.weight", [t.name for t in layout(DENSE_400M)])
        self.assertEqual(parameter_count(untied) - parameter_count(DENSE_400M),
                         DENSE_400M.vocab_size * DENSE_400M.d_model)

    def test_QkNormsAppearOnlyWhenAsked(self):
        without = ModelSpec(**{**DENSE_400M.__dict__, "qk_norm": False})
        self.assertEqual(sum(1 for t in layout(without) if "q_norm" in t.name), 0)
        self.assertEqual(sum(1 for t in layout(DENSE_400M) if "q_norm" in t.name),
                         DENSE_400M.n_layers)

    # -- mixture ------------------------------------------------------------

    def test_TheRouterAndSharedExpertsBelongToEveryone(self):
        """A holder without them could not tell whether a token was its business."""
        for t in layout(MOE_29B):
            if ".moe.router." in t.name or ".moe.shared." in t.name:
                self.assertIsNone(shard_of(MOE_29B, t), t.name)

    def test_EveryRoutedExpertBelongsToExactlyOneShard(self):
        counts = {}
        for t in layout(MOE_29B):
            if t.expert is None:
                continue
            s = shard_of(MOE_29B, t)
            self.assertIsNotNone(s)
            counts.setdefault(t.expert, set()).add(s)
        self.assertEqual(len(counts), MOE_29B.n_experts)
        for expert, shards in counts.items():
            self.assertEqual(len(shards), 1, f"expert {expert} in {shards}")

    def test_TheShardsCoverTheModelWithoutOverlapExceptWhatIsShared(self):
        every = layout(MOE_29B)
        shared = {t.name for t in every if t.expert is None}
        seen = set()
        for s in range(MOE_29B.expert_shard_count):
            held = {t.name for t in shard_layout(MOE_29B, s)}
            self.assertTrue(shared <= held, f"shard {s} is missing shared tensors")
            private = held - shared
            self.assertFalse(private & seen, f"shard {s} overlaps another")
            seen |= private
        self.assertEqual(seen | shared, {t.name for t in every})

    def test_EveryShardIsTheSameSize(self):
        sizes = {shard_parameter_count(MOE_29B, s)
                 for s in range(MOE_29B.expert_shard_count)}
        self.assertEqual(len(sizes), 1)
        self.assertEqual(sizes.pop(), 8_344_841_216)

    def test_AShardOutsideTheCountIsRefused(self):
        with self.assertRaises(ValueError):
            shard_layout(MOE_29B, MOE_29B.expert_shard_count)
        with self.assertRaises(ValueError):
            shard_layout(MOE_29B, -1)

    def test_ADenseModelHasOneShardHoldingEverything(self):
        self.assertEqual(shard_layout(DENSE_400M, 0), layout(DENSE_400M))
        for t in layout(DENSE_400M):
            self.assertIsNone(shard_of(DENSE_400M, t))

    # -- what the initialiser needs ----------------------------------------

    def test_EveryTensorSaysHowItStarts(self):
        for name, spec in MODELS.items():
            with self.subTest(name):
                for t in layout(spec):
                    self.assertIsInstance(t.kind, Kind)
                    self.assertGreater(t.numel, 0)
                    self.assertGreaterEqual(t.fan_in, 1)

    def test_NormsAreTheOnlyOneDimensionalTensors(self):
        for spec in MODELS.values():
            for t in layout(spec):
                self.assertEqual(len(t.shape) == 1, t.kind is Kind.NORM, t.name)

    def test_FanInIsTheTrailingDimension(self):
        spec = DENSE_400M
        wq = next(t for t in layout(spec) if t.name == "layers.0.attn.wq.weight")
        self.assertEqual(wq.shape, (spec.n_heads * spec.head_dim, spec.d_model))
        self.assertEqual(wq.fan_in, spec.d_model)
        wo = next(t for t in layout(spec) if t.name == "layers.0.attn.wo.weight")
        self.assertEqual(wo.fan_in, spec.n_heads * spec.head_dim)


if __name__ == "__main__":
    unittest.main(verbosity=2)

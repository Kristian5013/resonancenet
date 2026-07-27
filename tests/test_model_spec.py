"""Tests for the model shape — dense and mixture-of-experts.

The arithmetic here decides who can participate, so it is pinned against
literals rather than against itself.
"""

import unittest

from rnet.canon.stream import CanonError, Reader, Writer
from rnet.consensus.model_spec import ModelSpec

DENSE_400M = ModelSpec(
    d_model=1024, n_layers=24, n_heads=8, n_kv_heads=2, d_ff=4096,
    vocab_size=32000, seq_len=16384, rope_theta=1000000,
    tie_embeddings=True, qk_norm=True,
)

# 29.4B held by the network, 3.1B multiplied per token, 8.34B on any one holder.
# The configuration that answers "a 30B model on a 24 GB card": four shards, and
# a shard is 15.5 GiB in bf16 against 19.9 GiB of budget once activations are
# taken out. Pinned because these numbers are the reason the shape is this shape.
MOE_30B = ModelSpec(
    d_model=3072, n_layers=32, n_heads=24, n_kv_heads=4, d_ff=8192,
    vocab_size=32000, seq_len=8192, rope_theta=1000000,
    tie_embeddings=True, qk_norm=True,
    n_experts=64, n_experts_active=4, n_shared_experts=1, d_ff_expert=1536,
    moe_first_layer=1, moe_layer_stride=1, expert_shard_count=4,
)


def roundtrip(spec: ModelSpec) -> ModelSpec:
    r = Reader(spec.serialize(Writer()).take())
    out = ModelSpec.parse(r)
    r.expect_exhausted()
    return out


class ModelTests(unittest.TestCase):

    # -- dense --------------------------------------------------------------

    def test_TheDenseCountIsUnchangedFromRound0(self):
        """Pinned against the model that actually trained a round."""
        self.assertEqual(DENSE_400M.parameter_count(), 397_728_768)
        self.assertEqual(DENSE_400M.active_parameter_count(), 397_728_768)
        self.assertEqual(DENSE_400M.shard_parameter_count(), 397_728_768)

    def test_ADenseModelIsDenseInEveryField(self):
        """One byte pattern means dense; a half-filled mixture cannot pass."""
        for field, value in (("n_experts_active", 2), ("n_shared_experts", 1),
                             ("d_ff_expert", 512), ("moe_first_layer", 3)):
            with self.assertRaises(CanonError, msg=field):
                ModelSpec(**{**DENSE_400M.__dict__, field: value}).validate()
        with self.assertRaises(CanonError):
            ModelSpec(**{**DENSE_400M.__dict__, "moe_layer_stride": 2}).validate()
        with self.assertRaises(CanonError):
            ModelSpec(**{**DENSE_400M.__dict__, "expert_shard_count": 2}).validate()

    def test_DenseRoundTrips(self):
        self.assertEqual(roundtrip(DENSE_400M), DENSE_400M)

    # -- mixture ------------------------------------------------------------

    def test_TheThirtyBillionShapeFitsAConsumerCard(self):
        """The claim the shape exists to support, as numbers."""
        s = MOE_30B
        self.assertEqual(s.parameter_count(), 29_408_635_904)
        self.assertEqual(s.active_parameter_count(), 3_078_892_544)
        self.assertEqual(s.shard_parameter_count(), 8_344_841_216)

        weights_gib = s.bytes_per_shard(16) / 2**30
        self.assertLess(weights_gib, 16.0)   # measured: 15.54 GiB
        # 24 GB card, ~4 GiB of activations at this length with checkpointing.
        self.assertLess(weights_gib + 4.0, 23.9)

    def test_ActiveIsFarBelowTotalInAMixture(self):
        """The property that makes a mixture worth the complexity."""
        self.assertLess(MOE_30B.active_parameter_count() * 9, MOE_30B.parameter_count())
        # And a shard is smaller still than what the network holds.
        self.assertLess(MOE_30B.shard_parameter_count() * 3, MOE_30B.parameter_count())

    def test_MoreShardsMeansLessPerHolder(self):
        counts = [ModelSpec(**{**MOE_30B.__dict__, "expert_shard_count": n})
                  .shard_parameter_count() for n in (1, 2, 4, 8, 16)]
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertEqual(counts[0], MOE_30B.parameter_count())

    def test_WhichLayersAreMixtureLayers(self):
        s = MOE_30B  # first_layer 1, stride 1
        self.assertFalse(s.layer_is_moe(0))
        self.assertTrue(all(s.layer_is_moe(i) for i in range(1, s.n_layers)))
        self.assertEqual(s.n_moe_layers, s.n_layers - 1)

        every_other = ModelSpec(**{**MOE_30B.__dict__, "moe_layer_stride": 2})
        self.assertEqual([i for i in range(8) if every_other.layer_is_moe(i)],
                         [1, 3, 5, 7])

    def test_MoeRoundTrips(self):
        self.assertEqual(roundtrip(MOE_30B), MOE_30B)

    def test_ExpertsMustDivideIntoShardsEvenly(self):
        """Uneven shards would set the joinable card size by the largest one."""
        with self.assertRaises(CanonError):
            ModelSpec(**{**MOE_30B.__dict__, "expert_shard_count": 5}).validate()
        ModelSpec(**{**MOE_30B.__dict__, "expert_shard_count": 16}).validate()

    def test_RoutingBeyondTheExpertsIsRefused(self):
        with self.assertRaises(CanonError):
            ModelSpec(**{**MOE_30B.__dict__, "n_experts_active": 65}).validate()
        with self.assertRaises(CanonError):
            ModelSpec(**{**MOE_30B.__dict__, "n_experts_active": 0}).validate()

    def test_AFirstLayerOutsideTheModelIsRefused(self):
        """n_experts set with no layer to carry them is a contradiction."""
        with self.assertRaises(CanonError):
            ModelSpec(**{**MOE_30B.__dict__, "moe_first_layer": 40}).validate()
        with self.assertRaises(CanonError):
            ModelSpec(**{**MOE_30B.__dict__, "moe_first_layer": 32}).validate()

    def test_AFirstLayerInsideTheModelGuaranteesAMixtureLayer(self):
        """Why the range check is the only check the emptiness case needs.

        Layer `moe_first_layer` always satisfies the stride test against itself,
        so however large the stride, the mixture is never empty. Checked across
        the whole grid rather than argued, because "unreachable" is exactly the
        claim that rots quietly.
        """
        for first in range(MOE_30B.n_layers):
            for stride in (1, 2, 3, 7, 31, 64, 1000):
                spec = ModelSpec(**{**MOE_30B.__dict__, "moe_first_layer": first,
                                    "moe_layer_stride": stride})
                spec.validate()
                self.assertGreaterEqual(spec.n_moe_layers, 1, f"{first}/{stride}")
                self.assertTrue(spec.layer_is_moe(first))

    def test_SharedExpertsCountTowardEveryToken(self):
        """A shared expert is always routed, so it is always active."""
        without = ModelSpec(**{**MOE_30B.__dict__, "n_shared_experts": 0})
        self.assertGreater(MOE_30B.active_parameter_count(),
                           without.active_parameter_count())
        self.assertEqual(
            MOE_30B.active_parameter_count() - without.active_parameter_count(),
            MOE_30B.n_moe_layers * 3 * MOE_30B.d_model * MOE_30B.d_ff_expert)

    # -- shape invariants ---------------------------------------------------

    def test_AttentionShapeIsChecked(self):
        with self.assertRaises(CanonError):
            ModelSpec(**{**DENSE_400M.__dict__, "n_heads": 7}).validate()
        with self.assertRaises(CanonError):
            ModelSpec(**{**DENSE_400M.__dict__, "n_kv_heads": 3}).validate()

    def test_ZeroDimensionsAreRefused(self):
        for field in ("d_model", "n_layers", "n_heads", "n_kv_heads", "d_ff",
                      "vocab_size", "seq_len"):
            with self.assertRaises(CanonError, msg=field):
                ModelSpec(**{**DENSE_400M.__dict__, field: 0}).validate()

    def test_AnUnknownNormOrActivationIsRefused(self):
        with self.assertRaises(CanonError):
            ModelSpec(**{**DENSE_400M.__dict__, "norm_id": 2}).validate()
        with self.assertRaises(CanonError):
            ModelSpec(**{**DENSE_400M.__dict__, "activation_id": 2}).validate()

    def test_EveryFieldSurvivesTheWire(self):
        """A field that does not round-trip is a field the hash does not cover."""
        for field, value in (("d_model", 2048), ("n_layers", 12), ("n_heads", 16),
                             ("n_kv_heads", 8), ("d_ff", 5504), ("vocab_size", 128256),
                             ("seq_len", 4096), ("rope_theta", 500000),
                             ("tie_embeddings", False), ("qk_norm", False),
                             ("n_experts", 8), ("n_experts_active", 2),
                             ("n_shared_experts", 2), ("d_ff_expert", 704),
                             ("moe_first_layer", 2), ("moe_layer_stride", 3),
                             ("expert_shard_count", 2)):
            spec = ModelSpec(**{**MOE_30B.__dict__, field: value})
            try:
                spec.validate()
            except CanonError:
                continue
            self.assertEqual(roundtrip(spec), spec, field)
            self.assertNotEqual(spec.serialize(Writer()).take(),
                                MOE_30B.serialize(Writer()).take(), field)


if __name__ == "__main__":
    unittest.main(verbosity=2)

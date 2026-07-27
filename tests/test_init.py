"""Tests for deriving the first weights from the genesis hash.

Needs numpy — the only test module that needs anything outside the standard
library. That is a real weakening of "an auditor needs nothing", and it is
confined here on purpose: checking what a network CLAIMS stays stdlib-only,
while checking 400 million derived floats does not.
"""

import unittest

import numpy as np

from rnet.consensus import genesis, init
from rnet.consensus.params import DENSE_400M, MOE_29B, TINY_3M
from rnet.model.layout import Kind, layout, shard_layout

G = bytes.fromhex(genesis.GENESIS_HASH["regtest"])


class InitTests(unittest.TestCase):

    # -- the rounding -------------------------------------------------------

    def test_RoundingIsHalfToEvenAtTheTie(self):
        """Named because "round half up" is the default everywhere else, and
        the difference only shows on exact ties — which is where two
        implementations quietly diverge."""
        def bits(x):
            return int(init.float32_to_bf16(np.array([x], dtype=np.float32))[0])

        # 1.0 + one ulp of bf16 is 0x3F81; halfway to it is 0x3F808000, whose
        # low half is exactly half. The retained bit is odd, so it rounds up.
        self.assertEqual(bits(np.float32(np.uint32(0x3F808000).view(np.float32)
                                         if False else 0)), 0x0000)
        tie_odd = np.array([0x3F818000], dtype=np.uint32).view(np.float32)[0]
        self.assertEqual(bits(tie_odd), 0x3F82)      # 0x3F81 is odd -> up
        tie_even = np.array([0x3F808000], dtype=np.uint32).view(np.float32)[0]
        self.assertEqual(bits(tie_even), 0x3F80)     # 0x3F80 is even -> stays

    def test_RoundingIsExactWhenNothingIsLost(self):
        for value, expected in ((1.0, 0x3F80), (-1.0, 0xBF80), (0.0, 0x0000),
                                (2.0, 0x4000), (0.5, 0x3F00)):
            self.assertEqual(int(init.float32_to_bf16(
                np.array([value], dtype=np.float32))[0]), expected, value)

    def test_NanSurvivesRounding(self):
        """A NaN whose surviving mantissa bits are zero would become infinity,
        turning "no answer" into a very large answer."""
        sneaky = np.array([0x7F800001], dtype=np.uint32).view(np.float32)
        out = init.float32_to_bf16(sneaky)
        self.assertTrue(np.isnan(init.bf16_to_float32(out))[0])
        self.assertFalse(np.isinf(init.bf16_to_float32(out))[0])

    def test_Bf16RoundTripsThroughFloat32(self):
        raw = np.array([0x3F80, 0xBF80, 0x0000, 0x4049], dtype=np.uint16)
        self.assertTrue(np.array_equal(
            init.float32_to_bf16(init.bf16_to_float32(raw)), raw))

    # -- the stream ---------------------------------------------------------

    def test_TheStreamIsKeyedByNameNotByPosition(self):
        """So an architecture edit changes only what it touches.

        Turning off the qk norms removes two tensors from every layer. With a
        running counter that would move every value after the first layer; keyed
        by name, a tensor with the same name and shape is untouched.
        """
        from rnet.consensus.model_spec import ModelSpec
        without = ModelSpec(**{**TINY_3M.__dict__, "qk_norm": False})
        a = next(t for t in layout(TINY_3M) if t.name == "layers.3.ffn.down.weight")
        b = next(t for t in layout(without) if t.name == "layers.3.ffn.down.weight")
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.array_equal(init.derive_tensor(a, G),
                                       init.derive_tensor(b, G)))

    def test_DifferentTensorsGetDifferentValues(self):
        vals = {}
        for t in layout(TINY_3M):
            if t.kind is Kind.NORM:
                continue
            vals[t.name] = init.derive_tensor(t, G).tobytes()
        self.assertEqual(len(set(vals.values())), len(vals))

    def test_ADifferentGenesisGivesDifferentWeights(self):
        t = next(x for x in layout(TINY_3M) if x.name == "embed.weight")
        other = bytes.fromhex(genesis.GENESIS_HASH["main"])
        self.assertFalse(np.array_equal(init.derive_tensor(t, G),
                                        init.derive_tensor(t, other)))

    def test_DerivationIsDeterministic(self):
        t = next(x for x in layout(TINY_3M) if x.name == "layers.0.attn.wq.weight")
        self.assertTrue(np.array_equal(init.derive_tensor(t, G),
                                       init.derive_tensor(t, G)))

    # -- the values ---------------------------------------------------------

    def test_NormsAreOnesAndConsumeNoStream(self):
        for t in layout(TINY_3M):
            if t.kind is Kind.NORM:
                v = init.bf16_to_float32(init.derive_tensor(t, G))
                self.assertTrue(np.all(v == 1.0), t.name)

    def test_ValuesStayInsideTheFanInBound(self):
        for t in layout(TINY_3M):
            if t.kind is Kind.NORM:
                continue
            v = init.bf16_to_float32(init.derive_tensor(t, G))
            bound = 1.0 / np.sqrt(t.fan_in)
            # bf16 rounding can push a value one ulp past the open bound, which
            # is 1/256 of the magnitude at worst.
            self.assertLessEqual(float(np.abs(v).max()), bound * 1.004, t.name)

    def test_TheDistributionIsUniformNotNormal(self):
        """A uniform on ±b has standard deviation b/sqrt(3); a normal scaled to
        the same bound would not. Checks the shape, not just the range."""
        t = next(x for x in layout(TINY_3M) if x.name == "layers.0.attn.wq.weight")
        v = init.bf16_to_float32(init.derive_tensor(t, G))
        bound = 1.0 / np.sqrt(t.fan_in)
        self.assertAlmostEqual(float(v.std()), bound / np.sqrt(3), delta=bound * 0.02)
        self.assertAlmostEqual(float(v.mean()), 0.0, delta=bound * 0.02)

    # -- shards -------------------------------------------------------------

    def test_AShardDerivesItsOwnExpertsWithoutTheRest(self):
        """What makes a 29B mixture checkable by the people holding it."""
        held = shard_layout(MOE_29B, 2)
        names = {t.name for t in held}
        self.assertIn("layers.1.moe.experts.32.gate.weight", names)
        self.assertNotIn("layers.1.moe.experts.0.gate.weight", names)
        # And the value matches what the full layout would give for that tensor.
        one = next(t for t in held if t.name == "layers.1.moe.experts.32.gate.weight")
        full = next(t for t in layout(MOE_29B)
                    if t.name == "layers.1.moe.experts.32.gate.weight")
        self.assertTrue(np.array_equal(init.derive_tensor(one, G),
                                       init.derive_tensor(full, G)))

    def test_DeriveAllHonoursTheShard(self):
        """On a mixture small enough to hold. Deriving a shard of MOE_29B would
        be 8.3 billion values and 16 GB, which is a benchmark, not a test."""
        from rnet.consensus.model_spec import ModelSpec
        small = ModelSpec(d_model=64, n_layers=3, n_heads=4, n_kv_heads=2, d_ff=128,
                          vocab_size=99, seq_len=32, rope_theta=10000,
                          tie_embeddings=True, qk_norm=True,
                          n_experts=8, n_experts_active=2, n_shared_experts=1,
                          d_ff_expert=48, moe_first_layer=1, moe_layer_stride=1,
                          expert_shard_count=4)
        small.validate()
        every = init.derive_all(TINY_3M, G)
        self.assertEqual(set(every), {t.name for t in layout(TINY_3M)})
        for s in range(small.expert_shard_count):
            one = init.derive_all(small, G, shard=s)
            self.assertEqual(set(one), {t.name for t in shard_layout(small, s)})
            # A shard's values are the same values the full model would have.
            name = f"layers.1.moe.experts.{s * small.experts_per_shard}.gate.weight"
            full = next(t for t in layout(small) if t.name == name)
            self.assertTrue(np.array_equal(one[name], init.derive_tensor(full, G)))

    # -- the hash -----------------------------------------------------------

    def test_ParallelDerivationMatchesSequential(self):
        """Leaves come back in layout order, never as they finish — the root is
        defined by the order."""
        self.assertEqual(init.weights_hash(TINY_3M, G, workers=1),
                         init.weights_hash(TINY_3M, G, workers=8))
        self.assertEqual(init.leaves(TINY_3M, G, workers=1),
                         init.leaves(TINY_3M, G, workers=8))

    def test_TheRegtestWeightsHashIsPinned(self):
        """The fourth anchor, for the network small enough to check in a test."""
        self.assertEqual(
            init.weights_hash(TINY_3M, G).hex(),
            "f1f0c82d26889543586f6a5bbe76d301696edf017b159d66fb343eb21bd40cae")
        self.assertEqual(genesis.WEIGHTS_HASH["regtest"],
                         init.weights_hash(TINY_3M, G).hex())

    def test_OneTensorCanProveItBelongs(self):
        """What a flat digest could not offer at any price.

        A shard holder proves its experts are the ones the network settled on
        without producing a model that exists nowhere in one place.
        """
        from rnet.crypto import merkle
        root = init.weights_hash(TINY_3M, G)
        name = "layers.2.ffn.up.weight"
        proof = init.tensor_proof(TINY_3M, G, name)
        spec = next(t for t in layout(TINY_3M) if t.name == name)
        leaf = merkle.leaf_hash(init.tensor_bytes(spec, G))
        self.assertTrue(merkle.verify_proof(leaf, proof, root))

        # And a tensor that is not the one claimed does not prove.
        other = next(t for t in layout(TINY_3M) if t.name == "layers.2.ffn.gate.weight")
        self.assertFalse(merkle.verify_proof(
            merkle.leaf_hash(init.tensor_bytes(other, G)), proof, root))

    def test_AProofForAnAbsentTensorIsRefused(self):
        with self.assertRaises(KeyError):
            init.tensor_proof(TINY_3M, G, "layers.0.moe.router.weight")

    def test_TheStreamIsBlockedWithoutChangingTheValues(self):
        """Blocking bounds the largest allocation; it must not move a value.

        A tensor spanning several blocks and one that fits in a single block
        must agree wherever they overlap, or the block index would be silently
        part of the derivation for large tensors only.
        """
        big = init._stream("probe", G, init.BLOCK_ELEMS + 100)
        small = init._stream("probe", G, 64)
        self.assertTrue(np.array_equal(big[:64], small))
        # And the second block is not a repeat of the first.
        self.assertFalse(np.array_equal(big[:100], big[init.BLOCK_ELEMS:]))

    def test_HashingDerivedValuesMatchesDerivingTheHash(self):
        self.assertEqual(init.hash_of_values(TINY_3M, init.derive_all(TINY_3M, G)),
                         init.weights_hash(TINY_3M, G))

    def test_AMissingOrMisshapenTensorIsRefused(self):
        every = init.derive_all(TINY_3M, G)
        short = dict(every)
        del short["layers.0.attn.wq.weight"]
        with self.assertRaises(KeyError):
            init.hash_of_values(TINY_3M, short)
        wrong = dict(every)
        wrong["layers.0.attn.wq.weight"] = np.zeros(3, dtype=np.uint16)
        with self.assertRaises(ValueError):
            init.hash_of_values(TINY_3M, wrong)

    def test_ChangingOneWeightChangesTheHash(self):
        every = init.derive_all(TINY_3M, G)
        base = init.hash_of_values(TINY_3M, every)
        every["layers.0.ffn.up.weight"] = every["layers.0.ffn.up.weight"].copy()
        every["layers.0.ffn.up.weight"][0] ^= 1
        self.assertNotEqual(init.hash_of_values(TINY_3M, every), base)

    def test_TheHashDependsOnOrderNotJustContent(self):
        """Hashing in layout order is what makes the digest describe the model
        rather than one node's dictionary."""
        every = init.derive_all(TINY_3M, G)
        a, b = "layers.0.attn.wq.weight", "layers.0.attn.wk.weight"
        self.assertNotEqual(every[a].shape, every[b].shape)
        swapped = dict(every)
        swapped[a], swapped[b] = every[b], every[a]
        with self.assertRaises(ValueError):
            init.hash_of_values(TINY_3M, swapped)


if __name__ == "__main__":
    unittest.main(verbosity=2)

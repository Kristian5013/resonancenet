"""Tests for the model and the weights bridge.

Needs torch. The CUDA-only tests skip cleanly on a machine without a card,
because the claims they check are about kernels, and a claim about kernels
cannot be checked without one — asserting it on CPU would be a test that passes
by not testing.
"""

import unittest

import numpy as np
import torch

from rnet.consensus import genesis, init
from rnet.consensus.model_spec import ModelSpec
from rnet.consensus.numerics import (AttentionKernel, ContributionFormat, DType,
                                     Numerics)
from rnet.consensus.params import REGTEST_NUMERICS, TINY_3M
from rnet.model import weights as W
from rnet.model.layout import layout, shard_layout
from rnet.model.transformer import MoE, Transformer

CUDA = torch.cuda.is_available()
G = bytes.fromhex(genesis.GENESIS_HASH["regtest"])

SMALL = ModelSpec(d_model=128, n_layers=3, n_heads=4, n_kv_heads=2, d_ff=256,
                  vocab_size=128, seq_len=64, rope_theta=10000,
                  tie_embeddings=True, qk_norm=True)

SMALL_MOE = ModelSpec(d_model=128, n_layers=3, n_heads=4, n_kv_heads=2, d_ff=256,
                      vocab_size=128, seq_len=64, rope_theta=10000,
                      tie_embeddings=True, qk_norm=True,
                      n_experts=8, n_experts_active=2, n_shared_experts=1,
                      d_ff_expert=96, moe_first_layer=1, moe_layer_stride=1,
                      expert_shard_count=4)

MATH = Numerics(DType.BF16, DType.BF16, DType.FP32, AttentionKernel.MATH,
                ContributionFormat.INT8_POW2)
FLASH = Numerics(DType.BF16, DType.BF16, DType.FP32, AttentionKernel.FLASH,
                 ContributionFormat.INT8_POW2)


class ModuleTreeTests(unittest.TestCase):
    """The property that removes a whole class of failure: the module's
    parameter names ARE the canonical layout's names, not a second list kept in
    agreement with the first."""

    def test_TheModuleTreeMatchesTheCanonicalLayout(self):
        for name, spec in (("dense", SMALL), ("moe", SMALL_MOE),
                           ("regtest", TINY_3M)):
            with self.subTest(name):
                model = Transformer(spec, MATH)
                got = {n for n, _ in model.named_parameters()}
                want = {t.name for t in layout(spec)}
                self.assertEqual(got, want)

    def test_EveryParameterHasTheShapeTheLayoutSays(self):
        for spec in (SMALL, SMALL_MOE):
            model = Transformer(spec, MATH)
            params = dict(model.named_parameters())
            for t in layout(spec):
                self.assertEqual(tuple(params[t.name].shape), t.shape, t.name)

    def test_UntiedEmbeddingsAddTheHead(self):
        untied = ModelSpec(**{**SMALL.__dict__, "tie_embeddings": False})
        got = {n for n, _ in Transformer(untied, MATH).named_parameters()}
        self.assertIn("lm_head.weight", got)
        self.assertEqual(got, {t.name for t in layout(untied)})


class WeightsTests(unittest.TestCase):

    def test_LoadingDerivedWeightsReproducesTheAnchor(self):
        """The loop closed: genesis hash -> derived weights -> live model ->
        the weights anchor the network publishes."""
        model = W.build(TINY_3M, REGTEST_NUMERICS, G, device="cpu",
                        grad_checkpointing=False)
        self.assertEqual(W.weights_hash(model).hex(),
                         genesis.WEIGHTS_HASH["regtest"])

    def test_LoadingThenSavingIsTheIdentity(self):
        derived = init.derive_all(SMALL, G)
        model = Transformer(SMALL, MATH).to(dtype=torch.bfloat16)
        W.load_weights(model, derived)
        saved = W.save_weights(model)
        for name, words in derived.items():
            self.assertTrue(np.array_equal(saved[name], words), name)

    def test_AMissingTensorIsRefused(self):
        """A loader that skipped one would leave it at whatever the constructor
        produced, and train from weights the network never agreed on."""
        derived = init.derive_all(SMALL, G)
        del derived["layers.1.attn.wq.weight"]
        model = Transformer(SMALL, MATH).to(dtype=torch.bfloat16)
        with self.assertRaises(KeyError):
            W.load_weights(model, derived)

    def test_AnExtraTensorIsRefused(self):
        derived = init.derive_all(SMALL, G)
        derived["layers.9.made.up.weight"] = np.zeros(4, dtype=np.uint16)
        model = Transformer(SMALL, MATH).to(dtype=torch.bfloat16)
        with self.assertRaises(KeyError):
            W.load_weights(model, derived)

    def test_AMisshapenTensorIsRefused(self):
        derived = init.derive_all(SMALL, G)
        derived["layers.1.attn.wq.weight"] = np.zeros(7, dtype=np.uint16)
        model = Transformer(SMALL, MATH).to(dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            W.load_weights(model, derived)

    def test_AShardLoadsOnlyItsOwnExperts(self):
        for s in range(SMALL_MOE.expert_shard_count):
            with self.subTest(shard=s):
                derived = init.derive_all(SMALL_MOE, G, shard=s)
                model = Transformer(SMALL_MOE, MATH).to(dtype=torch.bfloat16)
                W.load_weights(model, derived, shard=s)
                saved = W.save_weights(model, shard=s)
                self.assertEqual(set(saved), {t.name for t in shard_layout(SMALL_MOE, s)})
                for name, words in derived.items():
                    self.assertTrue(np.array_equal(saved[name], words), name)

    def test_TheCanonicalBytesAreBigEndian(self):
        """Because "whatever this machine does" is not a specification."""
        model = Transformer(SMALL, MATH).to(dtype=torch.bfloat16)
        with torch.no_grad():
            dict(model.named_parameters())["norm.weight"].fill_(1.0)
        words = W.save_weights(model)["norm.weight"]
        # 1.0 is 0x3F80 in bf16. As big-endian bytes that is 3F 80.
        self.assertEqual(words.astype(">u2").tobytes()[:2], b"\x3f\x80")


class ForwardTests(unittest.TestCase):

    def _tokens(self, spec, device, batch=2):
        gen = torch.Generator(device="cpu").manual_seed(0)
        return torch.randint(0, spec.vocab_size, (batch, spec.seq_len + 1),
                             generator=gen).to(device)

    def test_ADenseForwardProducesALossNearChance(self):
        """Random weights should score about ln(vocab); far from it means the
        model is broken in a way a shape check would not catch."""
        model = W.build(SMALL, MATH, G, device="cpu", grad_checkpointing=False).eval()
        with torch.no_grad():
            loss = model.loss(self._tokens(SMALL, "cpu")).item()
        self.assertAlmostEqual(loss, np.log(SMALL.vocab_size), delta=0.6)

    def test_AMixtureForwardProducesALossNearChance(self):
        model = W.build(SMALL_MOE, MATH, G, device="cpu",
                        grad_checkpointing=False).eval()
        with torch.no_grad():
            loss = model.loss(self._tokens(SMALL_MOE, "cpu")).item()
        self.assertAlmostEqual(loss, np.log(SMALL_MOE.vocab_size), delta=0.6)

    def test_TheTargetIsTheNextToken(self):
        """The shift, pinned against a computation done outside the model.

        An off-by-one here trains the model to predict the token it was just
        given, which converges beautifully and learns nothing.
        """
        import torch.nn.functional as F
        model = W.build(SMALL, MATH, G, device="cpu", grad_checkpointing=False).eval()
        toks = self._tokens(SMALL, "cpu")
        with torch.no_grad():
            logits = model(toks[:, :-1])
            expected = F.cross_entropy(
                logits.reshape(-1, SMALL.vocab_size).float(), toks[:, 1:].reshape(-1))
            self.assertEqual(model.loss(toks).item(), expected.item())
            # And the test has teeth: predicting the CURRENT token scores
            # differently, so a wrong shift would not pass.
            wrong = F.cross_entropy(
                logits.reshape(-1, SMALL.vocab_size).float(), toks[:, :-1].reshape(-1))
            self.assertNotEqual(wrong.item(), expected.item())

    def test_AWindowTooShortToShiftIsRefused(self):
        """Not a NaN. An empty split returns one, and a NaN loss reads as a
        training failure rather than as the malformed input it is."""
        model = W.build(SMALL, MATH, G, device="cpu", grad_checkpointing=False).eval()
        for shape in ((2, 1), (2, 0)):
            with self.assertRaises(ValueError):
                model.loss(torch.zeros(*shape, dtype=torch.long))
        with self.assertRaises(ValueError):
            model.loss(torch.zeros(4, dtype=torch.long))


class MoETests(unittest.TestCase):

    def test_ARowSelectsAnExpertAtMostOnce(self):
        """The invariant transformer.py relies on but deliberately does not
        depend on: topk returns distinct indices."""
        gen = torch.Generator().manual_seed(0)
        logits = torch.randn(64, 8, generator=gen)
        _, chosen = torch.topk(logits, 3, dim=-1)
        for e in range(8):
            rows, _ = torch.where(chosen == e)
            self.assertEqual(len(rows), len(set(rows.tolist())), f"expert {e}")

    def test_RoutingWeightsSumToOnePerToken(self):
        model = W.build(SMALL_MOE, MATH, G, device="cpu",
                        grad_checkpointing=False).eval()
        moe = next(m for m in model.modules() if isinstance(m, MoE))
        x = torch.randn(2, 8, SMALL_MOE.d_model, dtype=torch.bfloat16)
        logits = moe.router(x.reshape(-1, SMALL_MOE.d_model)).to(torch.float32)
        weights, _ = torch.topk(logits, SMALL_MOE.n_experts_active, dim=-1)
        self.assertTrue(torch.allclose(torch.softmax(weights, -1).sum(-1),
                                       torch.ones(16), atol=1e-5))

    def test_SharedExpertsRunForEveryToken(self):
        """Removing them must change every token's output, not some."""
        with_shared = W.build(SMALL_MOE, MATH, G, device="cpu",
                              grad_checkpointing=False).eval()
        without = ModelSpec(**{**SMALL_MOE.__dict__, "n_shared_experts": 0})
        plain = W.build(without, MATH, G, device="cpu", grad_checkpointing=False).eval()
        toks = torch.randint(0, SMALL_MOE.vocab_size, (1, 16))
        with torch.no_grad():
            a, b = with_shared(toks), plain(toks)
        self.assertFalse(torch.allclose(a.float(), b.float()))


@unittest.skipUnless(CUDA, "the claim is about kernels; checking it needs one")
class KernelTests(unittest.TestCase):
    """Why the attention kernel is a consensus field and not a speed knob."""

    def _loss(self, numerics):
        model = W.build(SMALL, numerics, G, device="cuda",
                        grad_checkpointing=False).eval()
        gen = torch.Generator(device="cpu").manual_seed(0)
        toks = torch.randint(0, SMALL.vocab_size, (2, SMALL.seq_len + 1),
                             generator=gen).cuda()
        with torch.no_grad():
            return model.loss(toks).item()

    def test_TheSameKernelIsBitIdentical(self):
        for numerics in (FLASH, MATH):
            with self.subTest(numerics.attention_kernel.name):
                self.assertEqual(self._loss(numerics), self._loss(numerics))

    def test_FlashAndMathAreNotBitIdentical(self):
        """Measured, not assumed. This difference is the entire reason the
        kernel is pinned in a hashed artifact: a worker free to choose would
        compute an update nobody could reproduce, and would be
        indistinguishable from one that cheated."""
        flash, math = self._loss(FLASH), self._loss(MATH)
        self.assertNotEqual(flash, math)
        # Small enough that it would never be noticed as "wrong", which is
        # precisely why it has to be pinned rather than watched for.
        self.assertLess(abs(flash - math), 1e-2)

    def test_TheTwoKernelsLandInDifferentDeterminismClasses(self):
        self.assertNotEqual(FLASH.determinism_class, MATH.determinism_class)

    def test_TheAnchorHoldsOnTheGpuToo(self):
        model = W.build(TINY_3M, REGTEST_NUMERICS, G, device="cuda",
                        grad_checkpointing=False)
        self.assertEqual(W.weights_hash(model).hex(),
                         genesis.WEIGHTS_HASH["regtest"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

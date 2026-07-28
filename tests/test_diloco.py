"""Tests for the DiLoCo arithmetic: quantise, aggregate, outer step.

The property that runs through all of it: two nodes given the same
contributions must produce the same bytes, whatever order they arrived in and
whatever exponents the workers happened to choose.
"""

import unittest

import numpy as np

from rnet.consensus.numerics import ContributionFormat as F
from rnet.diloco import aggregate as A
from rnet.diloco import outer as O
from rnet.diloco import quantize as Q


def rng(seed=0):
    return np.random.default_rng(seed)


class QuantiseTests(unittest.TestCase):

    def test_TheRangeIsSymmetricAboutZero(self):
        """Two's complement would give one more negative value, biasing every
        update by half a step in a direction nobody chose."""
        for fmt, limit in ((F.INT4_POW2, 7), (F.INT6_POW2, 31), (F.INT8_POW2, 127)):
            self.assertEqual(fmt.max_magnitude, limit)
            values = np.array([-1e9, 1e9], dtype=np.float64)
            q = Q.quantize(values, 0, fmt)
            self.assertEqual(int(q.min()), -limit)
            self.assertEqual(int(q.max()), limit)

    def test_TiesRoundAwayFromZeroNotToEven(self):
        """numpy's default is to even; the rule has to be the stated one."""
        values = np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5])
        q = Q.quantize(values, 0, F.INT8_POW2)
        self.assertEqual(q.tolist(), [1, 2, 3, -1, -2, -3])
        # np.round would give [0, 2, 2, -0, -2, -2] — proof the rule differs.
        self.assertNotEqual(np.round(values).astype(np.int64).tolist(), q.tolist())

    def test_TheScaleIsChosenWithoutALogarithm(self):
        """frexp reads the exponent out of the bits; log2 computes it, and
        different C libraries differ in the last place. A worker one exponent
        off produces an update twice the size it meant."""
        for seed in range(20):
            v = rng(seed).normal(0, 10.0 ** rng(seed).integers(-30, 30), 500)
            exp = Q.choose_scale_exp(v, F.INT8_POW2)
            q = Q.quantize(v, exp, F.INT8_POW2)
            self.assertLessEqual(int(np.abs(q).max()), 127)
            # And it is the SMALLEST such exponent: one lower must overflow.
            self.assertGreater(
                int(np.abs(Q.quantize(v, exp - 1, F.INT8_POW2)).max()), 126)

    def test_DequantisingIsExact(self):
        """The scale is a power of two, so it touches only the exponent."""
        q = np.array([-127, -1, 0, 1, 127], dtype=np.int64)
        for exp in (-40, -1, 0, 1, 40):
            back = Q.dequantize(q, exp)
            self.assertTrue(np.array_equal(back, q.astype(np.float64) * 2.0 ** exp))

    def test_AZeroUpdateIsRepresentable(self):
        """A worker that changed nothing still has to be able to say so."""
        v = np.zeros(16)
        q, exp = Q.quantize_update(v, F.INT8_POW2)
        self.assertEqual(exp, 0)
        self.assertTrue(np.array_equal(q, np.zeros(16, dtype=np.int64)))
        self.assertTrue(np.array_equal(Q.dequantize(q, exp), v))

    def test_ANonFiniteUpdateIsRefused(self):
        """A diverged inner loop must not be encoded and spread to everyone
        who aggregates it."""
        for bad in (np.inf, -np.inf, np.nan):
            v = np.array([1.0, bad, 2.0])
            with self.assertRaises(Q.QuantiseError):
                Q.quantize_update(v, F.INT8_POW2)

    def test_AnEmptyUpdateIsRefused(self):
        with self.assertRaises(Q.QuantiseError):
            Q.quantize_update(np.array([]), F.INT8_POW2)

    def test_AnAbsurdExponentIsRefused(self):
        with self.assertRaises(Q.QuantiseError):
            Q.quantize(np.array([1.0]), 400, F.INT8_POW2)

    def test_SaturationIsCountedNotHidden(self):
        v = np.array([1.0, 200.0, -300.0])
        self.assertEqual(Q.clamped_count(v, 0, F.INT8_POW2), 2)
        self.assertEqual(Q.clamped_count(v, 8, F.INT8_POW2), 0)

    def test_EveryFormatPacksAndUnpacks(self):
        for fmt in F:
            with self.subTest(fmt.name):
                v = rng(1).normal(0, 1, 256)
                q, exp = Q.quantize_update(v, fmt)
                back = Q.unpack(Q.pack(q, fmt), q.size, fmt)
                self.assertTrue(np.array_equal(back, q))

    def test_Int4PackingHoldsTheFullSignedRange(self):
        q = np.array([-8, -7, -1, 0, 1, 7, -8, 7], dtype=np.int64)
        q = np.clip(q, -7, 7)
        back = Q.unpack(Q.pack(q, F.INT4_POW2), q.size, F.INT4_POW2)
        self.assertTrue(np.array_equal(back, q))

    def test_Int4PackingHalvesThePayload(self):
        q = Q.quantize(rng(2).normal(0, 1, 1024), 0, F.INT4_POW2)
        self.assertEqual(len(Q.pack(q, F.INT4_POW2)), 512)
        self.assertEqual(len(Q.pack(q, F.INT8_POW2)), 1024)

    def test_AWrongLengthUnpackIsRefused(self):
        with self.assertRaises(Q.QuantiseError):
            Q.unpack(b"\x00" * 4, 8, F.INT8_POW2)
        with self.assertRaises(Q.QuantiseError):
            Q.unpack(b"\x00" * 4, 4, F.INT4_POW2)


class AggregateTests(unittest.TestCase):

    def _contributions(self, n=4, size=32, seed=0):
        r = rng(seed)
        out = []
        for i in range(n):
            v = r.normal(0, 1e-3, size)
            q, exp = Q.quantize_update(v, F.INT8_POW2)
            out.append(A.Contribution(q, exp, F.INT8_POW2, worker_id=i))
        return out

    def test_TheSumDoesNotDependOnArrivalOrder(self):
        """The property the whole design is built around. Floating-point
        addition is not associative; integer addition is."""
        cs = self._contributions(6)
        base, exp = A.sum_contributions(cs)
        for seed in range(8):
            shuffled = list(cs)
            rng(seed).shuffle(shuffled)
            got, got_exp = A.sum_contributions(shuffled)
            self.assertTrue(np.array_equal(got, base))
            self.assertEqual(got_exp, exp)

    def test_EveryContributionIsInTheResult(self):
        """A sum, not a race: nobody's work is dropped for arriving second."""
        size = 8
        cs = [A.Contribution(np.full(size, v, dtype=np.int64), 0, F.INT8_POW2, i)
              for i, v in enumerate((1, 2, 3, 4))]
        total, _ = A.sum_contributions(cs)
        self.assertTrue(np.array_equal(total, np.full(size, 10)))

    def test_AlignmentGoesTowardTheFinestExponent(self):
        """Up, never down: shifting a coarse contribution up is exact."""
        fine = A.Contribution(np.array([4], dtype=np.int64), -10, F.INT8_POW2, 1)
        coarse = A.Contribution(np.array([1], dtype=np.int64), -8, F.INT8_POW2, 2)
        total, exp = A.sum_contributions([fine, coarse])
        # 4*2^-10 + 1*2^-8 = 8*2^-10, exactly.
        self.assertEqual(exp, -10)
        self.assertEqual(total.tolist(), [8])

    def test_AlignmentIsExactAndLosesNothing(self):
        """It rounds nowhere, because a left shift has nothing to round."""
        r = rng(11)
        for gap in range(0, A.MAX_ALIGN_SHIFT + 1):
            values = r.integers(-127, 128, 64).astype(np.int64)
            fine = A.Contribution(np.zeros(64, dtype=np.int64), -gap, F.INT8_POW2, 1)
            coarse = A.Contribution(values, 0, F.INT8_POW2, 2)
            total, exp = A.sum_contributions([fine, coarse])
            self.assertEqual(exp, -gap)
            self.assertTrue(np.array_equal(total, values << np.int64(gap)),
                            f"gap {gap}")

    def test_ACoarseContributorCannotEraseTheOthers(self):
        """The bug an audit found, as a test.

        Aligning toward the COARSEST exponent right-shifted every finer
        contribution, and at an eight-bit gap 127 >> 8 is 0 — so one participant
        choosing a coarse exponent silently zeroed everybody else's work, on
        every node that aggregated it, while the docstring promised the
        opposite.
        """
        honest = [A.Contribution(np.full(16, 100, dtype=np.int64), -20,
                                 F.INT8_POW2, worker_id=i) for i in range(1, 4)]
        coarse = A.Contribution(np.full(16, 1, dtype=np.int64), -4,
                                F.INT8_POW2, worker_id=9)
        total, exp = A.sum_contributions(honest + [coarse])
        self.assertEqual(exp, -20)

        # Every honest contribution survives intact, and the coarse one is
        # simply large — which is what a coarse exponent means.
        without, _ = A.sum_contributions(honest)
        self.assertTrue(np.all(total > without),
                        "the coarse contributor added nothing")
        self.assertTrue(np.all(without == 300),
                        "the honest contributions were altered")
        # And the honest work is still visible in the total rather than lost in
        # rounding: dropping one honest contributor changes the result.
        fewer, _ = A.sum_contributions(honest[:2] + [coarse])
        self.assertTrue(np.all(total - fewer == 100))

    def test_ADivergentExponentIsRefused(self):
        """Past a bound it is not a different scale, it is a different quantity
        — and a worker could produce it deliberately."""
        ok = A.Contribution(np.array([1], dtype=np.int64), 0, F.INT8_POW2, 1)
        far = A.Contribution(np.array([1], dtype=np.int64),
                             -A.MAX_ALIGN_SHIFT - 1, F.INT8_POW2, 2)
        with self.assertRaises(A.AggregateError):
            A.sum_contributions([ok, far])
        edge = A.Contribution(np.array([1], dtype=np.int64),
                              -A.MAX_ALIGN_SHIFT, F.INT8_POW2, 2)
        A.sum_contributions([ok, edge])

    def test_MismatchedLengthsAreRefused(self):
        a = A.Contribution(np.zeros(4, dtype=np.int64), 0, F.INT8_POW2, 1)
        b = A.Contribution(np.zeros(5, dtype=np.int64), 0, F.INT8_POW2, 2)
        with self.assertRaises(A.AggregateError):
            A.sum_contributions([a, b])

    def test_AveragingRoundsSymmetrically(self):
        self.assertEqual(A.rounded_div(np.array([3, -3, 5, -5]), 2).tolist(),
                         [2, -2, 3, -3])
        self.assertEqual(A.rounded_div(np.array([10, -10]), 4).tolist(), [3, -3])

    def test_TheContributionRootIsOrderIndependent(self):
        """Otherwise "every contribution was aggregated" would be a claim about
        arrival order rather than about content."""
        ids = [bytes([i]) * 32 for i in range(5)]
        base = A.contribution_root(ids)
        for seed in range(5):
            shuffled = list(ids)
            rng(seed).shuffle(shuffled)
            self.assertEqual(A.contribution_root(shuffled), base)

    def test_ARepeatedContributionIsRefused(self):
        ids = [bytes([1]) * 32, bytes([1]) * 32]
        with self.assertRaises(A.AggregateError):
            A.contribution_root(ids)


class OuterTests(unittest.TestCase):

    def _opt(self, nesterov=True):
        return O.OuterOptimizer(momentum_q16=58982, lr_q16=45875, nesterov=nesterov)

    def test_TheDomainHoldsAtTheWorstCaseAlignment(self):
        """Momentum lives in the aggregate's units, not in Q16.

        Driven to steady state against the largest aggregate the alignment rule
        permits, and measured against the domain rather than assumed to be
        inside it.
        """
        worst = np.full(64, 127 << A.MAX_ALIGN_SHIFT, dtype=np.int64)
        opt = self._opt()
        for _ in range(200):
            update, _ = opt.step(worst, -24)
        peak = int(np.abs(opt.momentum).max())
        self.assertLess(peak, O.DOMAIN_MAX)
        self.assertGreater(O.DOMAIN_MAX / peak, 1000)     # measured: 6,605x
        self.assertTrue(np.all(np.isfinite(update.astype(np.float64))))

    def test_MomentumHeldInQ16WouldOverflowTheMultiply(self):
        """The alternative design, exercised rather than described.

        The overflow is not in the stored value — 1.4e15 fits int64 fine — but
        in the intermediate of the next multiply, which is the kind that
        survives review.
        """
        worst = np.full(4, 127 << A.MAX_ALIGN_SHIFT, dtype=np.int64)
        opt = self._opt()
        for _ in range(200):
            opt.step(worst, -24)
        peak = int(np.abs(opt.momentum).max())
        int64_max = (1 << 63) - 1

        as_q16 = peak * O.Q16_ONE
        self.assertLess(as_q16, int64_max, "the stored value alone does fit")
        self.assertGreater(as_q16 * 58982, int64_max,
                           "but multiplying it by the momentum rate does not")
        # And numpy would not say so. It wraps — silently, modularly, and not
        # necessarily to a negative number, which is what makes it dangerous:
        # the result stays a plausible-looking positive integer.
        wrapped = int((np.array([as_q16], dtype=np.int64) * np.int64(58982))[0])
        self.assertNotEqual(wrapped, as_q16 * 58982)
        self.assertEqual(wrapped, ((as_q16 * 58982 + (1 << 63)) % (1 << 64)) - (1 << 63))
        self.assertGreater(wrapped, 0, "and it does not even look wrong")

    def test_AValueBeyondTheDomainRaisesRatherThanWrapping(self):
        """numpy int64 wraps silently, which is the hazard this replaces."""
        with self.assertRaises(O.OuterError):
            O.mul_q16(58982, np.array([O.DOMAIN_MAX + 1], dtype=np.int64))
        O.mul_q16(58982, np.array([O.DOMAIN_MAX], dtype=np.int64))

    def test_TheStepIsDeterministic(self):
        g = Q.quantize(rng(3).normal(0, 1e-3, 128), -20, F.INT8_POW2)
        a, b = self._opt(), self._opt()
        for _ in range(5):
            ua, _ = a.step(g, -20)
            ub, _ = b.step(g, -20)
            self.assertTrue(np.array_equal(ua, ub))
        self.assertEqual(a.state_hash(), b.state_hash())

    def test_MomentumAccumulates(self):
        g = np.full(8, 1000, dtype=np.int64)
        opt = self._opt(nesterov=False)
        first, _ = opt.step(g, 0)
        second, _ = opt.step(g, 0)
        self.assertTrue(np.all(np.abs(second) > np.abs(first)))

    def test_NesterovDiffersFromClassical(self):
        g = np.full(8, 1000, dtype=np.int64)
        n, c = self._opt(True), self._opt(False)
        self.assertFalse(np.array_equal(n.step(g, 0)[0], c.step(g, 0)[0]))

    def test_MomentumRealignsByShiftingWhenTheExponentMoves(self):
        opt = self._opt(nesterov=False)
        g = np.full(4, 64, dtype=np.int64)
        opt.step(g, 0)
        before = opt.momentum.copy()
        # A finer exponent: momentum shifts up, exactly.
        opt.step(np.zeros(4, dtype=np.int64), -2)
        self.assertEqual(opt.scale_exp, -2)
        self.assertTrue(np.all(np.abs(opt.momentum) >= np.abs(before)))

    def test_AResizedModelIsRefused(self):
        opt = self._opt()
        opt.step(np.zeros(4, dtype=np.int64), 0)
        with self.assertRaises(O.OuterError):
            opt.step(np.zeros(5, dtype=np.int64), 0)

    def test_TheStateHashCoversEveryField(self):
        """A field outside the hash is a field two producers could differ on
        while both reporting the same state."""
        base = self._opt()
        base.step(np.full(4, 7, dtype=np.int64), 0)
        digest = base.state_hash()
        for change in ({"momentum_q16": 1}, {"lr_q16": 1}, {"nesterov": False},
                       {"steps": 99}, {"scale_exp": 3}):
            other = self._opt()
            other.step(np.full(4, 7, dtype=np.int64), 0)
            for k, v in change.items():
                setattr(other, k, v)
            self.assertNotEqual(other.state_hash(), digest, change)
        moved = self._opt()
        moved.step(np.full(4, 8, dtype=np.int64), 0)
        self.assertNotEqual(moved.state_hash(), digest)

    def test_ApplyingAnUpdateIsExactUpToTheFinalRounding(self):
        weights = np.array([1.0, 2.0, -3.0], dtype=np.float64)
        update = np.array([1, 2, -4], dtype=np.int64)
        out = O.apply_update(weights, update, -2)
        self.assertTrue(np.array_equal(out, weights - update * 0.25))

    def test_ApplyingAMisshapenUpdateIsRefused(self):
        with self.assertRaises(A.AggregateError):
            O.apply_update(np.zeros(3), np.zeros(4, dtype=np.int64), 0)


class EndToEndTests(unittest.TestCase):

    def test_AWholeOuterStepIsOrderIndependent(self):
        """Quantise, aggregate, step, apply — twice, with the contributions
        shuffled. Same bytes."""
        r = rng(7)
        base = r.normal(0, 1.0, 256)
        cs = []
        for i in range(5):
            q, exp = Q.quantize_update(r.normal(0, 1e-3, 256), F.INT8_POW2)
            cs.append(A.Contribution(q, exp, F.INT8_POW2, worker_id=i))

        def run(order):
            opt = O.OuterOptimizer(58982, 45875, True)
            mean, exp = A.average_contributions(order)
            update, exp = opt.step(mean, exp)
            return O.apply_update(base, update, exp), opt.state_hash()

        want, want_hash = run(cs)
        for seed in range(6):
            shuffled = list(cs)
            rng(seed).shuffle(shuffled)
            got, got_hash = run(shuffled)
            self.assertTrue(np.array_equal(got, want))
            self.assertEqual(got_hash, want_hash)

    def test_TheUpdateMovesWeightsInTheDirectionWorkersReported(self):
        """A contribution is before-minus-after, so the network subtracts it."""
        base = np.zeros(8, dtype=np.float64)
        moved = A.Contribution(np.full(8, 100, dtype=np.int64), -4, F.INT8_POW2, 1)
        opt = O.OuterOptimizer(0, 65536, False)     # no momentum, lr 1.0
        mean, exp = A.average_contributions([moved])
        update, exp = opt.step(mean, exp)
        self.assertTrue(np.all(O.apply_update(base, update, exp) < 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class BlockedArithmeticIsTheSameArithmeticTests(unittest.TestCase):
    """Holding fewer arrays at once, and changing no value.

    The outer step allocated an array the size of the model on nearly every
    line — six across the step, on top of five inside each `mul_q16`, three
    times. It measured 27.6 GB for one 397,728,768-parameter round and the
    kernel killed the other worker on the machine. Everything below is the same
    arithmetic done in pieces, and a single differing integer here would be two
    nodes disagreeing about a checkpoint.
    """

    def reference_mul(self, constant, values):
        """What `mul_q16` was, kept as the definition."""
        sign = np.sign(values)
        magnitude = np.abs(values).astype(np.int64)
        return (sign * ((magnitude * constant + (1 << 15)) >> 16)).astype(np.int64)

    def test_BlockingDoesNotChangeTheProduct(self):
        rng = np.random.default_rng(4)
        for size in (0, 1, 3, O.BLOCK - 1, O.BLOCK, O.BLOCK + 5, 3 * O.BLOCK + 7):
            values = rng.integers(-(10 ** 12), 10 ** 12, size, dtype=np.int64)
            for constant in (0, 1, 45875, 58982, 65535, 65536):
                with self.subTest(size=size, constant=constant):
                    np.testing.assert_array_equal(
                        O.mul_q16(constant, values),
                        self.reference_mul(constant, values))

    def test_WritingInPlaceGivesTheSameAnswer(self):
        rng = np.random.default_rng(5)
        values = rng.integers(-(10 ** 10), 10 ** 10, O.BLOCK + 1234, dtype=np.int64)
        want = self.reference_mul(58982, values)
        scratch = values.copy()
        O.mul_q16(58982, scratch, out=scratch)
        np.testing.assert_array_equal(scratch, want)

    def test_TheDomainCheckStillBitesAcrossBlocks(self):
        """A value past the domain in the LAST block must raise as loudly as one
        in the first — a check that stopped early would let a wrapped, plausible
        number into a checkpoint."""
        values = np.zeros(2 * O.BLOCK + 3, dtype=np.int64)
        values[-1] = O.DOMAIN_MAX + 1
        with self.assertRaises(O.OuterError):
            O.mul_q16(45875, values)

    def test_AWholeOuterStepIsUnchanged(self):
        """The step itself, against an implementation that allocates freely."""
        def reference(momentum_q16, lr_q16, nesterov, momentum, aggregate):
            m = self.reference_mul(momentum_q16, momentum) + aggregate
            direction = (aggregate + self.reference_mul(momentum_q16, m)
                         if nesterov else m)
            return self.reference_mul(lr_q16, direction), m

        rng = np.random.default_rng(6)
        for nesterov in (True, False):
            with self.subTest(nesterov=nesterov):
                opt = O.OuterOptimizer(momentum_q16=58982, lr_q16=45875,
                                       nesterov=nesterov)
                momentum = np.zeros(9999, dtype=np.int64)
                for round_index in range(4):
                    aggregate = rng.integers(-(10 ** 6), 10 ** 6, 9999,
                                             dtype=np.int64)
                    want, momentum = reference(58982, 45875, nesterov,
                                               momentum, aggregate)
                    got, _ = opt.step(aggregate.copy(), 0)
                    np.testing.assert_array_equal(got, want, f"round {round_index}")
                    np.testing.assert_array_equal(opt.momentum, momentum,
                                                  f"momentum {round_index}")

    def test_ClassicalMomentumIsNotScaledInPlace(self):
        """Without nesterov the direction IS the momentum, and scaling it in
        place would multiply the carried state by the learning rate on every
        round — a recurrence nobody agreed to, and invisible for several rounds
        because the first one still looks right."""
        opt = O.OuterOptimizer(momentum_q16=58982, lr_q16=45875, nesterov=False)
        aggregate = np.full(64, 1 << 20, dtype=np.int64)
        first, _ = opt.step(aggregate.copy(), 0)
        self.assertTrue(np.array_equal(opt.momentum, aggregate),
                        "momentum should be the aggregate after one step")
        self.assertFalse(np.array_equal(first, opt.momentum),
                         "the returned update should be the scaled one")

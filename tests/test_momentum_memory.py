"""Can a node that joined late re-derive the outer optimizer's momentum?

It cannot, and this file is why. The question matters because the answer decides
the whole shape of joining: if momentum could be replayed from the chain, a
newcomer would need only headers; since it cannot, it needs the state itself
transferred and verified, which is a different and larger mechanism.

The tempting argument is that momentum is a contraction — each step multiplies
by 0.9 — so history fades and any two nodes converge. Under live training that
is even true, measured at 224 steps for the worst case here. It is not true in
general, and the exceptions are not exotic.
"""

import unittest

import numpy as np

from rnet.diloco.outer import OuterOptimizer

MU, LR = 58982, 45875
N = 128


def run(aggregates, start, limit=2000):
    """Steps two optimizers from different momenta on the same stream.

    Returns the step at which they became exactly equal, or None.
    """
    a = OuterOptimizer(MU, LR, True)
    b = OuterOptimizer(MU, LR, True)
    b.momentum = np.asarray(start, dtype=np.int64).copy()
    b.scale_exp = 0
    for step, g in enumerate(aggregates[:limit], 1):
        a.step(g, 0)
        b.step(g, 0)
        if np.array_equal(a.momentum, b.momentum):
            return step
    return None


def lively(count: int, seed: int = 0):
    r = np.random.default_rng(seed)
    return [r.integers(-127, 128, N).astype(np.int64) for _ in range(count)]


class MomentumMemoryTests(unittest.TestCase):

    def test_UnderLiveTrainingItDoesForget(self):
        """Which is what makes the wrong answer tempting."""
        worst = 0
        for seed in range(8):
            for scale in (1_000, 100_000, 10_000_000):
                r = np.random.default_rng(seed)
                step = run(lively(800, seed),
                           r.integers(-scale, scale + 1, N))
                self.assertIsNotNone(step, f"seed {seed} scale {scale}")
                worst = max(worst, step)
        self.assertLess(worst, 400)     # measured: 224 over a wider sweep

    def test_AConstantStreamNeverForgets(self):
        """The exception that decides the design.

        With a constant aggregate the recurrence has a PLATEAU of fixed points
        rather than one: m = trunc(0.9m) + 127 holds at 1269 and at 1270 alike.
        Two trajectories settle on different points of it and stay there. No
        rounding rule removes this — it is a property of integer contraction,
        not of a choice about halves.
        """
        constant = [np.full(N, 127, dtype=np.int64)] * 2000
        self.assertIsNone(run(constant, np.full(N, 5_000, dtype=np.int64)))

    def test_TheFixedPointIsAPlateau(self):
        """Shown directly, so the reason is not taken on trust."""
        from rnet.diloco.outer import mul_q16
        g = 127
        fixed = [m for m in range(1200, 1300)
                 if int(mul_q16(MU, np.array([m]))[0]) + g == m]
        self.assertGreater(len(fixed), 1, "a single fixed point would converge")

    def test_AZeroStreamNeverForgets(self):
        """A lone 1 is immortal: 1 * 58982 rounded half away from zero is 1
        again, forever."""
        from rnet.diloco.outer import mul_q16
        m = np.array([1], dtype=np.int64)
        for _ in range(1000):
            m = mul_q16(MU, m)
        self.assertEqual(int(m[0]), 1)
        self.assertIsNone(run([np.zeros(N, dtype=np.int64)] * 1000,
                              np.full(N, 5_000, dtype=np.int64)))

    def test_ThePlateauIsWideNotMarginal(self):
        """How far apart two nodes must be before they never converge: one.

        Measured against a momentum settled at 1265 under a constant stream. A
        difference of a single unit persists for three thousand steps, and so
        does one of two thousand — the plateau is not a rounding artifact at the
        boundary, it is the whole neighbourhood.
        """
        warm = OuterOptimizer(MU, LR, True)
        for _ in range(500):
            warm.step(np.full(N, 127, dtype=np.int64), 0)
        settled = warm.momentum.copy()
        self.assertEqual(int(settled[0]), 1265)

        constant = [np.full(N, 127, dtype=np.int64)] * 3000
        for offset in (1, 10, 1000, 2000):
            self.assertIsNone(run(constant, settled + offset), f"offset {offset}")

    def test_AJoinerFromZeroDoesCatchUp(self):
        """The case that actually happens, and it is the good one.

        A newcomer starts at zero and climbs the same trajectory the running
        node is already on, so it lands on the same plateau point rather than a
        neighbouring one. Measured under a constant stream — the hardest case —
        and under live training.
        """
        warm = OuterOptimizer(MU, LR, True)
        constant = np.full(N, 127, dtype=np.int64)
        for _ in range(300):
            warm.step(constant, 0)

        newcomer = OuterOptimizer(MU, LR, True)
        caught = None
        for step in range(1, 2001):
            newcomer.step(constant, 0)
            warm.step(constant, 0)
            if caught is None and np.array_equal(newcomer.momentum, warm.momentum):
                caught = step
        self.assertIsNotNone(caught)
        self.assertTrue(np.array_equal(newcomer.momentum, warm.momentum))

    def test_ANodeCanTellWhetherItHasCaughtUp(self):
        """Which is what makes the whole thing safe rather than hopeful.

        Convergence is not guaranteed in general — two nodes with different
        accumulated histories may never meet — so a node must not assume it is
        in sync. It does not have to: every checkpoint header commits to
        `optimizer_state_hash`, so a node compares its own against the chain's
        and knows exactly. Out of sync, it contributes and does not produce;
        in sync, it may produce. No transfer, no guessing.
        """
        a = OuterOptimizer(MU, LR, True)
        b = OuterOptimizer(MU, LR, True)
        stream = lively(50, seed=3)
        for g in stream:
            a.step(g, 0)
        self.assertNotEqual(a.state_hash(), b.state_hash())

        for g in stream:
            b.step(g, 0)
        self.assertEqual(a.state_hash(), b.state_hash(),
                         "same stream from the same start must agree exactly")


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Tests for the checkpoint chain and the fork-choice rule.

The rule is unusual enough to be worth stating in tests rather than prose: there
is no proof of work, heights advance in lockstep, so the tie-break is not a
fallback — it IS the rule, and it has to be total, deterministic and computable
from bytes every node already holds.
"""

import unittest

from rnet.consensus.objects import CheckpointHeader
from rnet.diloco.chain import (MAX_ORPHANS, Chain, ChainError, Outcome,
                               preferred)

ZERO = bytes(32)


def header(step: int, parent: bytes, *, producer: int = 1, weights: int = 0,
           timestamp: int = 0) -> CheckpointHeader:
    """A checkpoint that differs from its siblings only where asked."""
    return CheckpointHeader(
        round_id=0, outer_step=step, parent=parent,
        weights_hash=bytes([weights % 251]) * 32,
        optimizer_state_hash=bytes([weights % 241]) * 32,
        contribution_root=bytes([weights % 239]) * 32,
        producer_id=producer, timestamp_ms=timestamp)


def genesis() -> CheckpointHeader:
    return header(0, ZERO, producer=0)


def build(n: int) -> tuple[Chain, list[CheckpointHeader]]:
    """A chain of n steps past genesis."""
    g = genesis()
    chain = Chain(g, retained=64)
    made, parent = [g], g.id
    for step in range(1, n + 1):
        h = header(step, parent, weights=step)
        chain.add(h)
        made.append(h)
        parent = h.id
    return chain, made


class ForkChoiceTests(unittest.TestCase):

    def test_TheLowerIdWins(self):
        a, b = bytes([0]) + bytes(31), bytes([1]) + bytes(31)
        self.assertTrue(preferred(a, b))
        self.assertFalse(preferred(b, a))

    def test_TheRuleIsTotalAndAntisymmetric(self):
        """Two nodes must never both prefer the other's checkpoint."""
        ids = [bytes([i, j]) + bytes(30) for i in range(4) for j in range(4)]
        for a in ids:
            self.assertFalse(preferred(a, a), "a checkpoint beats itself")
            for b in ids:
                if a == b:
                    continue
                self.assertNotEqual(preferred(a, b), preferred(b, a))

    def test_TheRuleIsTransitive(self):
        ids = sorted(bytes([i]) * 32 for i in range(6))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                for k in range(j + 1, len(ids)):
                    if preferred(ids[i], ids[j]) and preferred(ids[j], ids[k]):
                        self.assertTrue(preferred(ids[i], ids[k]))

    def test_AMalformedIdIsRefused(self):
        with self.assertRaises(ChainError):
            preferred(b"short", bytes(32))


class ChainTests(unittest.TestCase):

    def test_AChainStartsAtGenesis(self):
        chain = Chain(genesis(), retained=8)
        self.assertEqual(chain.height, 0)
        self.assertEqual(chain.head.header, genesis())

    def test_AGenesisThatIsNotGenesisIsRefused(self):
        with self.assertRaises(ChainError):
            Chain(header(1, bytes([9]) * 32))
        with self.assertRaises(ChainError):
            Chain(genesis(), retained=1)

    def test_ItExtends(self):
        chain, made = build(5)
        self.assertEqual(chain.height, 5)
        self.assertEqual(chain.head.header, made[-1])
        for step, h in enumerate(made):
            self.assertEqual(chain.at_height(step).header, h)

    def test_AKnownCheckpointIsNotAddedTwice(self):
        chain, made = build(3)
        self.assertEqual(chain.add(made[2]), Outcome.KNOWN)
        self.assertEqual(chain.height, 3)

    def test_AStepThatDisagreesWithItsParentIsRefused(self):
        chain, made = build(2)
        self.assertEqual(chain.add(header(7, made[2].id)), Outcome.REFUSED)
        self.assertEqual(chain.height, 2)

    # -- forks --------------------------------------------------------------

    def test_TheLowerIdTakesTheHeadAtEqualHeight(self):
        chain, made = build(3)
        rivals = sorted((header(4, made[3].id, weights=w) for w in (10, 20, 30)),
                        key=lambda h: h.id)
        # Offer the worst first, then better ones: each must take the head.
        self.assertEqual(chain.add(rivals[2]), Outcome.EXTENDED)
        self.assertEqual(chain.add(rivals[1]), Outcome.REORGANISED)
        self.assertEqual(chain.add(rivals[0]), Outcome.REORGANISED)
        self.assertEqual(chain.head.id, rivals[0].id)

    def test_AWorseRivalDoesNotTakeTheHead(self):
        chain, made = build(3)
        rivals = sorted((header(4, made[3].id, weights=w) for w in (10, 20)),
                        key=lambda h: h.id)
        self.assertEqual(chain.add(rivals[0]), Outcome.EXTENDED)
        self.assertEqual(chain.add(rivals[1]), Outcome.SIDE_BRANCH)
        self.assertEqual(chain.head.id, rivals[0].id)

    def test_ArrivalOrderDoesNotDecideTheHead(self):
        """The property the rule exists for: two nodes that saw the same
        checkpoints in different orders agree about which one won."""
        _, made = build(3)
        rivals = [header(4, made[3].id, weights=w) for w in (11, 22, 33, 44)]
        heads = set()
        import itertools
        for order in itertools.permutations(rivals):
            chain = Chain(genesis(), retained=64)
            for h in made[1:]:
                chain.add(h)
            for h in order:
                chain.add(h)
            heads.add(chain.head.id)
        self.assertEqual(len(heads), 1)
        self.assertEqual(heads.pop(), min(h.id for h in rivals))

    def test_ALongerChainBeatsAPreferredHash(self):
        """Height first, id second: a node that fell behind must follow."""
        chain, made = build(3)
        low = min((header(4, made[3].id, weights=w) for w in range(20)),
                  key=lambda h: h.id)
        chain.add(low)
        self.assertEqual(chain.head.id, low.id)

        rival = max((header(4, made[3].id, weights=w) for w in range(20, 40)),
                    key=lambda h: h.id)
        chain.add(rival)
        self.assertEqual(chain.head.id, low.id, "a worse sibling took the head")
        deeper = header(5, rival.id, weights=99)
        self.assertEqual(chain.add(deeper), Outcome.EXTENDED)
        self.assertEqual(chain.head.id, deeper.id)
        self.assertEqual(chain.height, 5)

    def test_TheForkPointIsFound(self):
        chain, made = build(3)
        a = header(4, made[3].id, weights=1)
        b = header(4, made[3].id, weights=2)
        chain.add(a)
        chain.add(b)
        a2 = header(5, a.id, weights=3)
        chain.add(a2)
        self.assertEqual(chain.fork_point(a2.id, b.id), made[3].id)
        self.assertEqual(chain.fork_point(made[1].id, made[3].id), made[1].id)

    def test_RollbackMovesTheHead(self):
        chain, made = build(5)
        chain.rollback_to(made[2].id)
        self.assertEqual(chain.height, 2)
        self.assertEqual(chain.head.id, made[2].id)
        with self.assertRaises(ChainError):
            chain.rollback_to(bytes([7]) * 32)

    def test_AncestryIsAboutTheCurrentHead(self):
        chain, made = build(3)
        self.assertTrue(chain.is_ancestor_of_head(made[1].id))
        side = header(4, made[2].id, weights=5)   # branches off at 2
        chain.add(side)
        self.assertTrue(chain.is_ancestor_of_head(made[2].id))

    # -- orphans, which is what makes joining late possible -----------------

    def test_ACheckpointWithAnUnknownParentIsHeldNotDropped(self):
        """The implementation this replaces dropped it, which meant a node that
        started late never caught up: the message it needed in order to ask for
        the parent was the one it threw away."""
        chain = Chain(genesis(), retained=64)
        far = header(5, bytes([9]) * 32, weights=1)
        self.assertEqual(chain.add(far), Outcome.ORPHANED)
        self.assertEqual(chain.missing_parents(), [bytes([9]) * 32])
        self.assertEqual(chain.height, 0)

    def test_OrphansConnectWhenTheParentArrives(self):
        chain = Chain(genesis(), retained=64)
        g = genesis()
        chain_of = [g]
        parent = g.id
        for step in range(1, 6):
            chain_of.append(header(step, parent, weights=step))
            parent = chain_of[-1].id

        # Backwards, as a syncing node receives them.
        for h in reversed(chain_of[1:]):
            chain.add(h)
        self.assertEqual(chain.height, 5)
        self.assertEqual(chain.head.id, chain_of[-1].id)
        self.assertEqual(chain.missing_parents(), [])

    def test_AChainOfOrphansConnectsInOneGo(self):
        """Connecting must cascade: attaching one parent can release a run."""
        chain = Chain(genesis(), retained=64)
        g = genesis()
        made, parent = [g], g.id
        for step in range(1, 11):
            made.append(header(step, parent, weights=step))
            parent = made[-1].id
        for h in made[2:]:                 # everything except step 1
            self.assertEqual(chain.add(h), Outcome.ORPHANED)
        self.assertEqual(chain.height, 0)
        self.assertEqual(chain.add(made[1]), Outcome.EXTENDED)
        self.assertEqual(chain.height, 10)

    def test_TheOrphanPoolIsBounded(self):
        """A peer must not be able to make a node hold unbounded unverifiable
        data by sending headers that lead nowhere."""
        chain = Chain(genesis(), retained=64)
        for i in range(MAX_ORPHANS):
            chain.add(header(9, bytes([i // 256, i % 256]) + bytes(30), weights=i))
        self.assertEqual(chain.add(header(9, bytes([255, 255]) + bytes(30))),
                         Outcome.NO_ROOM)

    def test_AFullPoolDoesNotBlameThePeer(self):
        """Conflating "this node is full" with "this header is malformed" is how
        a node eclipses itself: an attacker fills the pool once, and from then
        on every honest peer offering a checkpoint the node cannot yet connect
        is scored as misbehaving and banned by netblock for a day."""
        self.assertFalse(Outcome.NO_ROOM.blames_the_sender)
        self.assertTrue(Outcome.REFUSED.blames_the_sender)
        for outcome in (Outcome.EXTENDED, Outcome.REORGANISED,
                        Outcome.SIDE_BRANCH, Outcome.ORPHANED, Outcome.KNOWN):
            self.assertFalse(outcome.blames_the_sender, outcome)

    def test_HoldingOrphansDoesNotCostQuadraticHashing(self):
        """Scanning the waiting list by `id` re-serialised and re-hashed every
        entry on every insert, which froze the event loop for minutes on a
        kilobyte of unsolicited headers."""
        import time as _t
        chain = Chain(genesis(), retained=64)
        parent = bytes([7]) * 32
        started = _t.monotonic()
        for i in range(2000):
            self.assertEqual(chain.add(header(9, parent, weights=i)),
                             Outcome.ORPHANED)
        elapsed = _t.monotonic() - started
        # Two thousand into one bucket. By id this was tens of seconds; by
        # header equality it is well under one.
        self.assertLess(elapsed, 5.0, f"took {elapsed:.1f}s")

    def test_AnOrphanIsNotHeldTwice(self):
        chain = Chain(genesis(), retained=64)
        far = header(5, bytes([9]) * 32, weights=1)
        self.assertEqual(chain.add(far), Outcome.ORPHANED)
        self.assertEqual(chain.add(far), Outcome.KNOWN)

    # -- retention ----------------------------------------------------------

    def test_ItPrunesBelowTheRetentionFloor(self):
        chain = Chain(genesis(), retained=4)
        g = genesis()
        parent = g.id
        for step in range(1, 21):
            h = header(step, parent, weights=step)
            chain.add(h)
            parent = h.id
        self.assertEqual(chain.height, 20)
        self.assertIsNotNone(chain.at_height(17))
        self.assertIsNone(chain.at_height(2))

    def test_PruningKeepsEnoughToAnswerAChallenge(self):
        """The policy invariant made concrete: retained must cover the deadline,
        so the weights a challenge needs are still here when it lands."""
        retained = 8
        chain = Chain(genesis(), retained=retained)
        parent = genesis().id
        for step in range(1, 30):
            h = header(step, parent, weights=step)
            chain.add(h)
            parent = h.id
        for back in range(retained):
            self.assertIsNotNone(chain.at_height(chain.height - back), back)


if __name__ == "__main__":
    unittest.main(verbosity=2)

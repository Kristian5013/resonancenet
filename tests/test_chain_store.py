"""The chain on disk.

A node that forgets its chain on restart has thrown away every checkpoint its
workers produced, and the symptom is a height of zero — far too quiet for what
it means. These tests pin the two halves of not doing that: what comes back,
and what happens when it cannot.
"""

import os
import shutil
import tempfile
import unittest

from rnet.consensus.objects import CheckpointHeader
from rnet.diloco.chain import Chain, Outcome
from rnet.node.daemon import (CHAIN_FILE, ChainStoreError, load_chain,
                              save_chain)


def genesis_header() -> CheckpointHeader:
    return CheckpointHeader(
        round_id=0, outer_step=0, parent=bytes(32), weights_hash=bytes([1]) * 32,
        optimizer_state_hash=bytes(32), contribution_root=bytes(32),
        producer_id=0, timestamp_ms=0)


def child(parent: CheckpointHeader, step: int, *, producer: int = 1,
          salt: int = 0) -> CheckpointHeader:
    return CheckpointHeader(
        round_id=0, outer_step=step, parent=parent.id,
        weights_hash=bytes([step % 256, salt % 256]) + bytes(30),
        optimizer_state_hash=bytes([salt % 256]) + bytes(31),
        contribution_root=bytes(32), producer_id=producer,
        timestamp_ms=1_000 * step)


class ChainStoreTests(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="rnet-chainstore-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.path = os.path.join(self.dir, CHAIN_FILE)
        self.genesis = genesis_header()

    def grown(self, steps: int, retained: int = 16) -> Chain:
        chain = Chain(self.genesis, retained=retained)
        head = self.genesis
        for step in range(1, steps + 1):
            head = child(head, step)
            self.assertIs(chain.add(head), Outcome.EXTENDED, step)
        return chain

    # -- the point of the exercise -------------------------------------------

    def test_ARestartKeepsTheWorkRatherThanTheGenesis(self):
        before = self.grown(9)
        save_chain(self.path, before)
        after = load_chain(self.path, self.genesis, 16)
        self.assertEqual(after.height, 9)
        self.assertEqual(after.head.id, before.head.id)
        self.assertEqual(after.head.header.weights_hash,
                         before.head.header.weights_hash)

    def test_EveryCheckpointComesBackNotJustTheHead(self):
        """A head alone cannot answer a challenge about the step before it,
        which is every challenge worth issuing."""
        before = self.grown(6)
        save_chain(self.path, before)
        after = load_chain(self.path, self.genesis, 16)
        for entry in before.entries():
            self.assertTrue(after.has(entry.id), entry.height)
            self.assertEqual(after.at_height(entry.height).id,
                             before.at_height(entry.height).id)

    def test_AMissingFileIsAFreshChainNotAnError(self):
        """The ordinary first start."""
        chain = load_chain(os.path.join(self.dir, "absent"), self.genesis, 16)
        self.assertEqual(chain.height, 0)
        self.assertEqual(chain.head.id, self.genesis.id)

    def test_SavingIsAtomic(self):
        save_chain(self.path, self.grown(3))
        self.assertEqual([f for f in os.listdir(self.dir) if f.endswith(".tmp")], [])

    def test_ItSurvivesBeingSavedAndLoadedRepeatedly(self):
        chain = self.grown(4)
        for _ in range(3):
            save_chain(self.path, chain)
            chain = load_chain(self.path, self.genesis, 16)
        self.assertEqual(chain.height, 4)

    # -- what happens when it cannot ------------------------------------------

    def test_ADamagedFileStopsTheNodeRatherThanResettingIt(self):
        """The whole reason this is not modelled on the address table.

        Addresses are a cache of hearsay and starting empty costs one round of
        seeding. Starting a chain again from genesis discards every checkpoint
        and every worker's training, and the only symptom would be a height of
        zero — so it fails loudly while the file is still there to look at.
        """
        save_chain(self.path, self.grown(5))
        with open(self.path, "r+b") as f:
            f.seek(24)
            f.write(b"\xff\xff\xff\xff")
        with self.assertRaises(ChainStoreError) as caught:
            load_chain(self.path, self.genesis, 16)
        self.assertIn(CHAIN_FILE, str(caught.exception))

    def test_AFileThatIsNotAChainFileIsRefused(self):
        with open(self.path, "wb") as f:
            f.write(b"this is not a chain")
        with self.assertRaises(ChainStoreError):
            load_chain(self.path, self.genesis, 16)

    def test_ADatadirFromAnotherNetworkIsNamedAsSuch(self):
        """Carrying a datadir across networks is a mistake with a specific
        cause, and saying so beats a checksum failure."""
        save_chain(self.path, self.grown(3))
        other = CheckpointHeader(
            round_id=0, outer_step=0, parent=bytes(32),
            weights_hash=bytes([9]) * 32, optimizer_state_hash=bytes(32),
            contribution_root=bytes(32), producer_id=0, timestamp_ms=0)
        with self.assertRaises(ChainStoreError) as caught:
            load_chain(self.path, other, 16)
        self.assertIn("different network", str(caught.exception))

    def test_ATamperedHeaderIsRefusedByTheSameCodeThatRefusesAPeer(self):
        """Nothing in the file is trusted: every header is replayed through
        `Chain.add`, so a checkpoint whose parent does not exist is refused
        exactly as one arriving over the wire would be."""
        chain = Chain(self.genesis, retained=16)
        head = child(self.genesis, 1)
        chain.add(head)
        # A checkpoint whose parent is nothing this chain has ever seen.
        stray = CheckpointHeader(
            round_id=0, outer_step=2, parent=bytes([7]) * 32,
            weights_hash=bytes(32), optimizer_state_hash=bytes(32),
            contribution_root=bytes(32), producer_id=1, timestamp_ms=2000)
        from rnet.canon.stream import Writer
        from rnet.node.daemon import CHAIN_MAGIC, CHAIN_VERSION
        w = Writer().raw(CHAIN_MAGIC).u16(CHAIN_VERSION).u32(2)
        for header in (self.genesis, stray):
            body = header.to_container()
            w.u32(len(body)).raw(body)
        with open(self.path, "wb") as f:
            f.write(w.take())
        with self.assertRaises(ChainStoreError) as caught:
            load_chain(self.path, self.genesis, 16)
        self.assertIn("ORPHANED", str(caught.exception))

    # -- what it holds --------------------------------------------------------

    def test_ItSavesWhatThePruningLeft(self):
        """`retained` bounds the chain in memory, so it bounds the file too —
        the store is not a place where an unbounded history accumulates."""
        chain = self.grown(40, retained=4)
        written = save_chain(self.path, chain)
        self.assertLessEqual(written, 8, written)
        after = load_chain(self.path, self.genesis, 4)
        self.assertEqual(after.head.id, chain.head.id)


if __name__ == "__main__":
    unittest.main(verbosity=2)

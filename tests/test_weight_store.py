"""Weights on disk.

The chain says what the weights became and holds none of them, so without this
a machine that rebooted after a week of training knew exactly where it had got
to and had no way to be there. What matters is that a file is only believed
when it hashes to its own name, and that a file which does not is thrown away
rather than stopped for.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

from rnet.consensus import genesis
from rnet.consensus.init import hash_of_values
from rnet.crypto import merkle
from rnet.model import store
from rnet.model.layout import layout

NETWORK = "regtest"


class WeightStoreTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.round_desc = genesis.round_descriptor(NETWORK)
        cls.spec = cls.round_desc.model
        cls.order = layout(cls.spec)

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="rnet-wstore-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def tensors(self, seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        return {t.name: rng.integers(0, 1 << 16, t.numel, dtype=np.uint16)
                for t in self.order}

    # -- the point ------------------------------------------------------------

    def test_WhatGoesInComesBackExactly(self):
        want = self.tensors()
        path, root = store.write(self.dir, self.spec, want)
        got = store.read(path, self.spec, expect=root)
        self.assertEqual(sorted(got), sorted(want))
        for name in want:
            np.testing.assert_array_equal(got[name], want[name])

    def test_TheNameIsTheRootTheNetworkAgreedOn(self):
        """Not a checksum of this file format: the same Merkle root a checkpoint
        header commits to, so a stored set can be checked against the chain
        without unpacking it."""
        want = self.tensors()
        _, root = store.write(self.dir, self.spec, want)
        self.assertEqual(root, hash_of_values(self.spec, want))
        self.assertTrue(store.has(self.dir, root))
        self.assertTrue(os.path.exists(
            os.path.join(self.dir, root.hex() + ".rnw")))

    def test_TheGenesisWeightsStoreUnderTheirAnchor(self):
        """The end-to-end claim in one line: what `genesis-weights` derives is
        what this stores, under the name the build compiled in."""
        from rnet.model import weights as W
        model = W.build(self.spec, self.round_desc.numerics,
                        bytes.fromhex(genesis.GENESIS_HASH[NETWORK]), device="cpu")
        _, root = store.write(self.dir, self.spec, W.save_weights(model))
        self.assertEqual(root.hex(), genesis.WEIGHTS_HASH[NETWORK])

    def test_FindReturnsThemAndNoneWhenAbsent(self):
        want = self.tensors()
        _, root = store.write(self.dir, self.spec, want)
        self.assertIsNotNone(store.find(self.dir, root, self.spec))
        self.assertIsNone(store.find(self.dir, bytes([9]) * 32, self.spec))

    # -- when it does not verify ----------------------------------------------

    def test_ADamagedFileIsDiscardedNotRaised(self):
        """A cache, unlike the chain: the weights can be derived again or asked
        for, so failing every start on a bad file would be worse than losing it.
        """
        want = self.tensors()
        path, root = store.write(self.dir, self.spec, want)
        with open(path, "r+b") as f:
            f.seek(60)
            f.write(b"\x00\x01\x02\x03")
        self.assertIsNone(store.find(self.dir, root, self.spec))
        self.assertFalse(os.path.exists(path), "the bad file should be gone")

    def test_ReadRefusesAFileWhoseNameLies(self):
        want = self.tensors()
        path, root = store.write(self.dir, self.spec, want)
        with self.assertRaises(store.StoreError):
            store.read(path, self.spec, expect=bytes([7]) * 32)

    def test_ATruncatedFileIsRefused(self):
        want = self.tensors()
        path, root = store.write(self.dir, self.spec, want)
        with open(path, "r+b") as f:
            f.truncate(os.path.getsize(path) - 64)
        with self.assertRaises(store.StoreError) as caught:
            store.read(path, self.spec, expect=root)
        self.assertIn("bytes of tensors", str(caught.exception))

    def test_AFileThatIsNotWeightsIsRefused(self):
        path = os.path.join(self.dir, "00" * 32 + ".rnw")
        with open(path, "wb") as f:
            f.write(b"not weights at all, not even close")
        with self.assertRaises(store.StoreError):
            store.read(path, self.spec)

    def test_WritingIsAtomic(self):
        store.write(self.dir, self.spec, self.tensors())
        self.assertEqual([f for f in os.listdir(self.dir) if f.endswith(".tmp")], [])

    def test_AMissingTensorIsRefusedBeforeAnythingIsWritten(self):
        broken = self.tensors()
        broken.pop(self.order[0].name)
        with self.assertRaises((store.StoreError, KeyError)):
            store.write(self.dir, self.spec, broken)
        self.assertEqual([f for f in os.listdir(self.dir)
                          if f.endswith(store.SUFFIX)], [])

    # -- bounded --------------------------------------------------------------

    def test_PruneKeepsTheRecentAndDropsTheRest(self):
        """759 MiB each for the dense 400M, so an unbounded store is a full
        disk in a day."""
        roots = []
        for seed in range(5):
            _, root = store.write(self.dir, self.spec, self.tensors(seed))
            roots.append(root)
        removed = store.prune(self.dir, roots, retain=2)
        self.assertEqual(removed, 3)
        for root in roots[-2:]:
            self.assertTrue(store.has(self.dir, root), root.hex()[:8])
        for root in roots[:-2]:
            self.assertFalse(store.has(self.dir, root), root.hex()[:8])

    def test_PruneIgnoresFilesItDidNotWrite(self):
        _, root = store.write(self.dir, self.spec, self.tensors())
        stranger = os.path.join(self.dir, "notes.txt")
        with open(stranger, "w") as f:
            f.write("nothing to do with weights")
        store.prune(self.dir, [], retain=0)
        self.assertTrue(os.path.exists(stranger))

    def test_WritingTheSameWeightsTwiceIsOneFile(self):
        want = self.tensors()
        first, root = store.write(self.dir, self.spec, want)
        second, again = store.write(self.dir, self.spec, want)
        self.assertEqual(first, second)
        self.assertEqual(root, again)
        self.assertEqual(len([f for f in os.listdir(self.dir)
                              if f.endswith(store.SUFFIX)]), 1)

    # -- what lets a peer be caught early -------------------------------------

    def test_EveryTensorProvesAgainstTheRoot(self):
        """759 MiB does not fit in one 2 MiB message, so it arrives in pieces —
        and a piece nobody can check is 759 MiB of trust."""
        want = self.tensors()
        _, root = store.write(self.dir, self.spec, want)
        for descriptor in self.order:
            proof = store.tensor_proof(self.spec, want, descriptor.name)
            leaf = merkle.leaf_hash(store.tensor_bytes(want, descriptor))
            self.assertTrue(merkle.verify_proof(leaf, proof, root), descriptor.name)

    def test_ATamperedTensorDoesNotProve(self):
        want = self.tensors()
        _, root = store.write(self.dir, self.spec, want)
        descriptor = self.order[0]
        proof = store.tensor_proof(self.spec, want, descriptor.name)
        tampered = dict(want)
        tampered[descriptor.name] = want[descriptor.name].copy()
        tampered[descriptor.name][0] ^= 1
        leaf = merkle.leaf_hash(store.tensor_bytes(tampered, descriptor))
        self.assertFalse(merkle.verify_proof(leaf, proof, root))

    def test_AnUnknownTensorNameIsRefused(self):
        with self.assertRaises(store.StoreError):
            store.tensor_proof(self.spec, self.tensors(), "no.such.tensor")


if __name__ == "__main__":
    unittest.main(verbosity=2)

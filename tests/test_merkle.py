"""Tests for the Merkle tree that addresses the corpus.

The properties here are what let a worker accept a megabyte of text from an
untrusted peer: the bytes either hash into the pinned root at the index the
schedule named, or they are not corpus.
"""

import hashlib
import unittest

from rnet.crypto.merkle import (LEAF_PREFIX, NODE_PREFIX, MerkleError, Proof,
                                build_proof, leaf_hash, node_hash, root,
                                root_of_chunks, verify_proof)


def chunks(n: int) -> list[bytes]:
    return [f"document {i}\n\n".encode() for i in range(n)]


class MerkleTests(unittest.TestCase):

    def test_ALeafCannotPoseAsANode(self):
        """The whole point of the 0x00/0x01 prefixes.

        Without them a 64-byte chunk that happens to be two concatenated hashes
        would verify as the interior node over them, letting a prover claim a
        subtree as data.
        """
        left, right = leaf_hash(b"a"), leaf_hash(b"b")
        interior = node_hash(left, right)
        forged = leaf_hash(left + right)
        self.assertNotEqual(interior, forged)
        # And the prefixes are the reason, stated as bytes.
        self.assertEqual(interior, hashlib.sha3_256(NODE_PREFIX + left + right).digest())
        self.assertEqual(forged, hashlib.sha3_256(LEAF_PREFIX + left + right).digest())

    def test_EmptyTreeHasADefinedRoot(self):
        """Not zero, so "no corpus" cannot be confused with an unset field."""
        self.assertEqual(root([]), leaf_hash(b""))
        self.assertNotEqual(root([]), bytes(32))

    def test_ASingleLeafIsItsOwnRoot(self):
        self.assertEqual(root([leaf_hash(b"only")]), leaf_hash(b"only"))

    def test_EveryLeafProvesAtEveryWidth(self):
        """Odd widths are where a promoted node could lose its position."""
        for n in list(range(1, 18)) + [31, 32, 33, 64, 100]:
            data = chunks(n)
            leaves = [leaf_hash(c) for c in data]
            r = root_of_chunks(data)
            for i in range(n):
                proof = build_proof(leaves, i)
                self.assertTrue(verify_proof(leaves[i], proof, r),
                                f"leaf {i} of {n} failed to prove")

    def test_AProofDoesNotVerifyTheWrongLeaf(self):
        data = chunks(9)
        leaves = [leaf_hash(c) for c in data]
        r = root_of_chunks(data)
        proof = build_proof(leaves, 3)
        self.assertFalse(verify_proof(leaves[4], proof, r))

    def test_TamperedChunkBytesAreRefused(self):
        data = chunks(12)
        leaves = [leaf_hash(c) for c in data]
        r = root_of_chunks(data)
        proof = build_proof(leaves, 5)
        tampered = bytearray(data[5])
        tampered[0] ^= 0x01
        self.assertFalse(verify_proof(leaf_hash(bytes(tampered)), proof, r))

    def test_AProofWithTrailingSiblingsIsRefused(self):
        """A malleable proof is the same claim at a different size on the wire."""
        data = chunks(8)
        leaves = [leaf_hash(c) for c in data]
        r = root_of_chunks(data)
        good = build_proof(leaves, 2)
        self.assertTrue(verify_proof(leaves[2], good, r))
        padded = Proof(index=good.index, leaf_count=good.leaf_count,
                       path=good.path + (bytes(32),))
        self.assertFalse(verify_proof(leaves[2], padded, r))

    def test_AShortProofIsRefused(self):
        data = chunks(8)
        leaves = [leaf_hash(c) for c in data]
        r = root_of_chunks(data)
        good = build_proof(leaves, 2)
        short = Proof(index=good.index, leaf_count=good.leaf_count, path=good.path[:-1])
        self.assertFalse(verify_proof(leaves[2], short, r))

    def test_ProofsAreBlindWithinAWidthEquivalenceClass(self):
        """Exactly what leaf_count does and does not pin, stated as a set.

        For a given index, every width that yields the same number of levels
        AND the same has-a-sibling decision at each one produces an identical
        walk. Index 2 with a three-level path is that way for widths 5 through
        8: below 5 a level disappears, above 8 one is added.

        Nothing is forged inside the class — the same siblings prove the same
        claim about the same root — but a caller must take the width from the
        dataset manifest rather than from the proof, because the proof does not
        pin it. See the note in crypto/merkle.py.
        """
        data = chunks(8)
        leaves = [leaf_hash(c) for c in data]
        r = root_of_chunks(data)
        good = build_proof(leaves, 2)
        self.assertEqual(len(good.path), 3)

        equivalent = {5, 6, 7, 8}
        for width in range(3, 20):
            accepted = verify_proof(leaves[2], Proof(2, width, good.path), r)
            self.assertEqual(accepted, width in equivalent,
                             f"width {width}: accepted={accepted}, expected "
                             f"{width in equivalent}")

    def test_AWidthOutsideTheClassIsRefused(self):
        """The boundary cases, named, so a shape change cannot pass silently."""
        data = chunks(8)
        leaves = [leaf_hash(c) for c in data]
        r = root_of_chunks(data)
        good = build_proof(leaves, 2)
        for width in (3, 4, 9, 16, 64):
            self.assertFalse(verify_proof(leaves[2], Proof(2, width, good.path), r),
                             f"width {width} reused the width-8 path")

    def test_AnOutOfRangeIndexIsRefused(self):
        data = chunks(5)
        leaves = [leaf_hash(c) for c in data]
        r = root_of_chunks(data)
        self.assertFalse(verify_proof(leaves[0], Proof(5, 5, ()), r))
        self.assertFalse(verify_proof(leaves[0], Proof(0, 0, ()), r))
        with self.assertRaises(MerkleError):
            build_proof(leaves, 5)
        with self.assertRaises(MerkleError):
            build_proof([], 0)

    def test_ChangingAnyChunkChangesTheRoot(self):
        data = chunks(20)
        base = root_of_chunks(data)
        for i in range(len(data)):
            mutated = list(data)
            mutated[i] = mutated[i] + b"!"
            self.assertNotEqual(root_of_chunks(mutated), base, f"chunk {i} did not move the root")

    def test_ReorderingChunksChangesTheRoot(self):
        """Position is part of the commitment, not just membership."""
        data = chunks(10)
        swapped = list(data)
        swapped[2], swapped[7] = swapped[7], swapped[2]
        self.assertNotEqual(root_of_chunks(swapped), root_of_chunks(data))

    def test_TheTreeShapeIsPairwiseWithOddPromotion(self):
        """Pinned against the shape the pinned dataset_root was built with.

        Three leaves fold to node(a,b) and a promoted c, then to
        node(node(a,b), c). If this ever becomes RFC-6962's power-of-two split,
        the 7.36 TB corpus would need reindexing to stay addressable.
        """
        a, b, c = leaf_hash(b"a"), leaf_hash(b"b"), leaf_hash(b"c")
        self.assertEqual(root([a, b, c]), node_hash(node_hash(a, b), c))
        d = leaf_hash(b"d")
        self.assertEqual(root([a, b, c, d]), node_hash(node_hash(a, b), node_hash(c, d)))
        e = leaf_hash(b"e")
        self.assertEqual(root([a, b, c, d, e]),
                         node_hash(node_hash(node_hash(a, b), node_hash(c, d)), e))


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Merkle trees over corpus chunks, with domain separation.

Leaves are hashed with a 0x00 prefix and interior nodes with 0x01, so no leaf
can ever be presented as an interior node. That is the attack the prefixes exist
to close: without them, a 64-byte "chunk" that happens to be two concatenated
hashes would verify as an interior node, and a prover could claim any subtree as
data. Proved by MerkleTests.ALeafCannotPoseAsANode.

The tree shape is pairwise doubling with the odd node promoted unchanged, NOT
RFC-6962's split at the largest power of two. This is a deliberate carry-over:
`dataset_root` 7195da13… was computed over 7.36 TB by the implementation this
one replaces, and the corpus that would be needed to recompute it lives on a
detached volume. Keeping the shape keeps the pin valid. Promotion is safe here
precisely because of the domain separation above — unlike Bitcoin's
duplicate-the-last-node padding, which admits two leaf sets with one root.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

LEAF_PREFIX = b"\x00"
NODE_PREFIX = b"\x01"


class MerkleError(Exception):
    pass


def leaf_hash(data: bytes) -> bytes:
    return hashlib.sha3_256(LEAF_PREFIX + data).digest()


def node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha3_256(NODE_PREFIX + left + right).digest()


def _fold(level: list[bytes]) -> list[bytes]:
    """One level up. The odd node rides unchanged, keeping its position."""
    nxt = [node_hash(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
    if len(level) % 2:
        nxt.append(level[-1])
    return nxt


def root(leaves: list[bytes]) -> bytes:
    """The root over already-hashed leaves.

    An empty tree is the hash of an empty leaf rather than a zero or an error,
    so "no corpus" has a definite root that cannot be confused with an unset
    field. Proved by MerkleTests.EmptyTreeHasADefinedRoot.
    """
    if not leaves:
        return leaf_hash(b"")
    level = list(leaves)
    while len(level) > 1:
        level = _fold(level)
    return level[0]


def root_of_chunks(chunks: list[bytes]) -> bytes:
    """The root over raw chunk bytes — hashes them into leaves first."""
    return root([leaf_hash(c) for c in chunks])


@dataclass(frozen=True)
class Proof:
    """An inclusion proof: which leaf, out of how many, and the siblings.

    `leaf_count` is carried because the walk's shape depends on it — how many
    levels there are, and at which of them this index has a sibling at all.

    But it is NOT a security field, and treating it as one would be a mistake
    worth stating plainly. Different widths can produce an identical walk for a
    given index — 7 and 8 do, for index 2 — so a proof does not pin the width it
    was built at. That forges nothing: an identical walk over identical siblings
    proves the identical claim about the identical root. What it means is that a
    caller must take the width from the artifact that pins it, the dataset
    manifest's `n_chunks`, and refuse a proof that disagrees.

    The equivalence classes are mapped exactly by
    MerkleTests.ProofsAreBlindWithinAWidthEquivalenceClass, and the boundaries
    by MerkleTests.AWidthOutsideTheClassIsRefused. Enforcing the manifest width
    at the call site is a duty of rnet/dataset/, and the first test that module
    owes.
    """

    index: int
    leaf_count: int
    path: tuple[bytes, ...]


def build_proof(leaves: list[bytes], index: int) -> Proof:
    if not leaves:
        raise MerkleError("merkle: an empty tree has no proofs")
    if not 0 <= index < len(leaves):
        raise MerkleError(f"merkle: leaf {index} of {len(leaves)}")

    path: list[bytes] = []
    level = list(leaves)
    pos = index
    while len(level) > 1:
        # A promoted odd node has no sibling at this level and contributes
        # nothing to the path; it simply keeps its relative position.
        if pos % 2 == 1 or pos + 1 < len(level):
            path.append(level[pos + 1 if pos % 2 == 0 else pos - 1])
        pos //= 2
        level = _fold(level)
    return Proof(index=index, leaf_count=len(leaves), path=tuple(path))


def verify_proof(leaf: bytes, proof: Proof, expected_root: bytes) -> bool:
    """True only if `leaf` sits at `proof.index` in a tree rooted at
    `expected_root`.

    Both a short path and a long one are rejected. A long one matters more than
    it looks: extra siblings that the walk never consumes would let a prover
    attach arbitrary bytes to a valid proof and have them ignored, which is a
    malleable proof — same meaning, different encoding, different size on the
    wire. Proved by MerkleTests.AProofWithTrailingSiblingsIsRefused.
    """
    if proof.leaf_count == 0 or not 0 <= proof.index < proof.leaf_count:
        return False

    acc = leaf
    pos = proof.index
    width = proof.leaf_count
    used = 0

    while width > 1:
        if pos % 2 == 1 or pos + 1 < width:
            if used >= len(proof.path):
                return False
            sibling = proof.path[used]
            used += 1
            acc = node_hash(acc, sibling) if pos % 2 == 0 else node_hash(sibling, acc)
        pos //= 2
        width = (width + 1) // 2

    return used == len(proof.path) and acc == expected_root

// Merkle trees with domain separation (RFC 6962 style).
//
//   leaf(d)      = SHA3-256(0x00 || d)
//   internal(l,r)= SHA3-256(0x01 || l || r)
//
// The 0x00/0x01 prefixes make a leaf hash and an internal-node hash structurally
// distinct. That closes two classic holes at once:
//
//   * second preimage — an attacker cannot present an internal node as if it were
//     a leaf,
//   * the duplicate-node ambiguity behind Bitcoin's CVE-2012-2459, where trees
//     with different leaf counts could produce the same root because the last
//     node was duplicated to pad an odd level.
//
// With domain separation an odd node can simply be promoted unchanged, which is
// what this implementation does — no padding, no duplication, no ambiguity.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "crypto/sha3.h"
#include "util/result.h"

namespace rnet::crypto {

Hash256 MerkleLeaf(std::span<const uint8_t> data);
Hash256 MerkleNode(const Hash256& left, const Hash256& right);

// Root over pre-computed leaves. An empty tree hashes to leaf(<empty>) so the
// "no data" case still has a well-defined, non-zero root.
Hash256 MerkleRoot(std::vector<Hash256> leaves);

// Inclusion proof: the sibling path from leaf `index` up to the root.
struct MerkleProof {
    uint64_t index{0};
    uint64_t leaf_count{0};
    std::vector<Hash256> path;
};

Result<MerkleProof> BuildMerkleProof(const std::vector<Hash256>& leaves, uint64_t index);

// Recomputes the root from a leaf hash plus its proof. Returns false if the proof
// does not reconstruct `expected_root` — callers must treat that as tampering.
bool VerifyMerkleProof(const Hash256& leaf, const MerkleProof& proof, const Hash256& expected_root);

}  // namespace rnet::crypto

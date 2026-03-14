// Copyright (c) 2024-2026 The ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "crypto/merkle.h"

#include "crypto/keccak.h"

#include <algorithm>
#include <cstring>

namespace rnet::crypto {

// ---------------------------------------------------------------------------
// Merkle tree construction
//
// Binary hash tree where each internal node is Keccak256d(left || right).
// Odd-count layers duplicate the last leaf before pairing.
//
//          root = H(H01, H23)
//         /                  \
//    H01 = H(L0, L1)    H23 = H(L2, L3)
//     /       \           /       \
//    L0       L1         L2       L3
//
// Used for block transaction commitment (txid merkle root).
// ---------------------------------------------------------------------------

static rnet::uint256 hash_pair(const rnet::uint256& a,
                                const rnet::uint256& b) {
    // 1. Concatenate the two 32-byte hashes.
    std::array<uint8_t, 64> combined;
    std::memcpy(combined.data(), a.data(), 32);
    std::memcpy(combined.data() + 32, b.data(), 32);

    // 2. Keccak256d(left || right).
    return keccak256d(std::span<const uint8_t>(combined));
}

// ---------------------------------------------------------------------------
// compute_merkle_root
//
// Build the tree bottom-up, reducing each layer by pairing adjacent leaves.
// Odd layers duplicate the last element.
// ---------------------------------------------------------------------------

rnet::uint256 compute_merkle_root(std::vector<rnet::uint256> leaves) {
    // 1. Empty tree -> zero hash.
    if (leaves.empty()) {
        return rnet::uint256{};
    }

    // 2. Reduce until one node remains.
    while (leaves.size() > 1) {
        if (leaves.size() % 2 != 0) {
            leaves.push_back(leaves.back());
        }
        std::vector<rnet::uint256> next;
        next.reserve(leaves.size() / 2);
        for (size_t i = 0; i < leaves.size(); i += 2) {
            next.push_back(hash_pair(leaves[i], leaves[i + 1]));
        }
        leaves = std::move(next);
    }
    return leaves[0];
}

// ---------------------------------------------------------------------------
// compute_merkle_branch
//
// Extract the authentication path (sibling hashes) for a given leaf index.
// The branch allows proving inclusion without the full tree.
// ---------------------------------------------------------------------------

std::vector<rnet::uint256> compute_merkle_branch(
    const std::vector<rnet::uint256>& leaves, size_t index) {
    std::vector<rnet::uint256> branch;
    if (leaves.empty()) return branch;

    auto working = leaves;
    size_t idx = index;

    // 1. At each level, record the sibling and reduce.
    while (working.size() > 1) {
        if (working.size() % 2 != 0) {
            working.push_back(working.back());
        }
        size_t sibling = (idx % 2 == 0) ? idx + 1 : idx - 1;
        branch.push_back(working[sibling]);

        std::vector<rnet::uint256> next;
        next.reserve(working.size() / 2);
        for (size_t i = 0; i < working.size(); i += 2) {
            next.push_back(hash_pair(working[i], working[i + 1]));
        }
        working = std::move(next);
        idx /= 2;
    }
    return branch;
}

// ---------------------------------------------------------------------------
// compute_merkle_root_from_branch
//
// Recompute the root from a leaf and its authentication path.
// At each level, the index parity determines left/right ordering.
// ---------------------------------------------------------------------------

rnet::uint256 compute_merkle_root_from_branch(
    const rnet::uint256& leaf,
    const std::vector<rnet::uint256>& branch,
    size_t index) {
    rnet::uint256 current = leaf;
    size_t idx = index;
    for (const auto& sibling : branch) {
        if (idx % 2 == 0) {
            current = hash_pair(current, sibling);
        } else {
            current = hash_pair(sibling, current);
        }
        idx /= 2;
    }
    return current;
}

// ---------------------------------------------------------------------------
// verify_merkle_branch
//
// Verify that a leaf is included in the tree with the given root.
// ---------------------------------------------------------------------------

bool verify_merkle_branch(const rnet::uint256& leaf,
                          const std::vector<rnet::uint256>& branch,
                          size_t index,
                          const rnet::uint256& root) {
    return compute_merkle_root_from_branch(leaf, branch, index) == root;
}

} // namespace rnet::crypto

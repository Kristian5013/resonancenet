#include "crypto/merkle.h"
#include "crypto/keccak.h"

#include <algorithm>
#include <cstring>

namespace rnet::crypto {

static rnet::uint256 hash_pair(const rnet::uint256& a,
                                const rnet::uint256& b) {
    // Concatenate the two 32-byte hashes and keccak256d the result
    std::array<uint8_t, 64> combined;
    std::memcpy(combined.data(), a.data(), 32);
    std::memcpy(combined.data() + 32, b.data(), 32);
    return keccak256d(std::span<const uint8_t>(combined));
}

rnet::uint256 compute_merkle_root(std::vector<rnet::uint256> leaves) {
    if (leaves.empty()) {
        return rnet::uint256{};
    }
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

std::vector<rnet::uint256> compute_merkle_branch(
    const std::vector<rnet::uint256>& leaves, size_t index) {
    std::vector<rnet::uint256> branch;
    if (leaves.empty()) return branch;

    auto working = leaves;
    size_t idx = index;

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

bool verify_merkle_branch(const rnet::uint256& leaf,
                          const std::vector<rnet::uint256>& branch,
                          size_t index,
                          const rnet::uint256& root) {
    return compute_merkle_root_from_branch(leaf, branch, index) == root;
}

}  // namespace rnet::crypto

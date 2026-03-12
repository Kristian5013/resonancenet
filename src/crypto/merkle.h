#pragma once

#include <vector>
#include "core/types.h"

namespace rnet::crypto {

/// Compute Merkle root from leaf hashes using keccak256d
rnet::uint256 compute_merkle_root(std::vector<rnet::uint256> leaves);

/// Compute Merkle branch (proof) for a given leaf index
std::vector<rnet::uint256> compute_merkle_branch(
    const std::vector<rnet::uint256>& leaves, size_t index);

/// Verify a Merkle branch
bool verify_merkle_branch(const rnet::uint256& leaf,
                          const std::vector<rnet::uint256>& branch,
                          size_t index,
                          const rnet::uint256& root);

/// Compute Merkle root from a branch and leaf
rnet::uint256 compute_merkle_root_from_branch(
    const rnet::uint256& leaf,
    const std::vector<rnet::uint256>& branch,
    size_t index);

}  // namespace rnet::crypto

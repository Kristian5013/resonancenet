// Copyright (c) 2025 The ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "consensus/merkle.h"

#include "crypto/merkle.h"

namespace rnet::consensus {

// ===========================================================================
//  Block-level Merkle root computation
// ===========================================================================

// ---------------------------------------------------------------------------
// block_merkle_root
//   Computes the Merkle root from the txids of all transactions in a block.
// ---------------------------------------------------------------------------
rnet::uint256 block_merkle_root(const primitives::CBlock& block)
{
    if (block.vtx.empty()) {
        return rnet::uint256{};
    }

    // 1. Collect txids as leaves
    std::vector<rnet::uint256> leaves;
    leaves.reserve(block.vtx.size());
    for (const auto& tx : block.vtx) {
        leaves.push_back(tx->txid());
    }

    return crypto::compute_merkle_root(std::move(leaves));
}

// ---------------------------------------------------------------------------
// block_witness_merkle_root
//   Same as block_merkle_root but uses wtxids and replaces the coinbase
//   entry (index 0) with a zero hash.
// ---------------------------------------------------------------------------
rnet::uint256 block_witness_merkle_root(const primitives::CBlock& block)
{
    if (block.vtx.empty()) {
        return rnet::uint256{};
    }

    std::vector<rnet::uint256> leaves;
    leaves.reserve(block.vtx.size());

    // 1. Coinbase wtxid is replaced with zero hash
    leaves.emplace_back();  // zero-initialized uint256

    // 2. Remaining transactions use their wtxid
    for (size_t i = 1; i < block.vtx.size(); ++i) {
        leaves.push_back(block.vtx[i]->wtxid());
    }

    return crypto::compute_merkle_root(std::move(leaves));
}

} // namespace rnet::consensus

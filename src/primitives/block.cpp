// Copyright (c) 2025 The ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "primitives/block.h"

#include "crypto/keccak.h"

namespace rnet::primitives {

// ===========================================================================
//  CBlock
// ===========================================================================

// ---------------------------------------------------------------------------
// compute_merkle_root
//   Builds a binary Merkle tree from transaction ids.  Odd-length levels
//   are padded by duplicating the last leaf.  Each pair is hashed with
//   Keccak-256d(left || right).
// ---------------------------------------------------------------------------
rnet::uint256 CBlock::compute_merkle_root() const
{
    if (vtx.empty()) {
        return rnet::uint256{};
    }

    // 1. Collect transaction ids as leaves
    std::vector<rnet::uint256> leaves;
    leaves.reserve(vtx.size());
    for (const auto& tx : vtx) {
        leaves.push_back(tx->txid());
    }

    // 2. Build merkle tree iteratively
    while (leaves.size() > 1) {
        if (leaves.size() % 2 != 0) {
            leaves.push_back(leaves.back());
        }

        std::vector<rnet::uint256> next_level;
        next_level.reserve(leaves.size() / 2);

        for (size_t i = 0; i < leaves.size(); i += 2) {
            std::vector<uint8_t> combined(64);
            std::memcpy(combined.data(), leaves[i].data(), 32);
            std::memcpy(combined.data() + 32, leaves[i + 1].data(), 32);
            next_level.push_back(
                crypto::keccak256d(std::span<const uint8_t>(combined)));
        }

        leaves = std::move(next_level);
    }

    return leaves[0];
}

// ---------------------------------------------------------------------------
// get_block_size
//   Sum of serialised sizes of all transactions (no witness discount).
// ---------------------------------------------------------------------------
size_t CBlock::get_block_size() const
{
    size_t total = 0;
    for (const auto& tx : vtx) {
        total += tx->get_total_size();
    }
    return total;
}

// ---------------------------------------------------------------------------
// get_block_weight
//   Sum of segwit-style weights of all transactions.
// ---------------------------------------------------------------------------
size_t CBlock::get_block_weight() const
{
    size_t total = 0;
    for (const auto& tx : vtx) {
        total += tx->get_weight();
    }
    return total;
}

// ---------------------------------------------------------------------------
// to_string
// ---------------------------------------------------------------------------
std::string CBlock::to_string() const
{
    std::string result = "CBlock(";
    result += CBlockHeader::to_string();
    result += ", txs=" + std::to_string(vtx.size());
    result += ")";
    return result;
}

} // namespace rnet::primitives

// Copyright (c) 2025 The ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "chain/block_index.h"

#include <sstream>

namespace rnet::chain {

// ===========================================================================
//  CBlockIndex -- in-memory block tree node
// ===========================================================================

// ---------------------------------------------------------------------------
// Constructor
//   Populates index fields from a deserialised block header.
// ---------------------------------------------------------------------------
CBlockIndex::CBlockIndex(const primitives::CBlockHeader& hdr)
    : header(hdr)
    , height(static_cast<int>(hdr.height))
    , timestamp(hdr.timestamp)
    , val_loss(hdr.val_loss)
    , d_model(hdr.d_model)
    , n_layers(hdr.n_layers)
{
    block_hash = hdr.hash();
}

// ---------------------------------------------------------------------------
// get_block_hash
//   Lazily computes the header hash if it has not been cached yet.
// ---------------------------------------------------------------------------
const rnet::uint256& CBlockIndex::get_block_hash() const
{
    if (block_hash.is_zero() && height > 0) {
        const_cast<CBlockIndex*>(this)->block_hash = header.hash();
    }
    return block_hash;
}

// ---------------------------------------------------------------------------
// get_ancestor
//   Walks the ->prev chain to find the ancestor at target_height.
//   Returns nullptr if target_height is out of range.
// ---------------------------------------------------------------------------
CBlockIndex* CBlockIndex::get_ancestor(int target_height)
{
    if (target_height > height || target_height < 0) {
        return nullptr;
    }
    CBlockIndex* walk = this;
    while (walk && walk->height != target_height) {
        walk = walk->prev;
    }
    return walk;
}

// ---------------------------------------------------------------------------
// get_ancestor (const overload)
// ---------------------------------------------------------------------------
const CBlockIndex* CBlockIndex::get_ancestor(int target_height) const
{
    return const_cast<CBlockIndex*>(this)->get_ancestor(target_height);
}

// ---------------------------------------------------------------------------
// get_locator
//   Builds a block-locator vector: the first 10 hashes are consecutive,
//   then spacing doubles exponentially.
// ---------------------------------------------------------------------------
std::vector<rnet::uint256> CBlockIndex::get_locator() const
{
    std::vector<rnet::uint256> have;
    const CBlockIndex* idx = this;
    int step = 1;
    while (idx) {
        have.push_back(idx->get_block_hash());
        // 1. Exponentially increase step after first 10 entries
        if (static_cast<int>(have.size()) > 10) {
            step *= 2;
        }
        for (int i = 0; idx && i < step; ++i) {
            idx = idx->prev;
        }
    }
    return have;
}

// ---------------------------------------------------------------------------
// is_ancestor_of
//   Returns true if *this is an ancestor of *other (i.e. on the same chain
//   at a lower or equal height with matching hash).
// ---------------------------------------------------------------------------
bool CBlockIndex::is_ancestor_of(const CBlockIndex* other) const
{
    if (!other) return false;
    if (other->height < height) return false;
    const CBlockIndex* walk = other->get_ancestor(height);
    return walk && walk->block_hash == block_hash;
}

// ---------------------------------------------------------------------------
// to_string
// ---------------------------------------------------------------------------
std::string CBlockIndex::to_string() const
{
    std::ostringstream oss;
    oss << "CBlockIndex(height=" << height
        << " hash=" << get_block_hash().to_hex().substr(0, 16) << "..."
        << " val_loss=" << val_loss
        << " status=" << static_cast<int>(status)
        << ")";
    return oss.str();
}

} // namespace rnet::chain

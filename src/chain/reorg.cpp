// Copyright (c) 2025 The ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "chain/reorg.h"

#include "chain/chainstate.h"
#include "core/logging.h"

#include <algorithm>

namespace rnet::chain {

// ===========================================================================
//  Chain reorganisation helpers
// ===========================================================================

// ---------------------------------------------------------------------------
// find_fork_point
//   Walks two block-index pointers back to the most recent common ancestor.
//   Both chains are first equalised to the same height, then stepped back
//   in lockstep until they converge.
// ---------------------------------------------------------------------------
CBlockIndex* find_fork_point(CBlockIndex* a, CBlockIndex* b)
{
    if (!a || !b) return nullptr;

    // 1. Walk both pointers to the same height
    while (a->height > b->height) a = a->prev;
    while (b->height > a->height) b = b->prev;

    // 2. Walk both back until they meet
    while (a != b) {
        if (!a || !b) return nullptr;
        a = a->prev;
        b = b->prev;
    }

    return a;
}

// ---------------------------------------------------------------------------
// compute_reorg_path
//   Given old_tip, new_tip, and their fork_point, fills two vectors:
//     disconnect -- blocks from old_tip down to (but excluding) fork_point
//     connect    -- blocks from fork_point up to new_tip, in connect order
// ---------------------------------------------------------------------------
void compute_reorg_path(CBlockIndex* old_tip,
                        CBlockIndex* new_tip,
                        CBlockIndex* fork_point,
                        std::vector<CBlockIndex*>& disconnect,
                        std::vector<CBlockIndex*>& connect)
{
    disconnect.clear();
    connect.clear();

    // 1. Blocks to disconnect: old_tip down to fork_point (exclusive)
    for (CBlockIndex* idx = old_tip; idx && idx != fork_point; idx = idx->prev) {
        disconnect.push_back(idx);
    }

    // 2. Blocks to connect: fork_point up to new_tip (reversed after collection)
    for (CBlockIndex* idx = new_tip; idx && idx != fork_point; idx = idx->prev) {
        connect.push_back(idx);
    }
    std::reverse(connect.begin(), connect.end());
}

// ---------------------------------------------------------------------------
// is_reorg_safe
//   Guard: rejects reorgs deeper than max_reorg_depth.
// ---------------------------------------------------------------------------
bool is_reorg_safe(const ReorgInfo& info, int max_reorg_depth)
{
    return info.disconnected_count <= max_reorg_depth;
}

// ---------------------------------------------------------------------------
// execute_reorg
//   Primarily handled by CChainState::activate_best_chain().  This function
//   produces the ReorgInfo record used for logging and notification.
// ---------------------------------------------------------------------------
Result<ReorgInfo> execute_reorg(CChainState& /*chainstate*/,
                                CBlockIndex* new_tip)
{
    ReorgInfo info;
    info.new_tip = new_tip;

    LogPrintf("Reorg to height %d, val_loss=%.6f",
              new_tip->height, static_cast<double>(new_tip->val_loss));

    return Result<ReorgInfo>::ok(info);
}

} // namespace rnet::chain

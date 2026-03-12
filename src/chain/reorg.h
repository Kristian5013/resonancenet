#pragma once

#include <vector>

#include "chain/block_index.h"
#include "core/error.h"

namespace rnet::chain {

class CChainState;

/// ReorgInfo — information about a chain reorganization.
struct ReorgInfo {
    CBlockIndex* old_tip = nullptr;
    CBlockIndex* new_tip = nullptr;
    CBlockIndex* fork_point = nullptr;
    int disconnected_count = 0;
    int connected_count = 0;
};

/// Find the fork point between two chain tips.
/// Returns nullptr if the chains share no common ancestor (should not happen).
CBlockIndex* find_fork_point(CBlockIndex* a, CBlockIndex* b);

/// Get the list of blocks to disconnect and connect for a reorg.
/// disconnect: blocks from old_tip down to (not including) fork_point
/// connect: blocks from fork_point up to new_tip
void compute_reorg_path(CBlockIndex* old_tip,
                        CBlockIndex* new_tip,
                        CBlockIndex* fork_point,
                        std::vector<CBlockIndex*>& disconnect,
                        std::vector<CBlockIndex*>& connect);

/// Check if a reorg is safe to perform (not too deep, etc.)
bool is_reorg_safe(const ReorgInfo& info, int max_reorg_depth = 100);

/// Execute a chain reorganization.
/// This is called internally by CChainState::activate_best_chain().
Result<ReorgInfo> execute_reorg(CChainState& chainstate,
                                CBlockIndex* new_tip);

}  // namespace rnet::chain

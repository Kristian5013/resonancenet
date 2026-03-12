#include "chain/reorg.h"

#include <algorithm>

#include "chain/chainstate.h"
#include "core/logging.h"

namespace rnet::chain {

CBlockIndex* find_fork_point(CBlockIndex* a, CBlockIndex* b) {
    if (!a || !b) return nullptr;

    // Walk both pointers to the same height
    while (a->height > b->height) a = a->prev;
    while (b->height > a->height) b = b->prev;

    // Walk both back until they meet
    while (a != b) {
        if (!a || !b) return nullptr;
        a = a->prev;
        b = b->prev;
    }

    return a;
}

void compute_reorg_path(CBlockIndex* old_tip,
                        CBlockIndex* new_tip,
                        CBlockIndex* fork_point,
                        std::vector<CBlockIndex*>& disconnect,
                        std::vector<CBlockIndex*>& connect)
{
    disconnect.clear();
    connect.clear();

    // Blocks to disconnect: old_tip down to fork_point (exclusive)
    for (CBlockIndex* idx = old_tip; idx && idx != fork_point; idx = idx->prev) {
        disconnect.push_back(idx);
    }

    // Blocks to connect: fork_point up to new_tip
    for (CBlockIndex* idx = new_tip; idx && idx != fork_point; idx = idx->prev) {
        connect.push_back(idx);
    }
    std::reverse(connect.begin(), connect.end());
}

bool is_reorg_safe(const ReorgInfo& info, int max_reorg_depth) {
    return info.disconnected_count <= max_reorg_depth;
}

Result<ReorgInfo> execute_reorg(CChainState& /*chainstate*/,
                                CBlockIndex* new_tip)
{
    // This is primarily handled by CChainState::activate_best_chain().
    // This function provides the reorg info for logging/notification.
    ReorgInfo info;
    info.new_tip = new_tip;

    LogPrintf("Reorg to height %d, val_loss=%.6f",
              new_tip->height, static_cast<double>(new_tip->val_loss));

    return Result<ReorgInfo>::ok(info);
}

}  // namespace rnet::chain

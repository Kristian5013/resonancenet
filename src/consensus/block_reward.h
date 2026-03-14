#pragma once

#include <cstdint>

#include "consensus/params.h"

namespace rnet::consensus {

/// Emission state tracked across the chain.
struct EmissionState {
    int64_t total_minted = 0;      ///< Total coins ever minted
    int64_t effective_supply = 0;  ///< Currently circulating supply
    int64_t estimated_lost = 0;    ///< Coins estimated lost / expired
};

/// Breakdown of a block's reward.
struct BlockReward {
    int64_t base = 0;       ///< Base subsidy (halving-aware)
    int64_t recovered = 0;  ///< Coins returned from expired UTXOs

    int64_t total() const { return base + recovered; }
};

/// Compute the base reward given the current emission state.
/// Counts how many halving thresholds the effective_supply has crossed
/// and halves the initial_reward accordingly.
int64_t get_base_reward(const EmissionState& state, const ConsensusParams& params);

/// Compute the full block reward (base subsidy + recovered coins).
/// @param height       Block height.
/// @param state        Current emission state.
/// @param params       Consensus parameters.
BlockReward compute_block_reward(uint64_t height,
                                 const EmissionState& state,
                                 const ConsensusParams& params);

}  // namespace rnet::consensus

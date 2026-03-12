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
    int64_t bonus = 0;      ///< Improvement bonus (up to 10% of base)
    int64_t recovered = 0;  ///< Coins returned from expired UTXOs

    int64_t total() const { return base + bonus + recovered; }
};

/// Compute the base reward given the current emission state.
/// Counts how many halving thresholds the effective_supply has crossed
/// and halves the initial_reward accordingly.
int64_t get_base_reward(const EmissionState& state, const ConsensusParams& params);

/// Compute the full block reward including bonus and recovered coins.
/// @param height       Block height.
/// @param improvement  Fractional loss improvement (0.0 to 1.0+, clamped).
/// @param state        Current emission state.
/// @param params       Consensus parameters.
BlockReward compute_block_reward(uint64_t height,
                                 float improvement,
                                 const EmissionState& state,
                                 const ConsensusParams& params);

}  // namespace rnet::consensus

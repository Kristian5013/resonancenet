#include "consensus/block_reward.h"

#include <algorithm>
#include <cmath>

namespace rnet::consensus {

int64_t get_base_reward(const EmissionState& state, const ConsensusParams& params) {
    int64_t reward = params.initial_reward;

    // Count how many halving thresholds the effective supply has crossed
    for (const auto& threshold : params.halving_thresholds) {
        if (state.effective_supply >= threshold) {
            reward /= 2;
        } else {
            break;
        }
    }

    // Reward cannot go below 1 resonance
    if (reward < 1) {
        reward = 1;
    }

    return reward;
}

BlockReward compute_block_reward(uint64_t height,
                                 float improvement,
                                 const EmissionState& state,
                                 const ConsensusParams& params) {
    BlockReward reward{};

    // Genesis block gets the full initial reward, no bonus
    if (height == 0) {
        reward.base = params.initial_reward;
        return reward;
    }

    // Base reward with halving
    reward.base = get_base_reward(state, params);

    // Cap total minted at max_supply
    if (state.total_minted >= params.max_supply) {
        reward.base = 0;
    } else if (state.total_minted + reward.base > params.max_supply) {
        reward.base = params.max_supply - state.total_minted;
    }

    // Bonus: 10% of base * improvement (clamped to [0, 1])
    float clamped_improvement = std::clamp(improvement, 0.0f, 1.0f);
    reward.bonus = static_cast<int64_t>(
        static_cast<double>(reward.base) * 0.10 * static_cast<double>(clamped_improvement));

    // Recovered coins from expired UTXOs are passed through directly
    // (set by the caller from chain state, not computed here)

    return reward;
}

}  // namespace rnet::consensus

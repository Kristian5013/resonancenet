// Copyright (c) 2024-2026 The ResonanceNet Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "consensus/block_reward.h"

#include <algorithm>
#include <cmath>

namespace rnet::consensus {

// ---------------------------------------------------------------------------
// get_base_reward
// ---------------------------------------------------------------------------
// Adaptive emission based on effective supply thresholds.
//
// Unlike fixed halving intervals (Bitcoin: every 210'000 blocks),
// ResonanceNet halves when the effective supply crosses predefined
// thresholds.  Effective supply accounts for expired UTXOs:
//
//   effective_supply = total_minted - total_expired
//
// This creates a self-regulating emission curve: as UTXO expiry
// reclaims coins (val_loss improves 10x), the effective supply drops,
// potentially reversing a halving and sustaining miner incentives.
//
// Threshold table (mainnet defaults):
//
//   Threshold #   Effective supply crossed       Reward
//   -----------   -------------------------      ------
//        0        < 5'250'000 RNET                50 RNET  (50 * 10^8 base units)
//        1        >= 5'250'000 RNET               25 RNET
//        2        >= 10'500'000 RNET              12.5 RNET
//        3        >= 15'750'000 RNET               6.25 RNET
//        4        >= 18'375'000 RNET               3.125 RNET
//
// Floor: reward never drops below 1 base unit (1 resonance).
// ---------------------------------------------------------------------------
int64_t get_base_reward(const EmissionState& state, const ConsensusParams& params)
{
    // 1. Start with the full initial reward.
    //    Default: 50 RNET = 5'000'000'000 base units (50 * 10^8).
    int64_t reward = params.initial_reward;

    // 2. Walk the halving threshold table.  Each threshold that the
    //    effective supply has reached or exceeded triggers one halving.
    //    Thresholds are sorted ascending; stop at the first unmet one.
    for (const auto& threshold : params.halving_thresholds) {
        if (state.effective_supply >= threshold) {
            reward /= 2;
        } else {
            break;
        }
    }

    // 3. Enforce a minimum reward of 1 resonance (1 base unit).
    //    This guarantees miners always receive something, even after
    //    many halvings or extreme integer truncation.
    if (reward < 1) {
        reward = 1;
    }

    return reward;
}

// ---------------------------------------------------------------------------
// compute_block_reward
// ---------------------------------------------------------------------------
// Assembles the full block reward from three components:
//
//   total = base + bonus + recovered
//
// Where:
//   base      — halving-aware subsidy from get_base_reward()
//   bonus     — training quality incentive (up to 10% of base)
//   recovered — coins returned from expired UTXOs (set by caller)
//
// Bonus formula (integer arithmetic, truncates toward zero):
//
//   bonus = base * 0.10 * clamp(improvement, 0, 1)
//
// The improvement value measures how much a miner's training step
// reduced the model's validation loss.  Clamping to [0, 1] prevents
// gaming via artificially inflated metrics while still rewarding
// genuine progress.
//
// Supply cap enforcement:
//   - If total_minted has already reached max_supply (21'000'000 RNET),
//     the base reward drops to zero.  Miners then rely on transaction
//     fees and recovered UTXO coins.
//   - If this block's reward would overshoot max_supply, the base is
//     trimmed to exactly fill the remaining gap.
//
// Genesis special case (height == 0):
//   Returns the full initial_reward with no bonus and no cap check,
//   since the genesis block bootstraps the chain before any emission
//   state exists.
// ---------------------------------------------------------------------------
BlockReward compute_block_reward(uint64_t height,
                                 float improvement,
                                 const EmissionState& state,
                                 const ConsensusParams& params)
{
    BlockReward reward{};

    // 1. Genesis block: return the full initial reward, no bonus.
    //    50 RNET (50 * 10^8 base units) on mainnet.
    if (height == 0) {
        reward.base = params.initial_reward;
        return reward;
    }

    // 2. Compute halving-aware base reward from effective supply.
    reward.base = get_base_reward(state, params);

    // 3. Enforce the hard supply cap of 21'000'000 RNET.
    //    Once total_minted reaches max_supply, no new coins are created.
    if (state.total_minted >= params.max_supply) {
        reward.base = 0;
    } else if (state.total_minted + reward.base > params.max_supply) {
        // Trim to the exact remaining supply so we never overshoot.
        reward.base = params.max_supply - state.total_minted;
    }

    // 4. Compute training quality bonus.
    //    bonus = base * 0.10 * clamp(improvement, 0, 1)
    //
    //    Cast through double to preserve precision before the final
    //    truncation to int64_t.
    float clamped_improvement = std::clamp(improvement, 0.0f, 1.0f);
    reward.bonus = static_cast<int64_t>(
        static_cast<double>(reward.base) * 0.10 * static_cast<double>(clamped_improvement));

    // 5. Recovered coins from expired UTXOs are not computed here;
    //    they are set by the caller from chain state before calling
    //    BlockReward::total().

    return reward;
}

} // namespace rnet::consensus

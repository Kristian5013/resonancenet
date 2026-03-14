// Copyright (c) 2024-2026 The ResonanceNet Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "consensus/growth_policy.h"

#include <algorithm>

namespace rnet::consensus {

// ---------------------------------------------------------------------------
// compute_growth
// ---------------------------------------------------------------------------
// Continuous model growth algorithm (CONSENSUS-CRITICAL).
//
// Stagnation counter tracks consecutive non-improving blocks:
//   if loss_improved:  stagnation' = 0,  delta = BASE_GROWTH (2 dims)
//   else:              stagnation' = stagnation + 1,  delta = 0
//
// On each improving block the hidden dimension grows by BASE_GROWTH:
//   d_model' = min(d_model + delta, MAX_D_MODEL)        [cap 4'096]
//
// Layer count is a deterministic function of cumulative growth:
//   cumulative = d_model' - GENESIS_D_MODEL              [384 base]
//   n_layers'  = GENESIS_N_LAYERS + cumulative / LAYER_THRESHOLD
//              = 6 + cumulative / 128
//   n_layers'  = min(n_layers', MAX_LAYERS)              [cap 48]
//
// Feed-forward dimension is always twice the model width:
//   d_ff' = 2 * d_model'
// ---------------------------------------------------------------------------
GrowthResult GrowthPolicy::compute_growth(const GrowthState& state,
                                          bool loss_improved)
{
    GrowthResult result{};

    // 1. Update stagnation counter and compute raw dimension delta.
    uint32_t new_stagnation{0};
    uint32_t delta{0};

    if (loss_improved) {
        new_stagnation = 0;
        delta = BASE_GROWTH;                       // +2 dims per improving block
    } else {
        new_stagnation = state.stagnation + 1;
        delta = 0;                                 // no growth without improvement
    }

    // 2. Grow d_model, clamped to MAX_D_MODEL (4'096).
    uint32_t new_d_model = std::min(state.d_model + delta, MAX_D_MODEL);
    delta = new_d_model - state.d_model;           // actual delta after clamping

    // 3. Derive layer count from cumulative growth since genesis.
    //    cumulative = d_model' - 384
    //    n_layers'  = 6 + cumulative / 128
    uint32_t cumulative_growth = new_d_model - GENESIS_D_MODEL;
    uint32_t expected_layers   = GENESIS_N_LAYERS
                               + cumulative_growth / LAYER_THRESHOLD;
    uint32_t new_n_layers = std::min(expected_layers, MAX_LAYERS);

    // 4. Feed-forward width = 2 * d_model'.
    uint32_t new_d_ff = 2 * new_d_model;

    // 5. Pack result.
    result.new_d_model    = new_d_model;
    result.new_n_layers   = new_n_layers;
    result.new_d_ff       = new_d_ff;
    result.delta_d_model  = delta;
    result.delta_n_layers = static_cast<int>(new_n_layers)
                          - static_cast<int>(state.n_layers);
    result.layer_added    = (new_n_layers > state.n_layers);
    result.new_stagnation = new_stagnation;

    return result;
}

// ---------------------------------------------------------------------------
// verify_growth
// ---------------------------------------------------------------------------
// Consensus rule: every block's growth fields must match the deterministic
// output of compute_growth given the parent header.  Genesis is special-
// cased — it must carry exactly the genesis constants and zero deltas.
// ---------------------------------------------------------------------------
bool GrowthPolicy::verify_growth(const primitives::CBlockHeader& header,
                                 const primitives::CBlockHeader& parent)
{
    // 1. Genesis block: fixed dimensions, no growth.
    if (header.is_genesis()) {
        return header.d_model         == GENESIS_D_MODEL      // 384
            && header.n_layers        == GENESIS_N_LAYERS      // 6
            && header.d_ff            == 2 * GENESIS_D_MODEL   // 768
            && header.stagnation_count == 0
            && header.growth_delta     == 0;
    }

    // 2. Determine whether this block improved the validation loss.
    bool loss_improved = header.val_loss < parent.val_loss;

    // 3. Reconstruct parent state.
    GrowthState state{};
    state.d_model   = parent.d_model;
    state.n_layers  = parent.n_layers;
    state.stagnation = parent.stagnation_count;
    state.last_loss = parent.val_loss;

    // 4. Compute expected growth from parent state.
    GrowthResult expected = compute_growth(state, loss_improved);

    // 5. Every growth field must match exactly (consensus-critical).
    if (header.d_model          != expected.new_d_model)    return false;
    if (header.n_layers         != expected.new_n_layers)   return false;
    if (header.d_ff             != expected.new_d_ff)       return false;
    if (header.growth_delta     != expected.delta_d_model)  return false;
    if (header.stagnation_count != expected.new_stagnation) return false;

    return true;
}

// ---------------------------------------------------------------------------
// expected_growth
// ---------------------------------------------------------------------------
// Convenience helper for miners: returns the growth result assuming the next
// block will improve the validation loss.  The caller is responsible for
// verifying whether the loss actually improved.
// ---------------------------------------------------------------------------
GrowthResult GrowthPolicy::expected_growth(const primitives::CBlockHeader& parent)
{
    // 1. Build state from the tip header.
    GrowthState state{};
    state.d_model   = parent.d_model;
    state.n_layers  = parent.n_layers;
    state.stagnation = parent.stagnation_count;
    state.last_loss = parent.val_loss;

    // 2. Optimistic: assume loss improved.
    return compute_growth(state, true);
}

} // namespace rnet::consensus

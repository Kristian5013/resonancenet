#include "consensus/growth_policy.h"

#include <algorithm>

namespace rnet::consensus {

GrowthResult GrowthPolicy::compute_growth(const GrowthState& state, bool loss_improved) {
    GrowthResult r{};

    // Stagnation tracking
    uint32_t new_stagnation;
    uint32_t delta;

    if (loss_improved) {
        new_stagnation = 0;
        delta = BASE_GROWTH;
    } else {
        new_stagnation = state.stagnation + 1;
        delta = 0;  // No growth without improvement
    }

    // New d_model (clamped to max)
    uint32_t new_d_model = std::min(state.d_model + delta, MAX_D_MODEL);
    delta = new_d_model - state.d_model;  // Actual delta after clamping

    // Layer computation based on cumulative growth from genesis
    uint32_t cumulative_growth = new_d_model - GENESIS_D_MODEL;
    uint32_t expected_layers = GENESIS_N_LAYERS + cumulative_growth / LAYER_THRESHOLD;
    uint32_t new_n_layers = std::min(expected_layers, MAX_LAYERS);

    // Feed-forward dimension is always 2x d_model
    uint32_t new_d_ff = 2 * new_d_model;

    r.new_d_model = new_d_model;
    r.new_n_layers = new_n_layers;
    r.new_d_ff = new_d_ff;
    r.delta_d_model = delta;
    r.delta_n_layers = static_cast<int>(new_n_layers) - static_cast<int>(state.n_layers);
    r.layer_added = (new_n_layers > state.n_layers);
    r.new_stagnation = new_stagnation;

    return r;
}

bool GrowthPolicy::verify_growth(const primitives::CBlockHeader& header,
                                  const primitives::CBlockHeader& parent) {
    // Genesis block has no parent to verify against
    if (header.is_genesis()) {
        return header.d_model == GENESIS_D_MODEL &&
               header.n_layers == GENESIS_N_LAYERS &&
               header.d_ff == 2 * GENESIS_D_MODEL &&
               header.stagnation_count == 0 &&
               header.growth_delta == 0;
    }

    // Determine if loss improved
    bool loss_improved = header.val_loss < parent.val_loss;

    GrowthState state{};
    state.d_model = parent.d_model;
    state.n_layers = parent.n_layers;
    state.stagnation = parent.stagnation_count;
    state.last_loss = parent.val_loss;

    GrowthResult expected = compute_growth(state, loss_improved);

    // Verify all growth fields match exactly
    if (header.d_model != expected.new_d_model) return false;
    if (header.n_layers != expected.new_n_layers) return false;
    if (header.d_ff != expected.new_d_ff) return false;
    if (header.growth_delta != expected.delta_d_model) return false;
    if (header.stagnation_count != expected.new_stagnation) return false;

    return true;
}

GrowthResult GrowthPolicy::expected_growth(const primitives::CBlockHeader& parent) {
    GrowthState state{};
    state.d_model = parent.d_model;
    state.n_layers = parent.n_layers;
    state.stagnation = parent.stagnation_count;
    state.last_loss = parent.val_loss;

    // Default: assume loss improved (caller verifies separately)
    return compute_growth(state, true);
}

}  // namespace rnet::consensus

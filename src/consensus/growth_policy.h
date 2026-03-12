#pragma once

#include <cstdint>

#include "primitives/block_header.h"

namespace rnet::consensus {

/// Snapshot of growth-relevant state from the parent block.
struct GrowthState {
    uint32_t d_model;
    uint32_t n_layers;
    uint32_t stagnation;
    float    last_loss;
};

/// Result of a growth computation — what the next block's model config must be.
struct GrowthResult {
    uint32_t new_d_model;
    uint32_t new_n_layers;
    uint32_t new_d_ff;
    uint32_t delta_d_model;
    int      delta_n_layers;
    bool     layer_added;
    uint32_t new_stagnation;
};

/// CONSENSUS-CRITICAL: Deterministic growth algorithm.
/// Every node must compute identical results from the same inputs.
class GrowthPolicy {
public:
    static constexpr uint32_t BASE_GROWTH    = 2;
    static constexpr uint32_t PATIENCE       = 10;
    static constexpr uint32_t MAX_D_MODEL    = 4096;
    static constexpr uint32_t MAX_LAYERS     = 48;
    static constexpr uint32_t LAYER_THRESHOLD = 128;
    static constexpr uint32_t GENESIS_D_MODEL = 384;
    static constexpr uint32_t GENESIS_N_LAYERS = 6;

    /// Compute the expected growth from the given state.
    /// @param state   Parent block's model state.
    /// @param loss_improved  True if current val_loss < parent val_loss.
    static GrowthResult compute_growth(const GrowthState& state, bool loss_improved);

    /// Verify that a block header's growth fields match the expected
    /// values derived from its parent header.
    static bool verify_growth(const primitives::CBlockHeader& header,
                              const primitives::CBlockHeader& parent);

    /// Compute the expected growth result from a parent header,
    /// assuming loss improved (conservative check — caller should
    /// also verify the loss_improved flag separately).
    static GrowthResult expected_growth(const primitives::CBlockHeader& parent);
};

}  // namespace rnet::consensus

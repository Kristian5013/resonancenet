#include "consensus/proof_of_training.h"
#include "consensus/growth_policy.h"

namespace rnet::consensus {

bool verify_pot_header(const primitives::CBlockHeader& header,
                       const primitives::CBlockHeader& parent,
                       const ConsensusParams& params) {
    ValidationState state;
    return verify_pot_header(header, parent, params, state);
}

bool verify_pot_header(const primitives::CBlockHeader& header,
                       const primitives::CBlockHeader& parent,
                       const ConsensusParams& params,
                       ValidationState& state) {
    // Genesis block passes trivially
    if (header.is_genesis()) {
        return true;
    }

    // prev_val_loss must match parent's val_loss
    if (header.prev_val_loss != parent.val_loss) {
        state.invalid("pot-prev-loss-mismatch");
        return false;
    }

    // val_loss must be positive
    if (header.val_loss <= 0.0f) {
        state.invalid("pot-invalid-loss");
        return false;
    }

    // Check if loss improved (within tolerance)
    // Loss is considered improved if: val_loss < prev_val_loss * (1 + tolerance)
    // This allows for minor floating-point variations
    bool loss_improved = header.val_loss < header.prev_val_loss;

    // Even if loss didn't improve, the block is valid (stagnation is tracked).
    // But if loss INCREASED beyond tolerance, that's suspicious — still valid
    // but stagnation must be correctly tracked.
    float tolerance_bound = header.prev_val_loss * (1.0f + params.loss_verify_tolerance);
    if (header.val_loss > tolerance_bound) {
        // Loss regressed beyond tolerance — this is allowed but unusual.
        // Growth policy handles it via stagnation.
    }

    // Verify train_steps is in valid range
    if (header.train_steps < static_cast<uint32_t>(params.min_steps_per_block)) {
        state.invalid("pot-too-few-steps");
        return false;
    }
    if (header.train_steps > static_cast<uint32_t>(params.max_steps_per_block)) {
        state.invalid("pot-too-many-steps");
        return false;
    }

    // Verify growth fields match expected values
    if (!GrowthPolicy::verify_growth(header, parent)) {
        state.invalid("pot-growth-mismatch");
        return false;
    }

    return true;
}

bool verify_pot_training(const primitives::CBlockHeader& /*header*/,
                         const ConsensusParams& /*params*/) {
    // Stub: full GPU verification not yet implemented.
    // Header-level checks are performed by verify_pot_header().
    // When rnet_training is complete, this will:
    //   1. Load the parent checkpoint
    //   2. Replay train_steps of training
    //   3. Run eval_batches of evaluation
    //   4. Compare the resulting val_loss within tolerance
    return true;
}

}  // namespace rnet::consensus

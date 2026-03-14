// Copyright (c) 2024-2026 The ResonanceNet Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "consensus/proof_of_training.h"

#include "consensus/growth_policy.h"

#include <cmath>

namespace rnet::consensus {

// ---------------------------------------------------------------------------
// verify_pot_header  (convenience overload)
// ---------------------------------------------------------------------------
// Delegates to the full overload with a throw-away ValidationState.
// ---------------------------------------------------------------------------

bool verify_pot_header(const primitives::CBlockHeader& header,
                       const primitives::CBlockHeader& parent,
                       const ConsensusParams& params) {
    ValidationState state;
    return verify_pot_header(header, parent, params, state);
}

// ---------------------------------------------------------------------------
// verify_pot_header
// ---------------------------------------------------------------------------
// Validates Proof-of-Training header fields against parent block.
//
// Checks performed (in order):
//   0. Genesis passthrough
//   1. Loss chain continuity:  header.prev_val_loss == parent.val_loss
//   2. Finiteness:             isfinite(val_loss)
//   3. Positivity:             val_loss > 0
//   4. Upper bound:            val_loss < 1'000  (ln(50'257) ~ 10.82)
//   5. Regression bound:       val_loss <= 2 * prev_val_loss
//   6. Training step range:    min_steps <= steps <= max_steps
//   7. Checkpoint hash present
//   8. Dataset hash present
//   9. Growth policy fields match deterministic computation
// ---------------------------------------------------------------------------

bool verify_pot_header(const primitives::CBlockHeader& header,
                       const primitives::CBlockHeader& parent,
                       const ConsensusParams& params,
                       ValidationState& state) {

    // 0. Genesis passthrough.
    if (header.is_genesis()) {
        return true;
    }

    // 1. Loss chain continuity — the block's prev_val_loss must exactly
    //    match the parent's val_loss so the loss chain is unbroken.
    if (header.prev_val_loss != parent.val_loss) {
        state.invalid("pot-prev-loss-mismatch");
        return false;
    }

    // 2. Finiteness — NaN and Inf silently pass some float comparisons;
    //    reject early.
    if (!std::isfinite(header.val_loss)) {
        state.invalid("pot-loss-not-finite");
        return false;
    }

    // 3. Positivity.
    if (header.val_loss <= 0.0f) {
        state.invalid("pot-invalid-loss");
        return false;
    }

    // 4. Upper bound — cross-entropy loss for a 50k-token vocabulary tops
    //    out at ln(50257) ~ 10.82.  Use 1000.0 as a generous ceiling;
    //    anything above is clearly invalid or fabricated.
    if (header.val_loss >= 1000.0f) {
        state.invalid("pot-loss-out-of-range");
        return false;
    }

    // 5. Regression bound — a block may report higher loss than its parent
    //    (stagnation), but more than 2x regression is implausible and could
    //    manipulate stagnation counters or growth triggers.
    if (header.val_loss > 2.0f * header.prev_val_loss) {
        state.invalid("pot-loss-regression");
        return false;
    }

    // 6. Training-step range.
    if (header.train_steps < static_cast<uint32_t>(params.min_steps_per_block)) {
        state.invalid("pot-too-few-steps");
        return false;
    }
    if (header.train_steps > static_cast<uint32_t>(params.max_steps_per_block)) {
        state.invalid("pot-too-many-steps");
        return false;
    }

    // 7. Checkpoint hash must be present — every non-genesis block must
    //    reference the model checkpoint produced by this block's training.
    if (header.checkpoint_hash.is_zero()) {
        state.invalid("pot-missing-checkpoint");
        return false;
    }

    // 8. Dataset hash must be present — pins the training data so every
    //    full node can replay and verify the claimed val_loss.
    if (header.dataset_hash.is_zero()) {
        state.invalid("pot-missing-dataset");
        return false;
    }

    // 9. Growth-policy fields — verify model_size, stagnation_counter, and
    //    any growth transitions are consistent with the parent block.
    if (!GrowthPolicy::verify_growth(header, parent)) {
        state.invalid("pot-growth-mismatch");
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// verify_pot_training
// ---------------------------------------------------------------------------
// Full GPU-based training replay verification (stub).
//
// Verification procedure:
//   1. Load parent checkpoint from header.prev_checkpoint_hash
//   2. Replay header.train_steps fwd/bwd passes on the consensus dataset
//      (header.dataset_hash) using deterministic OpenCL kernels
//   3. Evaluate on the validation split to compute replayed val_loss
//   4. Compare replayed val_loss against header.val_loss within
//      params.loss_verify_tolerance
//
// Until rnet_training's OpenCL backend is complete, header-level checks in
// verify_pot_header() are the only consensus gate.
// ---------------------------------------------------------------------------

bool verify_pot_training(const primitives::CBlockHeader& /*header*/,
                         const ConsensusParams& /*params*/) {
    // TODO(pot): Implement full GPU-based training replay verification.
    return true;
}

} // namespace rnet::consensus

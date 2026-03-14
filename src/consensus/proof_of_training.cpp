// ============================================================================
//  proof_of_training.cpp — Proof-of-Training consensus validation
// ============================================================================
//  Validates PoT block headers: loss continuity, range checks, regression
//  bounds, checkpoint/dataset presence, growth policy, and training-step
//  limits.  Full GPU replay verification is stubbed pending rnet_training.
// ============================================================================

#include "consensus/proof_of_training.h"
#include "consensus/growth_policy.h"

#include <cmath>

namespace rnet::consensus {

// ── convenience overload (creates a throw-away ValidationState) ─────────────

bool verify_pot_header(const primitives::CBlockHeader& header,
                       const primitives::CBlockHeader& parent,
                       const ConsensusParams& params) {
    ValidationState state;
    return verify_pot_header(header, parent, params, state);
}

// ── full header-level PoT validation ────────────────────────────────────────

bool verify_pot_header(const primitives::CBlockHeader& header,
                       const primitives::CBlockHeader& parent,
                       const ConsensusParams& params,
                       ValidationState& state) {

    // ── step 0: genesis passthrough ─────────────────────────────────────
    if (header.is_genesis()) {
        return true;
    }

    // ── step 1: loss chain continuity ───────────────────────────────────
    //  The block's prev_val_loss must exactly match the parent's val_loss
    //  so that the loss chain is an unbroken sequence.
    if (header.prev_val_loss != parent.val_loss) {
        state.invalid("pot-prev-loss-mismatch");
        return false;
    }

    // ── step 2: val_loss finiteness ─────────────────────────────────────
    //  NaN and Inf silently pass some float comparisons; reject early.
    if (!std::isfinite(header.val_loss)) {
        state.invalid("pot-loss-not-finite");
        return false;
    }

    // ── step 3: val_loss positivity ─────────────────────────────────────
    if (header.val_loss <= 0.0f) {
        state.invalid("pot-invalid-loss");
        return false;
    }

    // ── step 4: val_loss upper bound ────────────────────────────────────
    //  Cross-entropy loss for a 50k-token vocabulary tops out at
    //  ln(50257) ~ 10.82.  Use 1000.0 as a generous ceiling — anything
    //  above is clearly invalid or fabricated.
    if (header.val_loss >= 1000.0f) {
        state.invalid("pot-loss-out-of-range");
        return false;
    }

    // ── step 5: val_loss regression bound ───────────────────────────────
    //  A block may report higher loss than its parent (stagnation), but
    //  more than 2x regression is implausible and could be used to
    //  manipulate stagnation counters or growth triggers.
    if (header.val_loss > 2.0f * header.prev_val_loss) {
        state.invalid("pot-loss-regression");
        return false;
    }

    // ── step 6: training-step range ─────────────────────────────────────
    if (header.train_steps < static_cast<uint32_t>(params.min_steps_per_block)) {
        state.invalid("pot-too-few-steps");
        return false;
    }
    if (header.train_steps > static_cast<uint32_t>(params.max_steps_per_block)) {
        state.invalid("pot-too-many-steps");
        return false;
    }

    // ── step 7: checkpoint hash must be present ─────────────────────────
    //  Every non-genesis block must reference the model checkpoint that
    //  resulted from this block's training work.
    if (header.checkpoint_hash.is_zero()) {
        state.invalid("pot-missing-checkpoint");
        return false;
    }

    // ── step 8: dataset hash must be present ────────────────────────────
    //  The consensus dataset hash pins the training data so that every
    //  full node can replay and verify the claimed val_loss.
    if (header.dataset_hash.is_zero()) {
        state.invalid("pot-missing-dataset");
        return false;
    }

    // ── step 9: growth-policy fields ────────────────────────────────────
    //  Verify that model_size, stagnation_counter, and any growth
    //  transitions are consistent with the parent block.
    if (!GrowthPolicy::verify_growth(header, parent)) {
        state.invalid("pot-growth-mismatch");
        return false;
    }

    return true;
}

// ── full training replay verification (stub) ────────────────────────────────

bool verify_pot_training(const primitives::CBlockHeader& /*header*/,
                         const ConsensusParams& /*params*/) {
    // TODO(pot): Implement full GPU-based training replay verification.
    //
    // The verification procedure has four steps:
    //
    //   1. Load parent checkpoint — Deserialise the model state referenced
    //      by header.prev_checkpoint_hash (the parent block's checkpoint).
    //      This gives the exact weight tensor set the miner started from.
    //
    //   2. Replay training — Execute header.train_steps forward/backward
    //      passes on the consensus dataset (identified by header.dataset_hash)
    //      using the deterministic training kernel (fixed hyperparameters,
    //      fixed batch order derived from block hash).  The OpenCL kernels
    //      in rnet_training must produce bit-identical results across all
    //      compliant implementations.
    //
    //   3. Evaluate on validation set — Run eval_batches of forward-only
    //      inference on the held-out validation split of the consensus
    //      dataset to compute the resulting validation loss.
    //
    //   4. Compare val_loss — The replayed val_loss must match
    //      header.val_loss within params.loss_verify_tolerance.  If it
    //      falls outside that window the block is invalid.
    //
    // Until rnet_training's OpenCL backend is complete, header-level
    // checks in verify_pot_header() are the only consensus gate.

    return true;
}

}  // namespace rnet::consensus

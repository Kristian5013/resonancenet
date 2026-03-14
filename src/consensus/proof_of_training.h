#pragma once

#include "consensus/params.h"
#include "consensus/validation.h"
#include "primitives/block_header.h"

namespace rnet::consensus {

/// Verify the Proof-of-Training fields in a block header.
/// Checks:
///   - val_loss < prev_val_loss (within tolerance), OR stagnation tracked
///   - train_steps in valid range
///   - Growth fields match expected values from parent
///   - prev_val_loss matches parent's val_loss
/// @return true if the PoT header fields are valid.
bool verify_pot_header(const primitives::CBlockHeader& header,
                       const primitives::CBlockHeader& parent,
                       const ConsensusParams& params);

/// Verify the PoT header with a ValidationState for error reporting.
bool verify_pot_header(const primitives::CBlockHeader& header,
                       const primitives::CBlockHeader& parent,
                       const ConsensusParams& params,
                       ValidationState& state);

/// Stub: full GPU-based training verification.
/// Will be implemented when the rnet_training module is complete.
/// For now, returns true (header-only verification is sufficient).
bool verify_pot_training(const primitives::CBlockHeader& header,
                         const ConsensusParams& params);

// ---------------------------------------------------------------------------
// Adaptive difficulty retargeting (consensus-critical)
// ---------------------------------------------------------------------------

/// Check whether @p height is a difficulty retarget boundary.
bool is_retarget_height(uint64_t height, const ConsensusParams& params);

/// Compute the required difficulty_delta for the next block.
///
/// Every `difficulty_adjustment_interval` blocks, the minimum loss
/// improvement is adjusted so that blocks arrive approximately every
/// `target_block_time` seconds (default 600 s = 10 minutes):
///
///   ratio     = actual_elapsed / expected_elapsed
///   new_delta = prev_delta * ratio
///   clamped   to [prev_delta/4, prev_delta*4]  then [min, max]
///
/// Between retarget boundaries, delta carries forward unchanged.
///
/// @param height           Height of the block being computed.
/// @param parent_delta     Parent block's difficulty_delta.
/// @param period_start_ts  Timestamp at the start of the retarget period.
/// @param parent_ts        Timestamp of the parent block.
/// @param params           Consensus parameters.
float compute_next_difficulty(uint64_t height,
                              float parent_delta,
                              uint64_t period_start_ts,
                              uint64_t parent_ts,
                              const ConsensusParams& params);

}  // namespace rnet::consensus

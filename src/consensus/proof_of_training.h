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

}  // namespace rnet::consensus

#pragma once

#include "consensus/params.h"
#include "consensus/validation.h"
#include "primitives/block.h"
#include "primitives/block_header.h"

namespace rnet::consensus {

/// Context-free block validation.
/// Checks:
///   - Merkle root matches transactions
///   - At least one transaction
///   - First transaction is coinbase
///   - Only first transaction is coinbase
///   - No duplicate txids
///   - Block size within limits
///   - Sigop count within limits
///   - All transactions pass check_transaction
bool check_block(const primitives::CBlock& block,
                 ValidationState& state,
                 const ConsensusParams& params);

/// Block header validation against its parent.
/// Checks:
///   - Timestamp is greater than parent's timestamp
///   - Height is parent height + 1
///   - prev_hash matches parent's hash
///   - PoT fields are valid (via verify_pot_header)
///   - Growth fields are valid
bool check_block_header(const primitives::CBlockHeader& header,
                        const primitives::CBlockHeader& parent,
                        ValidationState& state,
                        const ConsensusParams& params);

}  // namespace rnet::consensus

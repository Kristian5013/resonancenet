#pragma once

#include "core/types.h"
#include "primitives/block.h"

namespace rnet::consensus {

/// Compute the merkle root from a block's transaction txids.
/// Returns a zero hash if the block has no transactions.
rnet::uint256 block_merkle_root(const primitives::CBlock& block);

/// Compute the witness merkle root from a block's transaction wtxids.
/// The coinbase wtxid is replaced with a zero hash (per segwit rules).
/// Returns a zero hash if the block has no transactions.
rnet::uint256 block_witness_merkle_root(const primitives::CBlock& block);

}  // namespace rnet::consensus

#pragma once

#include "consensus/params.h"
#include "primitives/block.h"

namespace rnet::consensus {

/// Create the genesis block for the given network.
/// The genesis block has:
///   - height = 0, prev_hash = 0
///   - d_model = 384, n_layers = 6, d_ff = 768
///   - val_loss = 10.0 (large initial value)
///   - A coinbase transaction with the genesis message
///   - Merkle root computed from the coinbase
primitives::CBlock create_genesis_block(const ConsensusParams& params);

/// The genesis coinbase message embedded in the scriptSig.
inline constexpr const char* GENESIS_MESSAGE =
    "OpenAI burns $14B in 2026, adds ads to ChatGPT as last resort - ResonanceNet";

}  // namespace rnet::consensus

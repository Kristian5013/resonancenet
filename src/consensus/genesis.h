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

/// Genesis miner Ed25519 public key (32 bytes).
/// The corresponding private key is held offline by the project founders.
/// Genesis coins are unspendable (all-zero script key), but this pubkey
/// anchors the identity of the genesis block for peer verification.
inline constexpr uint8_t GENESIS_PUBKEY[32] = {
    0xf3, 0x0f, 0x4e, 0xce, 0x79, 0x50, 0xf4, 0xf2,
    0x1f, 0x69, 0x21, 0xdf, 0xfe, 0x9b, 0x05, 0x4f,
    0xb7, 0x0c, 0xab, 0xb1, 0x4e, 0x1c, 0x88, 0xd2,
    0x1f, 0x41, 0x03, 0x1e, 0x1a, 0xd8, 0xed, 0xff
};

}  // namespace rnet::consensus

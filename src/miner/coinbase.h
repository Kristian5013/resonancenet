#pragma once

#include <cstdint>
#include <vector>

#include "crypto/ed25519.h"
#include "primitives/transaction.h"

namespace rnet::miner {

/// Create a coinbase scriptPubKey for Ed25519 mining reward.
/// Format: [0x20][32-byte Ed25519 pubkey][0xAC]
/// This matches crypto::ed25519_coinbase_script() but is the miner-side API.
std::vector<uint8_t> make_coinbase_script(const crypto::Ed25519PublicKey& pubkey);

/// Create a coinbase transaction for a new block.
///
/// @param height      Block height (encoded in scriptSig per BIP34 convention).
/// @param reward      Total block reward in resonances (base + bonus + recovered).
/// @param pubkey      Miner's Ed25519 public key for the reward output.
/// @param extra_nonce Optional extra nonce bytes for the scriptSig.
/// @return            Immutable coinbase transaction.
primitives::CTransactionRef create_coinbase_tx(
    uint64_t height,
    int64_t reward,
    const crypto::Ed25519PublicKey& pubkey,
    const std::vector<uint8_t>& extra_nonce = {});

/// Encode block height into coinbase scriptSig (BIP34-style).
/// Encodes as minimal push: [push_len][height LE bytes].
std::vector<uint8_t> encode_height_script(uint64_t height);

}  // namespace rnet::miner

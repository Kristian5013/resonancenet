// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

// Own header.
#include "miner/coinbase.h"

// Project headers.
#include "crypto/ed25519.h"
#include "primitives/outpoint.h"
#include "primitives/txin.h"
#include "primitives/txout.h"

// Standard library.
#include <cstring>

namespace rnet::miner {

// ---------------------------------------------------------------------------
// make_coinbase_script
// ---------------------------------------------------------------------------
// Builds the coinbase output script: [0x20][32-byte Ed25519 pubkey][0xAC].
// ---------------------------------------------------------------------------
std::vector<uint8_t> make_coinbase_script(const crypto::Ed25519PublicKey& pubkey)
{
    return crypto::ed25519_coinbase_script(pubkey);
}

// ---------------------------------------------------------------------------
// encode_height_script
// ---------------------------------------------------------------------------
// BIP34-style height encoding: encodes the block height as minimal
// little-endian bytes with a length prefix.  Appends a 0x00 byte if the
// high bit is set to prevent sign-magnitude misinterpretation.
// ---------------------------------------------------------------------------
std::vector<uint8_t> encode_height_script(uint64_t height)
{
    std::vector<uint8_t> script;

    if (height == 0) {
        // 1. Push OP_0 (0x00).
        script.push_back(0x00);
        return script;
    }

    // 2. Encode height as minimal little-endian bytes.
    std::vector<uint8_t> height_bytes;
    uint64_t h = height;
    while (h > 0) {
        height_bytes.push_back(static_cast<uint8_t>(h & 0xFF));
        h >>= 8;
    }

    // 3. Append 0x00 if the high bit of the last byte is set.
    if (height_bytes.back() & 0x80) {
        height_bytes.push_back(0x00);
    }

    // 4. Push the length prefix then the height bytes.
    script.push_back(static_cast<uint8_t>(height_bytes.size()));
    script.insert(script.end(), height_bytes.begin(), height_bytes.end());

    return script;
}

// ---------------------------------------------------------------------------
// create_coinbase_tx
// ---------------------------------------------------------------------------
// Constructs a complete coinbase transaction with a null outpoint,
// height-encoded scriptSig (optionally with extra nonce), and a single
// output paying the block reward to the miner's Ed25519 public key.
// ---------------------------------------------------------------------------
primitives::CTransactionRef create_coinbase_tx(
    uint64_t height,
    int64_t reward,
    const crypto::Ed25519PublicKey& pubkey,
    const std::vector<uint8_t>& extra_nonce)
{
    primitives::CMutableTransaction mtx;
    mtx.version = primitives::TX_VERSION_DEFAULT;

    // 1. Coinbase input: null outpoint + height-encoded scriptSig.
    primitives::COutPoint coinbase_outpoint;
    coinbase_outpoint.set_null();

    auto script_sig = encode_height_script(height);
    if (!extra_nonce.empty()) {
        script_sig.insert(script_sig.end(), extra_nonce.begin(), extra_nonce.end());
    }

    mtx.vin.emplace_back(coinbase_outpoint, std::move(script_sig));

    // 2. Coinbase output: reward to miner's Ed25519 pubkey script.
    auto coinbase_script = make_coinbase_script(pubkey);
    mtx.vout.emplace_back(reward, std::move(coinbase_script));

    return primitives::MakeTransactionRef(std::move(mtx));
}

} // namespace rnet::miner

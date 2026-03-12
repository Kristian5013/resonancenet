#include "miner/coinbase.h"

#include <cstring>

#include "crypto/ed25519.h"
#include "primitives/outpoint.h"
#include "primitives/txin.h"
#include "primitives/txout.h"

namespace rnet::miner {

std::vector<uint8_t> make_coinbase_script(const crypto::Ed25519PublicKey& pubkey) {
    return crypto::ed25519_coinbase_script(pubkey);
}

std::vector<uint8_t> encode_height_script(uint64_t height) {
    std::vector<uint8_t> script;

    if (height == 0) {
        // Push OP_0 (0x00)
        script.push_back(0x00);
        return script;
    }

    // Encode height as minimal little-endian bytes
    std::vector<uint8_t> height_bytes;
    uint64_t h = height;
    while (h > 0) {
        height_bytes.push_back(static_cast<uint8_t>(h & 0xFF));
        h >>= 8;
    }

    // If the high bit of the last byte is set, append a 0x00 byte
    // to prevent interpretation as negative (sign-magnitude encoding).
    if (height_bytes.back() & 0x80) {
        height_bytes.push_back(0x00);
    }

    // Push the length prefix then the height bytes
    script.push_back(static_cast<uint8_t>(height_bytes.size()));
    script.insert(script.end(), height_bytes.begin(), height_bytes.end());

    return script;
}

primitives::CTransactionRef create_coinbase_tx(
    uint64_t height,
    int64_t reward,
    const crypto::Ed25519PublicKey& pubkey,
    const std::vector<uint8_t>& extra_nonce) {

    primitives::CMutableTransaction mtx;
    mtx.version = primitives::TX_VERSION_DEFAULT;

    // Coinbase input: null outpoint + height-encoded scriptSig
    primitives::COutPoint coinbase_outpoint;
    coinbase_outpoint.set_null();

    auto script_sig = encode_height_script(height);
    if (!extra_nonce.empty()) {
        script_sig.insert(script_sig.end(), extra_nonce.begin(), extra_nonce.end());
    }

    mtx.vin.emplace_back(coinbase_outpoint, std::move(script_sig));

    // Coinbase output: reward to miner's Ed25519 pubkey script
    auto coinbase_script = make_coinbase_script(pubkey);
    mtx.vout.emplace_back(reward, std::move(coinbase_script));

    return primitives::MakeTransactionRef(std::move(mtx));
}

}  // namespace rnet::miner

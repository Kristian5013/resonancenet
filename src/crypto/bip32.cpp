// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "crypto/bip32.h"

#include "crypto/ed25519.h"
#include "crypto/hash.h"
#include "crypto/keccak.h"
#include "core/base58.h"

#include <cstring>
#include <sstream>

// ---------------------------------------------------------------------------
//  BIP-32 Hierarchical Deterministic Key Derivation (Ed25519 variant)
//
//  ResonanceNet derivation path:
//
//      m / 44' / 9555' / account' / change / index
//      |    |      |        |         |        |
//      |    |      |        |         |        +-- address sequence
//      |    |      |        |         +-- 0=external, 1=internal (change)
//      |    |      |        +-- account number (hardened)
//      |    |      +-- coin type 9555 = RNET (hardened)
//      |    +-- BIP-44 purpose (hardened)
//      +-- master key
//
//  Key derivation (SLIP-0010 / Ed25519 adaptation):
//
//      seed ---HMAC-SHA512("Bitcoin seed")--> (I_L || I_R)
//                                              |       |
//                                        master key  chain code
//
//  Child derivation:
//
//      Hardened (i >= 0x80000000):
//          I = HMAC-SHA512(chain_code, 0x00 || key || ser32(i))
//
//      Normal (i < 0x80000000):
//          I = HMAC-SHA512(chain_code, 0x00 || pubkey || ser32(i))
//
//      child_key  = I_L  XOR  parent_key   (key stretching)
//      child_cc   = I_R
//      fingerprint = first 4 bytes of Keccak256d(parent_pubkey)
// ---------------------------------------------------------------------------

namespace rnet::crypto {

// ---------------------------------------------------------------------------
// master_key_from_seed
// ---------------------------------------------------------------------------

rnet::Result<ExtKey> master_key_from_seed(
    std::span<const uint8_t> seed) {

    // 1. Validate seed length (BIP-32 requires 16-64 bytes).
    if (seed.size() < 16 || seed.size() > 64) {
        return rnet::Result<ExtKey>::err(
            "Seed must be 16-64 bytes");
    }

    // 2. Compute I = HMAC-SHA512(key="Bitcoin seed", data=seed).
    const uint8_t hmac_key[] = "Bitcoin seed";
    auto I = hmac_sha512(
        std::span<const uint8_t>(hmac_key, 12),
        seed);

    // 3. Split I into master secret (I_L) and chain code (I_R).
    ExtKey master;
    std::memcpy(master.key.data(), I.data(), 32);
    std::memcpy(master.chaincode.data(), I.data() + 32, 32);
    master.depth = 0;
    master.fingerprint = 0;
    master.child_num = 0;
    master.is_private = true;

    return rnet::Result<ExtKey>::ok(master);
}

// ---------------------------------------------------------------------------
// ExtKey::derive_child
// ---------------------------------------------------------------------------

rnet::Result<ExtKey> ExtKey::derive_child(uint32_t index) const {

    // 1. Only private-key derivation is supported.
    if (!is_private) {
        return rnet::Result<ExtKey>::err(
            "Cannot derive child from public key (not implemented)");
    }

    // 2. Build HMAC data depending on hardened vs normal derivation.
    //
    //    hardened (index >= 0x80000000):
    //        data = 0x00 || private_key || ser32(index)     [37 bytes]
    //
    //    normal  (index <  0x80000000):
    //        data = 0x00 || public_key  || ser32(index)     [37 bytes]
    //
    std::vector<uint8_t> data;
    if (index >= HARDENED) {
        // 2a. Hardened child -- use private key.
        data.reserve(37);
        data.push_back(0x00);
        data.insert(data.end(), key.begin(), key.end());
    } else {
        // 2b. Normal child -- use public key.
        auto pk_result = get_pubkey();
        if (pk_result.is_err()) {
            return rnet::Result<ExtKey>::err(pk_result.error());
        }
        data.reserve(37);
        data.push_back(0x00);
        data.insert(data.end(),
            pk_result.value().begin(), pk_result.value().end());
    }

    // 3. Append index as big-endian uint32.
    data.push_back(static_cast<uint8_t>((index >> 24) & 0xFF));
    data.push_back(static_cast<uint8_t>((index >> 16) & 0xFF));
    data.push_back(static_cast<uint8_t>((index >> 8) & 0xFF));
    data.push_back(static_cast<uint8_t>(index & 0xFF));

    // 4. I = HMAC-SHA512(chain_code, data).
    auto I = hmac_sha512(chaincode, data);

    // 5. Derive child key = I_L XOR parent_key (key stretching).
    ExtKey child;
    std::memcpy(child.key.data(), I.data(), 32);

    for (int i = 0; i < 32; ++i) {
        child.key[static_cast<size_t>(i)] ^=
            key[static_cast<size_t>(i)];
    }

    // 6. Child chain code = I_R.
    std::memcpy(child.chaincode.data(), I.data() + 32, 32);
    child.depth = depth + 1;
    child.child_num = index;
    child.is_private = true;

    // 7. Fingerprint = first 4 bytes of Keccak256d(parent_pubkey).
    auto pk = get_pubkey();
    if (pk.is_ok()) {
        auto hash = keccak256d(pk.value());
        std::memcpy(&child.fingerprint, hash.data(), 4);
    }

    return rnet::Result<ExtKey>::ok(child);
}

// ---------------------------------------------------------------------------
// ExtKey::derive_path
// ---------------------------------------------------------------------------

rnet::Result<ExtKey> ExtKey::derive_path(std::string_view path) const {

    // 1. Reject empty paths.
    if (path.empty()) {
        return rnet::Result<ExtKey>::err("Empty derivation path");
    }

    // 2. Skip the leading "m/" or "M/" prefix.
    size_t pos = 0;
    if (path[0] == 'm' || path[0] == 'M') {
        pos = 1;
        if (pos < path.size() && path[pos] == '/') pos++;
    }

    // 3. Walk each slash-delimited segment and derive iteratively.
    ExtKey current = *this;

    while (pos < path.size()) {
        auto slash = path.find('/', pos);
        auto segment = (slash == std::string_view::npos)
            ? path.substr(pos)
            : path.substr(pos, slash - pos);
        pos = (slash == std::string_view::npos)
            ? path.size()
            : slash + 1;

        if (segment.empty()) continue;

        // 4. Detect hardened marker (' or h or H suffix).
        bool hardened = false;
        if (segment.back() == '\'' || segment.back() == 'h' ||
            segment.back() == 'H') {
            hardened = true;
            segment = segment.substr(0, segment.size() - 1);
        }

        // 5. Parse the numeric index.
        uint32_t idx = 0;
        for (char c : segment) {
            if (c < '0' || c > '9') {
                return rnet::Result<ExtKey>::err(
                    "Invalid path component: " + std::string(segment));
            }
            idx = idx * 10 + static_cast<uint32_t>(c - '0');
        }

        if (hardened) idx += HARDENED;

        // 6. Derive child at this level.
        auto child = current.derive_child(idx);
        if (child.is_err()) return child;
        current = child.value();
    }

    return rnet::Result<ExtKey>::ok(current);
}

// ---------------------------------------------------------------------------
// ExtKey::get_pubkey
// ---------------------------------------------------------------------------

rnet::Result<std::array<uint8_t, 32>> ExtKey::get_pubkey() const {

    // 1. If already a public key, return directly.
    if (!is_private) {
        return rnet::Result<std::array<uint8_t, 32>>::ok(key);
    }

    // 2. Generate Ed25519 public key from private key seed.
    auto result = ed25519_from_seed(key);
    if (result.is_err()) {
        return rnet::Result<std::array<uint8_t, 32>>::err(
            result.error());
    }
    return rnet::Result<std::array<uint8_t, 32>>::ok(
        result.value().public_key.data);
}

// ---------------------------------------------------------------------------
// ExtKey::get_pubkey_hash
// ---------------------------------------------------------------------------

rnet::Result<rnet::uint160> ExtKey::get_pubkey_hash() const {

    // 1. Obtain the public key.
    auto pk = get_pubkey();
    if (pk.is_err()) {
        return rnet::Result<rnet::uint160>::err(pk.error());
    }

    // 2. Hash160 = first 20 bytes of Keccak256d(pubkey).
    auto hash = keccak256d(pk.value());
    rnet::uint160 result;
    std::memcpy(result.data(), hash.data(), 20);
    return rnet::Result<rnet::uint160>::ok(result);
}

// ---------------------------------------------------------------------------
// ExtKey::to_base58
// ---------------------------------------------------------------------------

std::string ExtKey::to_base58() const {

    // Serialisation layout (78 bytes):
    //   [0..3]   version     4 bytes   (xprv=0x0488ADE4, xpub=0x0488B21E)
    //   [4]      depth       1 byte
    //   [5..8]   fingerprint 4 bytes
    //   [9..12]  child_num   4 bytes
    //   [13..44] chain code  32 bytes
    //   [45]     0x00 prefix 1 byte
    //   [46..77] key         32 bytes

    std::vector<uint8_t> data;
    data.reserve(78);

    // 1. Version bytes (4).
    uint32_t version = is_private ? 0x0488ADE4 : 0x0488B21E;
    data.push_back(static_cast<uint8_t>((version >> 24) & 0xFF));
    data.push_back(static_cast<uint8_t>((version >> 16) & 0xFF));
    data.push_back(static_cast<uint8_t>((version >> 8) & 0xFF));
    data.push_back(static_cast<uint8_t>(version & 0xFF));

    // 2. Depth (1).
    data.push_back(depth);

    // 3. Parent fingerprint (4).
    data.push_back(static_cast<uint8_t>((fingerprint >> 24) & 0xFF));
    data.push_back(static_cast<uint8_t>((fingerprint >> 16) & 0xFF));
    data.push_back(static_cast<uint8_t>((fingerprint >> 8) & 0xFF));
    data.push_back(static_cast<uint8_t>(fingerprint & 0xFF));

    // 4. Child number (4).
    data.push_back(static_cast<uint8_t>((child_num >> 24) & 0xFF));
    data.push_back(static_cast<uint8_t>((child_num >> 16) & 0xFF));
    data.push_back(static_cast<uint8_t>((child_num >> 8) & 0xFF));
    data.push_back(static_cast<uint8_t>(child_num & 0xFF));

    // 5. Chain code (32).
    data.insert(data.end(), chaincode.begin(), chaincode.end());

    // 6. Key with 0x00 prefix (1 + 32).
    if (is_private) {
        data.push_back(0x00);
        data.insert(data.end(), key.begin(), key.end());
    } else {
        data.push_back(0x00);
        data.insert(data.end(), key.begin(), key.end());
    }

    return rnet::core::base58check_encode_simple(data);
}

// ---------------------------------------------------------------------------
// ExtKey::from_base58
// ---------------------------------------------------------------------------

rnet::Result<ExtKey> ExtKey::from_base58(std::string_view str) {

    // 1. Base58Check-decode and verify length.
    auto decoded = rnet::core::base58check_decode_simple(str);
    if (!decoded.has_value() || decoded->size() != 78) {
        return rnet::Result<ExtKey>::err("Invalid base58 extended key");
    }

    const auto& data = *decoded;
    ExtKey k;

    // 2. Parse version (bytes 0..3).
    uint32_t version = (static_cast<uint32_t>(data[0]) << 24)
                     | (static_cast<uint32_t>(data[1]) << 16)
                     | (static_cast<uint32_t>(data[2]) << 8)
                     | static_cast<uint32_t>(data[3]);

    k.is_private = (version == 0x0488ADE4);

    // 3. Parse depth (byte 4).
    k.depth = data[4];

    // 4. Parse parent fingerprint (bytes 5..8).
    k.fingerprint = (static_cast<uint32_t>(data[5]) << 24)
                  | (static_cast<uint32_t>(data[6]) << 16)
                  | (static_cast<uint32_t>(data[7]) << 8)
                  | static_cast<uint32_t>(data[8]);

    // 5. Parse child number (bytes 9..12).
    k.child_num = (static_cast<uint32_t>(data[9]) << 24)
                | (static_cast<uint32_t>(data[10]) << 16)
                | (static_cast<uint32_t>(data[11]) << 8)
                | static_cast<uint32_t>(data[12]);

    // 6. Chain code (bytes 13..44).
    std::memcpy(k.chaincode.data(), data.data() + 13, 32);

    // 7. Key (bytes 46..77, skipping 0x00 prefix at byte 45).
    std::memcpy(k.key.data(), data.data() + 46, 32);

    return rnet::Result<ExtKey>::ok(k);
}

// ---------------------------------------------------------------------------
// derive_rnet_key
// ---------------------------------------------------------------------------

rnet::Result<ExtKey> derive_rnet_key(
    const ExtKey& master,
    uint32_t account,
    uint32_t change,
    uint32_t index) {

    // Derive along the ResonanceNet BIP-44 path:
    //     m / 44' / 9555' / account' / change / index
    std::string path = "m/44'/9555'/"
        + std::to_string(account) + "'/"
        + std::to_string(change) + "/"
        + std::to_string(index);

    return master.derive_path(path);
}

} // namespace rnet::crypto

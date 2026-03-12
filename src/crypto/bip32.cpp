#include "crypto/bip32.h"
#include "crypto/ed25519.h"
#include "crypto/hash.h"
#include "crypto/keccak.h"
#include "core/base58.h"

#include <cstring>
#include <sstream>

namespace rnet::crypto {

// ─── Master key from seed ──────────────────────────────────────────

rnet::Result<ExtKey> master_key_from_seed(
    std::span<const uint8_t> seed) {

    if (seed.size() < 16 || seed.size() > 64) {
        return rnet::Result<ExtKey>::err(
            "Seed must be 16-64 bytes");
    }

    // HMAC-SHA512 with key "Bitcoin seed"
    const uint8_t hmac_key[] = "Bitcoin seed";
    auto I = hmac_sha512(
        std::span<const uint8_t>(hmac_key, 12),
        seed);

    ExtKey master;
    std::memcpy(master.key.data(), I.data(), 32);
    std::memcpy(master.chaincode.data(), I.data() + 32, 32);
    master.depth = 0;
    master.fingerprint = 0;
    master.child_num = 0;
    master.is_private = true;

    return rnet::Result<ExtKey>::ok(master);
}

// ─── Child derivation ──────────────────────────────────────────────

rnet::Result<ExtKey> ExtKey::derive_child(uint32_t index) const {
    if (!is_private) {
        return rnet::Result<ExtKey>::err(
            "Cannot derive child from public key (not implemented)");
    }

    std::vector<uint8_t> data;
    if (index >= HARDENED) {
        // Hardened: 0x00 || private_key || index
        data.reserve(37);
        data.push_back(0x00);
        data.insert(data.end(), key.begin(), key.end());
    } else {
        // Normal: public_key || index
        // Get compressed public key (33 bytes for secp256k1-style)
        // For Ed25519, use 0x00 + pubkey as the serialization
        auto pk_result = get_pubkey();
        if (pk_result.is_err()) {
            return rnet::Result<ExtKey>::err(pk_result.error());
        }
        data.reserve(37);
        data.push_back(0x00);
        data.insert(data.end(),
            pk_result.value().begin(), pk_result.value().end());
    }

    // Append index as big-endian 4 bytes
    data.push_back(static_cast<uint8_t>((index >> 24) & 0xFF));
    data.push_back(static_cast<uint8_t>((index >> 16) & 0xFF));
    data.push_back(static_cast<uint8_t>((index >> 8) & 0xFF));
    data.push_back(static_cast<uint8_t>(index & 0xFF));

    auto I = hmac_sha512(chaincode, data);

    ExtKey child;
    // For Ed25519, the child key is simply I_L
    // (no modular arithmetic since Ed25519 handles that internally)
    std::memcpy(child.key.data(), I.data(), 32);

    // XOR parent key into child for key stretching
    for (int i = 0; i < 32; ++i) {
        child.key[static_cast<size_t>(i)] ^=
            key[static_cast<size_t>(i)];
    }

    std::memcpy(child.chaincode.data(), I.data() + 32, 32);
    child.depth = depth + 1;
    child.child_num = index;
    child.is_private = true;

    // Fingerprint: first 4 bytes of Hash160(parent pubkey)
    auto pk = get_pubkey();
    if (pk.is_ok()) {
        auto hash = keccak256d(pk.value());
        std::memcpy(&child.fingerprint, hash.data(), 4);
    }

    return rnet::Result<ExtKey>::ok(child);
}

rnet::Result<ExtKey> ExtKey::derive_path(std::string_view path) const {
    // Parse path like "m/44'/9555'/0'/0/0"
    if (path.empty()) {
        return rnet::Result<ExtKey>::err("Empty derivation path");
    }

    size_t pos = 0;
    if (path[0] == 'm' || path[0] == 'M') {
        pos = 1;
        if (pos < path.size() && path[pos] == '/') pos++;
    }

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

        bool hardened = false;
        if (segment.back() == '\'' || segment.back() == 'h' ||
            segment.back() == 'H') {
            hardened = true;
            segment = segment.substr(0, segment.size() - 1);
        }

        uint32_t idx = 0;
        for (char c : segment) {
            if (c < '0' || c > '9') {
                return rnet::Result<ExtKey>::err(
                    "Invalid path component: " + std::string(segment));
            }
            idx = idx * 10 + static_cast<uint32_t>(c - '0');
        }

        if (hardened) idx += HARDENED;

        auto child = current.derive_child(idx);
        if (child.is_err()) return child;
        current = child.value();
    }

    return rnet::Result<ExtKey>::ok(current);
}

rnet::Result<std::array<uint8_t, 32>> ExtKey::get_pubkey() const {
    if (!is_private) {
        return rnet::Result<std::array<uint8_t, 32>>::ok(key);
    }

    // Generate Ed25519 public key from private key seed
    Ed25519KeyPair kp;
    auto result = ed25519_from_seed(key);
    if (result.is_err()) {
        return rnet::Result<std::array<uint8_t, 32>>::err(
            result.error());
    }
    return rnet::Result<std::array<uint8_t, 32>>::ok(
        result.value().public_key.data);
}

rnet::Result<rnet::uint160> ExtKey::get_pubkey_hash() const {
    auto pk = get_pubkey();
    if (pk.is_err()) {
        return rnet::Result<rnet::uint160>::err(pk.error());
    }

    // Hash160 = first 20 bytes of keccak256d(pubkey)
    auto hash = keccak256d(pk.value());
    rnet::uint160 result;
    std::memcpy(result.data(), hash.data(), 20);
    return rnet::Result<rnet::uint160>::ok(result);
}

std::string ExtKey::to_base58() const {
    // Simplified serialization
    std::vector<uint8_t> data;
    data.reserve(78);

    // Version bytes (4)
    uint32_t version = is_private ? 0x0488ADE4 : 0x0488B21E;
    data.push_back(static_cast<uint8_t>((version >> 24) & 0xFF));
    data.push_back(static_cast<uint8_t>((version >> 16) & 0xFF));
    data.push_back(static_cast<uint8_t>((version >> 8) & 0xFF));
    data.push_back(static_cast<uint8_t>(version & 0xFF));

    data.push_back(depth);

    data.push_back(static_cast<uint8_t>((fingerprint >> 24) & 0xFF));
    data.push_back(static_cast<uint8_t>((fingerprint >> 16) & 0xFF));
    data.push_back(static_cast<uint8_t>((fingerprint >> 8) & 0xFF));
    data.push_back(static_cast<uint8_t>(fingerprint & 0xFF));

    data.push_back(static_cast<uint8_t>((child_num >> 24) & 0xFF));
    data.push_back(static_cast<uint8_t>((child_num >> 16) & 0xFF));
    data.push_back(static_cast<uint8_t>((child_num >> 8) & 0xFF));
    data.push_back(static_cast<uint8_t>(child_num & 0xFF));

    data.insert(data.end(), chaincode.begin(), chaincode.end());

    if (is_private) {
        data.push_back(0x00);
        data.insert(data.end(), key.begin(), key.end());
    } else {
        data.push_back(0x00);
        data.insert(data.end(), key.begin(), key.end());
    }

    return rnet::core::base58check_encode_simple(data);
}

rnet::Result<ExtKey> ExtKey::from_base58(std::string_view str) {
    auto decoded = rnet::core::base58check_decode_simple(str);
    if (!decoded.has_value() || decoded->size() != 78) {
        return rnet::Result<ExtKey>::err("Invalid base58 extended key");
    }

    const auto& data = *decoded;
    ExtKey k;

    uint32_t version = (static_cast<uint32_t>(data[0]) << 24)
                     | (static_cast<uint32_t>(data[1]) << 16)
                     | (static_cast<uint32_t>(data[2]) << 8)
                     | static_cast<uint32_t>(data[3]);

    k.is_private = (version == 0x0488ADE4);
    k.depth = data[4];
    k.fingerprint = (static_cast<uint32_t>(data[5]) << 24)
                  | (static_cast<uint32_t>(data[6]) << 16)
                  | (static_cast<uint32_t>(data[7]) << 8)
                  | static_cast<uint32_t>(data[8]);
    k.child_num = (static_cast<uint32_t>(data[9]) << 24)
                | (static_cast<uint32_t>(data[10]) << 16)
                | (static_cast<uint32_t>(data[11]) << 8)
                | static_cast<uint32_t>(data[12]);

    std::memcpy(k.chaincode.data(), data.data() + 13, 32);
    // Skip the 0x00 prefix byte at position 45
    std::memcpy(k.key.data(), data.data() + 46, 32);

    return rnet::Result<ExtKey>::ok(k);
}

// ─── Convenience ────────────────────────────────────────────────────

rnet::Result<ExtKey> derive_rnet_key(
    const ExtKey& master,
    uint32_t account,
    uint32_t change,
    uint32_t index) {

    // m/44'/9555'/account'/change/index
    std::string path = "m/44'/9555'/"
        + std::to_string(account) + "'/"
        + std::to_string(change) + "/"
        + std::to_string(index);

    return master.derive_path(path);
}

}  // namespace rnet::crypto

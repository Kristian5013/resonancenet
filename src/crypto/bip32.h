#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/error.h"
#include "core/types.h"

namespace rnet::crypto {

/// BIP32 extended key (private or public)
struct ExtKey {
    std::array<uint8_t, 32> key{};      // Private key or public x-coord
    std::array<uint8_t, 32> chaincode{};
    uint8_t depth = 0;
    uint32_t fingerprint = 0;           // Parent fingerprint
    uint32_t child_num = 0;
    bool is_private = false;

    /// Derive child key (BIP32)
    /// Hardened derivation: index >= 0x80000000
    rnet::Result<ExtKey> derive_child(uint32_t index) const;

    /// Derive by path string (e.g., "m/44'/9555'/0'/0/0")
    rnet::Result<ExtKey> derive_path(std::string_view path) const;

    /// Get the Ed25519 public key from this private key
    rnet::Result<std::array<uint8_t, 32>> get_pubkey() const;

    /// Get the 20-byte Hash160 (first 20 bytes of keccak256d(pubkey))
    rnet::Result<rnet::uint160> get_pubkey_hash() const;

    /// Serialize to base58 (xprv/xpub format)
    std::string to_base58() const;

    /// Deserialize from base58
    static rnet::Result<ExtKey> from_base58(std::string_view str);
};

/// BIP32 hardened child index
inline constexpr uint32_t HARDENED = 0x80000000;

/// ResonanceNet BIP44 derivation path: m/44'/9555'/account'/change/index
inline constexpr uint32_t BIP44_PURPOSE = 44;
inline constexpr uint32_t BIP44_COIN_TYPE = 9555;

/// Create master key from BIP39 seed (64 bytes)
/// Uses HMAC-SHA512 with key "Bitcoin seed" (same as BTC for compatibility)
rnet::Result<ExtKey> master_key_from_seed(
    std::span<const uint8_t> seed);

/// Derive a ResonanceNet address key
/// path: m/44'/9555'/account'/change/index
rnet::Result<ExtKey> derive_rnet_key(
    const ExtKey& master,
    uint32_t account = 0,
    uint32_t change = 0,
    uint32_t index = 0);

}  // namespace rnet::crypto

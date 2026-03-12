#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/types.h"
#include "core/error.h"

namespace rnet::crypto {

// -----------------------------------------------------------------------
// Ed25519 types
// -----------------------------------------------------------------------

/// Ed25519 secret key: 64 bytes (seed[32] + public_key[32]).
/// The seed is the 32-byte private scalar.
/// The second 32 bytes are the corresponding public key.
struct Ed25519SecretKey {
    static constexpr size_t SIZE = 64;
    std::array<uint8_t, SIZE> data{};

    /// Get the 32-byte seed portion.
    std::span<const uint8_t> seed() const {
        return {data.data(), 32};
    }

    /// Get the embedded public key portion.
    std::span<const uint8_t> public_key_bytes() const {
        return {data.data() + 32, 32};
    }

    /// Securely wipe the key material.
    void wipe();

    bool operator==(const Ed25519SecretKey& other) const {
        return data == other.data;
    }
};

/// Ed25519 public key: 32 bytes (compressed point on the curve).
struct Ed25519PublicKey {
    static constexpr size_t SIZE = 32;
    std::array<uint8_t, SIZE> data{};

    /// Convert to uint256 for storage.
    rnet::uint256 to_uint256() const;

    /// Create from uint256.
    static Ed25519PublicKey from_uint256(const rnet::uint256& val);

    /// Create from raw bytes.
    static Ed25519PublicKey from_bytes(std::span<const uint8_t> bytes);

    /// Check if the key is all zeros (invalid).
    bool is_zero() const;

    /// Hex representation.
    std::string to_hex() const;

    /// Parse from hex string.
    static rnet::Result<Ed25519PublicKey> from_hex(std::string_view hex);

    bool operator==(const Ed25519PublicKey& other) const {
        return data == other.data;
    }

    auto operator<=>(const Ed25519PublicKey& other) const = default;
};

/// Ed25519 signature: 64 bytes (R[32] + S[32]).
struct Ed25519Signature {
    static constexpr size_t SIZE = 64;
    std::array<uint8_t, SIZE> data{};

    /// Check if the signature is all zeros.
    bool is_zero() const;

    /// Hex representation.
    std::string to_hex() const;

    /// Parse from hex string.
    static rnet::Result<Ed25519Signature> from_hex(std::string_view hex);

    bool operator==(const Ed25519Signature& other) const {
        return data == other.data;
    }
};

/// Ed25519 key pair (secret + public).
struct Ed25519KeyPair {
    Ed25519SecretKey secret;
    Ed25519PublicKey public_key;

    /// Securely wipe the secret key.
    void wipe();
};

// -----------------------------------------------------------------------
// Key generation
// -----------------------------------------------------------------------

/// Generate a new Ed25519 keypair from cryptographic random.
rnet::Result<Ed25519KeyPair> ed25519_generate();

/// Derive public key from a 32-byte seed.
rnet::Result<Ed25519KeyPair> ed25519_from_seed(
    std::span<const uint8_t> seed);

// -----------------------------------------------------------------------
// Signing
// -----------------------------------------------------------------------

/// Sign a message with an Ed25519 secret key.
/// Deterministic: the same (key, message) always produces the same sig.
rnet::Result<Ed25519Signature> ed25519_sign(
    const Ed25519SecretKey& secret,
    std::span<const uint8_t> message);

/// Sign a message (string_view convenience).
rnet::Result<Ed25519Signature> ed25519_sign(
    const Ed25519SecretKey& secret,
    std::string_view message);

// -----------------------------------------------------------------------
// Verification
// -----------------------------------------------------------------------

/// Verify an Ed25519 signature.
/// @return true if the signature is valid for the given public key
///         and message.
bool ed25519_verify(
    const Ed25519PublicKey& pubkey,
    std::span<const uint8_t> message,
    const Ed25519Signature& signature);

/// Verify (string_view convenience).
bool ed25519_verify(
    const Ed25519PublicKey& pubkey,
    std::string_view message,
    const Ed25519Signature& signature);

// -----------------------------------------------------------------------
// Batch verification
// -----------------------------------------------------------------------

/// Batch-verify multiple Ed25519 signatures.
/// More efficient than verifying individually when the OpenSSL
/// backend supports it. Falls back to sequential verification.
/// @return true if ALL signatures are valid.
bool ed25519_batch_verify(
    const std::vector<Ed25519PublicKey>& pubkeys,
    const std::vector<std::span<const uint8_t>>& messages,
    const std::vector<Ed25519Signature>& signatures);

// -----------------------------------------------------------------------
// Utility
// -----------------------------------------------------------------------

/// Check if a 32-byte buffer is a valid Ed25519 public key
/// (on the curve).
bool ed25519_is_valid_pubkey(std::span<const uint8_t> bytes);

/// Serialize the public key for coinbase scripts:
/// [0x20][32-byte pubkey][0xAC]
std::vector<uint8_t> ed25519_coinbase_script(
    const Ed25519PublicKey& pubkey);

/// Parse a coinbase script back to a public key.
rnet::Result<Ed25519PublicKey> ed25519_parse_coinbase_script(
    std::span<const uint8_t> script);

}  // namespace rnet::crypto

/// Hash support for Ed25519PublicKey
namespace std {
template<>
struct hash<rnet::crypto::Ed25519PublicKey> {
    size_t operator()(
        const rnet::crypto::Ed25519PublicKey& k) const noexcept {
        size_t h = 14695981039346656037ULL;
        for (size_t i = 0; i < 32; ++i) {
            h ^= static_cast<size_t>(k.data[i]);
            h *= 1099511628211ULL;
        }
        return h;
    }
};
}  // namespace std

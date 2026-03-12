#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "core/types.h"
#include "core/error.h"

namespace rnet::crypto {

// -----------------------------------------------------------------------
// secp256k1 types (for Bitcoin-compatible operations)
// -----------------------------------------------------------------------

/// Compressed public key: 33 bytes (02/03 prefix + 32-byte x coordinate).
struct Secp256k1PubKey {
    static constexpr size_t COMPRESSED_SIZE = 33;
    static constexpr size_t UNCOMPRESSED_SIZE = 65;

    std::array<uint8_t, COMPRESSED_SIZE> data{};
    bool valid = false;

    bool is_valid() const { return valid; }
    bool is_zero() const;

    std::string to_hex() const;
    static rnet::Result<Secp256k1PubKey> from_hex(std::string_view hex);
    static rnet::Result<Secp256k1PubKey> from_bytes(
        std::span<const uint8_t> bytes);

    bool operator==(const Secp256k1PubKey& other) const {
        return data == other.data && valid == other.valid;
    }
};

/// Secret key (scalar): 32 bytes.
struct Secp256k1SecretKey {
    static constexpr size_t SIZE = 32;
    std::array<uint8_t, SIZE> data{};

    void wipe();
    bool is_zero() const;

    bool operator==(const Secp256k1SecretKey& other) const {
        return data == other.data;
    }
};

/// ECDSA signature in DER format (variable length, max 72 bytes).
struct Secp256k1Signature {
    std::vector<uint8_t> der_data;

    bool is_empty() const { return der_data.empty(); }
    std::string to_hex() const;

    /// Compact signature (64 bytes, R||S).
    std::array<uint8_t, 64> to_compact() const;
    static rnet::Result<Secp256k1Signature> from_compact(
        std::span<const uint8_t> compact);
    static rnet::Result<Secp256k1Signature> from_der(
        std::span<const uint8_t> der);
};

/// Secp256k1 key pair.
struct Secp256k1KeyPair {
    Secp256k1SecretKey secret;
    Secp256k1PubKey public_key;

    void wipe();
};

// -----------------------------------------------------------------------
// secp256k1 context (singleton)
// -----------------------------------------------------------------------

/// Global secp256k1 context. Must be initialized before use.
/// Thread-safe after initialization.
class Secp256k1Context {
public:
    static Secp256k1Context& instance();

    /// Initialize the context. Call once at startup.
    rnet::Result<void> init();

    /// Shut down the context. Call at cleanup.
    void shutdown();

    /// Check if initialized.
    bool is_initialized() const;

    /// Get the raw secp256k1_context pointer (for internal use).
    void* raw_ctx() const;

private:
    Secp256k1Context();
    ~Secp256k1Context();

    Secp256k1Context(const Secp256k1Context&) = delete;
    Secp256k1Context& operator=(const Secp256k1Context&) = delete;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// -----------------------------------------------------------------------
// Key generation
// -----------------------------------------------------------------------

/// Generate a new secp256k1 key pair.
rnet::Result<Secp256k1KeyPair> secp256k1_generate();

/// Derive public key from a secret key.
rnet::Result<Secp256k1PubKey> secp256k1_pubkey_from_secret(
    const Secp256k1SecretKey& secret);

/// Tweak a secret key by adding a scalar (for BIP32 child derivation).
rnet::Result<Secp256k1SecretKey> secp256k1_secret_tweak_add(
    const Secp256k1SecretKey& secret,
    std::span<const uint8_t> tweak);

/// Tweak a public key by adding a point (for BIP32 child derivation).
rnet::Result<Secp256k1PubKey> secp256k1_pubkey_tweak_add(
    const Secp256k1PubKey& pubkey,
    std::span<const uint8_t> tweak);

// -----------------------------------------------------------------------
// ECDSA signing / verification
// -----------------------------------------------------------------------

/// Sign a 32-byte hash with ECDSA.
rnet::Result<Secp256k1Signature> secp256k1_sign(
    const Secp256k1SecretKey& secret,
    const rnet::uint256& hash);

/// Verify an ECDSA signature against a 32-byte hash.
bool secp256k1_verify(
    const Secp256k1PubKey& pubkey,
    const rnet::uint256& hash,
    const Secp256k1Signature& signature);

// -----------------------------------------------------------------------
// ECDH (shared secret)
// -----------------------------------------------------------------------

/// Compute ECDH shared secret.
rnet::Result<rnet::uint256> secp256k1_ecdh(
    const Secp256k1SecretKey& secret,
    const Secp256k1PubKey& pubkey);

// -----------------------------------------------------------------------
// Key validation
// -----------------------------------------------------------------------

/// Check if bytes form a valid secret key (non-zero, less than order).
bool secp256k1_is_valid_secret(std::span<const uint8_t> bytes);

/// Check if bytes form a valid compressed public key.
bool secp256k1_is_valid_pubkey(std::span<const uint8_t> bytes);

/// Decompress a public key from 33 bytes to 65 bytes.
rnet::Result<std::array<uint8_t, 65>> secp256k1_decompress_pubkey(
    const Secp256k1PubKey& pubkey);

/// Compress a public key from 65 bytes to 33 bytes.
rnet::Result<Secp256k1PubKey> secp256k1_compress_pubkey(
    std::span<const uint8_t> uncompressed);

}  // namespace rnet::crypto

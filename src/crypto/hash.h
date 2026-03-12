#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"
#include "core/error.h"
#include "crypto/keccak.h"

namespace rnet::crypto {

// -----------------------------------------------------------------------
// High-level hash wrappers
// -----------------------------------------------------------------------

/// Hash160(data) = first 20 bytes of keccak256d(data).
/// Used for P2WPKH addresses: Hash160(pubkey).
rnet::uint160 hash160(std::span<const uint8_t> data);
rnet::uint160 hash160(std::string_view data);

/// Hash256(data) = keccak256d(data). Alias for double-keccak.
/// Used for transaction hashing, block hashing, etc.
rnet::uint256 hash256(std::span<const uint8_t> data);
rnet::uint256 hash256(std::string_view data);

/// HMAC-SHA512 using OpenSSL.
/// Used by BIP32 key derivation.
/// @param key HMAC key.
/// @param data Message data.
/// @return 64-byte HMAC result.
rnet::uint512 hmac_sha512(std::span<const uint8_t> key,
                          std::span<const uint8_t> data);

/// HMAC-SHA512 convenience for string key.
rnet::uint512 hmac_sha512(std::string_view key,
                          std::span<const uint8_t> data);

/// SHA-512 (single hash) via OpenSSL.
/// Used internally by Ed25519 and BIP32.
rnet::uint512 sha512(std::span<const uint8_t> data);

/// SHA-256 (single hash) via OpenSSL.
/// Used for BIP39 checksum and compatibility.
rnet::uint256 sha256(std::span<const uint8_t> data);
rnet::uint256 sha256(std::string_view data);

/// HMAC-SHA256 via OpenSSL.
rnet::uint256 hmac_sha256(std::span<const uint8_t> key,
                          std::span<const uint8_t> data);

/// RIPEMD-160 via OpenSSL (for Bitcoin compatibility if needed).
rnet::uint160 ripemd160(std::span<const uint8_t> data);

// -----------------------------------------------------------------------
// Tagged hashing
// -----------------------------------------------------------------------

/// TaggedHash(tag, data) = keccak256d(keccak256(tag) || keccak256(tag) || data).
/// Provides domain separation for different hash contexts.
/// Similar to BIP340 tagged hashing but using Keccak.
rnet::uint256 tagged_hash(std::string_view tag,
                          std::span<const uint8_t> data);

/// Tagged hash with two data spans (common pattern).
rnet::uint256 tagged_hash(std::string_view tag,
                          std::span<const uint8_t> data1,
                          std::span<const uint8_t> data2);

// -----------------------------------------------------------------------
// Hash combiner (for multi-field hashing)
// -----------------------------------------------------------------------

/// HashWriter: accumulates data and produces a hash at the end.
/// Wraps KeccakHasher with a DataStream-like interface.
class HashWriter {
public:
    HashWriter();

    /// Write raw bytes.
    void write(const void* data, size_t len);

    /// Write a single byte.
    void write_byte(uint8_t b);

    /// Get single Keccak-256 hash.
    rnet::uint256 get_hash();

    /// Get double Keccak-256 hash (Hash256).
    rnet::uint256 get_hash256();

    /// Get Hash160 (first 20 bytes of keccak256d).
    rnet::uint160 get_hash160();

    /// Reset for reuse.
    void reset();

    /// Serialization support: operator<< for any serializable type.
    template<typename T>
    HashWriter& operator<<(const T& obj) {
        obj.serialize(*this);
        return *this;
    }

private:
    KeccakHasher hasher_;
};

/// Convenience: hash a serializable object with Hash256.
template<typename T>
rnet::uint256 serialize_hash256(const T& obj) {
    HashWriter writer;
    writer << obj;
    return writer.get_hash256();
}

/// Convenience: hash a serializable object with Hash160.
template<typename T>
rnet::uint160 serialize_hash160(const T& obj) {
    HashWriter writer;
    writer << obj;
    return writer.get_hash160();
}

// -----------------------------------------------------------------------
// PBKDF2-HMAC-SHA512 (for BIP39 seed derivation)
// -----------------------------------------------------------------------

/// PBKDF2 with HMAC-SHA512. Used by BIP39 mnemonic_to_seed().
/// @param password Password bytes.
/// @param salt Salt bytes.
/// @param iterations Number of iterations (2048 for BIP39).
/// @param out_len Desired output length in bytes.
/// @return Derived key.
std::vector<uint8_t> pbkdf2_hmac_sha512(
    std::span<const uint8_t> password,
    std::span<const uint8_t> salt,
    uint32_t iterations,
    size_t out_len);

/// PBKDF2-HMAC-SHA512 convenience overload for string inputs.
std::vector<uint8_t> pbkdf2_hmac_sha512(
    std::string_view password,
    std::string_view salt,
    uint32_t iterations,
    size_t out_len);

// -----------------------------------------------------------------------
// Utility
// -----------------------------------------------------------------------

/// Secure memory wipe (attempts to avoid compiler optimization).
void secure_wipe(void* ptr, size_t len);

/// Secure wipe for a vector.
template<typename T>
void secure_wipe_vec(std::vector<T>& vec) {
    if (!vec.empty()) {
        secure_wipe(vec.data(), vec.size() * sizeof(T));
    }
    vec.clear();
}

}  // namespace rnet::crypto

// Copyright (c) 2024-2026 The ResonanceNet Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "crypto/hash.h"

#include "core/logging.h"

#include <cstring>
#include <stdexcept>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/kdf.h>

#ifdef _WIN32
#include <windows.h>  // SecureZeroMemory
#endif

namespace rnet::crypto {

// ---------------------------------------------------------------------------
// hash160  --  first 20 bytes of Keccak-256d(data)
// ---------------------------------------------------------------------------
//
// Design note:
//   hash160 = first 20 bytes of Keccak-256d(data).
//   Used for P2WPKH address derivation: Hash160(pubkey).
//   Analogous to Bitcoin's RIPEMD-160(SHA-256(x)), but single-algorithm.

rnet::uint160 hash160(std::span<const uint8_t> data)
{
    // 1. Compute the double-Keccak-256 digest.
    rnet::uint256 dbl = keccak256d(data);

    // 2. Truncate to the first 20 bytes.
    rnet::uint160 result;
    std::memcpy(result.data(), dbl.data(), 20);
    return result;
}

rnet::uint160 hash160(std::string_view data)
{
    return hash160(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(data.data()), data.size()));
}

// ---------------------------------------------------------------------------
// hash256  --  Keccak-256d(data)  (double hash)
// ---------------------------------------------------------------------------
//
// Design note:
//   keccak256d = Keccak-256(Keccak-256(data)).
//   Used for transaction hashing, block hashing, and Merkle trees.

rnet::uint256 hash256(std::span<const uint8_t> data)
{
    return keccak256d(data);
}

rnet::uint256 hash256(std::string_view data)
{
    return keccak256d(data);
}

// ---------------------------------------------------------------------------
// sha256  --  SHA-256 wrapper for compatibility
// ---------------------------------------------------------------------------
//
// Design note:
//   sha256 = single SHA-256 via OpenSSL EVP.
//   Used for BIP39 checksum computation and external compatibility.

rnet::uint256 sha256(std::span<const uint8_t> data)
{
    rnet::uint256 result;
    unsigned int len = 32;

    // 1. Allocate an EVP digest context.
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) {
        return result;
    }

    // 2. Initialise, feed data, and finalise the SHA-256 digest.
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(ctx, data.data(), data.size()) != 1 ||
        EVP_DigestFinal_ex(ctx, result.data(), &len) != 1) {
        EVP_MD_CTX_free(ctx);
        return result;
    }

    // 3. Clean up the context.
    EVP_MD_CTX_free(ctx);
    return result;
}

rnet::uint256 sha256(std::string_view data)
{
    return sha256(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(data.data()), data.size()));
}

// ---------------------------------------------------------------------------
// sha512  --  SHA-512 via OpenSSL
// ---------------------------------------------------------------------------
//
// Design note:
//   sha512 = single SHA-512 via OpenSSL EVP.
//   Used internally by Ed25519 and BIP32 key derivation.

rnet::uint512 sha512(std::span<const uint8_t> data)
{
    rnet::uint512 result;
    unsigned int len = 64;

    // 1. Allocate an EVP digest context.
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) {
        return result;
    }

    // 2. Initialise, feed data, and finalise the SHA-512 digest.
    if (EVP_DigestInit_ex(ctx, EVP_sha512(), nullptr) != 1 ||
        EVP_DigestUpdate(ctx, data.data(), data.size()) != 1 ||
        EVP_DigestFinal_ex(ctx, result.data(), &len) != 1) {
        EVP_MD_CTX_free(ctx);
        return result;
    }

    // 3. Clean up the context.
    EVP_MD_CTX_free(ctx);
    return result;
}

// ---------------------------------------------------------------------------
// hmac_sha512  --  HMAC-SHA-512 for BIP32 key derivation
// ---------------------------------------------------------------------------
//
// Design note:
//   HMAC-SHA-512 via OpenSSL's one-shot HMAC() call.
//   Returns 64-byte MAC used by BIP32 CKDpriv / CKDpub.

rnet::uint512 hmac_sha512(std::span<const uint8_t> key,
                           std::span<const uint8_t> data)
{
    rnet::uint512 result;
    unsigned int len = 64;

    // 1. Compute the HMAC in a single call.
    uint8_t* out = HMAC(
        EVP_sha512(),
        key.data(), static_cast<int>(key.size()),
        data.data(), data.size(),
        result.data(), &len);

    // 2. Zero the result on failure.
    if (!out) {
        result.set_zero();
    }
    return result;
}

rnet::uint512 hmac_sha512(std::string_view key,
                           std::span<const uint8_t> data)
{
    return hmac_sha512(
        std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(key.data()), key.size()),
        data);
}

// ---------------------------------------------------------------------------
// hmac_sha256  --  HMAC-SHA-256
// ---------------------------------------------------------------------------

rnet::uint256 hmac_sha256(std::span<const uint8_t> key,
                           std::span<const uint8_t> data)
{
    rnet::uint256 result;
    unsigned int len = 32;

    // 1. Compute the HMAC in a single call.
    uint8_t* out = HMAC(
        EVP_sha256(),
        key.data(), static_cast<int>(key.size()),
        data.data(), data.size(),
        result.data(), &len);

    // 2. Zero the result on failure.
    if (!out) {
        result.set_zero();
    }
    return result;
}

// ---------------------------------------------------------------------------
// ripemd160  --  RIPEMD-160 via OpenSSL (Bitcoin compatibility)
// ---------------------------------------------------------------------------
//
// Design note:
//   RIPEMD-160 may be unavailable in FIPS-mode OpenSSL builds.
//   Falls back to the first 20 bytes of SHA-256 when the algorithm
//   is not present.

rnet::uint160 ripemd160(std::span<const uint8_t> data)
{
    rnet::uint160 result;
    unsigned int len = 20;

    // 1. Allocate an EVP digest context.
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) {
        return result;
    }

    // 2. Attempt to load the RIPEMD-160 algorithm.
    const EVP_MD* md = EVP_ripemd160();
    if (!md) {
        // 2a. Fallback: truncate SHA-256 to 20 bytes.
        EVP_MD_CTX_free(ctx);
        rnet::uint256 sha = sha256(data);
        std::memcpy(result.data(), sha.data(), 20);
        return result;
    }

    // 3. Initialise, feed data, and finalise the digest.
    if (EVP_DigestInit_ex(ctx, md, nullptr) != 1 ||
        EVP_DigestUpdate(ctx, data.data(), data.size()) != 1 ||
        EVP_DigestFinal_ex(ctx, result.data(), &len) != 1) {
        EVP_MD_CTX_free(ctx);
        result.set_zero();
        return result;
    }

    // 4. Clean up the context.
    EVP_MD_CTX_free(ctx);
    return result;
}

// ---------------------------------------------------------------------------
// tagged_hash  --  domain-separated Keccak hashing
// ---------------------------------------------------------------------------
//
// Design note:
//   tagged_hash = Keccak-256(Keccak-256(tag) || Keccak-256(tag) || msg).
//   The tag digest is prepended twice (BIP340 convention) to provide
//   collision-resistant domain separation between different hash usages.

rnet::uint256 tagged_hash(std::string_view tag,
                           std::span<const uint8_t> data)
{
    // 1. Compute the single-Keccak tag digest.
    rnet::uint256 tag_hash = keccak256(tag);

    // 2. Build the preimage: tag_hash || tag_hash || data.
    KeccakHasher hasher;
    hasher.write(tag_hash.span());
    hasher.write(tag_hash.span());
    hasher.write(data);

    // 3. First Keccak pass over the preimage.
    rnet::uint256 first = hasher.finalize();

    // 4. Second Keccak pass (double-hash).
    return keccak256(first.span());
}

rnet::uint256 tagged_hash(std::string_view tag,
                           std::span<const uint8_t> data1,
                           std::span<const uint8_t> data2)
{
    // 1. Compute the single-Keccak tag digest.
    rnet::uint256 tag_hash = keccak256(tag);

    // 2. Build the preimage: tag_hash || tag_hash || data1 || data2.
    KeccakHasher hasher;
    hasher.write(tag_hash.span());
    hasher.write(tag_hash.span());
    hasher.write(data1);
    hasher.write(data2);

    // 3. First Keccak pass over the preimage.
    rnet::uint256 first = hasher.finalize();

    // 4. Second Keccak pass (double-hash).
    return keccak256(first.span());
}

// ---------------------------------------------------------------------------
// HashWriter  --  streaming hash accumulator
// ---------------------------------------------------------------------------
//
// Design note:
//   Wraps KeccakHasher with a DataStream-like interface so that
//   serializable objects can be hashed via operator<<.

HashWriter::HashWriter() = default;

void HashWriter::write(const void* data, size_t len)
{
    hasher_.write(std::span<const uint8_t>(
        static_cast<const uint8_t*>(data), len));
}

void HashWriter::write_byte(uint8_t b)
{
    hasher_.write(std::span<const uint8_t>(&b, 1));
}

rnet::uint256 HashWriter::get_hash()
{
    // 1. Copy the hasher to preserve accumulated state.
    KeccakHasher copy = hasher_;

    // 2. Finalise the copy (single Keccak-256).
    return copy.finalize();
}

rnet::uint256 HashWriter::get_hash256()
{
    // 1. Copy the hasher to preserve accumulated state.
    KeccakHasher copy = hasher_;

    // 2. Finalise with double Keccak-256 (keccak256d).
    return copy.finalize_double();
}

rnet::uint160 HashWriter::get_hash160()
{
    // 1. Compute the double-Keccak-256 digest.
    rnet::uint256 h = get_hash256();

    // 2. Truncate to the first 20 bytes.
    rnet::uint160 result;
    std::memcpy(result.data(), h.data(), 20);
    return result;
}

void HashWriter::reset()
{
    hasher_.reset();
}

// ---------------------------------------------------------------------------
// pbkdf2_hmac_sha512  --  BIP39 seed derivation
// ---------------------------------------------------------------------------
//
// Design note:
//   PBKDF2 with HMAC-SHA-512 per RFC 8018.  BIP39 uses 2048 iterations
//   to derive a 64-byte seed from a mnemonic passphrase.
//   Prefers OpenSSL's PKCS5_PBKDF2_HMAC; falls back to a manual
//   implementation when that call fails.

std::vector<uint8_t> pbkdf2_hmac_sha512(
    std::span<const uint8_t> password,
    std::span<const uint8_t> salt,
    uint32_t iterations,
    size_t out_len)
{
    std::vector<uint8_t> result(out_len, 0);

    // 1. Try the OpenSSL built-in PBKDF2.
    int rc = PKCS5_PBKDF2_HMAC(
        reinterpret_cast<const char*>(password.data()),
        static_cast<int>(password.size()),
        salt.data(),
        static_cast<int>(salt.size()),
        static_cast<int>(iterations),
        EVP_sha512(),
        static_cast<int>(out_len),
        result.data());

    if (rc != 1) {
        // 2. Fallback: manual PBKDF2 with our HMAC-SHA-512.
        result.assign(out_len, 0);

        size_t blocks = (out_len + 63) / 64;  // SHA-512 = 64-byte blocks
        std::vector<uint8_t> derived;
        derived.reserve(blocks * 64);

        for (uint32_t block_idx = 1;
             block_idx <= static_cast<uint32_t>(blocks); ++block_idx) {
            // 2a. U_1 = HMAC(password, salt || INT_32_BE(block_idx))
            std::vector<uint8_t> salt_block(salt.begin(), salt.end());
            salt_block.push_back(
                static_cast<uint8_t>((block_idx >> 24) & 0xFF));
            salt_block.push_back(
                static_cast<uint8_t>((block_idx >> 16) & 0xFF));
            salt_block.push_back(
                static_cast<uint8_t>((block_idx >> 8) & 0xFF));
            salt_block.push_back(
                static_cast<uint8_t>(block_idx & 0xFF));

            rnet::uint512 u = hmac_sha512(password,
                std::span<const uint8_t>(salt_block));
            rnet::uint512 xor_sum = u;

            // 2b. U_2 .. U_c  and XOR accumulation.
            for (uint32_t iter = 1; iter < iterations; ++iter) {
                u = hmac_sha512(password, u.span());
                xor_sum ^= u;
            }

            derived.insert(derived.end(),
                           xor_sum.begin(), xor_sum.end());
        }

        // 3. Copy the requested number of bytes and wipe temporaries.
        std::memcpy(result.data(), derived.data(), out_len);
        secure_wipe(derived.data(), derived.size());
    }

    return result;
}

std::vector<uint8_t> pbkdf2_hmac_sha512(
    std::string_view password,
    std::string_view salt,
    uint32_t iterations,
    size_t out_len)
{
    return pbkdf2_hmac_sha512(
        std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(password.data()),
            password.size()),
        std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(salt.data()),
            salt.size()),
        iterations, out_len);
}

// ---------------------------------------------------------------------------
// secure_wipe  --  compiler-safe memory zeroing
// ---------------------------------------------------------------------------
//
// Design note:
//   Zeroes sensitive memory in a way the compiler cannot optimise away.
//   Uses SecureZeroMemory on Windows; volatile writes + compiler barrier
//   on other platforms.

void secure_wipe(void* ptr, size_t len)
{
    if (!ptr || len == 0) return;

#ifdef _WIN32
    // 1. Windows: use the SDK-provided secure wipe.
    SecureZeroMemory(ptr, len);
#else
    // 1. Write zeros through a volatile pointer.
    volatile uint8_t* p = static_cast<volatile uint8_t*>(ptr);
    for (size_t i = 0; i < len; ++i) {
        p[i] = 0;
    }

    // 2. Compiler barrier to prevent dead-store elimination.
    __asm__ __volatile__("" ::: "memory");
#endif
}

} // namespace rnet::crypto

#include "crypto/hash.h"

#include <cstring>
#include <stdexcept>

#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/kdf.h>

#include "core/logging.h"

#ifdef _WIN32
#include <windows.h>  // SecureZeroMemory
#endif

namespace rnet::crypto {

// -----------------------------------------------------------------------
// Hash160 / Hash256
// -----------------------------------------------------------------------

rnet::uint160 hash160(std::span<const uint8_t> data) {
    rnet::uint256 dbl = keccak256d(data);
    rnet::uint160 result;
    std::memcpy(result.data(), dbl.data(), 20);
    return result;
}

rnet::uint160 hash160(std::string_view data) {
    return hash160(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(data.data()), data.size()));
}

rnet::uint256 hash256(std::span<const uint8_t> data) {
    return keccak256d(data);
}

rnet::uint256 hash256(std::string_view data) {
    return keccak256d(data);
}

// -----------------------------------------------------------------------
// OpenSSL-based SHA-256
// -----------------------------------------------------------------------

rnet::uint256 sha256(std::span<const uint8_t> data) {
    rnet::uint256 result;
    unsigned int len = 32;

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) {
        return result;
    }

    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(ctx, data.data(), data.size()) != 1 ||
        EVP_DigestFinal_ex(ctx, result.data(), &len) != 1) {
        EVP_MD_CTX_free(ctx);
        return result;
    }

    EVP_MD_CTX_free(ctx);
    return result;
}

rnet::uint256 sha256(std::string_view data) {
    return sha256(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(data.data()), data.size()));
}

// -----------------------------------------------------------------------
// OpenSSL-based SHA-512
// -----------------------------------------------------------------------

rnet::uint512 sha512(std::span<const uint8_t> data) {
    rnet::uint512 result;
    unsigned int len = 64;

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) {
        return result;
    }

    if (EVP_DigestInit_ex(ctx, EVP_sha512(), nullptr) != 1 ||
        EVP_DigestUpdate(ctx, data.data(), data.size()) != 1 ||
        EVP_DigestFinal_ex(ctx, result.data(), &len) != 1) {
        EVP_MD_CTX_free(ctx);
        return result;
    }

    EVP_MD_CTX_free(ctx);
    return result;
}

// -----------------------------------------------------------------------
// HMAC-SHA512
// -----------------------------------------------------------------------

rnet::uint512 hmac_sha512(std::span<const uint8_t> key,
                          std::span<const uint8_t> data) {
    rnet::uint512 result;
    unsigned int len = 64;

    uint8_t* out = HMAC(
        EVP_sha512(),
        key.data(), static_cast<int>(key.size()),
        data.data(), data.size(),
        result.data(), &len);

    if (!out) {
        result.set_zero();
    }
    return result;
}

rnet::uint512 hmac_sha512(std::string_view key,
                          std::span<const uint8_t> data) {
    return hmac_sha512(
        std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(key.data()), key.size()),
        data);
}

// -----------------------------------------------------------------------
// HMAC-SHA256
// -----------------------------------------------------------------------

rnet::uint256 hmac_sha256(std::span<const uint8_t> key,
                          std::span<const uint8_t> data) {
    rnet::uint256 result;
    unsigned int len = 32;

    uint8_t* out = HMAC(
        EVP_sha256(),
        key.data(), static_cast<int>(key.size()),
        data.data(), data.size(),
        result.data(), &len);

    if (!out) {
        result.set_zero();
    }
    return result;
}

// -----------------------------------------------------------------------
// RIPEMD-160
// -----------------------------------------------------------------------

rnet::uint160 ripemd160(std::span<const uint8_t> data) {
    rnet::uint160 result;
    unsigned int len = 20;

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) {
        return result;
    }

    const EVP_MD* md = EVP_ripemd160();
    if (!md) {
        // RIPEMD-160 may not be available in all OpenSSL builds
        // (especially FIPS mode). Fall back to first 20 bytes of SHA-256.
        EVP_MD_CTX_free(ctx);
        rnet::uint256 sha = sha256(data);
        std::memcpy(result.data(), sha.data(), 20);
        return result;
    }

    if (EVP_DigestInit_ex(ctx, md, nullptr) != 1 ||
        EVP_DigestUpdate(ctx, data.data(), data.size()) != 1 ||
        EVP_DigestFinal_ex(ctx, result.data(), &len) != 1) {
        EVP_MD_CTX_free(ctx);
        result.set_zero();
        return result;
    }

    EVP_MD_CTX_free(ctx);
    return result;
}

// -----------------------------------------------------------------------
// Tagged hash
// -----------------------------------------------------------------------

rnet::uint256 tagged_hash(std::string_view tag,
                          std::span<const uint8_t> data) {
    // tag_hash = keccak256(tag)
    rnet::uint256 tag_hash = keccak256(tag);

    // Compute keccak256d(tag_hash || tag_hash || data)
    KeccakHasher hasher;
    hasher.write(tag_hash.span());
    hasher.write(tag_hash.span());
    hasher.write(data);
    rnet::uint256 first = hasher.finalize();
    return keccak256(first.span());
}

rnet::uint256 tagged_hash(std::string_view tag,
                          std::span<const uint8_t> data1,
                          std::span<const uint8_t> data2) {
    rnet::uint256 tag_hash = keccak256(tag);

    KeccakHasher hasher;
    hasher.write(tag_hash.span());
    hasher.write(tag_hash.span());
    hasher.write(data1);
    hasher.write(data2);
    rnet::uint256 first = hasher.finalize();
    return keccak256(first.span());
}

// -----------------------------------------------------------------------
// HashWriter
// -----------------------------------------------------------------------

HashWriter::HashWriter() = default;

void HashWriter::write(const void* data, size_t len) {
    hasher_.write(std::span<const uint8_t>(
        static_cast<const uint8_t*>(data), len));
}

void HashWriter::write_byte(uint8_t b) {
    hasher_.write(std::span<const uint8_t>(&b, 1));
}

rnet::uint256 HashWriter::get_hash() {
    // Copy the hasher to preserve state
    KeccakHasher copy = hasher_;
    return copy.finalize();
}

rnet::uint256 HashWriter::get_hash256() {
    KeccakHasher copy = hasher_;
    return copy.finalize_double();
}

rnet::uint160 HashWriter::get_hash160() {
    rnet::uint256 h = get_hash256();
    rnet::uint160 result;
    std::memcpy(result.data(), h.data(), 20);
    return result;
}

void HashWriter::reset() {
    hasher_.reset();
}

// -----------------------------------------------------------------------
// PBKDF2-HMAC-SHA512
// -----------------------------------------------------------------------

std::vector<uint8_t> pbkdf2_hmac_sha512(
    std::span<const uint8_t> password,
    std::span<const uint8_t> salt,
    uint32_t iterations,
    size_t out_len)
{
    std::vector<uint8_t> result(out_len, 0);

    // Use OpenSSL PKCS5_PBKDF2_HMAC
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
        // Fallback: manual PBKDF2 with our HMAC-SHA512
        result.assign(out_len, 0);

        size_t blocks = (out_len + 63) / 64;  // SHA-512 = 64 byte blocks
        std::vector<uint8_t> derived;
        derived.reserve(blocks * 64);

        for (uint32_t block_idx = 1;
             block_idx <= static_cast<uint32_t>(blocks); ++block_idx) {
            // U_1 = HMAC(password, salt || INT_32_BE(block_idx))
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

            for (uint32_t iter = 1; iter < iterations; ++iter) {
                u = hmac_sha512(password, u.span());
                xor_sum ^= u;
            }

            derived.insert(derived.end(),
                           xor_sum.begin(), xor_sum.end());
        }

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

// -----------------------------------------------------------------------
// Secure wipe
// -----------------------------------------------------------------------

void secure_wipe(void* ptr, size_t len) {
    if (!ptr || len == 0) return;

#ifdef _WIN32
    SecureZeroMemory(ptr, len);
#else
    // Use volatile to prevent compiler from optimizing away
    volatile uint8_t* p = static_cast<volatile uint8_t*>(ptr);
    for (size_t i = 0; i < len; ++i) {
        p[i] = 0;
    }
    // Compiler barrier
    __asm__ __volatile__("" ::: "memory");
#endif
}

}  // namespace rnet::crypto

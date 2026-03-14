// Copyright (c) 2024-2026 The ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "crypto/argon2.h"

#include "core/random.h"

#include <cstring>

#include <openssl/opensslv.h>

// ---------------------------------------------------------------------------
// Argon2id -- memory-hard key derivation function  (RFC 9106)
//
// Argon2id combines:
//   - Argon2d (data-dependent addressing, GPU/ASIC resistant)
//   - Argon2i (data-independent addressing, side-channel resistant)
//
// Parameters:
//   t_cost      -- number of passes over memory (time cost)
//   m_cost      -- memory usage in KiB (memory cost)
//   parallelism -- number of parallel lanes
//   output_len  -- desired derived key length
//
// The algorithm fills a memory matrix of (m_cost / 4p) x 4 blocks per lane,
// each block is 1024 bytes.  Memory hardness makes brute-force attacks
// proportionally expensive in both time and space.
//
// OpenSSL 3.2+ provides native Argon2id via EVP_KDF.
// Older versions fall back to PBKDF2-SHA512 with high iteration count.
// ---------------------------------------------------------------------------

#if OPENSSL_VERSION_NUMBER >= 0x30000000L
#include <openssl/core_names.h>
#include <openssl/kdf.h>
#include <openssl/params.h>
#endif

// Argon2 KDF params were added in OpenSSL 3.2
#if defined(OSSL_KDF_PARAM_ARGON2_MEMCOST)

namespace rnet::crypto {

// ---------------------------------------------------------------------------
// argon2id_derive  (native OpenSSL 3.2+ path)
// ---------------------------------------------------------------------------

rnet::Result<std::vector<uint8_t>> argon2id_derive(
    std::span<const uint8_t> password,
    std::span<const uint8_t> salt,
    const Argon2Params& params) {

    std::vector<uint8_t> output(params.output_len);

    // 1. Fetch the Argon2id KDF provider.
    EVP_KDF* kdf = EVP_KDF_fetch(nullptr, "ARGON2ID", nullptr);
    if (!kdf) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "Argon2id not available in this OpenSSL build");
    }

    // 2. Create KDF context.
    EVP_KDF_CTX* ctx = EVP_KDF_CTX_new(kdf);
    EVP_KDF_free(kdf);
    if (!ctx) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "Failed to create Argon2id context");
    }

    // 3. Configure parameters and derive.
    uint32_t threads = params.parallelism;
    uint32_t lanes = params.parallelism;

    OSSL_PARAM ossl_params[] = {
        OSSL_PARAM_construct_octet_string(
            OSSL_KDF_PARAM_PASSWORD,
            const_cast<uint8_t*>(password.data()),
            password.size()),
        OSSL_PARAM_construct_octet_string(
            OSSL_KDF_PARAM_SALT,
            const_cast<uint8_t*>(salt.data()),
            salt.size()),
        OSSL_PARAM_construct_uint32(
            OSSL_KDF_PARAM_ITER, const_cast<uint32_t*>(&params.t_cost)),
        OSSL_PARAM_construct_uint32(
            OSSL_KDF_PARAM_ARGON2_MEMCOST,
            const_cast<uint32_t*>(&params.m_cost)),
        OSSL_PARAM_construct_uint32(
            OSSL_KDF_PARAM_THREADS, &threads),
        OSSL_PARAM_construct_uint32(
            OSSL_KDF_PARAM_ARGON2_LANES, &lanes),
        OSSL_PARAM_construct_end()
    };

    int rc = EVP_KDF_derive(ctx, output.data(),
        output.size(), ossl_params);
    EVP_KDF_CTX_free(ctx);

    if (rc <= 0) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "Argon2id derivation failed");
    }

    return rnet::Result<std::vector<uint8_t>>::ok(std::move(output));
}

} // namespace rnet::crypto

#else

// ---------------------------------------------------------------------------
// argon2id_derive  (PBKDF2-SHA512 fallback for OpenSSL < 3.2)
//
// Less secure than Argon2id (not memory-hard) but provides a functional
// KDF on older systems.  Uses t_cost * 100000 PBKDF2 iterations minimum.
// ---------------------------------------------------------------------------

#include <openssl/evp.h>

namespace rnet::crypto {

rnet::Result<std::vector<uint8_t>> argon2id_derive(
    std::span<const uint8_t> password,
    std::span<const uint8_t> salt,
    const Argon2Params& params) {

    std::vector<uint8_t> output(params.output_len);

    // 1. Scale iterations from Argon2 t_cost.
    int iterations = static_cast<int>(params.t_cost) * 100000;
    if (iterations < 100000) iterations = 100000;

    // 2. Derive using PBKDF2-HMAC-SHA512.
    int rc = PKCS5_PBKDF2_HMAC(
        reinterpret_cast<const char*>(password.data()),
        static_cast<int>(password.size()),
        salt.data(), static_cast<int>(salt.size()),
        iterations,
        EVP_sha512(),
        static_cast<int>(output.size()),
        output.data());

    if (rc != 1) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "PBKDF2 fallback KDF failed");
    }

    return rnet::Result<std::vector<uint8_t>>::ok(std::move(output));
}

} // namespace rnet::crypto
#endif

namespace rnet::crypto {

// ---------------------------------------------------------------------------
// argon2id_derive  (string_view overload)
// ---------------------------------------------------------------------------

rnet::Result<std::vector<uint8_t>> argon2id_derive(
    std::string_view password,
    std::span<const uint8_t> salt,
    const Argon2Params& params) {
    return argon2id_derive(
        std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(password.data()),
            password.size()),
        salt, params);
}

// ---------------------------------------------------------------------------
// generate_argon2_salt
//
// Generate a cryptographically random 32-byte salt for Argon2id.
// ---------------------------------------------------------------------------

std::array<uint8_t, 32> generate_argon2_salt() {
    std::array<uint8_t, 32> salt;
    rnet::core::get_rand_bytes(salt);
    return salt;
}

} // namespace rnet::crypto

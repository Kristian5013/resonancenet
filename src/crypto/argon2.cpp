#include "crypto/argon2.h"
#include "core/random.h"

#include <cstring>

// OpenSSL 3.0+ has EVP_KDF for Argon2
#include <openssl/opensslv.h>

#if OPENSSL_VERSION_NUMBER >= 0x30000000L
#include <openssl/core_names.h>
#include <openssl/kdf.h>
#include <openssl/params.h>
#endif

// Argon2 KDF params were added in OpenSSL 3.2
#if defined(OSSL_KDF_PARAM_ARGON2_MEMCOST)

namespace rnet::crypto {

rnet::Result<std::vector<uint8_t>> argon2id_derive(
    std::span<const uint8_t> password,
    std::span<const uint8_t> salt,
    const Argon2Params& params) {

    std::vector<uint8_t> output(params.output_len);

    EVP_KDF* kdf = EVP_KDF_fetch(nullptr, "ARGON2ID", nullptr);
    if (!kdf) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "Argon2id not available in this OpenSSL build");
    }

    EVP_KDF_CTX* ctx = EVP_KDF_CTX_new(kdf);
    EVP_KDF_free(kdf);
    if (!ctx) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "Failed to create Argon2id context");
    }

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

}  // namespace rnet::crypto

#else
// Fallback: use PBKDF2-SHA512 when Argon2id is not available
// (OpenSSL < 3.2 or builds without Argon2 support)
#if OPENSSL_VERSION_NUMBER < 0x30000000L
#include <openssl/evp.h>
#endif

namespace rnet::crypto {

rnet::Result<std::vector<uint8_t>> argon2id_derive(
    std::span<const uint8_t> password,
    std::span<const uint8_t> salt,
    const Argon2Params& params) {

    // Fallback to PBKDF2-SHA512 with high iteration count
    // This is less secure than Argon2id but provides basic KDF
    std::vector<uint8_t> output(params.output_len);

    int iterations = static_cast<int>(params.t_cost) * 100000;
    if (iterations < 100000) iterations = 100000;

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

}  // namespace rnet::crypto
#endif

namespace rnet::crypto {

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

std::array<uint8_t, 32> generate_argon2_salt() {
    std::array<uint8_t, 32> salt;
    rnet::core::get_rand_bytes(salt);
    return salt;
}

}  // namespace rnet::crypto

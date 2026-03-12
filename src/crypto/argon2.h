#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/error.h"

namespace rnet::crypto {

/// Argon2id parameters for wallet backup key derivation
struct Argon2Params {
    uint32_t t_cost = 3;        // iterations
    uint32_t m_cost = 65536;    // memory in KiB (64 MB)
    uint32_t parallelism = 4;   // threads
    uint32_t output_len = 32;   // output key length in bytes
};

/// Default parameters for wallet backup
inline constexpr Argon2Params WALLET_BACKUP_PARAMS = {
    .t_cost = 3,
    .m_cost = 65536,
    .parallelism = 4,
    .output_len = 32
};

/// Derive key using Argon2id
/// Uses OpenSSL 3.0+ EVP_KDF if available
rnet::Result<std::vector<uint8_t>> argon2id_derive(
    std::span<const uint8_t> password,
    std::span<const uint8_t> salt,
    const Argon2Params& params = WALLET_BACKUP_PARAMS);

/// Convenience: derive from string password
rnet::Result<std::vector<uint8_t>> argon2id_derive(
    std::string_view password,
    std::span<const uint8_t> salt,
    const Argon2Params& params = WALLET_BACKUP_PARAMS);

/// Generate random salt (32 bytes)
std::array<uint8_t, 32> generate_argon2_salt();

}  // namespace rnet::crypto

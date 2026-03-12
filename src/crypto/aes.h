#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "core/error.h"

namespace rnet::crypto {

/// AES-256-CBC encryption/decryption via OpenSSL EVP
struct AES256CBC {
    static constexpr size_t KEY_SIZE = 32;
    static constexpr size_t IV_SIZE = 16;
    static constexpr size_t BLOCK_SIZE = 16;

    /// Encrypt with PKCS7 padding
    /// Returns IV (16 bytes) + ciphertext
    static rnet::Result<std::vector<uint8_t>> encrypt(
        std::span<const uint8_t> key,
        std::span<const uint8_t> iv,
        std::span<const uint8_t> plaintext);

    /// Decrypt with PKCS7 padding removal
    static rnet::Result<std::vector<uint8_t>> decrypt(
        std::span<const uint8_t> key,
        std::span<const uint8_t> iv,
        std::span<const uint8_t> ciphertext);

    /// Encrypt and prepend random IV
    static rnet::Result<std::vector<uint8_t>> encrypt_with_random_iv(
        std::span<const uint8_t> key,
        std::span<const uint8_t> plaintext);

    /// Decrypt where first 16 bytes are the IV
    static rnet::Result<std::vector<uint8_t>> decrypt_with_iv_prefix(
        std::span<const uint8_t> key,
        std::span<const uint8_t> iv_and_ciphertext);
};

}  // namespace rnet::crypto

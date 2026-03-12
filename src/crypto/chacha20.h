#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "core/error.h"

namespace rnet::crypto {

/// ChaCha20 stream cipher
class ChaCha20 {
public:
    static constexpr size_t KEY_SIZE = 32;
    static constexpr size_t NONCE_SIZE = 12;
    static constexpr size_t BLOCK_SIZE = 64;

    ChaCha20() = default;
    ChaCha20(std::span<const uint8_t, KEY_SIZE> key,
             std::span<const uint8_t, NONCE_SIZE> nonce);

    void set_key(std::span<const uint8_t, KEY_SIZE> key,
                 std::span<const uint8_t, NONCE_SIZE> nonce);

    /// Seek to a specific 64-byte block position
    void seek(uint32_t block_counter);

    /// Encrypt/decrypt in place (XOR with keystream)
    void crypt(std::span<uint8_t> data);

    /// Generate keystream bytes
    void keystream(std::span<uint8_t> out);

private:
    std::array<uint32_t, 16> state_{};
    std::array<uint8_t, BLOCK_SIZE> buffer_{};
    size_t buffer_pos_ = BLOCK_SIZE;  // Force generation on first use

    void generate_block();
    static void quarter_round(uint32_t& a, uint32_t& b,
                               uint32_t& c, uint32_t& d);
};

/// ChaCha20-Poly1305 AEAD
/// Uses OpenSSL EVP if available, otherwise pure implementation
struct ChaCha20Poly1305 {
    static constexpr size_t KEY_SIZE = 32;
    static constexpr size_t NONCE_SIZE = 12;
    static constexpr size_t TAG_SIZE = 16;

    /// Encrypt with associated data
    /// Returns ciphertext + 16-byte tag appended
    static rnet::Result<std::vector<uint8_t>> encrypt(
        std::span<const uint8_t> key,
        std::span<const uint8_t> nonce,
        std::span<const uint8_t> aad,
        std::span<const uint8_t> plaintext);

    /// Decrypt with associated data
    /// Input: ciphertext with 16-byte tag appended
    static rnet::Result<std::vector<uint8_t>> decrypt(
        std::span<const uint8_t> key,
        std::span<const uint8_t> nonce,
        std::span<const uint8_t> aad,
        std::span<const uint8_t> ciphertext_with_tag);
};

}  // namespace rnet::crypto

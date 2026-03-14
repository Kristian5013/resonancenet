// Copyright (c) 2024-2026 The ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "crypto/aes.h"

#include "core/random.h"

#include <cstring>

#include <openssl/evp.h>

namespace rnet::crypto {

// ---------------------------------------------------------------------------
// AES-256-CBC encrypt
//
// AES in Cipher Block Chaining mode with PKCS#7 padding (via OpenSSL EVP).
//
// AES-256 internals (handled by OpenSSL):
//   - SubBytes:    byte substitution via the Rijndael S-box
//   - ShiftRows:   cyclic left-shift of rows in the 4x4 state matrix
//   - MixColumns:  column-wise GF(2^8) matrix multiply
//   - AddRoundKey: XOR with the round key
//   - 14 rounds for 256-bit keys
//
// CBC chains blocks:  C_i = AES_K(P_i ^ C_{i-1}),  C_0 = IV
// ---------------------------------------------------------------------------

rnet::Result<std::vector<uint8_t>> AES256CBC::encrypt(
    std::span<const uint8_t> key,
    std::span<const uint8_t> iv,
    std::span<const uint8_t> plaintext) {

    // 1. Validate key and IV sizes.
    if (key.size() != KEY_SIZE) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "AES-256-CBC: key must be 32 bytes");
    }
    if (iv.size() != IV_SIZE) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "AES-256-CBC: IV must be 16 bytes");
    }

    // 2. Create cipher context.
    auto ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "Failed to create cipher context");
    }

    // 3. Output can be up to plaintext.size() + BLOCK_SIZE (PKCS#7 padding).
    std::vector<uint8_t> output(plaintext.size() + BLOCK_SIZE);
    int out_len = 0;

    // 4. Initialize, encrypt, finalize.
    bool ok = true;
    ok = ok && EVP_EncryptInit_ex(ctx, EVP_aes_256_cbc(),
        nullptr, key.data(), iv.data());
    ok = ok && EVP_EncryptUpdate(ctx, output.data(), &out_len,
        plaintext.data(), static_cast<int>(plaintext.size()));
    int total = out_len;
    ok = ok && EVP_EncryptFinal_ex(ctx,
        output.data() + total, &out_len);
    total += out_len;

    EVP_CIPHER_CTX_free(ctx);

    if (!ok) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "AES-256-CBC encryption failed");
    }

    output.resize(static_cast<size_t>(total));
    return rnet::Result<std::vector<uint8_t>>::ok(std::move(output));
}

// ---------------------------------------------------------------------------
// AES-256-CBC decrypt
//
// Inverse: strip PKCS#7 padding and reverse CBC chaining.
//   P_i = AES_K^{-1}(C_i) ^ C_{i-1}
// ---------------------------------------------------------------------------

rnet::Result<std::vector<uint8_t>> AES256CBC::decrypt(
    std::span<const uint8_t> key,
    std::span<const uint8_t> iv,
    std::span<const uint8_t> ciphertext) {

    // 1. Validate inputs.
    if (key.size() != KEY_SIZE) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "AES-256-CBC: key must be 32 bytes");
    }
    if (iv.size() != IV_SIZE) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "AES-256-CBC: IV must be 16 bytes");
    }
    if (ciphertext.empty() || ciphertext.size() % BLOCK_SIZE != 0) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "AES-256-CBC: ciphertext size must be multiple of 16");
    }

    // 2. Create cipher context.
    auto ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "Failed to create cipher context");
    }

    // 3. Decrypt and strip padding.
    std::vector<uint8_t> output(ciphertext.size() + BLOCK_SIZE);
    int out_len = 0;

    bool ok = true;
    ok = ok && EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(),
        nullptr, key.data(), iv.data());
    ok = ok && EVP_DecryptUpdate(ctx, output.data(), &out_len,
        ciphertext.data(), static_cast<int>(ciphertext.size()));
    int total = out_len;
    ok = ok && EVP_DecryptFinal_ex(ctx,
        output.data() + total, &out_len);
    total += out_len;

    EVP_CIPHER_CTX_free(ctx);

    if (!ok) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "AES-256-CBC decryption failed (bad key or corrupted data)");
    }

    output.resize(static_cast<size_t>(total));
    return rnet::Result<std::vector<uint8_t>>::ok(std::move(output));
}

// ---------------------------------------------------------------------------
// AES256CBC::encrypt_with_random_iv
//
// Generate a random 16-byte IV, encrypt, and prepend the IV to the output.
// Output format: [16-byte IV][ciphertext]
// ---------------------------------------------------------------------------

rnet::Result<std::vector<uint8_t>> AES256CBC::encrypt_with_random_iv(
    std::span<const uint8_t> key,
    std::span<const uint8_t> plaintext) {

    // 1. Generate random IV.
    std::array<uint8_t, IV_SIZE> iv;
    rnet::core::get_rand_bytes(iv);

    // 2. Encrypt with that IV.
    auto result = encrypt(key, iv, plaintext);
    if (result.is_err()) return result;

    // 3. Prepend IV to ciphertext.
    auto& ct = result.value();
    std::vector<uint8_t> output;
    output.reserve(IV_SIZE + ct.size());
    output.insert(output.end(), iv.begin(), iv.end());
    output.insert(output.end(), ct.begin(), ct.end());

    return rnet::Result<std::vector<uint8_t>>::ok(std::move(output));
}

// ---------------------------------------------------------------------------
// AES256CBC::decrypt_with_iv_prefix
//
// Split the first 16 bytes as IV, decrypt the remainder.
// ---------------------------------------------------------------------------

rnet::Result<std::vector<uint8_t>> AES256CBC::decrypt_with_iv_prefix(
    std::span<const uint8_t> key,
    std::span<const uint8_t> iv_and_ciphertext) {

    if (iv_and_ciphertext.size() < IV_SIZE + BLOCK_SIZE) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "AES-256-CBC: input too short (need IV + at least 1 block)");
    }

    auto iv = iv_and_ciphertext.subspan(0, IV_SIZE);
    auto ct = iv_and_ciphertext.subspan(IV_SIZE);

    return decrypt(key, iv, ct);
}

} // namespace rnet::crypto

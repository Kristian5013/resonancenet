#include "crypto/aes.h"
#include "core/random.h"

#include <cstring>
#include <openssl/evp.h>

namespace rnet::crypto {

rnet::Result<std::vector<uint8_t>> AES256CBC::encrypt(
    std::span<const uint8_t> key,
    std::span<const uint8_t> iv,
    std::span<const uint8_t> plaintext) {

    if (key.size() != KEY_SIZE) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "AES-256-CBC: key must be 32 bytes");
    }
    if (iv.size() != IV_SIZE) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "AES-256-CBC: IV must be 16 bytes");
    }

    auto ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "Failed to create cipher context");
    }

    // Output can be up to plaintext.size() + BLOCK_SIZE (padding)
    std::vector<uint8_t> output(plaintext.size() + BLOCK_SIZE);
    int out_len = 0;

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

rnet::Result<std::vector<uint8_t>> AES256CBC::decrypt(
    std::span<const uint8_t> key,
    std::span<const uint8_t> iv,
    std::span<const uint8_t> ciphertext) {

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

    auto ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "Failed to create cipher context");
    }

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

rnet::Result<std::vector<uint8_t>> AES256CBC::encrypt_with_random_iv(
    std::span<const uint8_t> key,
    std::span<const uint8_t> plaintext) {

    std::array<uint8_t, IV_SIZE> iv;
    rnet::core::get_rand_bytes(iv);

    auto result = encrypt(key, iv, plaintext);
    if (result.is_err()) return result;

    // Prepend IV to ciphertext
    auto& ct = result.value();
    std::vector<uint8_t> output;
    output.reserve(IV_SIZE + ct.size());
    output.insert(output.end(), iv.begin(), iv.end());
    output.insert(output.end(), ct.begin(), ct.end());

    return rnet::Result<std::vector<uint8_t>>::ok(std::move(output));
}

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

}  // namespace rnet::crypto

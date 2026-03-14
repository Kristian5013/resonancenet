// Copyright (c) 2024-2026 The ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "crypto/chacha20.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include <openssl/evp.h>

#ifdef _WIN32
#define NOMINMAX
#endif

namespace rnet::crypto {

// ---------------------------------------------------------------------------
// ChaCha20 quarter-round
//
// The core mixing function applied 80 times per block (4 per double-round,
// 10 double-rounds = 20 rounds total).
//
//   a += b;  d ^= a;  d <<<= 16;
//   c += d;  b ^= c;  b <<<= 12;
//   a += b;  d ^= a;  d <<=   8;
//   c += d;  b ^= c;  b <<=   7;
//
// Each step is an ARX (add-rotate-xor) operation -- no S-boxes, no table
// lookups, constant-time on all architectures.
// ---------------------------------------------------------------------------

inline void ChaCha20::quarter_round(uint32_t& a, uint32_t& b,
                                     uint32_t& c, uint32_t& d) {
    a += b; d ^= a; d = (d << 16) | (d >> 16);
    c += d; b ^= c; b = (b << 12) | (b >> 20);
    a += b; d ^= a; d = (d <<  8) | (d >> 24);
    c += d; b ^= c; b = (b <<  7) | (b >> 25);
}

// ---------------------------------------------------------------------------
// Little-endian load/store helpers
// ---------------------------------------------------------------------------

static uint32_t load32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0])
         | (static_cast<uint32_t>(p[1]) << 8)
         | (static_cast<uint32_t>(p[2]) << 16)
         | (static_cast<uint32_t>(p[3]) << 24);
}

static void store32_le(uint8_t* p, uint32_t v) {
    p[0] = static_cast<uint8_t>(v);
    p[1] = static_cast<uint8_t>(v >> 8);
    p[2] = static_cast<uint8_t>(v >> 16);
    p[3] = static_cast<uint8_t>(v >> 24);
}

// ---------------------------------------------------------------------------
// ChaCha20 constructor / set_key
//
// Initial state layout (16 x 32-bit words):
//
//   [ "expa"  "nd 3"  "2-by"  "te k" ]   constants
//   [  key0    key1    key2    key3   ]   key words 0-3
//   [  key4    key5    key6    key7   ]   key words 4-7
//   [ counter nonce0  nonce1  nonce2  ]   counter + nonce
// ---------------------------------------------------------------------------

ChaCha20::ChaCha20(std::span<const uint8_t, KEY_SIZE> key,
                   std::span<const uint8_t, NONCE_SIZE> nonce) {
    set_key(key, nonce);
}

void ChaCha20::set_key(std::span<const uint8_t, KEY_SIZE> key,
                       std::span<const uint8_t, NONCE_SIZE> nonce) {
    // 1. "expand 32-byte k" magic constants.
    state_[0] = 0x61707865;
    state_[1] = 0x3320646e;
    state_[2] = 0x79622d32;
    state_[3] = 0x6b206574;

    // 2. Load 256-bit key as 8 little-endian words.
    for (int i = 0; i < 8; ++i) {
        state_[4 + i] = load32_le(key.data() + i * 4);
    }

    // 3. Block counter starts at 0; load 96-bit nonce.
    state_[12] = 0;
    for (int i = 0; i < 3; ++i) {
        state_[13 + i] = load32_le(nonce.data() + i * 4);
    }

    // 4. Force block generation on next crypt/keystream call.
    buffer_pos_ = BLOCK_SIZE;
}

// ---------------------------------------------------------------------------
// ChaCha20::seek
//
// Jump to an arbitrary block counter position.
// ---------------------------------------------------------------------------

void ChaCha20::seek(uint32_t block_counter) {
    state_[12] = block_counter;
    buffer_pos_ = BLOCK_SIZE;
}

// ---------------------------------------------------------------------------
// ChaCha20::generate_block
//
// Run 20 rounds (10 double-rounds) of column + diagonal quarter-rounds,
// then add the original state and serialize to the keystream buffer.
//
// Double-round structure:
//   Column rounds:   QR(0,4,8,12)  QR(1,5,9,13)  QR(2,6,10,14)  QR(3,7,11,15)
//   Diagonal rounds: QR(0,5,10,15) QR(1,6,11,12) QR(2,7,8,13)   QR(3,4,9,14)
// ---------------------------------------------------------------------------

void ChaCha20::generate_block() {
    uint32_t x[16];
    std::memcpy(x, state_.data(), sizeof(x));

    // 1. 20 rounds (10 double-rounds).
    for (int i = 0; i < 10; ++i) {
        // Column rounds.
        quarter_round(x[0], x[4], x[ 8], x[12]);
        quarter_round(x[1], x[5], x[ 9], x[13]);
        quarter_round(x[2], x[6], x[10], x[14]);
        quarter_round(x[3], x[7], x[11], x[15]);
        // Diagonal rounds.
        quarter_round(x[0], x[5], x[10], x[15]);
        quarter_round(x[1], x[6], x[11], x[12]);
        quarter_round(x[2], x[7], x[ 8], x[13]);
        quarter_round(x[3], x[4], x[ 9], x[14]);
    }

    // 2. Add original state and serialize as little-endian.
    for (int i = 0; i < 16; ++i) {
        x[i] += state_[i];
        store32_le(buffer_.data() + i * 4, x[i]);
    }

    // 3. Increment block counter.
    state_[12]++;
    buffer_pos_ = 0;
}

// ---------------------------------------------------------------------------
// ChaCha20::crypt
//
// XOR data in-place with the keystream (encrypt == decrypt).
// ---------------------------------------------------------------------------

void ChaCha20::crypt(std::span<uint8_t> data) {
    for (size_t i = 0; i < data.size(); ++i) {
        if (buffer_pos_ >= BLOCK_SIZE) {
            generate_block();
        }
        data[i] ^= buffer_[buffer_pos_++];
    }
}

// ---------------------------------------------------------------------------
// ChaCha20::keystream
//
// Write raw keystream bytes to output buffer.
// ---------------------------------------------------------------------------

void ChaCha20::keystream(std::span<uint8_t> out) {
    for (size_t i = 0; i < out.size(); ++i) {
        if (buffer_pos_ >= BLOCK_SIZE) {
            generate_block();
        }
        out[i] = buffer_[buffer_pos_++];
    }
}

// ---------------------------------------------------------------------------
// ChaCha20-Poly1305 AEAD  (via OpenSSL EVP)
//
// Authenticated encryption with associated data.
//   encrypt: plaintext + AAD --> ciphertext || 16-byte tag
//   decrypt: ciphertext || tag + AAD --> plaintext (or auth failure)
// ---------------------------------------------------------------------------

rnet::Result<std::vector<uint8_t>> ChaCha20Poly1305::encrypt(
    std::span<const uint8_t> key,
    std::span<const uint8_t> nonce,
    std::span<const uint8_t> aad,
    std::span<const uint8_t> plaintext) {

    // 1. Validate key and nonce sizes.
    if (key.size() != KEY_SIZE) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "ChaCha20Poly1305: key must be 32 bytes");
    }
    if (nonce.size() != NONCE_SIZE) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "ChaCha20Poly1305: nonce must be 12 bytes");
    }

    // 2. Create cipher context.
    auto ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "Failed to create cipher context");
    }

    std::vector<uint8_t> output(plaintext.size() + TAG_SIZE);
    int out_len = 0;

    // 3. Initialize, process AAD, encrypt, finalize, extract tag.
    bool ok = true;
    ok = ok && EVP_EncryptInit_ex(ctx,
        EVP_chacha20_poly1305(), nullptr, nullptr, nullptr);
    ok = ok && EVP_CIPHER_CTX_ctrl(ctx,
        EVP_CTRL_AEAD_SET_IVLEN,
        static_cast<int>(NONCE_SIZE), nullptr);
    ok = ok && EVP_EncryptInit_ex(ctx, nullptr, nullptr,
        key.data(), nonce.data());

    if (!aad.empty()) {
        ok = ok && EVP_EncryptUpdate(ctx, nullptr, &out_len,
            aad.data(), static_cast<int>(aad.size()));
    }

    ok = ok && EVP_EncryptUpdate(ctx, output.data(), &out_len,
        plaintext.data(), static_cast<int>(plaintext.size()));
    int total = out_len;

    ok = ok && EVP_EncryptFinal_ex(ctx,
        output.data() + total, &out_len);
    total += out_len;

    ok = ok && EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_GET_TAG,
        TAG_SIZE, output.data() + plaintext.size());

    EVP_CIPHER_CTX_free(ctx);

    if (!ok) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "ChaCha20Poly1305 encryption failed");
    }

    return rnet::Result<std::vector<uint8_t>>::ok(std::move(output));
}

// ---------------------------------------------------------------------------
// ChaCha20Poly1305::decrypt
//
// Authenticated decryption.  Splits input into ciphertext and tag,
// verifies the Poly1305 MAC, and returns plaintext on success.
// ---------------------------------------------------------------------------

rnet::Result<std::vector<uint8_t>> ChaCha20Poly1305::decrypt(
    std::span<const uint8_t> key,
    std::span<const uint8_t> nonce,
    std::span<const uint8_t> aad,
    std::span<const uint8_t> ciphertext_with_tag) {

    // 1. Validate inputs.
    if (key.size() != KEY_SIZE) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "ChaCha20Poly1305: key must be 32 bytes");
    }
    if (ciphertext_with_tag.size() < TAG_SIZE) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "ChaCha20Poly1305: input too short");
    }

    // 2. Split ciphertext and tag.
    size_t ct_len = ciphertext_with_tag.size() - TAG_SIZE;
    auto ct = ciphertext_with_tag.subspan(0, ct_len);
    auto tag = ciphertext_with_tag.subspan(ct_len, TAG_SIZE);

    // 3. Create cipher context.
    auto ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "Failed to create cipher context");
    }

    std::vector<uint8_t> output(ct_len);
    int out_len = 0;

    // 4. Initialize, process AAD, decrypt, set tag, finalize.
    bool ok = true;
    ok = ok && EVP_DecryptInit_ex(ctx,
        EVP_chacha20_poly1305(), nullptr, nullptr, nullptr);
    ok = ok && EVP_CIPHER_CTX_ctrl(ctx,
        EVP_CTRL_AEAD_SET_IVLEN,
        static_cast<int>(NONCE_SIZE), nullptr);
    ok = ok && EVP_DecryptInit_ex(ctx, nullptr, nullptr,
        key.data(), nonce.data());

    if (!aad.empty()) {
        ok = ok && EVP_DecryptUpdate(ctx, nullptr, &out_len,
            aad.data(), static_cast<int>(aad.size()));
    }

    ok = ok && EVP_DecryptUpdate(ctx, output.data(), &out_len,
        ct.data(), static_cast<int>(ct.size()));

    ok = ok && EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_SET_TAG,
        TAG_SIZE,
        const_cast<uint8_t*>(tag.data()));

    int final_len = 0;
    ok = ok && EVP_DecryptFinal_ex(ctx,
        output.data() + out_len, &final_len);

    EVP_CIPHER_CTX_free(ctx);

    if (!ok) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "ChaCha20Poly1305 decryption failed (auth mismatch?)");
    }

    return rnet::Result<std::vector<uint8_t>>::ok(std::move(output));
}

} // namespace rnet::crypto

// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "crypto/bip39.h"

#include "crypto/bip39_wordlist.h"
#include "crypto/keccak.h"
#include "core/random.h"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <openssl/evp.h>

// ---------------------------------------------------------------------------
//  BIP-39  Mnemonic Code for Generating Deterministic Keys
//
//  Entropy -> Mnemonic:
//
//      entropy (ENT bits)
//          |
//          +--SHA-256--> checksum (CS bits = ENT / 32)
//          |
//      ENT || CS  (total bits always divisible by 11)
//          |
//      split into 11-bit groups --> word indices into 2048-word list
//
//      Word counts by entropy size:
//          128 bit -> 12 words    (CS = 4 bits)
//          160 bit -> 15 words    (CS = 5 bits)
//          192 bit -> 18 words    (CS = 6 bits)
//          224 bit -> 21 words    (CS = 7 bits)
//          256 bit -> 24 words    (CS = 8 bits)
//
//  Mnemonic -> Seed:
//
//      PBKDF2-HMAC-SHA512(
//          password   = mnemonic (UTF-8 NFKD),
//          salt       = "mnemonic" + passphrase (UTF-8 NFKD),
//          iterations = 2048,
//          dkLen      = 64 bytes
//      )  -->  512-bit seed  -->  BIP-32 master key
// ---------------------------------------------------------------------------

namespace rnet::crypto {

// ---------------------------------------------------------------------------
// bip39_wordlist
// ---------------------------------------------------------------------------

const std::array<const char*, 2048>& bip39_wordlist() {
    return BIP39_ENGLISH_WORDS;
}

// ---------------------------------------------------------------------------
// bip39_word_index
// ---------------------------------------------------------------------------

int bip39_word_index(std::string_view word) {
    const auto& wl = BIP39_ENGLISH_WORDS;
    for (int i = 0; i < 2048; ++i) {
        if (word == wl[static_cast<size_t>(i)]) return i;
    }
    return -1;
}

// ---------------------------------------------------------------------------
// sha256  (internal helper for BIP-39 checksum)
// ---------------------------------------------------------------------------

static std::array<uint8_t, 32> sha256(std::span<const uint8_t> data) {
    std::array<uint8_t, 32> hash;

    // 1. Create and run an EVP SHA-256 digest.
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);
    EVP_DigestUpdate(ctx, data.data(), data.size());
    unsigned int len = 32;
    EVP_DigestFinal_ex(ctx, hash.data(), &len);
    EVP_MD_CTX_free(ctx);

    return hash;
}

// ---------------------------------------------------------------------------
// entropy_to_mnemonic
// ---------------------------------------------------------------------------

rnet::Result<std::string> entropy_to_mnemonic(
    std::span<const uint8_t> entropy) {

    // 1. Validate entropy length (must be 128-256 bits, multiple of 32).
    size_t ent_bits = entropy.size() * 8;
    if (ent_bits < 128 || ent_bits > 256 || ent_bits % 32 != 0) {
        return rnet::Result<std::string>::err(
            "Invalid entropy size (must be 128-256 bits, multiple of 32)");
    }

    // 2. Checksum = first (ENT / 32) bits of SHA-256(entropy).
    //
    //        CS = ENT / 32
    //        total_bits = ENT + CS = ENT * 33 / 32
    //        n_words    = total_bits / 11
    //
    auto hash = sha256(entropy);
    size_t cs_bits = ent_bits / 32;

    // 3. Concatenate entropy bytes + first checksum byte.
    std::vector<uint8_t> bits;
    bits.reserve(entropy.size() + 1);
    bits.insert(bits.end(), entropy.begin(), entropy.end());
    bits.push_back(hash[0]);  // at most 8 checksum bits needed

    size_t total_bits = ent_bits + cs_bits;
    size_t n_words = total_bits / 11;

    // 4. Extract 11-bit groups and map each to a wordlist entry.
    std::vector<std::string> words;
    words.reserve(n_words);

    for (size_t i = 0; i < n_words; ++i) {
        size_t bit_start = i * 11;
        uint32_t idx = 0;
        for (size_t b = 0; b < 11; ++b) {
            size_t bit_pos = bit_start + b;
            size_t byte_pos = bit_pos / 8;
            size_t bit_off = 7 - (bit_pos % 8);
            if (byte_pos < bits.size()) {
                idx = (idx << 1) |
                    ((bits[byte_pos] >> bit_off) & 1);
            } else {
                idx <<= 1;
            }
        }
        if (idx >= 2048) idx = 0;
        words.push_back(BIP39_ENGLISH_WORDS[idx]);
    }

    // 5. Join words with single space separators.
    std::string result;
    for (size_t i = 0; i < words.size(); ++i) {
        if (i > 0) result += ' ';
        result += words[i];
    }

    return rnet::Result<std::string>::ok(std::move(result));
}

// ---------------------------------------------------------------------------
// generate_mnemonic
// ---------------------------------------------------------------------------

rnet::Result<std::string> generate_mnemonic(size_t word_count) {

    // 1. Map word count to entropy byte length.
    //
    //    words | entropy bytes | entropy bits | CS bits
    //    ------+---------------+--------------+--------
    //      12  |      16       |     128      |    4
    //      15  |      20       |     160      |    5
    //      18  |      24       |     192      |    6
    //      21  |      28       |     224      |    7
    //      24  |      32       |     256      |    8
    //
    size_t ent_bytes;
    switch (word_count) {
        case 12: ent_bytes = 16; break;
        case 15: ent_bytes = 20; break;
        case 18: ent_bytes = 24; break;
        case 21: ent_bytes = 28; break;
        case 24: ent_bytes = 32; break;
        default:
            return rnet::Result<std::string>::err(
                "Invalid word count (must be 12, 15, 18, 21, or 24)");
    }

    // 2. Fill entropy with cryptographically secure random bytes.
    std::vector<uint8_t> entropy(ent_bytes);
    rnet::core::get_rand_bytes(entropy);

    // 3. Convert entropy to mnemonic sentence.
    return entropy_to_mnemonic(entropy);
}

// ---------------------------------------------------------------------------
// mnemonic_to_entropy
// ---------------------------------------------------------------------------

rnet::Result<std::vector<uint8_t>> mnemonic_to_entropy(
    std::string_view mnemonic) {

    // 1. Split mnemonic string into individual words.
    std::vector<std::string_view> words;
    size_t start = 0;
    while (start < mnemonic.size()) {
        auto space = mnemonic.find(' ', start);
        if (space == std::string_view::npos) {
            words.push_back(mnemonic.substr(start));
            break;
        }
        words.push_back(mnemonic.substr(start, space - start));
        start = space + 1;
    }

    // 2. Validate word count (must be 12, 15, 18, 21, or 24).
    size_t n_words = words.size();
    if (n_words != 12 && n_words != 15 && n_words != 18 &&
        n_words != 21 && n_words != 24) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "Invalid mnemonic word count");
    }

    // 3. Look up each word in the BIP-39 wordlist.
    std::vector<uint32_t> indices;
    indices.reserve(n_words);
    for (const auto& word : words) {
        int idx = bip39_word_index(word);
        if (idx < 0) {
            return rnet::Result<std::vector<uint8_t>>::err(
                "Unknown BIP39 word: " + std::string(word));
        }
        indices.push_back(static_cast<uint32_t>(idx));
    }

    // 4. Reconstruct the bit stream from 11-bit word indices.
    //
    //    total_bits = n_words * 11
    //    ent_bits   = total_bits * 32 / 33
    //    cs_bits    = total_bits - ent_bits
    //
    size_t total_bits = n_words * 11;
    size_t ent_bits = (total_bits * 32) / 33;
    size_t cs_bits = total_bits - ent_bits;
    size_t ent_bytes = ent_bits / 8;

    std::vector<uint8_t> all_bytes((total_bits + 7) / 8, 0);
    for (size_t i = 0; i < n_words; ++i) {
        uint32_t idx = indices[i];
        for (size_t b = 0; b < 11; ++b) {
            size_t bit_pos = i * 11 + b;
            if (idx & (1 << (10 - b))) {
                all_bytes[bit_pos / 8] |=
                    (1 << (7 - (bit_pos % 8)));
            }
        }
    }

    // 5. Extract the entropy bytes (first ent_bytes of the bit stream).
    std::vector<uint8_t> entropy(all_bytes.begin(),
        all_bytes.begin() + static_cast<ptrdiff_t>(ent_bytes));

    // 6. Verify checksum: first cs_bits of SHA-256(entropy) must match.
    auto hash = sha256(entropy);
    uint8_t cs_mask = static_cast<uint8_t>(
        0xFF << (8 - cs_bits));
    uint8_t expected = hash[0] & cs_mask;
    uint8_t actual = all_bytes[ent_bytes] & cs_mask;

    if (expected != actual) {
        return rnet::Result<std::vector<uint8_t>>::err(
            "BIP39 checksum mismatch");
    }

    return rnet::Result<std::vector<uint8_t>>::ok(
        std::move(entropy));
}

// ---------------------------------------------------------------------------
// validate_mnemonic
// ---------------------------------------------------------------------------

bool validate_mnemonic(std::string_view mnemonic) {
    auto result = mnemonic_to_entropy(mnemonic);
    return result.is_ok();
}

// ---------------------------------------------------------------------------
// mnemonic_to_seed
// ---------------------------------------------------------------------------

rnet::Result<std::array<uint8_t, 64>> mnemonic_to_seed(
    std::string_view mnemonic,
    std::string_view passphrase) {

    // 1. Validate the mnemonic (word lookup + checksum).
    if (!validate_mnemonic(mnemonic)) {
        return rnet::Result<std::array<uint8_t, 64>>::err(
            "Invalid mnemonic");
    }

    // 2. Derive seed via PBKDF2-HMAC-SHA512.
    //
    //    password   = mnemonic        (UTF-8 NFKD)
    //    salt       = "mnemonic" + passphrase  (UTF-8 NFKD)
    //    iterations = 2048
    //    dkLen      = 64 bytes (512 bits)
    //
    std::string salt_str = "mnemonic";
    salt_str += passphrase;

    std::array<uint8_t, 64> seed;
    int rc = PKCS5_PBKDF2_HMAC(
        mnemonic.data(), static_cast<int>(mnemonic.size()),
        reinterpret_cast<const unsigned char*>(salt_str.data()),
        static_cast<int>(salt_str.size()),
        2048,
        EVP_sha512(),
        64,
        seed.data());

    if (rc != 1) {
        return rnet::Result<std::array<uint8_t, 64>>::err(
            "PBKDF2 failed");
    }

    return rnet::Result<std::array<uint8_t, 64>>::ok(seed);
}

} // namespace rnet::crypto

#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/error.h"

namespace rnet::crypto {

/// Generate a BIP39 mnemonic (24 words = 256 bits entropy)
rnet::Result<std::string> generate_mnemonic(size_t word_count = 24);

/// Convert mnemonic + optional passphrase to 512-bit seed
/// Uses PBKDF2-HMAC-SHA512 with "mnemonic" + passphrase as salt
rnet::Result<std::array<uint8_t, 64>> mnemonic_to_seed(
    std::string_view mnemonic,
    std::string_view passphrase = "");

/// Validate a mnemonic (checksum + word validity)
bool validate_mnemonic(std::string_view mnemonic);

/// Get the BIP39 wordlist (2048 English words)
const std::array<const char*, 2048>& bip39_wordlist();

/// Lookup word index in wordlist (-1 if not found)
int bip39_word_index(std::string_view word);

/// Convert entropy bytes to mnemonic words
rnet::Result<std::string> entropy_to_mnemonic(
    std::span<const uint8_t> entropy);

/// Convert mnemonic words back to entropy bytes
rnet::Result<std::vector<uint8_t>> mnemonic_to_entropy(
    std::string_view mnemonic);

}  // namespace rnet::crypto

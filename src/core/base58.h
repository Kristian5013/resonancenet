#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <optional>

namespace rnet::core {

/// Base58 alphabet (Bitcoin-compatible)
inline constexpr char BASE58_ALPHABET[] =
    "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

/// Encode raw bytes to Base58
std::string base58_encode(std::span<const uint8_t> data);

/// Decode Base58 string to raw bytes
std::optional<std::vector<uint8_t>> base58_decode(std::string_view str);

/// Encode with 4-byte checksum appended (Base58Check)
/// Checksum = first 4 bytes of double-hash (caller provides hash fn)
/// For simplicity, uses a basic checksum here; crypto module overrides.
using ChecksumFn = void(*)(const uint8_t* data, size_t len,
                           uint8_t out[32]);

std::string base58check_encode(std::span<const uint8_t> payload,
                               ChecksumFn hash_fn);

std::optional<std::vector<uint8_t>> base58check_decode(
    std::string_view str, ChecksumFn hash_fn);

/// Simple checksum (non-crypto, for testing without crypto dependency).
/// Real usage should pass Keccak256d from rnet_crypto.
void simple_checksum(const uint8_t* data, size_t len,
                     uint8_t out[32]);

/// Convenience: Base58Check with simple built-in checksum
std::string base58check_encode_simple(std::span<const uint8_t> payload);
std::optional<std::vector<uint8_t>> base58check_decode_simple(
    std::string_view str);

/// Validate that a string contains only valid Base58 characters
bool is_valid_base58(std::string_view str);

/// Get the Base58 character at a given index (0-57)
char base58_char_at(int index);

/// Get the index of a Base58 character (-1 if invalid)
int base58_char_index(char c);

}  // namespace rnet::core

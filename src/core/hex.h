#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <optional>

namespace rnet::core {

/// Convert bytes to lowercase hex string
std::string to_hex(std::span<const uint8_t> data);

/// Convert bytes to uppercase hex string
std::string to_hex_upper(std::span<const uint8_t> data);

/// Convert hex string to bytes. Returns empty vector on invalid input.
std::vector<uint8_t> from_hex(std::string_view hex);

/// Parse hex string, returning nullopt on invalid input
std::optional<std::vector<uint8_t>> parse_hex(std::string_view hex);

/// Check if string is valid hex (even length, all hex chars)
bool is_hex(std::string_view str);

/// Convert a single byte to two hex characters
void byte_to_hex(uint8_t byte, char& hi, char& lo);

/// Convert two hex characters to a byte. Returns -1 on invalid input.
int hex_to_byte(char hi, char lo);

/// Reverse a hex string (swap byte pairs, e.g. "aabb" -> "bbaa")
std::string reverse_hex(std::string_view hex);

// ─── String utilities ────────────────────────────────────────────────

/// Trim whitespace from both ends
std::string trim(std::string_view str);

/// Trim whitespace from left
std::string ltrim(std::string_view str);

/// Trim whitespace from right
std::string rtrim(std::string_view str);

/// Split string by delimiter
std::vector<std::string> split(std::string_view str, char delim);

/// Split string by string delimiter
std::vector<std::string> split(std::string_view str,
                               std::string_view delim);

/// Join strings with separator
std::string join(const std::vector<std::string>& parts,
                 std::string_view sep);

/// Convert string to lowercase
std::string to_lower(std::string_view str);

/// Convert string to uppercase
std::string to_upper(std::string_view str);

/// Check if string starts with prefix
bool starts_with(std::string_view str, std::string_view prefix);

/// Check if string ends with suffix
bool ends_with(std::string_view str, std::string_view suffix);

/// Replace all occurrences of 'from' with 'to' in str
std::string replace_all(std::string_view str, std::string_view from,
                        std::string_view to);

/// Format bytes as human-readable size (e.g., "1.5 MiB")
std::string format_bytes(uint64_t bytes);

/// Format a number with thousands separators
std::string format_number(int64_t number);

/// Parse a human-readable byte size (e.g., "100M", "2G")
/// Returns -1 on parse failure.
int64_t parse_byte_size(std::string_view str);

/// URL-encode a string
std::string url_encode(std::string_view str);

/// URL-decode a string
std::string url_decode(std::string_view str);

/// Sanitize a string for safe display (replace control chars)
std::string sanitize_string(std::string_view str,
                            size_t max_len = 256);

}  // namespace rnet::core

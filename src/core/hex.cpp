// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "core/hex.h"

#include <algorithm>
#include <cstdio>

namespace rnet::core {

// ===========================================================================
//  Lookup tables
// ===========================================================================

static constexpr char HEX_LOWER[] = "0123456789abcdef";
static constexpr char HEX_UPPER[] = "0123456789ABCDEF";

// ===========================================================================
//  Low-level hex helpers
// ===========================================================================

// ---------------------------------------------------------------------------
// hex_digit_val
//   Returns 0-15 for valid hex digits, -1 for anything else.
// ---------------------------------------------------------------------------
static int hex_digit_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// ---------------------------------------------------------------------------
// byte_to_hex
//   Splits a single byte into its two lowercase hex characters.
// ---------------------------------------------------------------------------
void byte_to_hex(uint8_t byte, char& hi, char& lo) {
    hi = HEX_LOWER[(byte >> 4) & 0x0F];
    lo = HEX_LOWER[byte & 0x0F];
}

// ---------------------------------------------------------------------------
// hex_to_byte
//   Combines two hex characters into a byte value (0-255), or -1 on error.
// ---------------------------------------------------------------------------
int hex_to_byte(char hi, char lo) {
    int h = hex_digit_val(hi);
    int l = hex_digit_val(lo);
    if (h < 0 || l < 0) return -1;
    return (h << 4) | l;
}

// ===========================================================================
//  Encode: bytes -> hex string
// ===========================================================================

// ---------------------------------------------------------------------------
// to_hex
//   Encodes a byte span as a lowercase hex string.
// ---------------------------------------------------------------------------
std::string to_hex(std::span<const uint8_t> data) {
    std::string result;
    result.reserve(data.size() * 2);
    for (auto b : data) {
        result.push_back(HEX_LOWER[(b >> 4) & 0x0F]);
        result.push_back(HEX_LOWER[b & 0x0F]);
    }
    return result;
}

// ---------------------------------------------------------------------------
// to_hex_upper
//   Encodes a byte span as an uppercase hex string.
// ---------------------------------------------------------------------------
std::string to_hex_upper(std::span<const uint8_t> data) {
    std::string result;
    result.reserve(data.size() * 2);
    for (auto b : data) {
        result.push_back(HEX_UPPER[(b >> 4) & 0x0F]);
        result.push_back(HEX_UPPER[b & 0x0F]);
    }
    return result;
}

// ===========================================================================
//  Decode: hex string -> bytes
// ===========================================================================

// ---------------------------------------------------------------------------
// from_hex
//   Decodes a hex string (with optional "0x" prefix) into bytes.
//   Returns an empty vector on odd length or invalid characters.
// ---------------------------------------------------------------------------
std::vector<uint8_t> from_hex(std::string_view hex) {
    // 1. Skip optional 0x prefix
    if (hex.size() >= 2 && hex[0] == '0' &&
        (hex[1] == 'x' || hex[1] == 'X')) {
        hex = hex.substr(2);
    }

    // 2. Reject odd-length input
    if (hex.size() % 2 != 0) return {};

    // 3. Decode pairs
    std::vector<uint8_t> result;
    result.reserve(hex.size() / 2);

    for (size_t i = 0; i < hex.size(); i += 2) {
        int byte = hex_to_byte(hex[i], hex[i + 1]);
        if (byte < 0) return {};
        result.push_back(static_cast<uint8_t>(byte));
    }
    return result;
}

// ---------------------------------------------------------------------------
// parse_hex
//   Like from_hex but returns nullopt on decode failure (rather than an
//   ambiguous empty vector).
// ---------------------------------------------------------------------------
std::optional<std::vector<uint8_t>> parse_hex(std::string_view hex) {
    auto result = from_hex(hex);
    // 1. Distinguish empty input from error
    if (result.empty() && !hex.empty() && hex != "0x" && hex != "0X") {
        if (hex.size() > 0) {
            // 2. Check if it was actually valid empty
            std::string_view check = hex;
            if (check.size() >= 2 && check[0] == '0' &&
                (check[1] == 'x' || check[1] == 'X')) {
                check = check.substr(2);
            }
            if (!check.empty()) {
                return std::nullopt;
            }
        }
    }
    return result;
}

// ===========================================================================
//  Hex validation and manipulation
// ===========================================================================

// ---------------------------------------------------------------------------
// is_hex
//   Returns true if str is a valid even-length hex string.
// ---------------------------------------------------------------------------
bool is_hex(std::string_view str) {
    if (str.size() % 2 != 0) return false;
    for (char c : str) {
        if (hex_digit_val(c) < 0) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// reverse_hex
//   Byte-reverses a hex string (swaps pairs).  Used for endian conversion
//   of hash displays.
// ---------------------------------------------------------------------------
std::string reverse_hex(std::string_view hex) {
    if (hex.size() % 2 != 0) return std::string(hex);
    std::string result;
    result.reserve(hex.size());
    for (size_t i = hex.size(); i >= 2; i -= 2) {
        result.push_back(hex[i - 2]);
        result.push_back(hex[i - 1]);
    }
    return result;
}

// ===========================================================================
//  String utilities -- trim / split / join
// ===========================================================================

// ---------------------------------------------------------------------------
// trim
//   Strips leading and trailing whitespace.
// ---------------------------------------------------------------------------
std::string trim(std::string_view str) {
    auto start = str.find_first_not_of(" \t\r\n");
    if (start == std::string_view::npos) return {};
    auto end = str.find_last_not_of(" \t\r\n");
    return std::string(str.substr(start, end - start + 1));
}

// ---------------------------------------------------------------------------
// ltrim
//   Strips leading whitespace only.
// ---------------------------------------------------------------------------
std::string ltrim(std::string_view str) {
    auto start = str.find_first_not_of(" \t\r\n");
    if (start == std::string_view::npos) return {};
    return std::string(str.substr(start));
}

// ---------------------------------------------------------------------------
// rtrim
//   Strips trailing whitespace only.
// ---------------------------------------------------------------------------
std::string rtrim(std::string_view str) {
    auto end = str.find_last_not_of(" \t\r\n");
    if (end == std::string_view::npos) return {};
    return std::string(str.substr(0, end + 1));
}

// ---------------------------------------------------------------------------
// split  (char delimiter)
//   Splits a string on a single character delimiter.  Trailing delimiters
//   produce an empty final element.
// ---------------------------------------------------------------------------
std::vector<std::string> split(std::string_view str, char delim) {
    std::vector<std::string> result;
    size_t start = 0;
    while (start < str.size()) {
        auto pos = str.find(delim, start);
        if (pos == std::string_view::npos) {
            result.emplace_back(str.substr(start));
            break;
        }
        result.emplace_back(str.substr(start, pos - start));
        start = pos + 1;
    }
    if (str.empty() || (!str.empty() && str.back() == delim)) {
        result.emplace_back();
    }
    return result;
}

// ---------------------------------------------------------------------------
// split  (string_view delimiter)
//   Splits on a multi-character delimiter.  Empty delimiter returns the
//   input as a single element.
// ---------------------------------------------------------------------------
std::vector<std::string> split(std::string_view str,
                               std::string_view delim) {
    std::vector<std::string> result;
    if (delim.empty()) {
        result.emplace_back(str);
        return result;
    }
    size_t start = 0;
    while (start < str.size()) {
        auto pos = str.find(delim, start);
        if (pos == std::string_view::npos) {
            result.emplace_back(str.substr(start));
            break;
        }
        result.emplace_back(str.substr(start, pos - start));
        start = pos + delim.size();
    }
    return result;
}

// ---------------------------------------------------------------------------
// join
//   Concatenates strings with a separator between each pair.
// ---------------------------------------------------------------------------
std::string join(const std::vector<std::string>& parts,
                 std::string_view sep) {
    if (parts.empty()) return {};
    std::string result = parts[0];
    for (size_t i = 1; i < parts.size(); ++i) {
        result.append(sep);
        result.append(parts[i]);
    }
    return result;
}

// ===========================================================================
//  String utilities -- case conversion
// ===========================================================================

// ---------------------------------------------------------------------------
// to_lower
//   ASCII-only lowercase conversion (no locale dependency).
// ---------------------------------------------------------------------------
std::string to_lower(std::string_view str) {
    std::string result;
    result.reserve(str.size());
    for (char c : str) {
        if (c >= 'A' && c <= 'Z') {
            result.push_back(static_cast<char>(c + ('a' - 'A')));
        } else {
            result.push_back(c);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// to_upper
//   ASCII-only uppercase conversion (no locale dependency).
// ---------------------------------------------------------------------------
std::string to_upper(std::string_view str) {
    std::string result;
    result.reserve(str.size());
    for (char c : str) {
        if (c >= 'a' && c <= 'z') {
            result.push_back(static_cast<char>(c - ('a' - 'A')));
        } else {
            result.push_back(c);
        }
    }
    return result;
}

// ===========================================================================
//  String utilities -- prefix / suffix / replace
// ===========================================================================

// ---------------------------------------------------------------------------
// starts_with
//   Returns true if str begins with prefix.
// ---------------------------------------------------------------------------
bool starts_with(std::string_view str, std::string_view prefix) {
    if (prefix.size() > str.size()) return false;
    return str.substr(0, prefix.size()) == prefix;
}

// ---------------------------------------------------------------------------
// ends_with
//   Returns true if str ends with suffix.
// ---------------------------------------------------------------------------
bool ends_with(std::string_view str, std::string_view suffix) {
    if (suffix.size() > str.size()) return false;
    return str.substr(str.size() - suffix.size()) == suffix;
}

// ---------------------------------------------------------------------------
// replace_all
//   Replaces every occurrence of `from` with `to` in the input string.
//   Empty `from` returns the input unchanged.
// ---------------------------------------------------------------------------
std::string replace_all(std::string_view str, std::string_view from,
                        std::string_view to) {
    if (from.empty()) return std::string(str);
    std::string result;
    result.reserve(str.size());
    size_t pos = 0;
    while (pos < str.size()) {
        auto found = str.find(from, pos);
        if (found == std::string_view::npos) {
            result.append(str.substr(pos));
            break;
        }
        result.append(str.substr(pos, found - pos));
        result.append(to);
        pos = found + from.size();
    }
    return result;
}

// ===========================================================================
//  Formatting utilities
// ===========================================================================

// ---------------------------------------------------------------------------
// format_bytes
//   Converts a raw byte count into a human-readable string with binary
//   unit suffixes (KiB, MiB, GiB, ...).
// ---------------------------------------------------------------------------
std::string format_bytes(uint64_t bytes) {
    static constexpr const char* units[] = {
        "B", "KiB", "MiB", "GiB", "TiB", "PiB"};
    int unit = 0;
    double val = static_cast<double>(bytes);
    while (val >= 1024.0 && unit < 5) {
        val /= 1024.0;
        ++unit;
    }
    char buf[64];
    if (unit == 0) {
        std::snprintf(buf, sizeof(buf), "%llu %s",
                      static_cast<unsigned long long>(bytes),
                      units[unit]);
    } else {
        std::snprintf(buf, sizeof(buf), "%.2f %s", val, units[unit]);
    }
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// format_number
//   Formats an integer with thousands separators (commas).
//   Handles negative values and the int64_t minimum correctly.
// ---------------------------------------------------------------------------
std::string format_number(int64_t number) {
    bool negative = number < 0;
    uint64_t abs_val = negative
        ? static_cast<uint64_t>(-(number + 1)) + 1
        : static_cast<uint64_t>(number);

    std::string digits = std::to_string(abs_val);
    std::string result;
    int count = 0;
    for (auto it = digits.rbegin(); it != digits.rend(); ++it) {
        if (count > 0 && count % 3 == 0) {
            result.push_back(',');
        }
        result.push_back(*it);
        ++count;
    }
    if (negative) result.push_back('-');
    std::reverse(result.begin(), result.end());
    return result;
}

// ---------------------------------------------------------------------------
// parse_byte_size
//   Parses a human-readable size string (e.g. "512M", "2.5GB") into bytes.
//   Returns -1 on invalid input.
// ---------------------------------------------------------------------------
int64_t parse_byte_size(std::string_view str) {
    if (str.empty()) return -1;

    // 1. Trim whitespace
    auto trimmed = trim(str);
    if (trimmed.empty()) return -1;

    // 2. Determine multiplier from suffix
    int64_t multiplier = 1;
    size_t num_end = trimmed.size();

    char suffix = trimmed.back();
    if (suffix == 'B' || suffix == 'b') {
        num_end--;
        if (num_end > 0) {
            suffix = trimmed[num_end - 1];
        } else {
            suffix = 0;
        }
    }

    switch (suffix) {
        case 'K': case 'k':
            multiplier = 1024;
            num_end--;
            break;
        case 'M': case 'm':
            multiplier = 1024LL * 1024;
            num_end--;
            break;
        case 'G': case 'g':
            multiplier = 1024LL * 1024 * 1024;
            num_end--;
            break;
        case 'T': case 't':
            multiplier = 1024LL * 1024 * 1024 * 1024;
            num_end--;
            break;
        default:
            break;
    }

    // 3. Parse the numeric part
    auto num_str = trimmed.substr(0, num_end);
    if (num_str.empty()) return -1;

    try {
        double val = std::stod(std::string(num_str));
        return static_cast<int64_t>(val * static_cast<double>(
            multiplier));
    } catch (...) {
        return -1;
    }
}

// ===========================================================================
//  URL encoding / decoding
// ===========================================================================

// ---------------------------------------------------------------------------
// url_encode
//   Percent-encodes a string per RFC 3986.  Unreserved characters
//   (A-Z, a-z, 0-9, '-', '_', '.', '~') are passed through unchanged.
// ---------------------------------------------------------------------------
std::string url_encode(std::string_view str) {
    std::string result;
    result.reserve(str.size() * 3);
    for (unsigned char c : str) {
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_' ||
            c == '.' || c == '~') {
            result.push_back(static_cast<char>(c));
        } else {
            result.push_back('%');
            result.push_back(HEX_UPPER[(c >> 4) & 0x0F]);
            result.push_back(HEX_UPPER[c & 0x0F]);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// url_decode
//   Decodes percent-encoded sequences and '+' (as space).
// ---------------------------------------------------------------------------
std::string url_decode(std::string_view str) {
    std::string result;
    result.reserve(str.size());
    for (size_t i = 0; i < str.size(); ++i) {
        if (str[i] == '%' && i + 2 < str.size()) {
            int byte = hex_to_byte(str[i + 1], str[i + 2]);
            if (byte >= 0) {
                result.push_back(static_cast<char>(byte));
                i += 2;
                continue;
            }
        }
        if (str[i] == '+') {
            result.push_back(' ');
        } else {
            result.push_back(str[i]);
        }
    }
    return result;
}

// ===========================================================================
//  Sanitisation
// ===========================================================================

// ---------------------------------------------------------------------------
// sanitize_string
//   Replaces non-printable characters with escape sequences and truncates
//   at max_len.  Safe for logging untrusted input.
// ---------------------------------------------------------------------------
std::string sanitize_string(std::string_view str, size_t max_len) {
    std::string result;
    result.reserve(std::min(str.size(), max_len));
    for (size_t i = 0; i < str.size() && result.size() < max_len; ++i) {
        unsigned char c = static_cast<unsigned char>(str[i]);
        if (c >= 32 && c < 127) {
            result.push_back(static_cast<char>(c));
        } else if (c == '\n') {
            result.append("\\n");
        } else if (c == '\r') {
            result.append("\\r");
        } else if (c == '\t') {
            result.append("\\t");
        } else {
            // 1. Replace non-printable with hex escape
            char buf[5];
            std::snprintf(buf, sizeof(buf), "\\x%02x", c);
            result.append(buf);
        }
    }
    if (str.size() > max_len) {
        result.append("...");
    }
    return result;
}

} // namespace rnet::core

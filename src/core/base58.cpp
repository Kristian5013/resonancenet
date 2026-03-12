#include "core/base58.h"

#include <algorithm>
#include <cstring>
#include <numeric>

namespace rnet::core {

// Reverse lookup table: ASCII char -> base58 digit value (-1 = invalid)
static constexpr int8_t BASE58_MAP[256] = {
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1, 0, 1, 2, 3, 4, 5, 6, 7, 8,-1,-1,-1,-1,-1,-1, // '1'-'9'
    -1, 9,10,11,12,13,14,15,16,-1,17,18,19,20,21,-1,  // 'A'-'O'
    22,23,24,25,26,27,28,29,30,31,32,-1,-1,-1,-1,-1,  // 'P'-'Z'
    -1,33,34,35,36,37,38,39,40,41,42,43,-1,44,45,46,  // 'a'-'o'
    47,48,49,50,51,52,53,54,55,56,57,-1,-1,-1,-1,-1,  // 'p'-'z'
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
};

std::string base58_encode(std::span<const uint8_t> data) {
    // Count leading zeros
    size_t leading_zeros = 0;
    for (auto b : data) {
        if (b != 0) break;
        ++leading_zeros;
    }

    // Allocate enough space in big-endian base58 representation.
    // log(256) / log(58) ~= 1.3656..., so we need ~138% of input size
    size_t output_size = data.size() * 138 / 100 + 1;
    std::vector<uint8_t> b58(output_size, 0);

    for (auto byte : data) {
        int carry = static_cast<int>(byte);
        for (auto it = b58.rbegin(); it != b58.rend(); ++it) {
            carry += 256 * static_cast<int>(*it);
            *it = static_cast<uint8_t>(carry % 58);
            carry /= 58;
        }
    }

    // Skip leading zeros in b58
    auto it = b58.begin();
    while (it != b58.end() && *it == 0) {
        ++it;
    }

    // Build result: leading '1's + encoded digits
    std::string result;
    result.reserve(leading_zeros + std::distance(it, b58.end()));
    result.assign(leading_zeros, '1');
    for (; it != b58.end(); ++it) {
        result.push_back(BASE58_ALPHABET[*it]);
    }

    return result;
}

std::optional<std::vector<uint8_t>> base58_decode(std::string_view str) {
    if (str.empty()) return std::vector<uint8_t>{};

    // Count leading '1's (represent zero bytes)
    size_t leading_ones = 0;
    for (char c : str) {
        if (c != '1') break;
        ++leading_ones;
    }

    // Allocate enough space for binary result
    size_t output_size = str.size() * 733 / 1000 + 1;  // log(58)/log(256)
    std::vector<uint8_t> b256(output_size, 0);

    for (char c : str) {
        int digit = BASE58_MAP[static_cast<uint8_t>(c)];
        if (digit < 0) return std::nullopt;

        int carry = digit;
        for (auto it = b256.rbegin(); it != b256.rend(); ++it) {
            carry += 58 * static_cast<int>(*it);
            *it = static_cast<uint8_t>(carry % 256);
            carry /= 256;
        }

        if (carry != 0) return std::nullopt;
    }

    // Skip leading zeros in b256
    auto it = b256.begin();
    while (it != b256.end() && *it == 0) {
        ++it;
    }

    // Build result: leading zero bytes + decoded data
    std::vector<uint8_t> result;
    result.reserve(leading_ones + std::distance(it, b256.end()));
    result.assign(leading_ones, 0x00);
    result.insert(result.end(), it, b256.end());

    return result;
}

void simple_checksum(const uint8_t* data, size_t len,
                     uint8_t out[32]) {
    // Non-crypto double-hash for standalone testing.
    // FNV-1a variant, applied twice. NOT secure — placeholder only.
    auto fnv_hash = [](const uint8_t* d, size_t n,
                       uint8_t result[32]) {
        std::memset(result, 0, 32);
        uint64_t h = 14695981039346656037ULL;
        for (size_t i = 0; i < n; ++i) {
            h ^= static_cast<uint64_t>(d[i]);
            h *= 1099511628211ULL;
        }
        // Spread 64-bit hash across 32 bytes
        for (int i = 0; i < 4; ++i) {
            uint64_t val = h ^ (h >> (i * 7 + 3));
            val *= 0x9E3779B97F4A7C15ULL;
            for (int j = 0; j < 8; ++j) {
                result[i * 8 + j] =
                    static_cast<uint8_t>((val >> (j * 8)) & 0xFF);
            }
        }
    };

    uint8_t first[32];
    fnv_hash(data, len, first);
    fnv_hash(first, 32, out);
}

std::string base58check_encode(std::span<const uint8_t> payload,
                               ChecksumFn hash_fn) {
    uint8_t hash[32];
    hash_fn(payload.data(), payload.size(), hash);

    // Append first 4 bytes of hash as checksum
    std::vector<uint8_t> with_checksum(payload.begin(), payload.end());
    with_checksum.insert(with_checksum.end(), hash, hash + 4);

    return base58_encode(with_checksum);
}

std::optional<std::vector<uint8_t>> base58check_decode(
    std::string_view str, ChecksumFn hash_fn) {
    auto decoded = base58_decode(str);
    if (!decoded || decoded->size() < 4) {
        return std::nullopt;
    }

    size_t payload_len = decoded->size() - 4;
    uint8_t hash[32];
    hash_fn(decoded->data(), payload_len, hash);

    // Verify checksum
    if (std::memcmp(hash, decoded->data() + payload_len, 4) != 0) {
        return std::nullopt;
    }

    decoded->resize(payload_len);
    return decoded;
}

std::string base58check_encode_simple(
    std::span<const uint8_t> payload) {
    return base58check_encode(payload, simple_checksum);
}

std::optional<std::vector<uint8_t>> base58check_decode_simple(
    std::string_view str) {
    return base58check_decode(str, simple_checksum);
}

bool is_valid_base58(std::string_view str) {
    for (char c : str) {
        if (BASE58_MAP[static_cast<uint8_t>(c)] < 0) return false;
    }
    return true;
}

char base58_char_at(int index) {
    if (index < 0 || index > 57) return '?';
    return BASE58_ALPHABET[index];
}

int base58_char_index(char c) {
    return BASE58_MAP[static_cast<uint8_t>(c)];
}

}  // namespace rnet::core

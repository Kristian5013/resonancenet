// Hex encoding for hashes and wire dumps. Decoding is strict: any non-hex
// character or odd length is an error, never a silently truncated value.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "util/result.h"

namespace rnet::util {

std::string ToHex(std::span<const uint8_t> bytes);

template <size_t N>
std::string ToHex(const std::array<uint8_t, N>& bytes) {
    return ToHex(std::span<const uint8_t>(bytes.data(), bytes.size()));
}

Result<std::vector<uint8_t>> FromHex(std::string_view hex);

// Fixed-width variant used for the 32-byte hashes that appear in consensus
// objects; rejects anything that is not exactly N bytes of hex.
template <size_t N>
Result<std::array<uint8_t, N>> FromHexFixed(std::string_view hex) {
    auto bytes = FromHex(hex);
    if (!bytes) return Err(bytes.error());
    if (bytes.value().size() != N) {
        return Err("hex: expected " + std::to_string(N) + " bytes, got " +
                   std::to_string(bytes.value().size()));
    }
    std::array<uint8_t, N> out{};
    std::copy(bytes.value().begin(), bytes.value().end(), out.begin());
    return out;
}

}  // namespace rnet::util

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <optional>
#include <utility>

namespace rnet::core {

/// Bech32 encoding types per BIP-173 and BIP-350
enum class Bech32Encoding {
    BECH32,   // BIP-173 (SegWit v0)
    BECH32M,  // BIP-350 (SegWit v1+)
    INVALID
};

/// Result of decoding a bech32 string
struct Bech32DecodeResult {
    Bech32Encoding encoding = Bech32Encoding::INVALID;
    std::string hrp;                    // Human-readable part
    std::vector<uint8_t> data;          // 5-bit values (witness program)
};

/// Encode data as bech32/bech32m
/// hrp: human-readable part (e.g., "rn" for ResonanceNet)
/// values: 5-bit data values
std::string bech32_encode(std::string_view hrp,
                          const std::vector<uint8_t>& values,
                          Bech32Encoding encoding);

/// Decode a bech32/bech32m string
Bech32DecodeResult bech32_decode(std::string_view str);

/// Convert between bit groups (e.g., 8-bit to 5-bit and back)
/// frombits/tobits: number of bits per group
/// pad: whether to pad the last group
std::vector<uint8_t> convert_bits(std::span<const uint8_t> data,
                                  int frombits, int tobits, bool pad);

/// High-level: encode a segwit-style address
/// hrp: "rn" for mainnet, "trn" for testnet
/// witness_version: 0-16
/// witness_program: 20 or 32 bytes typically
std::string encode_segwit_addr(std::string_view hrp,
                               int witness_version,
                               std::span<const uint8_t> witness_prog);

/// High-level: decode a segwit-style address
struct SegwitAddrResult {
    bool valid = false;
    int witness_version = -1;
    std::vector<uint8_t> witness_program;
};

SegwitAddrResult decode_segwit_addr(std::string_view hrp,
                                    std::string_view addr);

/// Validate a bech32/bech32m string without fully decoding
bool is_valid_bech32(std::string_view str);

/// Get the HRP from a bech32 string without full decode
std::string get_bech32_hrp(std::string_view str);

/// Locate a potential bech32 error position (returns -1 if none found)
int locate_bech32_error(std::string_view str);

}  // namespace rnet::core

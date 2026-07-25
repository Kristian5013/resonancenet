#include "util/hex.h"

namespace rnet::util {

namespace {
constexpr char kDigits[] = "0123456789abcdef";

int NibbleValue(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
}  // namespace

std::string ToHex(std::span<const uint8_t> bytes) {
    std::string out;
    out.reserve(bytes.size() * 2);
    for (uint8_t b : bytes) {
        out.push_back(kDigits[b >> 4]);
        out.push_back(kDigits[b & 0x0F]);
    }
    return out;
}

Result<std::vector<uint8_t>> FromHex(std::string_view hex) {
    if (hex.size() % 2 != 0) return Err("hex: odd length");
    std::vector<uint8_t> out;
    out.reserve(hex.size() / 2);
    for (size_t i = 0; i < hex.size(); i += 2) {
        const int hi = NibbleValue(hex[i]);
        const int lo = NibbleValue(hex[i + 1]);
        if (hi < 0 || lo < 0) return Err("hex: invalid character");
        out.push_back(static_cast<uint8_t>((hi << 4) | lo));
    }
    return out;
}

}  // namespace rnet::util

#include "core/stream.h"

namespace rnet::core {

std::string DataStream::to_hex() const {
    static constexpr char hex_chars[] = "0123456789abcdef";
    std::string result;
    result.reserve(data_.size() * 2);
    for (auto b : data_) {
        result.push_back(hex_chars[(b >> 4) & 0x0F]);
        result.push_back(hex_chars[b & 0x0F]);
    }
    return result;
}

// Helper: create a DataStream from a hex string
DataStream DataStream_from_hex(std::string_view hex) {
    // Skip 0x prefix
    if (hex.size() >= 2 && hex[0] == '0' &&
        (hex[1] == 'x' || hex[1] == 'X')) {
        hex = hex.substr(2);
    }

    std::vector<uint8_t> data;
    data.reserve(hex.size() / 2);

    auto hex_val = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };

    for (size_t i = 0; i + 1 < hex.size(); i += 2) {
        int hi = hex_val(hex[i]);
        int lo = hex_val(hex[i + 1]);
        if (hi < 0 || lo < 0) break;
        data.push_back(static_cast<uint8_t>((hi << 4) | lo));
    }

    return DataStream(std::move(data));
}

}  // namespace rnet::core

#include "core/bech32.h"

#include <algorithm>
#include <cctype>
#include <span>

namespace rnet::core {

static constexpr char BECH32_CHARSET[] =
    "qpzry9x8gf2tvdw0s3jn54khce6mua7l";

static constexpr int8_t BECH32_CHARSET_REV[128] = {
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    15,-1,10,17,21,20,26,30, 7, 5,-1,-1,-1,-1,-1,-1,
    -1,29,-1,24,13,25, 9, 8,23,-1,18,22,31,27,19,-1,
     1, 0, 3,16,11,28,12,14, 6, 4, 2,-1,-1,-1,-1,-1,
    -1,29,-1,24,13,25, 9, 8,23,-1,18,22,31,27,19,-1,
     1, 0, 3,16,11,28,12,14, 6, 4, 2,-1,-1,-1,-1,-1,
};

static constexpr uint32_t BECH32_CONST  = 1;
static constexpr uint32_t BECH32M_CONST = 0x2bc830a3;

static uint32_t bech32_polymod(const std::vector<uint8_t>& values) {
    static constexpr uint32_t GEN[5] = {
        0x3b6a57b2, 0x26508e6d, 0x1ea119fa,
        0x3d4233dd, 0x2a1462b3
    };
    uint32_t chk = 1;
    for (auto v : values) {
        uint32_t top = chk >> 25;
        chk = ((chk & 0x1ffffff) << 5) ^ v;
        for (int i = 0; i < 5; ++i) {
            if ((top >> i) & 1) {
                chk ^= GEN[i];
            }
        }
    }
    return chk;
}

static std::vector<uint8_t> hrp_expand(std::string_view hrp) {
    std::vector<uint8_t> ret;
    ret.reserve(hrp.size() * 2 + 1);
    for (char c : hrp) {
        ret.push_back(static_cast<uint8_t>(c >> 5));
    }
    ret.push_back(0);
    for (char c : hrp) {
        ret.push_back(static_cast<uint8_t>(c & 0x1f));
    }
    return ret;
}

static bool verify_checksum(std::string_view hrp,
                             const std::vector<uint8_t>& data,
                             Bech32Encoding& enc) {
    auto exp = hrp_expand(hrp);
    exp.insert(exp.end(), data.begin(), data.end());
    uint32_t poly = bech32_polymod(exp);
    if (poly == BECH32_CONST) {
        enc = Bech32Encoding::BECH32;
        return true;
    }
    if (poly == BECH32M_CONST) {
        enc = Bech32Encoding::BECH32M;
        return true;
    }
    enc = Bech32Encoding::INVALID;
    return false;
}

static std::vector<uint8_t> create_checksum(
    std::string_view hrp,
    const std::vector<uint8_t>& values,
    Bech32Encoding encoding) {

    auto exp = hrp_expand(hrp);
    exp.insert(exp.end(), values.begin(), values.end());
    exp.resize(exp.size() + 6, 0);

    uint32_t target = (encoding == Bech32Encoding::BECH32M)
                          ? BECH32M_CONST
                          : BECH32_CONST;
    uint32_t poly = bech32_polymod(exp) ^ target;

    std::vector<uint8_t> ret(6);
    for (int i = 0; i < 6; ++i) {
        ret[i] = static_cast<uint8_t>((poly >> (5 * (5 - i))) & 31);
    }
    return ret;
}

std::string bech32_encode(std::string_view hrp,
                          const std::vector<uint8_t>& values,
                          Bech32Encoding encoding) {
    auto checksum = create_checksum(hrp, values, encoding);

    std::string result;
    result.reserve(hrp.size() + 1 + values.size() + 6);

    for (char c : hrp) {
        result.push_back(static_cast<char>(std::tolower(
            static_cast<unsigned char>(c))));
    }
    result.push_back('1');

    for (auto v : values) {
        result.push_back(BECH32_CHARSET[v]);
    }
    for (auto v : checksum) {
        result.push_back(BECH32_CHARSET[v]);
    }

    return result;
}

Bech32DecodeResult bech32_decode(std::string_view str) {
    Bech32DecodeResult result;

    if (str.size() < 8 || str.size() > 90) return result;

    // Check mixed case
    bool has_lower = false, has_upper = false;
    for (char c : str) {
        auto uc = static_cast<unsigned char>(c);
        if (uc < 33 || uc > 126) return result;
        if (c >= 'a' && c <= 'z') has_lower = true;
        if (c >= 'A' && c <= 'Z') has_upper = true;
    }
    if (has_lower && has_upper) return result;

    // Find separator '1'
    auto sep_pos = str.rfind('1');
    if (sep_pos == std::string_view::npos ||
        sep_pos < 1 || sep_pos + 7 > str.size()) {
        return result;
    }

    std::string hrp_lower;
    for (size_t i = 0; i < sep_pos; ++i) {
        hrp_lower.push_back(static_cast<char>(std::tolower(
            static_cast<unsigned char>(str[i]))));
    }

    std::vector<uint8_t> data;
    data.reserve(str.size() - sep_pos - 1);

    for (size_t i = sep_pos + 1; i < str.size(); ++i) {
        auto c = static_cast<unsigned char>(
            std::tolower(static_cast<unsigned char>(str[i])));
        if (c >= 128) return result;
        int8_t val = BECH32_CHARSET_REV[c];
        if (val < 0) return result;
        data.push_back(static_cast<uint8_t>(val));
    }

    Bech32Encoding enc;
    if (!verify_checksum(hrp_lower, data, enc)) {
        return result;
    }

    // Remove checksum (last 6 values)
    data.resize(data.size() - 6);

    result.encoding = enc;
    result.hrp = std::move(hrp_lower);
    result.data = std::move(data);
    return result;
}

std::vector<uint8_t> convert_bits(std::span<const uint8_t> data,
                                  int frombits, int tobits, bool pad) {
    int acc = 0;
    int bits = 0;
    int maxv = (1 << tobits) - 1;
    std::vector<uint8_t> ret;
    ret.reserve(data.size() * frombits / tobits + 1);

    for (auto value : data) {
        if (value < 0 || (value >> frombits) != 0) {
            return {};
        }
        acc = (acc << frombits) | value;
        bits += frombits;
        while (bits >= tobits) {
            bits -= tobits;
            ret.push_back(static_cast<uint8_t>((acc >> bits) & maxv));
        }
    }

    if (pad) {
        if (bits > 0) {
            ret.push_back(
                static_cast<uint8_t>((acc << (tobits - bits)) & maxv));
        }
    } else if (bits >= frombits ||
               ((acc << (tobits - bits)) & maxv) != 0) {
        return {};
    }

    return ret;
}

std::string encode_segwit_addr(std::string_view hrp,
                               int witness_version,
                               std::span<const uint8_t> witness_prog) {
    auto conv = convert_bits(witness_prog, 8, 5, true);
    if (conv.empty() && !witness_prog.empty()) return {};

    std::vector<uint8_t> data;
    data.reserve(1 + conv.size());
    data.push_back(static_cast<uint8_t>(witness_version));
    data.insert(data.end(), conv.begin(), conv.end());

    auto encoding = (witness_version == 0)
                        ? Bech32Encoding::BECH32
                        : Bech32Encoding::BECH32M;

    return bech32_encode(hrp, data, encoding);
}

SegwitAddrResult decode_segwit_addr(std::string_view hrp,
                                    std::string_view addr) {
    SegwitAddrResult result;

    auto dec = bech32_decode(addr);
    if (dec.encoding == Bech32Encoding::INVALID) return result;
    if (dec.hrp != hrp) return result;
    if (dec.data.empty()) return result;

    int witness_version = dec.data[0];
    if (witness_version > 16) return result;

    // Check encoding type matches witness version
    if (witness_version == 0 &&
        dec.encoding != Bech32Encoding::BECH32) {
        return result;
    }
    if (witness_version > 0 &&
        dec.encoding != Bech32Encoding::BECH32M) {
        return result;
    }

    auto prog = convert_bits(
        std::span<const uint8_t>(dec.data.data() + 1,
                                 dec.data.size() - 1),
        5, 8, false);
    if (prog.empty() && dec.data.size() > 1) return result;

    // BIP-141 constraints on witness program length
    if (prog.size() < 2 || prog.size() > 40) return result;
    if (witness_version == 0 &&
        prog.size() != 20 && prog.size() != 32) {
        return result;
    }

    result.valid = true;
    result.witness_version = witness_version;
    result.witness_program = std::move(prog);
    return result;
}

bool is_valid_bech32(std::string_view str) {
    auto result = bech32_decode(str);
    return result.encoding != Bech32Encoding::INVALID;
}

std::string get_bech32_hrp(std::string_view str) {
    auto sep = str.rfind('1');
    if (sep == std::string_view::npos || sep < 1) return {};
    std::string hrp;
    for (size_t i = 0; i < sep; ++i) {
        hrp.push_back(static_cast<char>(std::tolower(
            static_cast<unsigned char>(str[i]))));
    }
    return hrp;
}

int locate_bech32_error(std::string_view str) {
    // Simple approach: try decoding and if it fails, do a per-char
    // substitution to find which character is wrong.
    auto result = bech32_decode(str);
    if (result.encoding != Bech32Encoding::INVALID) return -1;

    // Find separator
    auto sep = str.rfind('1');
    if (sep == std::string_view::npos || sep < 1) return -1;

    // Check each data character after the separator
    std::string mutable_str(str);
    for (size_t i = sep + 1; i < mutable_str.size(); ++i) {
        char original = mutable_str[i];
        // Try all valid bech32 characters
        for (int j = 0; j < 32; ++j) {
            char replacement = BECH32_CHARSET[j];
            if (replacement == std::tolower(
                    static_cast<unsigned char>(original))) {
                continue;
            }
            mutable_str[i] = replacement;
            auto test = bech32_decode(mutable_str);
            if (test.encoding != Bech32Encoding::INVALID) {
                return static_cast<int>(i);
            }
        }
        mutable_str[i] = original;
    }
    return -1;
}

}  // namespace rnet::core

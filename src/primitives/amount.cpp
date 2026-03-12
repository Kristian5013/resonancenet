#include "primitives/amount.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>

namespace rnet::primitives {

std::string FormatMoney(int64_t amount) {
    bool negative = amount < 0;
    int64_t abs_val = negative ? -amount : amount;

    int64_t whole = abs_val / COIN;
    int64_t frac = abs_val % COIN;

    // Format fractional part with 8 digits, stripping trailing zeros
    char frac_buf[16];
    std::snprintf(frac_buf, sizeof(frac_buf), "%08lld",
                  static_cast<long long>(frac));
    std::string frac_str(frac_buf);

    // Remove trailing zeros but keep at least 2 decimal places
    size_t last_nonzero = frac_str.find_last_not_of('0');
    if (last_nonzero == std::string::npos || last_nonzero < 1) {
        last_nonzero = 1;
    }
    frac_str.resize(last_nonzero + 1);

    std::string result;
    if (negative) {
        result += '-';
    }
    result += std::to_string(whole);
    result += '.';
    result += frac_str;
    return result;
}

bool ParseMoney(const std::string& str, int64_t& amount_out) {
    if (str.empty()) return false;

    bool negative = false;
    size_t pos = 0;
    if (str[0] == '-') {
        negative = true;
        pos = 1;
    }

    // Find decimal point
    auto dot_pos = str.find('.', pos);

    std::string whole_str;
    std::string frac_str;

    if (dot_pos == std::string::npos) {
        whole_str = str.substr(pos);
        frac_str = "0";
    } else {
        whole_str = str.substr(pos, dot_pos - pos);
        frac_str = str.substr(dot_pos + 1);
    }

    if (whole_str.empty() && frac_str.empty()) return false;
    if (whole_str.empty()) whole_str = "0";

    // Validate digits
    for (char c : whole_str) {
        if (c < '0' || c > '9') return false;
    }
    for (char c : frac_str) {
        if (c < '0' || c > '9') return false;
    }

    // Pad or truncate fractional part to 8 digits
    if (frac_str.size() > 8) {
        return false;  // Too many decimal places
    }
    while (frac_str.size() < 8) {
        frac_str += '0';
    }

    int64_t whole = 0;
    for (char c : whole_str) {
        int64_t prev = whole;
        whole = whole * 10 + (c - '0');
        if (whole / 10 != prev) return false;  // Overflow
    }

    int64_t frac = 0;
    for (char c : frac_str) {
        frac = frac * 10 + (c - '0');
    }

    // Check overflow before multiplication
    if (whole > MAX_MONEY / COIN) return false;
    int64_t result = whole * COIN + frac;
    if (result < 0 || result > MAX_MONEY) return false;

    amount_out = negative ? -result : result;
    return true;
}

}  // namespace rnet::primitives

// Copyright (c) 2025 The ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "primitives/amount.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>

namespace rnet::primitives {

// ===========================================================================
//  Money formatting and parsing
// ===========================================================================

// ---------------------------------------------------------------------------
// FormatMoney
//   Converts a satoshi-denominated int64 to a fixed-point string with
//   exactly 8 decimal places (e.g. 123456789 -> "1.23456789").
// ---------------------------------------------------------------------------
std::string FormatMoney(int64_t amount)
{
    bool negative = amount < 0;
    int64_t abs_val = negative ? -amount : amount;

    int64_t whole = abs_val / COIN;
    int64_t frac = abs_val % COIN;

    // 1. Format fractional part with exactly 8 digits
    char frac_buf[16];
    std::snprintf(frac_buf, sizeof(frac_buf), "%08lld",
                  static_cast<long long>(frac));

    // 2. Assemble result
    std::string result;
    if (negative) {
        result += '-';
    }
    result += std::to_string(whole);
    result += '.';
    result += frac_buf;
    return result;
}

// ---------------------------------------------------------------------------
// ParseMoney
//   Parses a decimal string into a satoshi-denominated int64.  Rejects
//   more than 8 fractional digits and values exceeding MAX_MONEY.
// ---------------------------------------------------------------------------
bool ParseMoney(const std::string& str, int64_t& amount_out)
{
    if (str.empty()) return false;

    // 1. Handle optional leading minus sign
    bool negative = false;
    size_t pos = 0;
    if (str[0] == '-') {
        negative = true;
        pos = 1;
    }

    // 2. Find decimal point
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

    // 3. Validate digits
    for (char c : whole_str) {
        if (c < '0' || c > '9') return false;
    }
    for (char c : frac_str) {
        if (c < '0' || c > '9') return false;
    }

    // 4. Pad or reject fractional part (must be <= 8 digits)
    if (frac_str.size() > 8) {
        return false;
    }
    while (frac_str.size() < 8) {
        frac_str += '0';
    }

    // 5. Convert whole part with overflow detection
    int64_t whole = 0;
    for (char c : whole_str) {
        int64_t prev = whole;
        whole = whole * 10 + (c - '0');
        if (whole / 10 != prev) return false;
    }

    // 6. Convert fractional part
    int64_t frac = 0;
    for (char c : frac_str) {
        frac = frac * 10 + (c - '0');
    }

    // 7. Check overflow before final multiplication
    if (whole > MAX_MONEY / COIN) return false;
    int64_t result = whole * COIN + frac;
    if (result < 0 || result > MAX_MONEY) return false;

    amount_out = negative ? -result : result;
    return true;
}

} // namespace rnet::primitives

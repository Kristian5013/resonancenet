#pragma once

#include <cstdint>
#include <string>

namespace rnet::primitives {

/// 1 RNT = 100,000,000 resonances
static constexpr int64_t COIN = 100'000'000;

/// Maximum total supply: 21 million RNT
static constexpr int64_t MAX_MONEY = 21'000'000 * COIN;

/// Check whether a monetary value is within the valid range [0, MAX_MONEY].
inline bool MoneyRange(int64_t value) {
    return value >= 0 && value <= MAX_MONEY;
}

/// Format an amount in resonances as a human-readable string (e.g. "1.23456789").
std::string FormatMoney(int64_t amount);

/// Parse a decimal string (e.g. "1.5") into resonances.
/// Returns false on overflow or invalid input.
bool ParseMoney(const std::string& str, int64_t& amount_out);

}  // namespace rnet::primitives

// Copyright (c) 2025 The ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "primitives/fees.h"

#include <sstream>

namespace rnet::primitives {

// ===========================================================================
//  Fee rate and dust threshold utilities
// ===========================================================================

// ---------------------------------------------------------------------------
// CFeeRate::to_string
//   Displays the rate in resonances per virtual byte.
// ---------------------------------------------------------------------------
std::string CFeeRate::to_string() const
{
    double rate_per_vb = static_cast<double>(resonances_per_kvb_) / 1000.0;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.2f res/vB", rate_per_vb);
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// get_dust_threshold
//   Returns the minimum output value that is not considered dust.
//   Dust = value < (cost to create the output + cost to spend it).
//   Assumes P2WPKH spend size of ~68 vbytes.
// ---------------------------------------------------------------------------
int64_t get_dust_threshold(size_t script_size, const CFeeRate& fee_rate)
{
    // 1. Witness spend cost (~68 vbytes for P2WPKH)
    size_t spend_size = 68;

    // 2. Output itself: 8 (value) + 1 (script length) + script_size
    size_t output_size = 9 + script_size;

    // 3. Dust if output value < cost to create + cost to spend
    int64_t cost = fee_rate.get_fee(spend_size + output_size);
    return cost;
}

// ---------------------------------------------------------------------------
// is_dust
//   Convenience wrapper around get_dust_threshold.
// ---------------------------------------------------------------------------
bool is_dust(int64_t value, size_t script_size, const CFeeRate& fee_rate)
{
    return value < get_dust_threshold(script_size, fee_rate);
}

} // namespace rnet::primitives

#include "primitives/fees.h"

#include <sstream>

namespace rnet::primitives {

std::string CFeeRate::to_string() const {
    // Display as resonances per virtual byte (rate / 1000)
    double rate_per_vb = static_cast<double>(resonances_per_kvb_) / 1000.0;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.2f res/vB", rate_per_vb);
    return std::string(buf);
}

int64_t get_dust_threshold(size_t script_size, const CFeeRate& fee_rate) {
    // The cost to spend an output is approximately:
    //   input_size = 32 (txid) + 4 (vout) + 1 (scriptSig len) + 107 (sig+pubkey) + 4 (seq) = 148
    // For witness outputs, input is ~68 vbytes.
    // Use 68 vbytes for witness (P2WPKH) as default.
    size_t spend_size = 68;

    // Also account for the output itself: 8 (value) + 1 + script_size
    size_t output_size = 9 + script_size;

    // Dust if output value < cost to create + cost to spend
    int64_t cost = fee_rate.get_fee(spend_size + output_size);
    return cost;
}

bool is_dust(int64_t value, size_t script_size, const CFeeRate& fee_rate) {
    return value < get_dust_threshold(script_size, fee_rate);
}

}  // namespace rnet::primitives

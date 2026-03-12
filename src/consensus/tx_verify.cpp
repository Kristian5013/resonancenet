#include "consensus/tx_verify.h"

#include <set>

#include "primitives/amount.h"

namespace rnet::consensus {

bool check_transaction(const primitives::CTransaction& tx, ValidationState& state) {
    // Must have at least one input and one output
    if (tx.vin().empty()) {
        state.invalid("bad-txns-vin-empty");
        return false;
    }
    if (tx.vout().empty()) {
        state.invalid("bad-txns-vout-empty");
        return false;
    }

    // Check output values
    int64_t total_out = 0;
    for (const auto& txout : tx.vout()) {
        if (txout.value < 0) {
            state.invalid("bad-txns-vout-negative");
            return false;
        }
        if (!primitives::MoneyRange(txout.value)) {
            state.invalid("bad-txns-vout-toolarge");
            return false;
        }
        total_out += txout.value;
        if (!primitives::MoneyRange(total_out)) {
            state.invalid("bad-txns-txouttotal-toolarge");
            return false;
        }
    }

    // Check for duplicate inputs
    std::set<primitives::COutPoint> seen_inputs;
    for (const auto& txin : tx.vin()) {
        if (!seen_inputs.insert(txin.prevout).second) {
            state.invalid("bad-txns-inputs-duplicate");
            return false;
        }
    }

    if (tx.is_coinbase()) {
        // Coinbase scriptSig must be between 2 and 100 bytes
        const auto& script_sig = tx.vin()[0].script_sig;
        if (script_sig.size() < 2 || script_sig.size() > 100) {
            state.invalid("bad-cb-length");
            return false;
        }
    } else {
        // Non-coinbase inputs must not reference null outpoints
        for (const auto& txin : tx.vin()) {
            if (txin.prevout.is_null()) {
                state.invalid("bad-txns-prevout-null");
                return false;
            }
        }
    }

    return true;
}

}  // namespace rnet::consensus

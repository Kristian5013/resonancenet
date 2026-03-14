#include "consensus/tx_verify.h"

#include <set>

#include "primitives/amount.h"
#include "script/interpreter.h"
#include "script/verify.h"

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

bool check_inputs(const primitives::CTransaction& tx,
                  const chain::CCoinsViewCache& coins,
                  ValidationState& state) {
    // Coinbase transactions have no inputs to check
    if (tx.is_coinbase()) {
        return true;
    }

    int64_t total_in = 0;

    for (unsigned int i = 0; i < tx.vin().size(); ++i) {
        const auto& txin = tx.vin()[i];

        // 1. Look up the coin in the UTXO set
        chain::Coin coin;
        if (!coins.get_coin(txin.prevout, coin)) {
            state.invalid("bad-txns-inputs-missingorspent");
            return false;
        }

        // 2. Verify the coin is not spent
        if (coin.is_spent()) {
            state.invalid("bad-txns-inputs-missingorspent");
            return false;
        }

        // 3. Sum input values and check for overflow
        total_in += coin.out.value;
        if (!primitives::MoneyRange(coin.out.value) ||
            !primitives::MoneyRange(total_in)) {
            state.invalid("bad-txns-inputvalues-outofrange");
            return false;
        }

        // 4. Run script verification
        script::CScript script_sig(txin.script_sig.begin(),
                                   txin.script_sig.end());
        script::CScript script_pub_key(coin.out.script_pub_key.begin(),
                                       coin.out.script_pub_key.end());

        script::TransactionSignatureChecker checker(&tx, i, coin.out.value);
        script::ScriptError serror = script::ScriptError::OK;

        const primitives::CScriptWitness* witness =
            txin.witness.is_null() ? nullptr : &txin.witness;

        if (!script::verify_script(script_sig, script_pub_key, witness,
                                   script::STANDARD_SCRIPT_VERIFY_FLAGS,
                                   checker, &serror)) {
            state.invalid("mandatory-script-verify-flag-failed ("
                          + std::string(script::script_error_string(serror))
                          + ")");
            return false;
        }
    }

    // 5. Verify inputs cover outputs (no inflation)
    if (total_in < tx.get_value_out()) {
        state.invalid("bad-txns-in-belowout");
        return false;
    }

    return true;
}

}  // namespace rnet::consensus

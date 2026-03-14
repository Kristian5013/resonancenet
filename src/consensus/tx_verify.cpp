// Copyright (c) 2024-2026 The ResonanceNet Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "consensus/tx_verify.h"

#include "primitives/amount.h"
#include "script/interpreter.h"
#include "script/verify.h"

#include <set>

namespace rnet::consensus {

// ---------------------------------------------------------------------------
// check_transaction
// ---------------------------------------------------------------------------
//
// Context-free validation of a transaction's internal consistency.
//
// Checks:
//   [1] vin is non-empty
//   [2] vout is non-empty
//   [3] every output value >= 0
//   [4] every output value within money range
//   [5] cumulative output value within money range
//   [6] no duplicate inputs (by outpoint)
//   [7] coinbase scriptSig between 2 and 100 bytes
//   [8] non-coinbase inputs must not reference null outpoints
// ---------------------------------------------------------------------------

bool check_transaction(const primitives::CTransaction& tx, ValidationState& state)
{
    // 1. Transaction must have at least one input.
    if (tx.vin().empty()) {
        state.invalid("bad-txns-vin-empty");
        return false;
    }

    // 2. Transaction must have at least one output.
    if (tx.vout().empty()) {
        state.invalid("bad-txns-vout-empty");
        return false;
    }

    // 3-5. Validate every output value and the running total.
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

    // 6. No two inputs may spend the same outpoint.
    std::set<primitives::COutPoint> seen_inputs;
    for (const auto& txin : tx.vin()) {
        if (!seen_inputs.insert(txin.prevout).second) {
            state.invalid("bad-txns-inputs-duplicate");
            return false;
        }
    }

    // 7-8. Coinbase vs regular-transaction rules.
    if (tx.is_coinbase()) {
        static constexpr size_t kMinCoinbaseScriptSize = 2;
        static constexpr size_t kMaxCoinbaseScriptSize = 100;

        const auto& script_sig = tx.vin()[0].script_sig;
        if (script_sig.size() < kMinCoinbaseScriptSize ||
            script_sig.size() > kMaxCoinbaseScriptSize) {
            state.invalid("bad-cb-length");
            return false;
        }
    } else {
        for (const auto& txin : tx.vin()) {
            if (txin.prevout.is_null()) {
                state.invalid("bad-txns-prevout-null");
                return false;
            }
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// check_inputs
// ---------------------------------------------------------------------------
//
// Contextual validation: verify every input against the UTXO set and run
// script verification.
//
// Checks:
//   [1] each referenced coin exists in the UTXO set
//   [2] each referenced coin is unspent
//   [3] individual and cumulative input values within money range
//   [4] script signature satisfies the locking script
//   [5] total inputs >= total outputs (no inflation)
// ---------------------------------------------------------------------------

bool check_inputs(const primitives::CTransaction& tx,
                  const chain::CCoinsViewCache& coins,
                  ValidationState& state)
{
    // Coinbase transactions have no inputs to verify.
    if (tx.is_coinbase()) {
        return true;
    }

    int64_t total_in = 0;

    for (unsigned int i = 0; i < tx.vin().size(); ++i) {
        const auto& txin = tx.vin()[i];

        // 1. Look up the coin in the UTXO set.
        chain::Coin coin;
        if (!coins.get_coin(txin.prevout, coin)) {
            state.invalid("bad-txns-inputs-missingorspent");
            return false;
        }

        // 2. Verify the coin is not already spent.
        if (coin.is_spent()) {
            state.invalid("bad-txns-inputs-missingorspent");
            return false;
        }

        // 3. Accumulate input values; reject on overflow or out-of-range.
        total_in += coin.out.value;
        if (!primitives::MoneyRange(coin.out.value) ||
            !primitives::MoneyRange(total_in)) {
            state.invalid("bad-txns-inputvalues-outofrange");
            return false;
        }

        // 4. Run the script interpreter to verify the signature.
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

    // 5. Total inputs must cover total outputs (no inflation).
    if (total_in < tx.get_value_out()) {
        state.invalid("bad-txns-in-belowout");
        return false;
    }

    return true;
}

} // namespace rnet::consensus

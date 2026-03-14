// Copyright (c) 2025-2026 The ResonanceNet Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "wallet/balance.h"

#include "primitives/amount.h"

namespace rnet::wallet {

// ===========================================================================
//  Balance computation -- confirmed / unconfirmed / immature buckets
// ===========================================================================
//
//  Coins are classified into three buckets:
//    confirmed   -- confirmed non-coinbase, or mature coinbase
//    unconfirmed -- in mempool (height < 0)
//    immature    -- coinbase outputs below COINBASE_MATURITY depth
//
//  Coinbase outputs are identified by the raw Ed25519 script pattern:
//    [0x20][32-byte pubkey][0xAC]  (34 bytes total, first byte 0x20).
//
//  total = confirmed + unconfirmed  (immature excluded until mature).

// ---------------------------------------------------------------------------
// WalletBalance::to_string -- human-readable summary
// ---------------------------------------------------------------------------

std::string WalletBalance::to_string() const {
    return "confirmed=" + primitives::FormatMoney(confirmed) +
           " unconfirmed=" + primitives::FormatMoney(unconfirmed) +
           " immature=" + primitives::FormatMoney(immature) +
           " total=" + primitives::FormatMoney(total);
}

// ---------------------------------------------------------------------------
// compute_from_coins -- classify UTXOs into balance buckets
// ---------------------------------------------------------------------------

static WalletBalance compute_from_coins(const std::vector<WalletCoin>& utxos,
                                        int32_t current_height,
                                        int32_t coinbase_maturity) {
    WalletBalance bal;

    for (const auto& coin : utxos) {
        if (coin.is_spent) continue;

        int64_t value = coin.txout.value;

        if (coin.height < 0) {
            // 1. Unconfirmed (mempool).
            bal.unconfirmed += value;
        } else {
            int32_t depth = current_height - coin.height + 1;

            // 2. Check if this is a coinbase output that hasn't matured.
            //    Coinbase outputs have a special script starting with 0x20.
            bool is_coinbase_output = !coin.txout.script_pub_key.empty() &&
                                      coin.txout.script_pub_key[0] == 0x20 &&
                                      coin.txout.script_pub_key.size() == 34;

            if (is_coinbase_output && depth < coinbase_maturity) {
                // 3. Immature coinbase.
                bal.immature += value;
            } else {
                // 4. Confirmed and spendable.
                bal.confirmed += value;
            }
        }
    }

    bal.total = bal.confirmed + bal.unconfirmed;
    return bal;
}

// ---------------------------------------------------------------------------
// compute_balance -- whole-wallet balance
// ---------------------------------------------------------------------------

WalletBalance compute_balance(const CoinTracker& coins,
                              int32_t current_height,
                              int32_t coinbase_maturity) {
    auto utxos = coins.get_unspent();
    return compute_from_coins(utxos, current_height, coinbase_maturity);
}

// ---------------------------------------------------------------------------
// compute_balance_for -- balance for a single pubkey hash
// ---------------------------------------------------------------------------

WalletBalance compute_balance_for(const CoinTracker& coins,
                                  const uint160& pubkey_hash,
                                  int32_t current_height,
                                  int32_t coinbase_maturity) {
    auto utxos = coins.get_unspent_for(pubkey_hash);
    return compute_from_coins(utxos, current_height, coinbase_maturity);
}

} // namespace rnet::wallet

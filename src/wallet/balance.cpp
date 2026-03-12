#include "wallet/balance.h"

#include "primitives/amount.h"

namespace rnet::wallet {

std::string WalletBalance::to_string() const {
    return "confirmed=" + primitives::FormatMoney(confirmed) +
           " unconfirmed=" + primitives::FormatMoney(unconfirmed) +
           " immature=" + primitives::FormatMoney(immature) +
           " total=" + primitives::FormatMoney(total);
}

static WalletBalance compute_from_coins(const std::vector<WalletCoin>& utxos,
                                        int32_t current_height,
                                        int32_t coinbase_maturity) {
    WalletBalance bal;

    for (const auto& coin : utxos) {
        if (coin.is_spent) continue;

        int64_t value = coin.txout.value;

        if (coin.height < 0) {
            // Unconfirmed (mempool)
            bal.unconfirmed += value;
        } else {
            int32_t depth = current_height - coin.height + 1;

            // Check if this is a coinbase output that hasn't matured
            // Coinbase outputs have a special script starting with 0x20
            bool is_coinbase_output = !coin.txout.script_pub_key.empty() &&
                                      coin.txout.script_pub_key[0] == 0x20 &&
                                      coin.txout.script_pub_key.size() == 34;

            if (is_coinbase_output && depth < coinbase_maturity) {
                bal.immature += value;
            } else {
                bal.confirmed += value;
            }
        }
    }

    bal.total = bal.confirmed + bal.unconfirmed;
    return bal;
}

WalletBalance compute_balance(const CoinTracker& coins,
                              int32_t current_height,
                              int32_t coinbase_maturity) {
    auto utxos = coins.get_unspent();
    return compute_from_coins(utxos, current_height, coinbase_maturity);
}

WalletBalance compute_balance_for(const CoinTracker& coins,
                                  const uint160& pubkey_hash,
                                  int32_t current_height,
                                  int32_t coinbase_maturity) {
    auto utxos = coins.get_unspent_for(pubkey_hash);
    return compute_from_coins(utxos, current_height, coinbase_maturity);
}

}  // namespace rnet::wallet

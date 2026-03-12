#pragma once

#include <cstdint>
#include <string>

#include "core/types.h"
#include "wallet/coins.h"

namespace rnet::wallet {

/// Wallet balance breakdown.
struct WalletBalance {
    int64_t confirmed = 0;       ///< Confirmed (depth >= 1)
    int64_t unconfirmed = 0;     ///< In mempool (depth == 0)
    int64_t immature = 0;        ///< Coinbase < 100 blocks deep
    int64_t total = 0;           ///< confirmed + unconfirmed

    /// Human-readable balance string.
    std::string to_string() const;
};

/// Compute balance from a CoinTracker.
/// @param coins         The wallet's UTXO tracker.
/// @param current_height Current blockchain height.
/// @param coinbase_maturity Minimum depth for coinbase maturity (default 100).
WalletBalance compute_balance(const CoinTracker& coins,
                              int32_t current_height,
                              int32_t coinbase_maturity = 100);

/// Compute balance for a specific pubkey hash.
WalletBalance compute_balance_for(const CoinTracker& coins,
                                  const uint160& pubkey_hash,
                                  int32_t current_height,
                                  int32_t coinbase_maturity = 100);

}  // namespace rnet::wallet

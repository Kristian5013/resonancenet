#pragma once

#include <cstdint>
#include <map>
#include <vector>

#include "core/error.h"
#include "core/sync.h"
#include "core/types.h"
#include "primitives/outpoint.h"
#include "primitives/txout.h"

namespace rnet::wallet {

/// A wallet UTXO entry.
struct WalletCoin {
    primitives::COutPoint outpoint;
    primitives::CTxOut txout;
    int32_t height = -1;          ///< Block height where confirmed (-1 = mempool)
    uint160 pubkey_hash;          ///< Owner pubkey hash
    bool is_spent = false;
    bool is_change = false;
    int64_t creation_time = 0;
    double val_loss_at_creation = 0.0;  ///< For UTXO expiry tracking
};

/// CoinTracker: manages the wallet's UTXO set.
class CoinTracker {
public:
    CoinTracker() = default;

    /// Add a new unspent coin.
    Result<void> add_coin(const WalletCoin& coin);

    /// Mark a coin as spent.
    Result<void> spend_coin(const primitives::COutPoint& outpoint);

    /// Remove a coin entirely (e.g., on reorg).
    Result<void> remove_coin(const primitives::COutPoint& outpoint);

    /// Get a specific coin by outpoint.
    Result<WalletCoin> get_coin(const primitives::COutPoint& outpoint) const;

    /// Check if we have a specific coin.
    bool have_coin(const primitives::COutPoint& outpoint) const;

    /// Get all unspent coins.
    std::vector<WalletCoin> get_unspent() const;

    /// Get unspent coins for a specific pubkey hash.
    std::vector<WalletCoin> get_unspent_for(const uint160& pubkey_hash) const;

    /// Get all coins (including spent).
    std::vector<WalletCoin> get_all() const;

    /// Get total number of unspent coins.
    size_t unspent_count() const;

    /// Clear all coins.
    void clear();

private:
    mutable core::Mutex mutex_;
    std::map<primitives::COutPoint, WalletCoin> coins_;
};

}  // namespace rnet::wallet

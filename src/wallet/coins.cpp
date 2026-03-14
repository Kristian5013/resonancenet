// Copyright (c) 2025-2026 The ResonanceNet Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "wallet/coins.h"

#include "core/logging.h"

namespace rnet::wallet {

// ===========================================================================
//  CoinTracker -- in-memory UTXO set owned by this wallet
// ===========================================================================
//
//  Mirrors the subset of the global UTXO set that belongs to our keys.
//  Each WalletCoin carries val_loss_at_creation for the UTXO expiry
//  protocol: when current_val_loss / val_loss_at_creation exceeds the
//  RECLAIM_RATIO (10x), the coin expires and returns to mining rewards.
//  A heartbeat (send-to-self) resets the timer.

// ---------------------------------------------------------------------------
// add_coin -- insert a new UTXO (rejects duplicates)
// ---------------------------------------------------------------------------

Result<void> CoinTracker::add_coin(const WalletCoin& coin) {
    LOCK(mutex_);
    if (coins_.count(coin.outpoint)) {
        return Result<void>::err("coin already exists");
    }
    coins_[coin.outpoint] = coin;
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// spend_coin -- mark a UTXO as spent (keeps it in the map for history)
// ---------------------------------------------------------------------------

Result<void> CoinTracker::spend_coin(const primitives::COutPoint& outpoint) {
    LOCK(mutex_);
    auto it = coins_.find(outpoint);
    if (it == coins_.end()) {
        return Result<void>::err("coin not found");
    }
    it->second.is_spent = true;
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// remove_coin -- erase a UTXO entirely (e.g. on reorg)
// ---------------------------------------------------------------------------

Result<void> CoinTracker::remove_coin(const primitives::COutPoint& outpoint) {
    LOCK(mutex_);
    auto it = coins_.find(outpoint);
    if (it == coins_.end()) {
        return Result<void>::err("coin not found");
    }
    coins_.erase(it);
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// Lookups
// ---------------------------------------------------------------------------

Result<WalletCoin> CoinTracker::get_coin(const primitives::COutPoint& outpoint) const {
    LOCK(mutex_);
    auto it = coins_.find(outpoint);
    if (it == coins_.end()) {
        return Result<WalletCoin>::err("coin not found");
    }
    return Result<WalletCoin>::ok(it->second);
}

bool CoinTracker::have_coin(const primitives::COutPoint& outpoint) const {
    LOCK(mutex_);
    return coins_.count(outpoint) > 0;
}

// ---------------------------------------------------------------------------
// Unspent queries
// ---------------------------------------------------------------------------

std::vector<WalletCoin> CoinTracker::get_unspent() const {
    LOCK(mutex_);
    std::vector<WalletCoin> result;
    for (const auto& [_, coin] : coins_) {
        if (!coin.is_spent) {
            result.push_back(coin);
        }
    }
    return result;
}

std::vector<WalletCoin> CoinTracker::get_unspent_for(const uint160& pubkey_hash) const {
    LOCK(mutex_);
    std::vector<WalletCoin> result;
    for (const auto& [_, coin] : coins_) {
        if (!coin.is_spent && coin.pubkey_hash == pubkey_hash) {
            result.push_back(coin);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Bulk retrieval
// ---------------------------------------------------------------------------

std::vector<WalletCoin> CoinTracker::get_all() const {
    LOCK(mutex_);
    std::vector<WalletCoin> result;
    result.reserve(coins_.size());
    for (const auto& [_, coin] : coins_) {
        result.push_back(coin);
    }
    return result;
}

// ---------------------------------------------------------------------------
// unspent_count / clear
// ---------------------------------------------------------------------------

size_t CoinTracker::unspent_count() const {
    LOCK(mutex_);
    size_t count = 0;
    for (const auto& [_, coin] : coins_) {
        if (!coin.is_spent) ++count;
    }
    return count;
}

void CoinTracker::clear() {
    LOCK(mutex_);
    coins_.clear();
}

} // namespace rnet::wallet

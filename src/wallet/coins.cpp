#include "wallet/coins.h"

#include "core/logging.h"

namespace rnet::wallet {

Result<void> CoinTracker::add_coin(const WalletCoin& coin) {
    LOCK(mutex_);
    if (coins_.count(coin.outpoint)) {
        return Result<void>::err("coin already exists");
    }
    coins_[coin.outpoint] = coin;
    return Result<void>::ok();
}

Result<void> CoinTracker::spend_coin(const primitives::COutPoint& outpoint) {
    LOCK(mutex_);
    auto it = coins_.find(outpoint);
    if (it == coins_.end()) {
        return Result<void>::err("coin not found");
    }
    it->second.is_spent = true;
    return Result<void>::ok();
}

Result<void> CoinTracker::remove_coin(const primitives::COutPoint& outpoint) {
    LOCK(mutex_);
    auto it = coins_.find(outpoint);
    if (it == coins_.end()) {
        return Result<void>::err("coin not found");
    }
    coins_.erase(it);
    return Result<void>::ok();
}

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

std::vector<WalletCoin> CoinTracker::get_all() const {
    LOCK(mutex_);
    std::vector<WalletCoin> result;
    result.reserve(coins_.size());
    for (const auto& [_, coin] : coins_) {
        result.push_back(coin);
    }
    return result;
}

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

}  // namespace rnet::wallet

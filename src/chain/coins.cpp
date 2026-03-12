#include "chain/coins.h"

namespace rnet::chain {

CCoinsViewCache::CCoinsViewCache(CCoinsView* base)
    : CCoinsViewBacked(base) {}

bool CCoinsViewCache::get_coin(const primitives::COutPoint& outpoint,
                               Coin& coin) const {
    auto it = cache_.find(outpoint);
    if (it != cache_.end()) {
        if (it->second.is_spent()) return false;
        coin = it->second;
        return true;
    }
    // Try backing store
    auto fetched = fetch_coin(outpoint);
    if (fetched != cache_.end() && !fetched->second.is_spent()) {
        coin = fetched->second;
        return true;
    }
    return false;
}

bool CCoinsViewCache::have_coin(const primitives::COutPoint& outpoint) const {
    auto it = cache_.find(outpoint);
    if (it != cache_.end()) {
        return !it->second.is_spent();
    }
    auto fetched = fetch_coin(outpoint);
    return fetched != cache_.end() && !fetched->second.is_spent();
}

rnet::uint256 CCoinsViewCache::get_best_block() const {
    if (best_block_set_) return best_block_;
    if (base_) {
        best_block_ = base_->get_best_block();
        best_block_set_ = true;
    }
    return best_block_;
}

size_t CCoinsViewCache::estimate_size() const {
    return cache_.size();
}

void CCoinsViewCache::add_coin(const primitives::COutPoint& outpoint,
                               Coin coin, bool /*possible_overwrite*/) {
    cache_[outpoint] = std::move(coin);
}

bool CCoinsViewCache::spend_coin(const primitives::COutPoint& outpoint,
                                 Coin* moved_out) {
    auto it = cache_.find(outpoint);
    if (it == cache_.end()) {
        // Try to fetch from backing store first
        it = fetch_coin(outpoint);
        if (it == cache_.end()) return false;
    }
    if (it->second.is_spent()) return false;
    if (moved_out) {
        *moved_out = std::move(it->second);
    }
    it->second.out.set_null();
    return true;
}

void CCoinsViewCache::set_best_block(const rnet::uint256& hash) {
    best_block_ = hash;
    best_block_set_ = true;
}

Result<void> CCoinsViewCache::flush() {
    // In a full implementation, this would write all dirty entries
    // to the backing CCoinsViewDB. For now, just clear the cache.
    cache_.clear();
    return Result<void>::ok();
}

int64_t CCoinsViewCache::get_cached_value() const {
    int64_t total = 0;
    for (const auto& [op, coin] : cache_) {
        if (!coin.is_spent()) {
            total += coin.out.value;
        }
    }
    return total;
}

size_t CCoinsViewCache::cache_size() const {
    return cache_.size();
}

bool CCoinsViewCache::is_coin_expired(const Coin& coin,
                                      float current_val_loss,
                                      float reclaim_ratio) const {
    if (coin.val_loss_at_creation <= 0.0f || current_val_loss <= 0.0f) {
        return false;
    }
    // Coin expires when val_loss_at_creation / current_val_loss > reclaim_ratio
    return (coin.val_loss_at_creation / current_val_loss) > reclaim_ratio;
}

CCoinsViewCache::CacheMap::iterator
CCoinsViewCache::fetch_coin(const primitives::COutPoint& outpoint) const {
    if (!base_) return cache_.end();
    Coin coin;
    if (!base_->get_coin(outpoint, coin)) {
        return cache_.end();
    }
    auto [it, inserted] = cache_.emplace(outpoint, std::move(coin));
    return it;
}

}  // namespace rnet::chain

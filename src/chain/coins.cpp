// Copyright (c) 2025 The ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "chain/coins.h"

namespace rnet::chain {

// ===========================================================================
//  CCoinsViewCache -- in-memory UTXO layer backed by a persistent store
// ===========================================================================

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
CCoinsViewCache::CCoinsViewCache(CCoinsView* base)
    : CCoinsViewBacked(base) {}

// ---------------------------------------------------------------------------
// get_coin
//   Looks up a coin, checking the local cache first, then the backing store.
//   Returns false for spent entries.
// ---------------------------------------------------------------------------
bool CCoinsViewCache::get_coin(const primitives::COutPoint& outpoint,
                               Coin& coin) const
{
    // 1. Check local cache
    auto it = cache_.find(outpoint);
    if (it != cache_.end()) {
        if (it->second.is_spent()) return false;
        coin = it->second;
        return true;
    }

    // 2. Try backing store
    auto fetched = fetch_coin(outpoint);
    if (fetched != cache_.end() && !fetched->second.is_spent()) {
        coin = fetched->second;
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// have_coin
//   Existence check without copying the coin data.
// ---------------------------------------------------------------------------
bool CCoinsViewCache::have_coin(const primitives::COutPoint& outpoint) const
{
    auto it = cache_.find(outpoint);
    if (it != cache_.end()) {
        return !it->second.is_spent();
    }
    auto fetched = fetch_coin(outpoint);
    return fetched != cache_.end() && !fetched->second.is_spent();
}

// ---------------------------------------------------------------------------
// get_best_block
//   Returns the cached best-block hash, fetching from the backing store on
//   first access.
// ---------------------------------------------------------------------------
rnet::uint256 CCoinsViewCache::get_best_block() const
{
    if (best_block_set_) return best_block_;
    if (base_) {
        best_block_ = base_->get_best_block();
        best_block_set_ = true;
    }
    return best_block_;
}

// ---------------------------------------------------------------------------
// estimate_size
// ---------------------------------------------------------------------------
size_t CCoinsViewCache::estimate_size() const
{
    return cache_.size();
}

// ---------------------------------------------------------------------------
// add_coin
//   Inserts or overwrites a coin in the cache.
// ---------------------------------------------------------------------------
void CCoinsViewCache::add_coin(const primitives::COutPoint& outpoint,
                               Coin coin, bool /*possible_overwrite*/)
{
    cache_[outpoint] = std::move(coin);
}

// ---------------------------------------------------------------------------
// spend_coin
//   Marks a coin as spent by nulling its output.  Optionally moves the old
//   value into *moved_out before spending.
// ---------------------------------------------------------------------------
bool CCoinsViewCache::spend_coin(const primitives::COutPoint& outpoint,
                                 Coin* moved_out)
{
    // 1. Look in local cache
    auto it = cache_.find(outpoint);
    if (it == cache_.end()) {
        // 2. Try to fetch from backing store
        it = fetch_coin(outpoint);
        if (it == cache_.end()) return false;
    }
    if (it->second.is_spent()) return false;

    // 3. Optionally extract before nulling
    if (moved_out) {
        *moved_out = std::move(it->second);
    }
    it->second.out.set_null();
    return true;
}

// ---------------------------------------------------------------------------
// set_best_block
// ---------------------------------------------------------------------------
void CCoinsViewCache::set_best_block(const rnet::uint256& hash)
{
    best_block_ = hash;
    best_block_set_ = true;
}

// ---------------------------------------------------------------------------
// flush
//   Stub: a full implementation writes all dirty entries to the backing
//   CCoinsViewDB.  For now, just clears the cache.
// ---------------------------------------------------------------------------
Result<void> CCoinsViewCache::flush()
{
    cache_.clear();
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// get_cached_value
//   Sums the value of all unspent coins currently held in the cache.
// ---------------------------------------------------------------------------
int64_t CCoinsViewCache::get_cached_value() const
{
    int64_t total = 0;
    for (const auto& [op, coin] : cache_) {
        if (!coin.is_spent()) {
            total += coin.out.value;
        }
    }
    return total;
}

// ---------------------------------------------------------------------------
// cache_size
// ---------------------------------------------------------------------------
size_t CCoinsViewCache::cache_size() const
{
    return cache_.size();
}

// ---------------------------------------------------------------------------
// is_coin_expired
//   UTXO expiry: a coin expires when its creation-time validation loss
//   divided by the current validation loss exceeds the reclaim ratio.
// ---------------------------------------------------------------------------
bool CCoinsViewCache::is_coin_expired(const Coin& coin,
                                      float current_val_loss,
                                      float reclaim_ratio) const
{
    if (coin.val_loss_at_creation <= 0.0f || current_val_loss <= 0.0f) {
        return false;
    }
    return (coin.val_loss_at_creation / current_val_loss) > reclaim_ratio;
}

// ---------------------------------------------------------------------------
// fetch_coin  (private)
//   Pulls a coin from the backing store into the local cache and returns an
//   iterator to the newly-inserted entry.
// ---------------------------------------------------------------------------
CCoinsViewCache::CacheMap::iterator
CCoinsViewCache::fetch_coin(const primitives::COutPoint& outpoint) const
{
    if (!base_) return cache_.end();
    Coin coin;
    if (!base_->get_coin(outpoint, coin)) {
        return cache_.end();
    }
    auto [it, inserted] = cache_.emplace(outpoint, std::move(coin));
    return it;
}

} // namespace rnet::chain

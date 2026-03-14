// Copyright (c) 2025 The ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "consensus/sigcache.h"

#include "crypto/keccak.h"

#include <cstring>
#include <vector>

namespace rnet::consensus {

// ===========================================================================
//  SignatureCache -- LRU cache for Ed25519 signature verification results
// ===========================================================================

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
SignatureCache::SignatureCache(size_t max_entries)
    : max_entries_(max_entries) {}

// ---------------------------------------------------------------------------
// get
//   Looks up a cached verification result by key.  On hit, promotes the
//   entry to the front of the LRU list and writes the result to valid_out.
// ---------------------------------------------------------------------------
bool SignatureCache::get(const rnet::uint256& key, bool& valid_out)
{
    LOCK(mutex_);

    auto it = map_.find(key);
    if (it == map_.end()) {
        return false;
    }

    // 1. Move to front (most recently used)
    lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
    valid_out = it->second->second;
    return true;
}

// ---------------------------------------------------------------------------
// put
//   Inserts or updates a cache entry.  Evicts the least-recently-used entry
//   when the cache is at capacity.
// ---------------------------------------------------------------------------
void SignatureCache::put(const rnet::uint256& key, bool valid)
{
    LOCK(mutex_);

    auto it = map_.find(key);
    if (it != map_.end()) {
        // 1. Update existing entry and move to front
        it->second->second = valid;
        lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
        return;
    }

    // 2. Evict if at capacity
    while (lru_list_.size() >= max_entries_ && !lru_list_.empty()) {
        evict_lru();
    }

    // 3. Insert new entry at front
    lru_list_.emplace_front(key, valid);
    map_[key] = lru_list_.begin();
}

// ---------------------------------------------------------------------------
// clear
// ---------------------------------------------------------------------------
void SignatureCache::clear()
{
    LOCK(mutex_);
    map_.clear();
    lru_list_.clear();
}

// ---------------------------------------------------------------------------
// size
// ---------------------------------------------------------------------------
size_t SignatureCache::size() const
{
    LOCK(mutex_);
    return map_.size();
}

// ---------------------------------------------------------------------------
// compute_key
//   Derives a deterministic cache key from (pubkey, signature, sighash)
//   via Keccak-256d of the concatenated inputs.
// ---------------------------------------------------------------------------
rnet::uint256 SignatureCache::compute_key(const uint8_t* pubkey, size_t pubkey_len,
                                          const uint8_t* sig, size_t sig_len,
                                          const uint8_t* sighash, size_t sighash_len)
{
    std::vector<uint8_t> data;
    data.reserve(pubkey_len + sig_len + sighash_len);
    data.insert(data.end(), pubkey, pubkey + pubkey_len);
    data.insert(data.end(), sig, sig + sig_len);
    data.insert(data.end(), sighash, sighash + sighash_len);

    return crypto::keccak256d(std::span<const uint8_t>(data));
}

// ---------------------------------------------------------------------------
// evict_lru  (private)
//   Removes the least-recently-used entry (back of the list).
//   Caller must hold mutex_.
// ---------------------------------------------------------------------------
void SignatureCache::evict_lru()
{
    if (lru_list_.empty()) return;

    auto& back = lru_list_.back();
    map_.erase(back.first);
    lru_list_.pop_back();
}

// ---------------------------------------------------------------------------
// get_signature_cache
//   Process-wide singleton with 100 000 entry capacity.
// ---------------------------------------------------------------------------
SignatureCache& get_signature_cache()
{
    static SignatureCache cache(100000);
    return cache;
}

} // namespace rnet::consensus

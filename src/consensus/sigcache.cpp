#include "consensus/sigcache.h"

#include <cstring>
#include <vector>

#include "crypto/keccak.h"

namespace rnet::consensus {

SignatureCache::SignatureCache(size_t max_entries)
    : max_entries_(max_entries) {}

bool SignatureCache::get(const rnet::uint256& key, bool& valid_out) {
    LOCK(mutex_);

    auto it = map_.find(key);
    if (it == map_.end()) {
        return false;
    }

    // Move to front (most recently used)
    lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
    valid_out = it->second->second;
    return true;
}

void SignatureCache::put(const rnet::uint256& key, bool valid) {
    LOCK(mutex_);

    auto it = map_.find(key);
    if (it != map_.end()) {
        // Update existing entry and move to front
        it->second->second = valid;
        lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
        return;
    }

    // Evict if at capacity
    while (lru_list_.size() >= max_entries_ && !lru_list_.empty()) {
        evict_lru();
    }

    // Insert new entry at front
    lru_list_.emplace_front(key, valid);
    map_[key] = lru_list_.begin();
}

void SignatureCache::clear() {
    LOCK(mutex_);
    map_.clear();
    lru_list_.clear();
}

size_t SignatureCache::size() const {
    LOCK(mutex_);
    return map_.size();
}

rnet::uint256 SignatureCache::compute_key(const uint8_t* pubkey, size_t pubkey_len,
                                          const uint8_t* sig, size_t sig_len,
                                          const uint8_t* sighash, size_t sighash_len) {
    std::vector<uint8_t> data;
    data.reserve(pubkey_len + sig_len + sighash_len);
    data.insert(data.end(), pubkey, pubkey + pubkey_len);
    data.insert(data.end(), sig, sig + sig_len);
    data.insert(data.end(), sighash, sighash + sighash_len);

    return crypto::keccak256d(std::span<const uint8_t>(data));
}

void SignatureCache::evict_lru() {
    // Caller must hold mutex_
    if (lru_list_.empty()) return;

    auto& back = lru_list_.back();
    map_.erase(back.first);
    lru_list_.pop_back();
}

SignatureCache& get_signature_cache() {
    static SignatureCache cache(100000);
    return cache;
}

}  // namespace rnet::consensus

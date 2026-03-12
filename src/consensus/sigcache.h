#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <unordered_map>

#include "core/sync.h"
#include "core/types.h"

namespace rnet::consensus {

/// Thread-safe LRU signature verification cache.
/// Caches the result of Ed25519 signature verifications to avoid
/// redundant work when the same transaction is seen multiple times
/// (e.g., in mempool and then in a block).
///
/// Key = keccak256d(pubkey || signature || sighash)
/// Value = bool (valid or not)
class SignatureCache {
public:
    /// @param max_entries  Maximum number of cached entries (LRU eviction).
    explicit SignatureCache(size_t max_entries = 100000);

    /// Check if a verification result is cached.
    /// @param key   Cache key: keccak256d(pubkey || sig || sighash).
    /// @param valid_out  Set to the cached result if found.
    /// @return true if the entry was found in the cache.
    bool get(const rnet::uint256& key, bool& valid_out);

    /// Insert a verification result into the cache.
    void put(const rnet::uint256& key, bool valid);

    /// Remove all entries.
    void clear();

    /// Current number of cached entries.
    size_t size() const;

    /// Compute a cache key from signature verification inputs.
    static rnet::uint256 compute_key(const uint8_t* pubkey, size_t pubkey_len,
                                     const uint8_t* sig, size_t sig_len,
                                     const uint8_t* sighash, size_t sighash_len);

private:
    /// Evict the least-recently-used entry (caller must hold mutex).
    void evict_lru();

    size_t max_entries_;

    mutable core::Mutex mutex_;

    // LRU list: front = most recently used, back = least recently used
    using LruList = std::list<std::pair<rnet::uint256, bool>>;
    LruList lru_list_;

    // Map from key to iterator into the LRU list
    std::unordered_map<rnet::uint256, LruList::iterator> map_;
};

/// Global signature cache singleton.
SignatureCache& get_signature_cache();

}  // namespace rnet::consensus

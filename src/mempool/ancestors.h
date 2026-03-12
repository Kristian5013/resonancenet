#pragma once

#include <cstdint>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "core/types.h"

namespace rnet::mempool {

class CTxMemPoolEntry;

/// Limits on ancestor/descendant chain length and size
struct AncestorLimits {
    int64_t max_ancestors = 25;
    int64_t max_ancestor_size = 101'000;   // ~100 kB
    int64_t max_descendants = 25;
    int64_t max_descendant_size = 101'000;
};

/// AncestorTracker — tracks parent/child relationships between
/// mempool transactions for package-aware mining and eviction.
class AncestorTracker {
public:
    AncestorTracker() = default;

    /// Add a transaction and its parent txids
    void add_tx(const rnet::uint256& txid,
                const std::set<rnet::uint256>& parent_txids,
                int64_t size, int64_t fee);

    /// Remove a transaction (and update its ancestors/descendants)
    void remove_tx(const rnet::uint256& txid);

    /// Get all ancestor txids (transitive)
    std::set<rnet::uint256> get_ancestors(const rnet::uint256& txid) const;

    /// Get all descendant txids (transitive)
    std::set<rnet::uint256> get_descendants(const rnet::uint256& txid) const;

    /// Get direct parents
    std::set<rnet::uint256> get_parents(const rnet::uint256& txid) const;

    /// Get direct children
    std::set<rnet::uint256> get_children(const rnet::uint256& txid) const;

    /// Check if adding a tx would exceed ancestor/descendant limits
    bool check_limits(const rnet::uint256& txid,
                      const std::set<rnet::uint256>& parent_txids,
                      int64_t size,
                      const AncestorLimits& limits) const;

    /// Get ancestor count for a transaction
    int64_t ancestor_count(const rnet::uint256& txid) const;

    /// Get descendant count for a transaction
    int64_t descendant_count(const rnet::uint256& txid) const;

    /// Clear all tracking data
    void clear();

private:
    struct TxLinks {
        std::set<rnet::uint256> parents;
        std::set<rnet::uint256> children;
        int64_t size = 0;
        int64_t fee = 0;
    };

    std::unordered_map<rnet::uint256, TxLinks> links_;

    void collect_ancestors(const rnet::uint256& txid,
                           std::set<rnet::uint256>& result) const;
    void collect_descendants(const rnet::uint256& txid,
                             std::set<rnet::uint256>& result) const;
};

}  // namespace rnet::mempool

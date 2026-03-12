#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <vector>

#include "core/sync.h"
#include "core/types.h"
#include "primitives/outpoint.h"
#include "primitives/transaction.h"

namespace rnet::mempool {

/// OrphanPool — holds transactions whose parents are not yet known.
/// When a parent transaction arrives, orphans that depend on it
/// can be re-evaluated for mempool admission.
class OrphanPool {
public:
    static constexpr size_t MAX_ORPHAN_TXS = 100;
    static constexpr size_t MAX_ORPHAN_TX_SIZE = 100'000;

    OrphanPool() = default;

    /// Add an orphan transaction. Returns true if added.
    bool add_tx(primitives::CTransactionRef tx, uint64_t peer_id);

    /// Remove an orphan transaction by txid
    bool erase_tx(const rnet::uint256& txid);

    /// Remove all orphans from a specific peer
    void erase_for_peer(uint64_t peer_id);

    /// Get orphans that spend a given outpoint (i.e., depend on a parent txid)
    std::vector<primitives::CTransactionRef> get_children_of(
        const rnet::uint256& parent_txid) const;

    /// Remove random orphans to keep pool within MAX_ORPHAN_TXS
    void limit_orphans();

    /// Check if a transaction is in the orphan pool
    bool have_tx(const rnet::uint256& txid) const;

    /// Get an orphan transaction by txid
    primitives::CTransactionRef get_tx(const rnet::uint256& txid) const;

    /// Number of orphan transactions
    size_t size() const;

    /// Clear all orphans
    void clear();

private:
    struct OrphanEntry {
        primitives::CTransactionRef tx;
        uint64_t from_peer = 0;
        int64_t time_added = 0;
    };

    mutable core::Mutex mutex_;
    std::map<rnet::uint256, OrphanEntry> orphans_;

    /// Index: outpoint -> set of orphan txids that spend it
    std::map<primitives::COutPoint, std::set<rnet::uint256>> outpoint_index_;
};

}  // namespace rnet::mempool

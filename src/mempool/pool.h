#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include "chain/coins.h"
#include "core/error.h"
#include "core/signal.h"
#include "core/sync.h"
#include "core/types.h"
#include "mempool/ancestors.h"
#include "mempool/entry.h"
#include "mempool/fee_estimator.h"
#include "mempool/orphans.h"
#include "mempool/rbf.h"
#include "primitives/transaction.h"

namespace rnet::mempool {

/// CTxMemPool — the transaction memory pool.
/// Holds unconfirmed transactions waiting for inclusion in blocks.
class CTxMemPool {
public:
    /// Maximum mempool size in bytes (default 300 MB)
    static constexpr size_t DEFAULT_MAX_MEMPOOL_SIZE = 300 * 1024 * 1024;

    explicit CTxMemPool(size_t max_size = DEFAULT_MAX_MEMPOOL_SIZE);
    ~CTxMemPool();

    // Non-copyable
    CTxMemPool(const CTxMemPool&) = delete;
    CTxMemPool& operator=(const CTxMemPool&) = delete;

    /// Add a transaction to the mempool.
    /// Validates against the UTXO set and mempool policy.
    Result<void> add_tx(primitives::CTransactionRef tx,
                        int64_t fee, int height,
                        float current_val_loss = 0.0f);

    /// Remove a transaction from the mempool
    void remove_tx(const rnet::uint256& txid);

    /// Remove transactions confirmed in a block
    void remove_for_block(const std::vector<primitives::CTransactionRef>& txs,
                          int block_height);

    /// Check if a transaction is in the mempool
    bool exists(const rnet::uint256& txid) const;

    /// Get a transaction from the mempool
    primitives::CTransactionRef get(const rnet::uint256& txid) const;

    /// Get the mempool entry for a transaction
    const CTxMemPoolEntry* get_entry(const rnet::uint256& txid) const;

    /// Get all transactions sorted by ancestor fee rate (for mining)
    std::vector<primitives::CTransactionRef> get_sorted_txs() const;

    /// Get number of transactions in the mempool
    size_t size() const;

    /// Get total size in bytes of all mempool transactions
    size_t bytes() const;

    /// Get the minimum fee rate to get into the mempool
    primitives::CFeeRate get_min_fee() const;

    /// Trim the mempool to the maximum size by evicting lowest-fee txs
    void trim_to_size(size_t max_size);

    /// Clear all transactions
    void clear();

    /// Get the fee estimator
    FeeEstimator& fee_estimator() { return fee_estimator_; }
    const FeeEstimator& fee_estimator() const { return fee_estimator_; }

    /// Get the orphan pool
    OrphanPool& orphan_pool() { return orphan_pool_; }
    const OrphanPool& orphan_pool() const { return orphan_pool_; }

    /// Get the ancestor tracker
    AncestorTracker& ancestor_tracker() { return ancestor_tracker_; }

    /// Check if an outpoint is spent by any mempool transaction
    bool is_spent_by_mempool(const primitives::COutPoint& outpoint) const;

    /// Get all txids
    std::vector<rnet::uint256> get_txids() const;

    /// Prioritize a transaction (adjust its effective fee)
    void prioritize_tx(const rnet::uint256& txid, int64_t fee_delta);

    /// Signals
    core::Signal<const primitives::CTransactionRef&> on_tx_added;
    core::Signal<const rnet::uint256&> on_tx_removed;

private:
    mutable core::Mutex mutex_;
    size_t max_size_;

    /// Main storage: txid -> entry
    std::unordered_map<rnet::uint256, CTxMemPoolEntry> entries_;

    /// Outpoints spent by mempool transactions
    std::unordered_map<primitives::COutPoint, rnet::uint256> spent_outpoints_;

    /// Total bytes of all transactions
    size_t total_bytes_ = 0;

    FeeEstimator fee_estimator_;
    OrphanPool orphan_pool_;
    AncestorTracker ancestor_tracker_;

    /// Remove a transaction and update internal indices (must hold mutex_)
    void remove_tx_locked(const rnet::uint256& txid);
};

}  // namespace rnet::mempool

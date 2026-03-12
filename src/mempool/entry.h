#pragma once

#include <chrono>
#include <cstdint>

#include "primitives/fees.h"
#include "primitives/transaction.h"

namespace rnet::mempool {

/// CTxMemPoolEntry — a transaction entry in the mempool.
/// Tracks fee, size, time, and ancestor/descendant metadata.
class CTxMemPoolEntry {
public:
    CTxMemPoolEntry() = default;

    explicit CTxMemPoolEntry(primitives::CTransactionRef tx,
                             int64_t fee,
                             int64_t time,
                             int entry_height,
                             float val_loss_at_entry = 0.0f);

    /// The transaction
    const primitives::CTransactionRef& get_tx() const { return tx_; }
    const primitives::CTransaction& tx() const { return *tx_; }
    const rnet::uint256& txid() const { return tx_->txid(); }

    /// Fee paid by this transaction
    int64_t get_fee() const { return fee_; }

    /// Virtual size of the transaction
    size_t get_tx_size() const { return tx_size_; }

    /// Weight of the transaction
    size_t get_tx_weight() const { return tx_weight_; }

    /// Fee rate
    primitives::CFeeRate get_fee_rate() const {
        return primitives::CFeeRate(fee_, tx_size_);
    }

    /// Time this tx entered the mempool (epoch seconds)
    int64_t get_time() const { return time_; }

    /// Block height when this entry was added
    int get_entry_height() const { return entry_height_; }

    /// Val loss at time of entry (for UTXO expiry tracking)
    float get_val_loss() const { return val_loss_at_entry_; }

    /// Modified fee (fee + fee delta from prioritization)
    int64_t get_modified_fee() const { return fee_ + fee_delta_; }

    /// Set fee delta (for prioritization)
    void update_fee_delta(int64_t delta) { fee_delta_ = delta; }
    int64_t get_fee_delta() const { return fee_delta_; }

    // --- Ancestor/descendant tracking ---
    int64_t ancestor_count = 1;
    int64_t ancestor_size = 0;
    int64_t ancestor_fee = 0;

    int64_t descendant_count = 1;
    int64_t descendant_size = 0;
    int64_t descendant_fee = 0;

    /// Update ancestor stats
    void update_ancestors(int64_t count, int64_t size, int64_t fee);

    /// Update descendant stats
    void update_descendants(int64_t count, int64_t size, int64_t fee);

    /// Ancestor fee rate (for package-aware mining)
    primitives::CFeeRate ancestor_fee_rate() const {
        if (ancestor_size == 0) return primitives::CFeeRate(0);
        return primitives::CFeeRate(ancestor_fee,
                                    static_cast<size_t>(ancestor_size));
    }

private:
    primitives::CTransactionRef tx_;
    int64_t fee_ = 0;
    size_t tx_size_ = 0;
    size_t tx_weight_ = 0;
    int64_t time_ = 0;
    int entry_height_ = 0;
    float val_loss_at_entry_ = 0.0f;
    int64_t fee_delta_ = 0;
};

}  // namespace rnet::mempool

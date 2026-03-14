// Copyright (c) 2025 The ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "mempool/entry.h"

namespace rnet::mempool {

// ===========================================================================
//  CTxMemPoolEntry -- mempool bookkeeping for a single transaction
// ===========================================================================

// ---------------------------------------------------------------------------
// Constructor
//   Initialises fee, size, weight, and ancestor/descendant tracking from the
//   transaction itself.
// ---------------------------------------------------------------------------
CTxMemPoolEntry::CTxMemPoolEntry(primitives::CTransactionRef tx,
                                 int64_t fee, int64_t time,
                                 int entry_height, float val_loss_at_entry)
    : tx_(std::move(tx))
    , fee_(fee)
    , time_(time)
    , entry_height_(entry_height)
    , val_loss_at_entry_(val_loss_at_entry)
{
    if (tx_) {
        tx_size_ = tx_->get_virtual_size();
        tx_weight_ = tx_->get_weight();
        ancestor_size = static_cast<int64_t>(tx_size_);
        ancestor_fee = fee_;
        descendant_size = static_cast<int64_t>(tx_size_);
        descendant_fee = fee_;
    }
}

// ---------------------------------------------------------------------------
// update_ancestors
//   Adjusts aggregate ancestor statistics when a parent is added or removed.
// ---------------------------------------------------------------------------
void CTxMemPoolEntry::update_ancestors(int64_t count, int64_t size,
                                       int64_t fee)
{
    ancestor_count += count;
    ancestor_size += size;
    ancestor_fee += fee;
}

// ---------------------------------------------------------------------------
// update_descendants
//   Adjusts aggregate descendant statistics when a child is added or removed.
// ---------------------------------------------------------------------------
void CTxMemPoolEntry::update_descendants(int64_t count, int64_t size,
                                         int64_t fee)
{
    descendant_count += count;
    descendant_size += size;
    descendant_fee += fee;
}

} // namespace rnet::mempool

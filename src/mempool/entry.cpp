#include "mempool/entry.h"

namespace rnet::mempool {

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

void CTxMemPoolEntry::update_ancestors(int64_t count, int64_t size,
                                       int64_t fee) {
    ancestor_count += count;
    ancestor_size += size;
    ancestor_fee += fee;
}

void CTxMemPoolEntry::update_descendants(int64_t count, int64_t size,
                                         int64_t fee) {
    descendant_count += count;
    descendant_size += size;
    descendant_fee += fee;
}

}  // namespace rnet::mempool

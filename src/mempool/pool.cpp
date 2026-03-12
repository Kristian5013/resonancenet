#include "mempool/pool.h"

#include <algorithm>

#include "core/logging.h"
#include "core/time.h"

namespace rnet::mempool {

CTxMemPool::CTxMemPool(size_t max_size)
    : max_size_(max_size) {}

CTxMemPool::~CTxMemPool() = default;

Result<void> CTxMemPool::add_tx(primitives::CTransactionRef tx,
                                int64_t fee, int height,
                                float current_val_loss) {
    if (!tx) {
        return Result<void>::err("Null transaction");
    }

    const auto& txid = tx->txid();

    LOCK(mutex_);

    // Already in mempool?
    if (entries_.count(txid)) {
        return Result<void>::err("tx already in mempool");
    }

    // Check for double-spends within mempool
    for (const auto& txin : tx->vin()) {
        auto sit = spent_outpoints_.find(txin.prevout);
        if (sit != spent_outpoints_.end()) {
            // Check RBF
            if (!signals_opt_in_rbf(*tx)) {
                return Result<void>::err(
                    "txn-mempool-conflict: " + txin.prevout.to_string());
            }
            // TODO: full RBF evaluation
        }
    }

    // Create entry
    CTxMemPoolEntry entry(tx, fee, core::get_time(), height, current_val_loss);

    // Update parent tracking
    std::set<rnet::uint256> parent_txids;
    for (const auto& txin : tx->vin()) {
        if (entries_.count(txin.prevout.hash)) {
            parent_txids.insert(txin.prevout.hash);
        }
    }

    // Record spent outpoints
    for (const auto& txin : tx->vin()) {
        spent_outpoints_[txin.prevout] = txid;
    }

    total_bytes_ += entry.get_tx_size();

    ancestor_tracker_.add_tx(txid, parent_txids,
                             static_cast<int64_t>(entry.get_tx_size()), fee);

    entries_.emplace(txid, std::move(entry));

    // Track for fee estimation
    fee_estimator_.track_tx(txid, fee, height);

    // Trim if over size limit
    if (total_bytes_ > max_size_) {
        trim_to_size(max_size_);
    }

    on_tx_added.emit(tx);

    return Result<void>::ok();
}

void CTxMemPool::remove_tx(const rnet::uint256& txid) {
    LOCK(mutex_);
    remove_tx_locked(txid);
}

void CTxMemPool::remove_tx_locked(const rnet::uint256& txid) {
    auto it = entries_.find(txid);
    if (it == entries_.end()) return;

    // Remove spent outpoints
    for (const auto& txin : it->second.tx().vin()) {
        auto sit = spent_outpoints_.find(txin.prevout);
        if (sit != spent_outpoints_.end() && sit->second == txid) {
            spent_outpoints_.erase(sit);
        }
    }

    total_bytes_ -= it->second.get_tx_size();
    ancestor_tracker_.remove_tx(txid);
    entries_.erase(it);

    on_tx_removed.emit(txid);
}

void CTxMemPool::remove_for_block(
    const std::vector<primitives::CTransactionRef>& txs,
    int block_height)
{
    LOCK(mutex_);

    std::vector<rnet::uint256> confirmed_txids;
    for (const auto& tx : txs) {
        const auto& txid = tx->txid();
        confirmed_txids.push_back(txid);
        remove_tx_locked(txid);
    }

    fee_estimator_.process_block(block_height, confirmed_txids);
}

bool CTxMemPool::exists(const rnet::uint256& txid) const {
    LOCK(mutex_);
    return entries_.count(txid) > 0;
}

primitives::CTransactionRef CTxMemPool::get(
    const rnet::uint256& txid) const
{
    LOCK(mutex_);
    auto it = entries_.find(txid);
    if (it != entries_.end()) return it->second.get_tx();
    return nullptr;
}

const CTxMemPoolEntry* CTxMemPool::get_entry(
    const rnet::uint256& txid) const
{
    LOCK(mutex_);
    auto it = entries_.find(txid);
    if (it != entries_.end()) return &it->second;
    return nullptr;
}

std::vector<primitives::CTransactionRef> CTxMemPool::get_sorted_txs() const {
    LOCK(mutex_);

    // Sort by ancestor fee rate (descending) for mining
    struct SortEntry {
        primitives::CTransactionRef tx;
        int64_t ancestor_fee_rate;
    };

    std::vector<SortEntry> sorted;
    sorted.reserve(entries_.size());
    for (const auto& [txid, entry] : entries_) {
        sorted.push_back({
            entry.get_tx(),
            entry.ancestor_fee_rate().get_fee_per_kvb()
        });
    }

    std::sort(sorted.begin(), sorted.end(),
              [](const SortEntry& a, const SortEntry& b) {
                  return a.ancestor_fee_rate > b.ancestor_fee_rate;
              });

    std::vector<primitives::CTransactionRef> result;
    result.reserve(sorted.size());
    for (auto& s : sorted) {
        result.push_back(std::move(s.tx));
    }
    return result;
}

size_t CTxMemPool::size() const {
    LOCK(mutex_);
    return entries_.size();
}

size_t CTxMemPool::bytes() const {
    LOCK(mutex_);
    return total_bytes_;
}

primitives::CFeeRate CTxMemPool::get_min_fee() const {
    LOCK(mutex_);

    if (total_bytes_ < max_size_ / 2) {
        return primitives::MIN_RELAY_TX_FEE;
    }

    // As the mempool fills up, increase the minimum fee
    double fill_ratio = static_cast<double>(total_bytes_) /
                        static_cast<double>(max_size_);
    int64_t base = primitives::MIN_RELAY_TX_FEE.get_fee_per_kvb();
    int64_t scaled = static_cast<int64_t>(
        static_cast<double>(base) * fill_ratio * 2.0);
    return primitives::CFeeRate(std::max(base, scaled));
}

void CTxMemPool::trim_to_size(size_t max_size) {
    // Caller must hold mutex_

    // Evict by lowest fee rate until under budget
    while (total_bytes_ > max_size && !entries_.empty()) {
        // Find the entry with lowest fee rate
        auto worst = entries_.begin();
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->second.get_fee_rate() < worst->second.get_fee_rate()) {
                worst = it;
            }
        }
        auto txid = worst->first;
        remove_tx_locked(txid);
    }
}

void CTxMemPool::clear() {
    LOCK(mutex_);
    entries_.clear();
    spent_outpoints_.clear();
    total_bytes_ = 0;
    ancestor_tracker_.clear();
    orphan_pool_.clear();
}

bool CTxMemPool::is_spent_by_mempool(
    const primitives::COutPoint& outpoint) const
{
    LOCK(mutex_);
    return spent_outpoints_.count(outpoint) > 0;
}

std::vector<rnet::uint256> CTxMemPool::get_txids() const {
    LOCK(mutex_);
    std::vector<rnet::uint256> result;
    result.reserve(entries_.size());
    for (const auto& [txid, entry] : entries_) {
        result.push_back(txid);
    }
    return result;
}

void CTxMemPool::prioritize_tx(const rnet::uint256& txid,
                               int64_t fee_delta) {
    LOCK(mutex_);
    auto it = entries_.find(txid);
    if (it != entries_.end()) {
        it->second.update_fee_delta(fee_delta);
    }
}

}  // namespace rnet::mempool

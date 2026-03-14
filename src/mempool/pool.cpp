// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "mempool/pool.h"

#include "core/logging.h"
#include "core/time.h"

#include <algorithm>

namespace rnet::mempool {

// ---------------------------------------------------------------------------
// CTxMemPool :: constructor / destructor
// ---------------------------------------------------------------------------

CTxMemPool::CTxMemPool(size_t max_size)
    : max_size_(max_size) {}

CTxMemPool::~CTxMemPool() = default;

// ---------------------------------------------------------------------------
// CTxMemPool :: add_tx
// ---------------------------------------------------------------------------
//
// Design note — mempool admission
//   Accepting a new transaction into the mempool involves several
//   policy checks that mirror Bitcoin Core's AcceptToMemoryPool flow:
//     (a) reject nulls and duplicates,
//     (b) detect double-spends (with optional RBF override),
//     (c) build the ancestor/descendant graph,
//     (d) enforce the size ceiling by evicting the cheapest txs.
//   The fee estimator is updated last so that only admitted txs are
//   tracked for future fee-rate predictions.
// ---------------------------------------------------------------------------

Result<void> CTxMemPool::add_tx(primitives::CTransactionRef tx,
                                int64_t fee, int height,
                                float current_val_loss) {
    // 1. Reject null transactions early.
    if (!tx) {
        return Result<void>::err("Null transaction");
    }

    const auto& txid = tx->txid();

    LOCK(mutex_);

    // 2. Reject duplicates already present in the pool.
    if (entries_.count(txid)) {
        return Result<void>::err("tx already in mempool");
    }

    // 3. Scan inputs for double-spends against existing mempool entries.
    for (const auto& txin : tx->vin()) {
        auto sit = spent_outpoints_.find(txin.prevout);
        if (sit != spent_outpoints_.end()) {
            // 3a. Allow replacement only when the new tx signals RBF.
            if (!signals_opt_in_rbf(*tx)) {
                return Result<void>::err(
                    "txn-mempool-conflict: " + txin.prevout.to_string());
            }
            // TODO: full RBF evaluation
        }
    }

    // 4. Create the mempool entry with current timestamp.
    CTxMemPoolEntry entry(tx, fee, core::get_time(), height, current_val_loss);

    // 5. Collect in-mempool parent txids for ancestor tracking.
    std::set<rnet::uint256> parent_txids;
    for (const auto& txin : tx->vin()) {
        if (entries_.count(txin.prevout.hash)) {
            parent_txids.insert(txin.prevout.hash);
        }
    }

    // 6. Record every input as spent within the mempool.
    for (const auto& txin : tx->vin()) {
        spent_outpoints_[txin.prevout] = txid;
    }

    // 7. Update cumulative byte counter and ancestor graph.
    total_bytes_ += entry.get_tx_size();

    ancestor_tracker_.add_tx(txid, parent_txids,
                             static_cast<int64_t>(entry.get_tx_size()), fee);

    entries_.emplace(txid, std::move(entry));

    // 8. Feed the fee estimator so future estimates reflect this tx.
    fee_estimator_.track_tx(txid, fee, height);

    // 9. Evict low-fee transactions if the pool exceeds the size budget.
    if (total_bytes_ > max_size_) {
        trim_to_size(max_size_);
    }

    // 10. Notify subscribers (e.g. wallet, GUI).
    on_tx_added.emit(tx);

    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// CTxMemPool :: remove_tx
// ---------------------------------------------------------------------------

void CTxMemPool::remove_tx(const rnet::uint256& txid) {
    LOCK(mutex_);
    remove_tx_locked(txid);
}

// ---------------------------------------------------------------------------
// CTxMemPool :: remove_tx_locked
// ---------------------------------------------------------------------------
//
// Design note — internal removal (caller must hold mutex_)
//   Cleans up every secondary index that references the transaction:
//   spent-outpoints map, byte counter, and the ancestor graph.
//   Emits on_tx_removed so listeners can react (e.g. GUI refresh).
// ---------------------------------------------------------------------------

void CTxMemPool::remove_tx_locked(const rnet::uint256& txid) {
    auto it = entries_.find(txid);
    if (it == entries_.end()) return;

    // 1. Release all outpoints that this transaction was spending.
    for (const auto& txin : it->second.tx().vin()) {
        auto sit = spent_outpoints_.find(txin.prevout);
        if (sit != spent_outpoints_.end() && sit->second == txid) {
            spent_outpoints_.erase(sit);
        }
    }

    // 2. Subtract from the cumulative byte total.
    total_bytes_ -= it->second.get_tx_size();

    // 3. Detach from the ancestor/descendant graph.
    ancestor_tracker_.remove_tx(txid);

    // 4. Erase the entry itself.
    entries_.erase(it);

    // 5. Broadcast removal signal.
    on_tx_removed.emit(txid);
}

// ---------------------------------------------------------------------------
// CTxMemPool :: remove_for_block
// ---------------------------------------------------------------------------
//
// Design note — block connection
//   When a new block arrives every confirmed transaction must leave the
//   mempool.  The fee estimator is updated with the block height so it
//   can record how many blocks each tracked tx took to confirm.
// ---------------------------------------------------------------------------

void CTxMemPool::remove_for_block(
    const std::vector<primitives::CTransactionRef>& txs,
    int block_height)
{
    LOCK(mutex_);

    // 1. Collect txids and remove each confirmed transaction.
    std::vector<rnet::uint256> confirmed_txids;
    for (const auto& tx : txs) {
        const auto& txid = tx->txid();
        confirmed_txids.push_back(txid);
        remove_tx_locked(txid);
    }

    // 2. Let the fee estimator learn from the confirmed set.
    fee_estimator_.process_block(block_height, confirmed_txids);
}

// ---------------------------------------------------------------------------
// CTxMemPool :: exists
// ---------------------------------------------------------------------------

bool CTxMemPool::exists(const rnet::uint256& txid) const {
    LOCK(mutex_);
    return entries_.count(txid) > 0;
}

// ---------------------------------------------------------------------------
// CTxMemPool :: get
// ---------------------------------------------------------------------------

primitives::CTransactionRef CTxMemPool::get(
    const rnet::uint256& txid) const
{
    LOCK(mutex_);
    auto it = entries_.find(txid);
    if (it != entries_.end()) return it->second.get_tx();
    return nullptr;
}

// ---------------------------------------------------------------------------
// CTxMemPool :: get_entry
// ---------------------------------------------------------------------------

const CTxMemPoolEntry* CTxMemPool::get_entry(
    const rnet::uint256& txid) const
{
    LOCK(mutex_);
    auto it = entries_.find(txid);
    if (it != entries_.end()) return &it->second;
    return nullptr;
}

// ---------------------------------------------------------------------------
// CTxMemPool :: get_sorted_txs
// ---------------------------------------------------------------------------
//
// Design note — mining order
//   The miner calls this to obtain the best transaction ordering.
//   Transactions are sorted by descending ancestor-fee-rate so that
//   high-value packages (CPFP chains) float to the top of the block
//   template, maximising total collected fees.
// ---------------------------------------------------------------------------

std::vector<primitives::CTransactionRef> CTxMemPool::get_sorted_txs() const {
    LOCK(mutex_);

    // 1. Build a lightweight vector with fee-rate metadata.
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

    // 2. Sort descending by ancestor fee rate.
    std::sort(sorted.begin(), sorted.end(),
              [](const SortEntry& a, const SortEntry& b) {
                  return a.ancestor_fee_rate > b.ancestor_fee_rate;
              });

    // 3. Extract the bare transaction pointers.
    std::vector<primitives::CTransactionRef> result;
    result.reserve(sorted.size());
    for (auto& s : sorted) {
        result.push_back(std::move(s.tx));
    }
    return result;
}

// ---------------------------------------------------------------------------
// CTxMemPool :: size
// ---------------------------------------------------------------------------

size_t CTxMemPool::size() const {
    LOCK(mutex_);
    return entries_.size();
}

// ---------------------------------------------------------------------------
// CTxMemPool :: bytes
// ---------------------------------------------------------------------------

size_t CTxMemPool::bytes() const {
    LOCK(mutex_);
    return total_bytes_;
}

// ---------------------------------------------------------------------------
// CTxMemPool :: get_min_fee
// ---------------------------------------------------------------------------
//
// Design note — dynamic minimum fee
//   When the pool is less than half full the minimum relay fee applies.
//   As utilisation rises the minimum scales linearly so that senders
//   are encouraged to bid higher during congestion, preventing the
//   pool from being flooded with cheap transactions.
// ---------------------------------------------------------------------------

primitives::CFeeRate CTxMemPool::get_min_fee() const {
    LOCK(mutex_);

    // 1. Below 50 % capacity the base relay fee is sufficient.
    if (total_bytes_ < max_size_ / 2) {
        return primitives::MIN_RELAY_TX_FEE;
    }

    // 2. Scale linearly with fill ratio (2x at 100 % full).
    double fill_ratio = static_cast<double>(total_bytes_) /
                        static_cast<double>(max_size_);
    int64_t base = primitives::MIN_RELAY_TX_FEE.get_fee_per_kvb();
    int64_t scaled = static_cast<int64_t>(
        static_cast<double>(base) * fill_ratio * 2.0);
    return primitives::CFeeRate(std::max(base, scaled));
}

// ---------------------------------------------------------------------------
// CTxMemPool :: trim_to_size
// ---------------------------------------------------------------------------
//
// Design note — eviction policy
//   Repeatedly evicts the transaction with the lowest individual fee
//   rate until the pool fits within the byte budget.  This is a
//   simplified strategy; a production implementation would consider
//   ancestor-package fee rates and descendant scoring.
//   Caller must already hold mutex_.
// ---------------------------------------------------------------------------

void CTxMemPool::trim_to_size(size_t max_size) {
    // 1. Evict the cheapest transaction on each iteration.
    while (total_bytes_ > max_size && !entries_.empty()) {
        // 2. Linear scan for the lowest fee-rate entry.
        auto worst = entries_.begin();
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->second.get_fee_rate() < worst->second.get_fee_rate()) {
                worst = it;
            }
        }
        // 3. Remove the victim and repeat.
        auto txid = worst->first;
        remove_tx_locked(txid);
    }
}

// ---------------------------------------------------------------------------
// CTxMemPool :: clear
// ---------------------------------------------------------------------------

void CTxMemPool::clear() {
    LOCK(mutex_);

    // 1. Wipe every internal data structure.
    entries_.clear();
    spent_outpoints_.clear();
    total_bytes_ = 0;
    ancestor_tracker_.clear();
    orphan_pool_.clear();
}

// ---------------------------------------------------------------------------
// CTxMemPool :: is_spent_by_mempool
// ---------------------------------------------------------------------------

bool CTxMemPool::is_spent_by_mempool(
    const primitives::COutPoint& outpoint) const
{
    LOCK(mutex_);
    return spent_outpoints_.count(outpoint) > 0;
}

// ---------------------------------------------------------------------------
// CTxMemPool :: get_txids
// ---------------------------------------------------------------------------

std::vector<rnet::uint256> CTxMemPool::get_txids() const {
    LOCK(mutex_);

    std::vector<rnet::uint256> result;
    result.reserve(entries_.size());
    for (const auto& [txid, entry] : entries_) {
        result.push_back(txid);
    }
    return result;
}

// ---------------------------------------------------------------------------
// CTxMemPool :: prioritize_tx
// ---------------------------------------------------------------------------

void CTxMemPool::prioritize_tx(const rnet::uint256& txid,
                               int64_t fee_delta) {
    LOCK(mutex_);

    auto it = entries_.find(txid);
    if (it != entries_.end()) {
        it->second.update_fee_delta(fee_delta);
    }
}

} // namespace rnet::mempool

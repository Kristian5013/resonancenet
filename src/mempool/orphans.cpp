// Copyright (c) 2025 The ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "mempool/orphans.h"

#include "core/random.h"
#include "core/time.h"

namespace rnet::mempool {

// ===========================================================================
//  OrphanPool -- holds transactions whose parents are not yet known
// ===========================================================================

// ---------------------------------------------------------------------------
// add_tx
//   Inserts a transaction into the orphan pool.  Rejects duplicates and
//   oversized transactions.  Maintains the outpoint index and enforces
//   the MAX_ORPHAN_TXS size limit.
// ---------------------------------------------------------------------------
bool OrphanPool::add_tx(primitives::CTransactionRef tx, uint64_t peer_id)
{
    LOCK(mutex_);

    const auto& txid = tx->txid();

    // 1. Already have it?
    if (orphans_.count(txid)) return false;

    // 2. Size check
    if (tx->get_total_size() > MAX_ORPHAN_TX_SIZE) return false;

    // 3. Build entry
    OrphanEntry entry;
    entry.tx = tx;
    entry.from_peer = peer_id;
    entry.time_added = core::get_time();

    // 4. Index by outpoints this tx spends
    for (const auto& txin : tx->vin()) {
        outpoint_index_[txin.prevout].insert(txid);
    }

    orphans_[txid] = std::move(entry);

    // 5. Enforce size limit
    limit_orphans();

    return true;
}

// ---------------------------------------------------------------------------
// erase_tx
//   Removes a single orphan by txid, cleaning up the outpoint index.
// ---------------------------------------------------------------------------
bool OrphanPool::erase_tx(const rnet::uint256& txid)
{
    LOCK(mutex_);

    auto it = orphans_.find(txid);
    if (it == orphans_.end()) return false;

    // 1. Remove outpoint index entries
    for (const auto& txin : it->second.tx->vin()) {
        auto oit = outpoint_index_.find(txin.prevout);
        if (oit != outpoint_index_.end()) {
            oit->second.erase(txid);
            if (oit->second.empty()) {
                outpoint_index_.erase(oit);
            }
        }
    }

    orphans_.erase(it);
    return true;
}

// ---------------------------------------------------------------------------
// erase_for_peer
//   Removes all orphans submitted by a specific peer (e.g. on disconnect).
// ---------------------------------------------------------------------------
void OrphanPool::erase_for_peer(uint64_t peer_id)
{
    LOCK(mutex_);

    // 1. Collect txids to remove
    std::vector<rnet::uint256> to_remove;
    for (const auto& [txid, entry] : orphans_) {
        if (entry.from_peer == peer_id) {
            to_remove.push_back(txid);
        }
    }

    // 2. Remove each orphan and its outpoint index entries
    for (const auto& txid : to_remove) {
        auto it = orphans_.find(txid);
        if (it != orphans_.end()) {
            for (const auto& txin : it->second.tx->vin()) {
                auto oit = outpoint_index_.find(txin.prevout);
                if (oit != outpoint_index_.end()) {
                    oit->second.erase(txid);
                    if (oit->second.empty()) {
                        outpoint_index_.erase(oit);
                    }
                }
            }
            orphans_.erase(it);
        }
    }
}

// ---------------------------------------------------------------------------
// get_children_of
//   Returns all orphan transactions that spend an output of parent_txid.
//   Used to re-evaluate orphans after a parent transaction is accepted.
// ---------------------------------------------------------------------------
std::vector<primitives::CTransactionRef> OrphanPool::get_children_of(
    const rnet::uint256& parent_txid) const
{
    LOCK(mutex_);

    std::vector<primitives::CTransactionRef> result;

    // 1. Scan outpoint_index for any outpoint with hash == parent_txid
    for (const auto& [outpoint, txids] : outpoint_index_) {
        if (outpoint.hash == parent_txid) {
            for (const auto& txid : txids) {
                auto it = orphans_.find(txid);
                if (it != orphans_.end()) {
                    result.push_back(it->second.tx);
                }
            }
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// limit_orphans  (private)
//   Evicts random orphans until pool size is at or below MAX_ORPHAN_TXS.
//   Caller must hold mutex_.
// ---------------------------------------------------------------------------
void OrphanPool::limit_orphans()
{
    while (orphans_.size() > MAX_ORPHAN_TXS) {
        // 1. Pick a random orphan
        auto rand_idx = core::get_rand_range(orphans_.size());
        auto it = orphans_.begin();
        std::advance(it, static_cast<ptrdiff_t>(rand_idx));

        // 2. Remove its outpoint indices
        auto txid = it->first;
        for (const auto& txin : it->second.tx->vin()) {
            auto oit = outpoint_index_.find(txin.prevout);
            if (oit != outpoint_index_.end()) {
                oit->second.erase(txid);
                if (oit->second.empty()) {
                    outpoint_index_.erase(oit);
                }
            }
        }
        orphans_.erase(it);
    }
}

// ---------------------------------------------------------------------------
// have_tx
// ---------------------------------------------------------------------------
bool OrphanPool::have_tx(const rnet::uint256& txid) const
{
    LOCK(mutex_);
    return orphans_.count(txid) > 0;
}

// ---------------------------------------------------------------------------
// get_tx
// ---------------------------------------------------------------------------
primitives::CTransactionRef OrphanPool::get_tx(
    const rnet::uint256& txid) const
{
    LOCK(mutex_);
    auto it = orphans_.find(txid);
    if (it != orphans_.end()) return it->second.tx;
    return nullptr;
}

// ---------------------------------------------------------------------------
// size
// ---------------------------------------------------------------------------
size_t OrphanPool::size() const
{
    LOCK(mutex_);
    return orphans_.size();
}

// ---------------------------------------------------------------------------
// clear
// ---------------------------------------------------------------------------
void OrphanPool::clear()
{
    LOCK(mutex_);
    orphans_.clear();
    outpoint_index_.clear();
}

} // namespace rnet::mempool

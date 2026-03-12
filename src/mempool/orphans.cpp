#include "mempool/orphans.h"

#include "core/random.h"
#include "core/time.h"

namespace rnet::mempool {

bool OrphanPool::add_tx(primitives::CTransactionRef tx, uint64_t peer_id) {
    LOCK(mutex_);

    const auto& txid = tx->txid();

    // Already have it?
    if (orphans_.count(txid)) return false;

    // Size check
    if (tx->get_total_size() > MAX_ORPHAN_TX_SIZE) return false;

    OrphanEntry entry;
    entry.tx = tx;
    entry.from_peer = peer_id;
    entry.time_added = core::get_time();

    // Index by outpoints this tx spends
    for (const auto& txin : tx->vin()) {
        outpoint_index_[txin.prevout].insert(txid);
    }

    orphans_[txid] = std::move(entry);

    // Enforce size limit
    limit_orphans();

    return true;
}

bool OrphanPool::erase_tx(const rnet::uint256& txid) {
    LOCK(mutex_);

    auto it = orphans_.find(txid);
    if (it == orphans_.end()) return false;

    // Remove outpoint index entries
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

void OrphanPool::erase_for_peer(uint64_t peer_id) {
    LOCK(mutex_);

    std::vector<rnet::uint256> to_remove;
    for (const auto& [txid, entry] : orphans_) {
        if (entry.from_peer == peer_id) {
            to_remove.push_back(txid);
        }
    }

    // Unlock for removal (use the public method which re-locks)
    // Actually, since we hold the lock, do it inline:
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

std::vector<primitives::CTransactionRef> OrphanPool::get_children_of(
    const rnet::uint256& parent_txid) const
{
    LOCK(mutex_);

    std::vector<primitives::CTransactionRef> result;

    // Check all outpoints that could come from this parent
    // We need to scan outpoint_index for any outpoint with hash == parent_txid
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

void OrphanPool::limit_orphans() {
    // Caller must hold mutex_
    while (orphans_.size() > MAX_ORPHAN_TXS) {
        // Remove a random orphan
        auto rand_idx = core::get_rand_range(orphans_.size());
        auto it = orphans_.begin();
        std::advance(it, static_cast<ptrdiff_t>(rand_idx));

        auto txid = it->first;
        // Remove outpoint indices
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

bool OrphanPool::have_tx(const rnet::uint256& txid) const {
    LOCK(mutex_);
    return orphans_.count(txid) > 0;
}

primitives::CTransactionRef OrphanPool::get_tx(
    const rnet::uint256& txid) const
{
    LOCK(mutex_);
    auto it = orphans_.find(txid);
    if (it != orphans_.end()) return it->second.tx;
    return nullptr;
}

size_t OrphanPool::size() const {
    LOCK(mutex_);
    return orphans_.size();
}

void OrphanPool::clear() {
    LOCK(mutex_);
    orphans_.clear();
    outpoint_index_.clear();
}

}  // namespace rnet::mempool

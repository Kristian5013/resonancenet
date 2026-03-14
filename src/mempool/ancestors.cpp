// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "mempool/ancestors.h"

namespace rnet::mempool {

// ---------------------------------------------------------------------------
// AncestorTracker :: add_tx
// ---------------------------------------------------------------------------
//
// Design note — bidirectional graph insertion
//   Each transaction keeps a set of parents and children.  When a new
//   tx is inserted we also update every existing parent so that its
//   children set reflects the new link.  This bidirectional structure
//   allows O(1) local look-ups for both ancestor and descendant
//   traversals without re-scanning the entire pool.
// ---------------------------------------------------------------------------

void AncestorTracker::add_tx(const rnet::uint256& txid,
                             const std::set<rnet::uint256>& parent_txids,
                             int64_t size, int64_t fee) {
    // 1. Create the link record for the new transaction.
    auto& links = links_[txid];
    links.parents = parent_txids;
    links.size = size;
    links.fee = fee;

    // 2. Register as a child of every in-pool parent.
    for (const auto& parent : parent_txids) {
        auto it = links_.find(parent);
        if (it != links_.end()) {
            it->second.children.insert(txid);
        }
    }
}

// ---------------------------------------------------------------------------
// AncestorTracker :: remove_tx
// ---------------------------------------------------------------------------
//
// Design note — graph detachment
//   Removing a tx requires severing both directions: the tx must be
//   erased from every parent's children set and from every child's
//   parents set.  Failure to do so would leave dangling references
//   that corrupt subsequent ancestor/descendant queries.
// ---------------------------------------------------------------------------

void AncestorTracker::remove_tx(const rnet::uint256& txid) {
    auto it = links_.find(txid);
    if (it == links_.end()) return;

    // 1. Detach from each parent's children set.
    for (const auto& parent : it->second.parents) {
        auto pit = links_.find(parent);
        if (pit != links_.end()) {
            pit->second.children.erase(txid);
        }
    }

    // 2. Detach from each child's parents set.
    for (const auto& child : it->second.children) {
        auto cit = links_.find(child);
        if (cit != links_.end()) {
            cit->second.parents.erase(txid);
        }
    }

    // 3. Erase the link record itself.
    links_.erase(it);
}

// ---------------------------------------------------------------------------
// AncestorTracker :: get_ancestors
// ---------------------------------------------------------------------------

std::set<rnet::uint256> AncestorTracker::get_ancestors(
    const rnet::uint256& txid) const
{
    std::set<rnet::uint256> result;
    collect_ancestors(txid, result);
    return result;
}

// ---------------------------------------------------------------------------
// AncestorTracker :: get_descendants
// ---------------------------------------------------------------------------

std::set<rnet::uint256> AncestorTracker::get_descendants(
    const rnet::uint256& txid) const
{
    std::set<rnet::uint256> result;
    collect_descendants(txid, result);
    return result;
}

// ---------------------------------------------------------------------------
// AncestorTracker :: get_parents
// ---------------------------------------------------------------------------

std::set<rnet::uint256> AncestorTracker::get_parents(
    const rnet::uint256& txid) const
{
    auto it = links_.find(txid);
    if (it == links_.end()) return {};
    return it->second.parents;
}

// ---------------------------------------------------------------------------
// AncestorTracker :: get_children
// ---------------------------------------------------------------------------

std::set<rnet::uint256> AncestorTracker::get_children(
    const rnet::uint256& txid) const
{
    auto it = links_.find(txid);
    if (it == links_.end()) return {};
    return it->second.children;
}

// ---------------------------------------------------------------------------
// AncestorTracker :: check_limits
// ---------------------------------------------------------------------------
//
// Design note — package-limit enforcement
//   Before admitting a new transaction we simulate what the ancestor
//   and descendant counts would become.  If any limit would be
//   exceeded the function returns false and the caller should reject
//   the tx.  This prevents long CPFP chains that could degrade
//   mining-sort performance and increase eviction complexity.
// ---------------------------------------------------------------------------

bool AncestorTracker::check_limits(
    const rnet::uint256& /*txid*/,
    const std::set<rnet::uint256>& parent_txids,
    int64_t size,
    const AncestorLimits& limits) const
{
    // 1. Compute projected ancestor count (self + all transitive ancestors).
    int64_t total_ancestors = 1;  // self
    int64_t total_ancestor_size = size;

    for (const auto& parent : parent_txids) {
        auto ancestors = get_ancestors(parent);
        total_ancestors += 1 + static_cast<int64_t>(ancestors.size());
    }

    // 2. Reject if ancestor limits would be breached.
    if (total_ancestors > limits.max_ancestors) return false;
    if (total_ancestor_size > limits.max_ancestor_size) return false;

    // 3. Check descendant limits on each parent.
    for (const auto& parent : parent_txids) {
        auto desc = get_descendants(parent);
        if (static_cast<int64_t>(desc.size()) + 1 > limits.max_descendants) {
            return false;
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// AncestorTracker :: ancestor_count
// ---------------------------------------------------------------------------

int64_t AncestorTracker::ancestor_count(const rnet::uint256& txid) const {
    auto ancestors = get_ancestors(txid);
    return static_cast<int64_t>(ancestors.size());
}

// ---------------------------------------------------------------------------
// AncestorTracker :: descendant_count
// ---------------------------------------------------------------------------

int64_t AncestorTracker::descendant_count(const rnet::uint256& txid) const {
    auto descendants = get_descendants(txid);
    return static_cast<int64_t>(descendants.size());
}

// ---------------------------------------------------------------------------
// AncestorTracker :: clear
// ---------------------------------------------------------------------------

void AncestorTracker::clear() {
    links_.clear();
}

// ---------------------------------------------------------------------------
// AncestorTracker :: collect_ancestors  (private recursive helper)
// ---------------------------------------------------------------------------

void AncestorTracker::collect_ancestors(
    const rnet::uint256& txid, std::set<rnet::uint256>& result) const
{
    auto it = links_.find(txid);
    if (it == links_.end()) return;

    for (const auto& parent : it->second.parents) {
        if (result.insert(parent).second) {
            collect_ancestors(parent, result);
        }
    }
}

// ---------------------------------------------------------------------------
// AncestorTracker :: collect_descendants  (private recursive helper)
// ---------------------------------------------------------------------------

void AncestorTracker::collect_descendants(
    const rnet::uint256& txid, std::set<rnet::uint256>& result) const
{
    auto it = links_.find(txid);
    if (it == links_.end()) return;

    for (const auto& child : it->second.children) {
        if (result.insert(child).second) {
            collect_descendants(child, result);
        }
    }
}

} // namespace rnet::mempool

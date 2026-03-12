#include "mempool/ancestors.h"

namespace rnet::mempool {

void AncestorTracker::add_tx(const rnet::uint256& txid,
                             const std::set<rnet::uint256>& parent_txids,
                             int64_t size, int64_t fee) {
    auto& links = links_[txid];
    links.parents = parent_txids;
    links.size = size;
    links.fee = fee;

    // Register as child of each parent
    for (const auto& parent : parent_txids) {
        auto it = links_.find(parent);
        if (it != links_.end()) {
            it->second.children.insert(txid);
        }
    }
}

void AncestorTracker::remove_tx(const rnet::uint256& txid) {
    auto it = links_.find(txid);
    if (it == links_.end()) return;

    // Remove from parents' children lists
    for (const auto& parent : it->second.parents) {
        auto pit = links_.find(parent);
        if (pit != links_.end()) {
            pit->second.children.erase(txid);
        }
    }

    // Remove from children's parents lists
    for (const auto& child : it->second.children) {
        auto cit = links_.find(child);
        if (cit != links_.end()) {
            cit->second.parents.erase(txid);
        }
    }

    links_.erase(it);
}

std::set<rnet::uint256> AncestorTracker::get_ancestors(
    const rnet::uint256& txid) const
{
    std::set<rnet::uint256> result;
    collect_ancestors(txid, result);
    return result;
}

std::set<rnet::uint256> AncestorTracker::get_descendants(
    const rnet::uint256& txid) const
{
    std::set<rnet::uint256> result;
    collect_descendants(txid, result);
    return result;
}

std::set<rnet::uint256> AncestorTracker::get_parents(
    const rnet::uint256& txid) const
{
    auto it = links_.find(txid);
    if (it == links_.end()) return {};
    return it->second.parents;
}

std::set<rnet::uint256> AncestorTracker::get_children(
    const rnet::uint256& txid) const
{
    auto it = links_.find(txid);
    if (it == links_.end()) return {};
    return it->second.children;
}

bool AncestorTracker::check_limits(
    const rnet::uint256& /*txid*/,
    const std::set<rnet::uint256>& parent_txids,
    int64_t size,
    const AncestorLimits& limits) const
{
    // Calculate what the ancestor stats would be
    int64_t total_ancestors = 1;  // self
    int64_t total_ancestor_size = size;

    for (const auto& parent : parent_txids) {
        auto ancestors = get_ancestors(parent);
        total_ancestors += 1 + static_cast<int64_t>(ancestors.size());
    }

    if (total_ancestors > limits.max_ancestors) return false;
    if (total_ancestor_size > limits.max_ancestor_size) return false;

    // Check descendant limits on parents
    for (const auto& parent : parent_txids) {
        auto desc = get_descendants(parent);
        if (static_cast<int64_t>(desc.size()) + 1 > limits.max_descendants) {
            return false;
        }
    }

    return true;
}

int64_t AncestorTracker::ancestor_count(const rnet::uint256& txid) const {
    auto ancestors = get_ancestors(txid);
    return static_cast<int64_t>(ancestors.size());
}

int64_t AncestorTracker::descendant_count(const rnet::uint256& txid) const {
    auto descendants = get_descendants(txid);
    return static_cast<int64_t>(descendants.size());
}

void AncestorTracker::clear() {
    links_.clear();
}

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

}  // namespace rnet::mempool

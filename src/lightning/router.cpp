#include "lightning/router.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <set>
#include <unordered_set>

#include "core/logging.h"

namespace rnet::lightning {

// ── ChannelEdge ─────────────────────────────────────────────────────

int64_t ChannelEdge::compute_fee(int64_t amount) const {
    return fee_base + (amount * fee_rate_ppm) / 1'000'000;
}

// ── Route ───────────────────────────────────────────────────────────

int64_t Route::total_amount() const {
    if (hops.empty()) return 0;
    return hops.front().amount;
}

// ── Graph management ────────────────────────────────────────────────

void Router::add_node(GraphNode node) {
    LOCK(mutex_);
    auto key = node.node_id;
    nodes_[key] = std::move(node);
}

void Router::remove_node(const crypto::Ed25519PublicKey& node_id) {
    LOCK(mutex_);
    nodes_.erase(node_id);
    adjacency_.erase(node_id);
}

void Router::add_channel(ChannelEdge edge) {
    LOCK(mutex_);
    uint64_t scid = edge.short_channel_id;
    auto node1 = edge.node1;
    auto node2 = edge.node2;
    channels_[scid] = std::move(edge);

    // Update adjacency for both nodes
    auto& adj1 = adjacency_[node1];
    // Remove existing entry for this channel
    adj1.erase(std::remove_if(adj1.begin(), adj1.end(),
        [scid](const AdjEntry& e) { return e.short_channel_id == scid; }),
        adj1.end());
    adj1.push_back({scid, node2});

    auto& adj2 = adjacency_[node2];
    adj2.erase(std::remove_if(adj2.begin(), adj2.end(),
        [scid](const AdjEntry& e) { return e.short_channel_id == scid; }),
        adj2.end());
    adj2.push_back({scid, node1});
}

void Router::remove_channel(uint64_t short_channel_id) {
    LOCK(mutex_);
    auto it = channels_.find(short_channel_id);
    if (it == channels_.end()) return;

    auto node1 = it->second.node1;
    auto node2 = it->second.node2;
    channels_.erase(it);

    // Clean adjacency
    auto clean = [short_channel_id](std::vector<AdjEntry>& adj) {
        adj.erase(std::remove_if(adj.begin(), adj.end(),
            [short_channel_id](const AdjEntry& e) {
                return e.short_channel_id == short_channel_id;
            }), adj.end());
    };

    if (auto ait = adjacency_.find(node1); ait != adjacency_.end()) {
        clean(ait->second);
    }
    if (auto ait = adjacency_.find(node2); ait != adjacency_.end()) {
        clean(ait->second);
    }
}

void Router::set_channel_disabled(uint64_t short_channel_id, bool disabled) {
    LOCK(mutex_);
    auto it = channels_.find(short_channel_id);
    if (it != channels_.end()) {
        it->second.disabled = disabled;
    }
}

const GraphNode* Router::get_node(
    const crypto::Ed25519PublicKey& node_id) const {
    LOCK(mutex_);
    auto it = nodes_.find(node_id);
    return it != nodes_.end() ? &it->second : nullptr;
}

const ChannelEdge* Router::get_channel(uint64_t short_channel_id) const {
    LOCK(mutex_);
    auto it = channels_.find(short_channel_id);
    return it != channels_.end() ? &it->second : nullptr;
}

size_t Router::node_count() const {
    LOCK(mutex_);
    return nodes_.size();
}

size_t Router::channel_count() const {
    LOCK(mutex_);
    return channels_.size();
}

// ── Dijkstra pathfinding ────────────────────────────────────────────

Result<Route> Router::find_route(
    const crypto::Ed25519PublicKey& source,
    const crypto::Ed25519PublicKey& destination,
    int64_t amount,
    uint32_t final_cltv_expiry,
    uint32_t max_hops,
    const std::vector<uint64_t>& excluded_channels) const {

    LOCK(mutex_);

    if (source == destination) {
        return Result<Route>::err("Source and destination are the same");
    }
    if (amount <= 0) {
        return Result<Route>::err("Amount must be positive");
    }

    std::unordered_set<uint64_t> excluded(excluded_channels.begin(),
                                           excluded_channels.end());

    // Dijkstra from destination to source (backward search)
    // This way we can correctly account for fees at each hop
    struct DijkState {
        int64_t  cost;      // Total cost (amount + fees) to reach dest
        uint32_t cltv;      // Total CLTV needed
        uint32_t hops;      // Number of hops
        crypto::Ed25519PublicKey node;
    };

    auto cmp = [](const DijkState& a, const DijkState& b) {
        return a.cost > b.cost;
    };
    std::priority_queue<DijkState, std::vector<DijkState>, decltype(cmp)> pq(cmp);

    // dist[node] = best cost to reach destination from node
    std::unordered_map<crypto::Ed25519PublicKey, DijkState> dist;
    // prev[node] = (channel_id, next_node_toward_dest)
    std::unordered_map<crypto::Ed25519PublicKey,
                       std::pair<uint64_t, crypto::Ed25519PublicKey>> prev;

    DijkState init_state{amount, final_cltv_expiry, 0, destination};
    dist[destination] = init_state;
    pq.push(init_state);

    while (!pq.empty()) {
        auto current = pq.top();
        pq.pop();

        if (current.node == source) break;
        if (current.hops >= max_hops) continue;

        // Check if this is stale
        auto dit = dist.find(current.node);
        if (dit != dist.end() && current.cost > dit->second.cost) continue;

        // Explore neighbors
        auto ait = adjacency_.find(current.node);
        if (ait == adjacency_.end()) continue;

        for (const auto& adj : ait->second) {
            if (excluded.count(adj.short_channel_id)) continue;

            auto cit = channels_.find(adj.short_channel_id);
            if (cit == channels_.end()) continue;
            const auto& edge = cit->second;
            if (edge.disabled) continue;
            if (edge.capacity < current.cost) continue;

            int64_t fee = edge.compute_fee(current.cost);
            int64_t new_cost = current.cost + fee;
            uint32_t new_cltv = current.cltv + edge.cltv_expiry_delta;
            uint32_t new_hops = current.hops + 1;

            auto nit = dist.find(adj.neighbor);
            if (nit == dist.end() || new_cost < nit->second.cost) {
                DijkState new_state{new_cost, new_cltv, new_hops,
                                     adj.neighbor};
                dist[adj.neighbor] = new_state;
                prev[adj.neighbor] = {adj.short_channel_id, current.node};
                pq.push(new_state);
            }
        }
    }

    // Reconstruct path from source to destination
    if (dist.find(source) == dist.end()) {
        return Result<Route>::err("No route found from source to destination");
    }

    Route route;
    auto current_node = source;
    int64_t remaining_amount = dist[source].cost;
    uint32_t remaining_cltv = dist[source].cltv;

    while (current_node != destination) {
        auto pit = prev.find(current_node);
        if (pit == prev.end()) {
            return Result<Route>::err("Route reconstruction failed");
        }

        auto [scid, next_node] = pit->second;
        auto cit = channels_.find(scid);
        if (cit == channels_.end()) {
            return Result<Route>::err("Channel disappeared during routing");
        }

        RouteHop hop;
        hop.node_id = next_node;
        hop.short_channel_id = scid;
        hop.amount = remaining_amount;
        hop.cltv_expiry = remaining_cltv;

        if (next_node == destination) {
            hop.fee = 0;
            hop.amount = amount;
            hop.cltv_expiry = final_cltv_expiry;
        } else {
            auto next_dist = dist.find(next_node);
            if (next_dist != dist.end()) {
                hop.fee = remaining_amount - next_dist->second.cost;
                remaining_amount = next_dist->second.cost;
                remaining_cltv = next_dist->second.cltv;
            }
        }

        route.total_fees += hop.fee;
        route.hops.push_back(std::move(hop));
        current_node = next_node;
    }

    route.total_cltv_delta = dist[source].cltv - final_cltv_expiry;

    LogPrint(LIGHTNING, "Found route: %zu hops, fee=%lld, cltv_delta=%u",
             route.hops.size(), route.total_fees, route.total_cltv_delta);

    return Result<Route>::ok(std::move(route));
}

Result<std::vector<Route>> Router::find_routes(
    const crypto::Ed25519PublicKey& source,
    const crypto::Ed25519PublicKey& destination,
    int64_t amount,
    uint32_t final_cltv_expiry,
    uint32_t max_routes,
    uint32_t max_hops) const {

    std::vector<Route> routes;
    std::vector<uint64_t> excluded;

    for (uint32_t i = 0; i < max_routes; ++i) {
        auto result = find_route(source, destination, amount,
                                  final_cltv_expiry, max_hops, excluded);
        if (!result) break;

        auto& route = result.value();
        // Exclude the first channel of this route for diversity
        if (!route.hops.empty()) {
            excluded.push_back(route.hops[0].short_channel_id);
        }
        routes.push_back(std::move(route));
    }

    if (routes.empty()) {
        return Result<std::vector<Route>>::err("No routes found");
    }

    return Result<std::vector<Route>>::ok(std::move(routes));
}

// ── Pruning ─────────────────────────────────────────────────────────

size_t Router::prune_stale(uint32_t max_age_seconds, uint32_t current_time) {
    LOCK(mutex_);
    size_t pruned = 0;
    uint32_t cutoff = current_time > max_age_seconds
                          ? current_time - max_age_seconds : 0;

    std::vector<uint64_t> to_remove;
    for (const auto& [scid, edge] : channels_) {
        if (edge.last_update < cutoff) {
            to_remove.push_back(scid);
        }
    }

    for (uint64_t scid : to_remove) {
        auto it = channels_.find(scid);
        if (it != channels_.end()) {
            auto n1 = it->second.node1;
            auto n2 = it->second.node2;
            channels_.erase(it);

            auto clean = [scid](std::vector<AdjEntry>& adj) {
                adj.erase(std::remove_if(adj.begin(), adj.end(),
                    [scid](const AdjEntry& e) {
                        return e.short_channel_id == scid;
                    }), adj.end());
            };

            if (auto ait = adjacency_.find(n1); ait != adjacency_.end()) {
                clean(ait->second);
            }
            if (auto ait = adjacency_.find(n2); ait != adjacency_.end()) {
                clean(ait->second);
            }
            ++pruned;
        }
    }

    return pruned;
}

void Router::clear() {
    LOCK(mutex_);
    nodes_.clear();
    channels_.clear();
    adjacency_.clear();
}

void Router::rebuild_adjacency_for(const crypto::Ed25519PublicKey& node_id) {
    auto& adj = adjacency_[node_id];
    adj.clear();
    for (const auto& [scid, edge] : channels_) {
        if (edge.node1 == node_id) {
            adj.push_back({scid, edge.node2});
        } else if (edge.node2 == node_id) {
            adj.push_back({scid, edge.node1});
        }
    }
}

}  // namespace rnet::lightning

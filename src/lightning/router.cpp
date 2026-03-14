// Copyright (c) 2024-2026 The ResonanceNet Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "lightning/router.h"

#include "core/logging.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <set>
#include <unordered_set>

namespace rnet::lightning {

// ===========================================================================
// Fee Computation
// ===========================================================================

// ---------------------------------------------------------------------------
// ChannelEdge::compute_fee
// ---------------------------------------------------------------------------
// Computes the routing fee a forwarding node charges for relaying `amount`
// resonances through this channel.
//
//   fee = fee_base + amount * fee_rate_ppm / 10^6
//
// fee_base is a flat per-HTLC charge (in millisatoshi-equivalent units).
// fee_rate_ppm is a proportional charge expressed in parts-per-million:
// e.g. fee_rate_ppm = 1'000 means 0.1% of the forwarded amount.
// Integer division truncates toward zero, matching BOLT #7 behavior.
// ---------------------------------------------------------------------------
int64_t ChannelEdge::compute_fee(int64_t amount) const {
    return fee_base + (amount * fee_rate_ppm) / 1'000'000;
}

// ---------------------------------------------------------------------------
// Route::total_amount
// ---------------------------------------------------------------------------
// Returns the total amount that must leave the sender, which equals the
// amount field of the first hop (payment amount plus all accumulated fees).
// An empty route returns zero.
// ---------------------------------------------------------------------------
int64_t Route::total_amount() const {
    if (hops.empty()) return 0;
    return hops.front().amount;
}

// ===========================================================================
// Graph Management
// ===========================================================================

// ---------------------------------------------------------------------------
// Router::add_node
// ---------------------------------------------------------------------------
// Inserts or replaces a node in the channel graph.  The node's public key
// is used as the map key; the GraphNode struct is moved into storage.
// ---------------------------------------------------------------------------
void Router::add_node(GraphNode node) {
    LOCK(mutex_);
    // 1. Copy the key before moving the node.
    auto key = node.node_id;
    // 2. Insert or overwrite the node entry.
    nodes_[key] = std::move(node);
}

// ---------------------------------------------------------------------------
// Router::remove_node
// ---------------------------------------------------------------------------
// Removes a node and its adjacency list from the graph.  Channels that
// reference this node are NOT removed here -- the caller must remove them
// separately to keep the graph consistent.
// ---------------------------------------------------------------------------
void Router::remove_node(const crypto::Ed25519PublicKey& node_id) {
    LOCK(mutex_);
    // 1. Erase the node record.
    nodes_.erase(node_id);
    // 2. Erase its adjacency list.
    adjacency_.erase(node_id);
}

// ---------------------------------------------------------------------------
// Router::add_channel
// ---------------------------------------------------------------------------
// Inserts or replaces a channel edge and updates the adjacency lists of
// both endpoint nodes.  The adjacency list is a bidirectional structure:
// each channel (node1 <-> node2) produces two entries so that Dijkstra
// can traverse the undirected channel graph in either direction.
//
// If a channel with the same short_channel_id already exists in a node's
// adjacency list, the old entry is removed before the new one is appended
// (erase-remove idiom) to prevent duplicates.
// ---------------------------------------------------------------------------
void Router::add_channel(ChannelEdge edge) {
    LOCK(mutex_);
    // 1. Extract identifiers before moving the edge.
    uint64_t scid = edge.short_channel_id;
    auto node1 = edge.node1;
    auto node2 = edge.node2;
    // 2. Store the channel edge.
    channels_[scid] = std::move(edge);

    // 3. Update adjacency for node1 -> node2.
    auto& adj1 = adjacency_[node1];
    adj1.erase(std::remove_if(adj1.begin(), adj1.end(),
        [scid](const AdjEntry& e) { return e.short_channel_id == scid; }),
        adj1.end());
    adj1.push_back({scid, node2});

    // 4. Update adjacency for node2 -> node1.
    auto& adj2 = adjacency_[node2];
    adj2.erase(std::remove_if(adj2.begin(), adj2.end(),
        [scid](const AdjEntry& e) { return e.short_channel_id == scid; }),
        adj2.end());
    adj2.push_back({scid, node1});
}

// ---------------------------------------------------------------------------
// Router::remove_channel
// ---------------------------------------------------------------------------
// Removes a channel edge by short_channel_id and cleans both endpoint
// nodes' adjacency lists.  Uses erase-remove to strip the matching entry
// from each adjacency vector.
// ---------------------------------------------------------------------------
void Router::remove_channel(uint64_t short_channel_id) {
    LOCK(mutex_);
    // 1. Find the channel.
    auto it = channels_.find(short_channel_id);
    if (it == channels_.end()) return;

    // 2. Remember both endpoints before erasing.
    auto node1 = it->second.node1;
    auto node2 = it->second.node2;
    channels_.erase(it);

    // 3. Clean adjacency lists for both endpoints.
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

// ---------------------------------------------------------------------------
// Router::set_channel_disabled
// ---------------------------------------------------------------------------
// Marks a channel as disabled (or re-enables it).  Disabled channels are
// skipped during Dijkstra traversal but remain in the graph so they can
// be re-enabled without re-announcing.
// ---------------------------------------------------------------------------
void Router::set_channel_disabled(uint64_t short_channel_id, bool disabled) {
    LOCK(mutex_);
    auto it = channels_.find(short_channel_id);
    if (it != channels_.end()) {
        it->second.disabled = disabled;
    }
}

// ---------------------------------------------------------------------------
// Router::get_node
// ---------------------------------------------------------------------------
// Returns a pointer to the GraphNode for the given public key, or nullptr
// if the node is not in the graph.  The pointer is only valid while the
// caller holds no intervening lock operations.
// ---------------------------------------------------------------------------
const GraphNode* Router::get_node(
    const crypto::Ed25519PublicKey& node_id) const {
    LOCK(mutex_);
    auto it = nodes_.find(node_id);
    return it != nodes_.end() ? &it->second : nullptr;
}

// ---------------------------------------------------------------------------
// Router::get_channel
// ---------------------------------------------------------------------------
// Returns a pointer to the ChannelEdge for the given short_channel_id,
// or nullptr if the channel is not in the graph.
// ---------------------------------------------------------------------------
const ChannelEdge* Router::get_channel(uint64_t short_channel_id) const {
    LOCK(mutex_);
    auto it = channels_.find(short_channel_id);
    return it != channels_.end() ? &it->second : nullptr;
}

// ---------------------------------------------------------------------------
// Router::node_count
// ---------------------------------------------------------------------------
// Returns the number of nodes currently in the channel graph.
// ---------------------------------------------------------------------------
size_t Router::node_count() const {
    LOCK(mutex_);
    return nodes_.size();
}

// ---------------------------------------------------------------------------
// Router::channel_count
// ---------------------------------------------------------------------------
// Returns the number of channels currently in the channel graph.
// ---------------------------------------------------------------------------
size_t Router::channel_count() const {
    LOCK(mutex_);
    return channels_.size();
}

// ===========================================================================
// Dijkstra Pathfinding
// ===========================================================================

// ---------------------------------------------------------------------------
// Router::find_route
// ---------------------------------------------------------------------------
// Finds the cheapest route from `source` to `destination` using a modified
// Dijkstra shortest-path search run BACKWARD from destination to source.
//
// Why backward search:
//   Routing fees depend on the amount being forwarded through a channel.
//   The receiver expects exactly `amount` resonances, so fees must be
//   layered on top starting from the destination.  At each hop i:
//
//     cost_i = cost_{i+1} + fee_base_i + cost_{i+1} * fee_rate_i / 10^6
//
//   By searching from dest -> source, we always know cost_{i+1} (the
//   amount that must arrive at the next node toward the destination)
//   before computing the fee at hop i.  A forward search would require
//   the total downstream amount, which is unknown until the full path
//   is found -- making greedy relaxation incorrect.
//
// The priority queue is a min-heap keyed on total cost (amount + fees).
// dist[node] records the best known cost to deliver `amount` from that
// node to the destination.  prev[node] records the next-hop channel and
// node used to reconstruct the path.
//
// After Dijkstra converges, the path is reconstructed forward from
// source to destination.  Each hop's amount and CLTV are filled from
// the dist table; the final hop receives exactly `amount` with the
// caller's `final_cltv_expiry`.
// ---------------------------------------------------------------------------
Result<Route> Router::find_route(
    const crypto::Ed25519PublicKey& source,
    const crypto::Ed25519PublicKey& destination,
    int64_t amount,
    uint32_t final_cltv_expiry,
    uint32_t max_hops,
    const std::vector<uint64_t>& excluded_channels) const {

    LOCK(mutex_);

    // 1. Validate inputs.
    if (source == destination) {
        return Result<Route>::err("Source and destination are the same");
    }
    if (amount <= 0) {
        return Result<Route>::err("Amount must be positive");
    }

    // 2. Build the exclusion set for O(1) lookup.
    std::unordered_set<uint64_t> excluded(excluded_channels.begin(),
                                           excluded_channels.end());

    // 3. Define the Dijkstra state: cost to reach dest, cumulative CLTV,
    //    hop count, and the current node being explored.
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

    // 4. dist[node] = best cost to reach destination from node.
    std::unordered_map<crypto::Ed25519PublicKey, DijkState> dist;
    // 5. prev[node] = (channel_id, next_node_toward_dest).
    std::unordered_map<crypto::Ed25519PublicKey,
                       std::pair<uint64_t, crypto::Ed25519PublicKey>> prev;

    // 6. Seed the search at the destination with the exact payment amount.
    DijkState init_state{amount, final_cltv_expiry, 0, destination};
    dist[destination] = init_state;
    pq.push(init_state);

    // 7. Dijkstra main loop -- relax edges backward from dest toward source.
    while (!pq.empty()) {
        auto current = pq.top();
        pq.pop();

        // 7a. Stop if we reached the source.
        if (current.node == source) break;
        // 7b. Skip if this path exceeds the hop limit.
        if (current.hops >= max_hops) continue;

        // 7c. Skip stale entries (a better path was already found).
        auto dit = dist.find(current.node);
        if (dit != dist.end() && current.cost > dit->second.cost) continue;

        // 7d. Explore neighbors via the adjacency list.
        auto ait = adjacency_.find(current.node);
        if (ait == adjacency_.end()) continue;

        for (const auto& adj : ait->second) {
            // 7e. Skip excluded channels.
            if (excluded.count(adj.short_channel_id)) continue;

            auto cit = channels_.find(adj.short_channel_id);
            if (cit == channels_.end()) continue;
            const auto& edge = cit->second;
            // 7f. Skip disabled channels.
            if (edge.disabled) continue;
            // 7g. Skip channels with insufficient capacity.
            if (edge.capacity < current.cost) continue;

            // 7h. Compute fee and new cumulative cost.
            //     fee = fee_base + current.cost * fee_rate_ppm / 10^6
            int64_t fee = edge.compute_fee(current.cost);
            int64_t new_cost = current.cost + fee;
            uint32_t new_cltv = current.cltv + edge.cltv_expiry_delta;
            uint32_t new_hops = current.hops + 1;

            // 7i. Relax the edge if this path is cheaper.
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

    // 8. Check that a path to the source was found.
    if (dist.find(source) == dist.end()) {
        return Result<Route>::err("No route found from source to destination");
    }

    // 9. Reconstruct the path forward from source to destination.
    Route route;
    auto current_node = source;
    int64_t remaining_amount = dist[source].cost;
    uint32_t remaining_cltv = dist[source].cltv;

    while (current_node != destination) {
        // 9a. Look up the next hop toward the destination.
        auto pit = prev.find(current_node);
        if (pit == prev.end()) {
            return Result<Route>::err("Route reconstruction failed");
        }

        auto [scid, next_node] = pit->second;
        auto cit = channels_.find(scid);
        if (cit == channels_.end()) {
            return Result<Route>::err("Channel disappeared during routing");
        }

        // 9b. Build the hop with amount and CLTV for this segment.
        RouteHop hop;
        hop.node_id = next_node;
        hop.short_channel_id = scid;
        hop.amount = remaining_amount;
        hop.cltv_expiry = remaining_cltv;

        // 9c. Final hop delivers exactly the payment amount with no fee.
        if (next_node == destination) {
            hop.fee = 0;
            hop.amount = amount;
            hop.cltv_expiry = final_cltv_expiry;
        } else {
            // 9d. Intermediate hops: fee = amount entering - amount leaving.
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

    // 10. Compute the total CLTV delta across all hops.
    route.total_cltv_delta = dist[source].cltv - final_cltv_expiry;

    LogPrint(LIGHTNING, "Found route: %zu hops, fee=%lld, cltv_delta=%u",
             route.hops.size(), route.total_fees, route.total_cltv_delta);

    return Result<Route>::ok(std::move(route));
}

// ===========================================================================
// Multi-Route
// ===========================================================================

// ---------------------------------------------------------------------------
// Router::find_routes
// ---------------------------------------------------------------------------
// Finds up to `max_routes` alternative payment routes by iteratively
// calling find_route and excluding the first channel of each found route.
//
// Route diversity strategy:
//   After finding route R_i, the first channel (hops[0].short_channel_id)
//   of R_i is added to the exclusion list.  This forces Dijkstra to pick
//   a different initial outbound channel for each subsequent route,
//   producing topologically diverse paths.  This improves payment
//   reliability: if one path's first hop is congested, the sender can
//   fall back to an alternative that starts on a different channel.
//
// The search stops early if find_route returns an error (no more paths
// exist).
// ---------------------------------------------------------------------------
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
        // 1. Attempt to find a route with current exclusions.
        auto result = find_route(source, destination, amount,
                                  final_cltv_expiry, max_hops, excluded);
        if (!result) break;

        auto& route = result.value();
        // 2. Exclude the first channel of this route for diversity.
        if (!route.hops.empty()) {
            excluded.push_back(route.hops[0].short_channel_id);
        }
        // 3. Collect the route.
        routes.push_back(std::move(route));
    }

    if (routes.empty()) {
        return Result<std::vector<Route>>::err("No routes found");
    }

    return Result<std::vector<Route>>::ok(std::move(routes));
}

// ===========================================================================
// Maintenance
// ===========================================================================

// ---------------------------------------------------------------------------
// Router::prune_stale
// ---------------------------------------------------------------------------
// Removes channels whose last_update timestamp is older than the cutoff
// time (current_time - max_age_seconds).  This garbage-collects channels
// whose operators have gone offline or stopped broadcasting gossip
// updates.
//
// Pruning algorithm:
//   cutoff = current_time - max_age_seconds   (clamped to 0)
//   For each channel where last_update < cutoff:
//     1. Remove the channel from the channel map.
//     2. Remove corresponding entries from both endpoint adjacency lists.
//
// A two-pass approach is used: first collect IDs to remove, then erase.
// This avoids iterator invalidation during traversal of channels_.
// ---------------------------------------------------------------------------
size_t Router::prune_stale(uint32_t max_age_seconds, uint32_t current_time) {
    LOCK(mutex_);
    size_t pruned = 0;
    // 1. Compute the cutoff timestamp, clamped to zero.
    uint32_t cutoff = current_time > max_age_seconds
                          ? current_time - max_age_seconds : 0;

    // 2. Collect stale channel IDs (avoids iterator invalidation).
    std::vector<uint64_t> to_remove;
    for (const auto& [scid, edge] : channels_) {
        if (edge.last_update < cutoff) {
            to_remove.push_back(scid);
        }
    }

    // 3. Remove each stale channel and clean adjacency lists.
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

// ---------------------------------------------------------------------------
// Router::clear
// ---------------------------------------------------------------------------
// Wipes all nodes, channels, and adjacency data from the graph.  Used
// during shutdown or when re-syncing the gossip store from scratch.
// ---------------------------------------------------------------------------
void Router::clear() {
    LOCK(mutex_);
    // 1. Clear all graph structures.
    nodes_.clear();
    channels_.clear();
    adjacency_.clear();
}

// ---------------------------------------------------------------------------
// Router::rebuild_adjacency_for
// ---------------------------------------------------------------------------
// Rebuilds the adjacency list for a single node by scanning all channels.
// Called when the adjacency list may be out of sync (e.g. after a bulk
// channel import).  For each channel that has `node_id` as either
// endpoint, an AdjEntry pointing to the other endpoint is appended.
// ---------------------------------------------------------------------------
void Router::rebuild_adjacency_for(const crypto::Ed25519PublicKey& node_id) {
    // 1. Clear existing adjacency entries for this node.
    auto& adj = adjacency_[node_id];
    adj.clear();
    // 2. Scan all channels for edges involving this node.
    for (const auto& [scid, edge] : channels_) {
        if (edge.node1 == node_id) {
            adj.push_back({scid, edge.node2});
        } else if (edge.node2 == node_id) {
            adj.push_back({scid, edge.node1});
        }
    }
}

} // namespace rnet::lightning

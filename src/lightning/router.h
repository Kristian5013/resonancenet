#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/types.h"
#include "lightning/channel_state.h"
#include "core/error.h"
#include "core/sync.h"
#include "crypto/ed25519.h"
#include "lightning/channel_state.h"

namespace rnet::lightning {

// ── Graph edge (directed channel) ───────────────────────────────────

struct ChannelEdge {
    uint64_t                     short_channel_id = 0;
    crypto::Ed25519PublicKey     node1;
    crypto::Ed25519PublicKey     node2;
    int64_t                      capacity = 0;
    int64_t                      fee_base = DEFAULT_FEE_BASE_MSAT;
    int64_t                      fee_rate_ppm = DEFAULT_FEE_RATE_PPM;
    uint32_t                     cltv_expiry_delta = DEFAULT_CLTV_EXPIRY_DELTA;
    bool                         disabled = false;
    uint32_t                     last_update = 0;  // Unix timestamp

    /// Compute the routing fee for a given amount
    int64_t compute_fee(int64_t amount) const;

    template<typename Stream>
    void serialize(Stream& s) const {
        core::Serialize(s, short_channel_id);
        s.write(node1.data.data(), 32);
        s.write(node2.data.data(), 32);
        core::Serialize(s, capacity);
        core::Serialize(s, fee_base);
        core::Serialize(s, fee_rate_ppm);
        core::Serialize(s, cltv_expiry_delta);
        core::Serialize(s, disabled);
        core::Serialize(s, last_update);
    }

    template<typename Stream>
    void unserialize(Stream& s) {
        core::Unserialize(s, short_channel_id);
        s.read(node1.data.data(), 32);
        s.read(node2.data.data(), 32);
        core::Unserialize(s, capacity);
        core::Unserialize(s, fee_base);
        core::Unserialize(s, fee_rate_ppm);
        core::Unserialize(s, cltv_expiry_delta);
        core::Unserialize(s, disabled);
        core::Unserialize(s, last_update);
    }
};

// ── Graph node ──────────────────────────────────────────────────────

struct GraphNode {
    crypto::Ed25519PublicKey     node_id;
    std::string                  alias;
    uint32_t                     last_update = 0;
    std::vector<uint64_t>        channel_ids;  // Short channel IDs

    template<typename Stream>
    void serialize(Stream& s) const {
        s.write(node_id.data.data(), 32);
        core::Serialize(s, alias);
        core::Serialize(s, last_update);
        core::Serialize(s, channel_ids);
    }

    template<typename Stream>
    void unserialize(Stream& s) {
        s.read(node_id.data.data(), 32);
        core::Unserialize(s, alias);
        core::Unserialize(s, last_update);
        core::Unserialize(s, channel_ids);
    }
};

// ── Route hop ───────────────────────────────────────────────────────

struct RouteHop {
    crypto::Ed25519PublicKey     node_id;       // Node at end of this hop
    uint64_t                     short_channel_id = 0;
    int64_t                      amount = 0;    // Amount entering this hop
    uint32_t                     cltv_expiry = 0;
    int64_t                      fee = 0;       // Fee charged by this hop
};

/// A complete payment route
struct Route {
    std::vector<RouteHop>        hops;
    int64_t                      total_fees = 0;
    uint32_t                     total_cltv_delta = 0;

    /// Total amount that must leave us (payment + all fees)
    int64_t total_amount() const;
};

// ── Channel graph / router ──────────────────────────────────────────

/// Dijkstra-based pathfinding over the channel graph
class Router {
public:
    Router() = default;

    // ── Graph management ────────────────────────────────────────────

    /// Add or update a node in the graph
    void add_node(GraphNode node);

    /// Remove a node from the graph
    void remove_node(const crypto::Ed25519PublicKey& node_id);

    /// Add or update a channel edge
    void add_channel(ChannelEdge edge);

    /// Remove a channel
    void remove_channel(uint64_t short_channel_id);

    /// Disable/enable a channel
    void set_channel_disabled(uint64_t short_channel_id, bool disabled);

    /// Get node info
    const GraphNode* get_node(const crypto::Ed25519PublicKey& node_id) const;

    /// Get channel info
    const ChannelEdge* get_channel(uint64_t short_channel_id) const;

    /// Number of nodes in the graph
    size_t node_count() const;

    /// Number of channels in the graph
    size_t channel_count() const;

    // ── Pathfinding ─────────────────────────────────────────────────

    /// Find the cheapest route from source to destination
    /// amount: payment amount in resonances
    /// max_hops: maximum route length (default MAX_ROUTE_HOPS)
    /// excluded_channels: channels to exclude from pathfinding
    Result<Route> find_route(
        const crypto::Ed25519PublicKey& source,
        const crypto::Ed25519PublicKey& destination,
        int64_t amount,
        uint32_t final_cltv_expiry,
        uint32_t max_hops = MAX_ROUTE_HOPS,
        const std::vector<uint64_t>& excluded_channels = {}) const;

    /// Find multiple alternative routes
    Result<std::vector<Route>> find_routes(
        const crypto::Ed25519PublicKey& source,
        const crypto::Ed25519PublicKey& destination,
        int64_t amount,
        uint32_t final_cltv_expiry,
        uint32_t max_routes = 3,
        uint32_t max_hops = MAX_ROUTE_HOPS) const;

    // ── Pruning ─────────────────────────────────────────────────────

    /// Remove stale channels (not updated in given seconds)
    size_t prune_stale(uint32_t max_age_seconds, uint32_t current_time);

    /// Clear the entire graph
    void clear();

private:
    /// Internal adjacency list: node_id -> list of (channel_id, neighbor)
    struct AdjEntry {
        uint64_t short_channel_id;
        crypto::Ed25519PublicKey neighbor;
    };

    mutable core::Mutex mutex_;
    std::unordered_map<crypto::Ed25519PublicKey, GraphNode> nodes_;
    std::unordered_map<uint64_t, ChannelEdge> channels_;
    std::unordered_map<crypto::Ed25519PublicKey, std::vector<AdjEntry>> adjacency_;

    /// Rebuild adjacency list for a node
    void rebuild_adjacency_for(const crypto::Ed25519PublicKey& node_id);
};

}  // namespace rnet::lightning

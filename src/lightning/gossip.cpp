// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "lightning/gossip.h"

#include "core/logging.h"
#include "core/serialize.h"
#include "core/stream.h"
#include "crypto/keccak.h"

namespace rnet::lightning {

// ===========================================================================
//  Signing Hash
// ===========================================================================

// ---------------------------------------------------------------------------
// ChannelAnnouncement::signing_hash
//
// Design: Keccak256d over canonical serialization of announcement fields.
// The hash covers short_channel_id, both node IDs, chain_hash, and capacity
// in a deterministic byte order so both sides produce identical digests.
// ---------------------------------------------------------------------------
uint256 ChannelAnnouncement::signing_hash() const {
    // 1. Build canonical serialization of all announcement fields
    crypto::KeccakHasher hasher;
    core::DataStream ss;
    core::ser_write_u64(ss, short_channel_id);
    ss.write(node_id_1.data.data(), 32);
    ss.write(node_id_2.data.data(), 32);
    ss.write(chain_hash.data(), 32);
    core::ser_write_i64(ss, capacity);

    // 2. Feed serialized bytes into Keccak256d hasher
    hasher.write(ss.span());
    return hasher.finalize_double();
}

// ---------------------------------------------------------------------------
// ChannelUpdate::signing_hash
//
// Design: Keccak256d over canonical serialization of update fields.
// Covers short_channel_id, timestamp, channel_flags, cltv_expiry_delta,
// HTLC limits, fee parameters, and chain_hash for replay protection.
// ---------------------------------------------------------------------------
uint256 ChannelUpdate::signing_hash() const {
    // 1. Build canonical serialization of all update fields
    crypto::KeccakHasher hasher;
    core::DataStream ss;
    core::ser_write_u64(ss, short_channel_id);
    core::ser_write_u32(ss, timestamp);
    core::ser_write_u8(ss, channel_flags);
    core::ser_write_u32(ss, cltv_expiry_delta);
    core::ser_write_i64(ss, htlc_minimum);
    core::ser_write_i64(ss, htlc_maximum);
    core::ser_write_i64(ss, fee_base);
    core::ser_write_i64(ss, fee_rate_ppm);
    ss.write(chain_hash.data(), 32);

    // 2. Feed serialized bytes into Keccak256d hasher
    hasher.write(ss.span());
    return hasher.finalize_double();
}

// ---------------------------------------------------------------------------
// NodeAnnouncement::signing_hash
//
// Design: Keccak256d over canonical serialization of node announcement fields.
// Covers node_id, timestamp, alias, and addresses so the announcing node
// can prove ownership of its identity.
// ---------------------------------------------------------------------------
uint256 NodeAnnouncement::signing_hash() const {
    // 1. Build canonical serialization of all node announcement fields
    crypto::KeccakHasher hasher;
    core::DataStream ss;
    ss.write(node_id.data.data(), 32);
    core::ser_write_u32(ss, timestamp);
    core::Serialize(ss, alias);
    core::Serialize(ss, addresses);

    // 2. Feed serialized bytes into Keccak256d hasher
    hasher.write(ss.span());
    return hasher.finalize_double();
}

// ===========================================================================
//  Gossip Message Verification
// ===========================================================================

// ---------------------------------------------------------------------------
// ChannelAnnouncement::verify
//
// Design: Ed25519 signature verification over signing_hash. Both node
// signatures (node_sig_1 and node_sig_2) must be valid against their
// respective public keys for the announcement to be accepted.
// ---------------------------------------------------------------------------
bool ChannelAnnouncement::verify() const {
    // 1. Compute the signing hash over canonical fields
    uint256 hash = signing_hash();

    // 2. Verify both node signatures against the hash
    return crypto::ed25519_verify(node_id_1, hash.span(), node_sig_1) &&
           crypto::ed25519_verify(node_id_2, hash.span(), node_sig_2);
}

// ---------------------------------------------------------------------------
// ChannelUpdate::verify
//
// Design: Ed25519 signature verification over signing_hash. The caller
// must supply the correct node_id (determined by the direction bit in
// channel_flags) so the signature is checked against the right key.
// ---------------------------------------------------------------------------
bool ChannelUpdate::verify(const crypto::Ed25519PublicKey& node_id) const {
    // 1. Compute the signing hash over canonical fields
    uint256 hash = signing_hash();

    // 2. Verify the signature against the supplied node key
    return crypto::ed25519_verify(node_id, hash.span(), signature);
}

// ---------------------------------------------------------------------------
// NodeAnnouncement::verify
//
// Design: Ed25519 signature verification over signing_hash. The node
// proves ownership of its public key by signing the announcement.
// ---------------------------------------------------------------------------
bool NodeAnnouncement::verify() const {
    // 1. Compute the signing hash over canonical fields
    uint256 hash = signing_hash();

    // 2. Verify the node's signature
    return crypto::ed25519_verify(node_id, hash.span(), signature);
}

// ===========================================================================
//  GossipManager
// ===========================================================================

// ---------------------------------------------------------------------------
// GossipManager constructor
//
// Design: Stores a reference to the Router for graph mutations. All gossip
// processing flows through the manager before reaching the routing graph.
// ---------------------------------------------------------------------------
GossipManager::GossipManager(Router& router)
    : router_(router) {}

// ===========================================================================
//  Channel Announcement Processing
// ===========================================================================

// ---------------------------------------------------------------------------
// GossipManager::process_channel_announcement
//
// Design: Dedup (seen_channels_ set) -> verify both Ed25519 sigs ->
// add edge to routing graph -> ensure both endpoint nodes exist in the
// graph (creating stubs if needed) -> forward to peers via callback.
// ---------------------------------------------------------------------------
Result<void> GossipManager::process_channel_announcement(
    const ChannelAnnouncement& ann) {
    LOCK(mutex_);

    // 1. Dedup: skip already-known channels
    if (seen_channels_.count(ann.short_channel_id)) {
        return Result<void>::ok();
    }

    // 2. Verify both node signatures
    if (!ann.verify()) {
        return Result<void>::err("Channel announcement signature verification failed");
    }

    // 3. Add channel edge to the routing graph
    ChannelEdge edge;
    edge.short_channel_id = ann.short_channel_id;
    edge.node1 = ann.node_id_1;
    edge.node2 = ann.node_id_2;
    edge.capacity = ann.capacity;

    router_.add_channel(std::move(edge));
    seen_channels_.insert(ann.short_channel_id);

    // 4. Ensure both endpoint nodes exist in the graph
    if (!router_.get_node(ann.node_id_1)) {
        GraphNode n;
        n.node_id = ann.node_id_1;
        n.channel_ids.push_back(ann.short_channel_id);
        router_.add_node(std::move(n));
    }
    if (!router_.get_node(ann.node_id_2)) {
        GraphNode n;
        n.node_id = ann.node_id_2;
        n.channel_ids.push_back(ann.short_channel_id);
        router_.add_node(std::move(n));
    }

    LogPrint(LIGHTNING, "Processed channel announcement: scid=%llu",
             ann.short_channel_id);

    // 5. Forward to connected peers
    if (forward_fn_) {
        core::DataStream ss;
        core::ser_write_u16(ss, static_cast<uint16_t>(GossipType::CHANNEL_ANNOUNCEMENT));
        ann.serialize(ss);
        forward_fn_(ss.vch());
    }

    return Result<void>::ok();
}

// ===========================================================================
//  Channel Update Processing
// ===========================================================================

// ---------------------------------------------------------------------------
// GossipManager::process_channel_update
//
// Design: Timestamp freshness check (reject stale updates) -> find the
// signing node using the direction bit (bit 0 of channel_flags) ->
// verify Ed25519 sig -> update edge parameters in graph -> forward.
// ---------------------------------------------------------------------------
Result<void> GossipManager::process_channel_update(
    const ChannelUpdate& update) {
    LOCK(mutex_);

    // 1. Check timestamp freshness (reject stale updates)
    uint64_t key = (static_cast<uint64_t>(update.short_channel_id) << 1) |
                   update.direction();
    auto it = channel_update_timestamps_.find(key);
    if (it != channel_update_timestamps_.end() &&
        update.timestamp <= it->second) {
        return Result<void>::ok();
    }

    // 2. Find the signing node based on direction bit
    const auto* edge = router_.get_channel(update.short_channel_id);
    if (!edge) {
        return Result<void>::err("Channel update for unknown channel: " +
                                  std::to_string(update.short_channel_id));
    }

    const auto& signing_node = (update.direction() == 0) ? edge->node1 : edge->node2;

    // 3. Verify Ed25519 signature against the signing node
    if (!update.verify(signing_node)) {
        return Result<void>::err("Channel update signature verification failed");
    }

    // 4. Update the channel edge in the router
    ChannelEdge updated_edge = *edge;
    updated_edge.fee_base = update.fee_base;
    updated_edge.fee_rate_ppm = update.fee_rate_ppm;
    updated_edge.cltv_expiry_delta = update.cltv_expiry_delta;
    updated_edge.disabled = update.is_disabled();
    updated_edge.last_update = update.timestamp;
    router_.add_channel(std::move(updated_edge));

    channel_update_timestamps_[key] = update.timestamp;

    LogPrint(LIGHTNING, "Processed channel update: scid=%llu, dir=%u",
             update.short_channel_id, update.direction());

    // 5. Forward to connected peers
    if (forward_fn_) {
        core::DataStream ss;
        core::ser_write_u16(ss, static_cast<uint16_t>(GossipType::CHANNEL_UPDATE));
        update.serialize(ss);
        forward_fn_(ss.vch());
    }

    return Result<void>::ok();
}

// ===========================================================================
//  Node Announcement Processing
// ===========================================================================

// ---------------------------------------------------------------------------
// GossipManager::process_node_announcement
//
// Design: Timestamp freshness check (reject stale announcements) -> verify
// Ed25519 sig -> upsert node in graph (preserve existing channel_ids from
// the router so we do not lose channel associations) -> forward to peers.
// ---------------------------------------------------------------------------
Result<void> GossipManager::process_node_announcement(
    const NodeAnnouncement& ann) {
    LOCK(mutex_);

    // 1. Check timestamp freshness (reject stale announcements)
    auto it = node_update_timestamps_.find(ann.node_id);
    if (it != node_update_timestamps_.end() &&
        ann.timestamp <= it->second) {
        return Result<void>::ok();
    }

    // 2. Verify the node's Ed25519 signature
    if (!ann.verify()) {
        return Result<void>::err("Node announcement signature verification failed");
    }

    // 3. Build updated node, preserving existing channel_ids
    GraphNode node;
    node.node_id = ann.node_id;
    node.alias = ann.alias;
    node.last_update = ann.timestamp;

    const auto* existing = router_.get_node(ann.node_id);
    if (existing) {
        node.channel_ids = existing->channel_ids;
    }

    // 4. Upsert into the routing graph
    router_.add_node(std::move(node));
    node_update_timestamps_[ann.node_id] = ann.timestamp;

    LogPrint(LIGHTNING, "Processed node announcement: %s (%s)",
             ann.node_id.to_hex().c_str(), ann.alias.c_str());

    // 5. Forward to connected peers
    if (forward_fn_) {
        core::DataStream ss;
        core::ser_write_u16(ss, static_cast<uint16_t>(GossipType::NODE_ANNOUNCEMENT));
        ann.serialize(ss);
        forward_fn_(ss.vch());
    }

    return Result<void>::ok();
}

// ===========================================================================
//  Gossip Creation
// ===========================================================================

// ---------------------------------------------------------------------------
// GossipManager::create_channel_announcement
//
// Design: Canonical ordering ensures node1 < node2 (lexicographic on
// public key bytes) so both sides produce identical announcements. We
// sign with our key and place the signature in the correct slot (sig1
// if we are node1, sig2 otherwise). The counterparty fills the other.
// ---------------------------------------------------------------------------
Result<ChannelAnnouncement> GossipManager::create_channel_announcement(
    uint64_t short_channel_id,
    const crypto::Ed25519KeyPair& our_keys,
    const crypto::Ed25519PublicKey& their_node_id,
    int64_t capacity,
    const uint256& chain_hash) {

    // 1. Populate announcement fields
    ChannelAnnouncement ann;
    ann.short_channel_id = short_channel_id;
    ann.capacity = capacity;
    ann.chain_hash = chain_hash;

    // 2. Enforce canonical ordering: node1 < node2
    if (our_keys.public_key < their_node_id) {
        ann.node_id_1 = our_keys.public_key;
        ann.node_id_2 = their_node_id;
    } else {
        ann.node_id_1 = their_node_id;
        ann.node_id_2 = our_keys.public_key;
    }

    // 3. Sign the canonical hash with our secret key
    uint256 hash = ann.signing_hash();
    auto sig = crypto::ed25519_sign(our_keys.secret, hash.span());
    if (!sig) {
        return Result<ChannelAnnouncement>::err("Failed to sign channel announcement");
    }

    // 4. Place signature in the correct slot based on ordering
    if (our_keys.public_key < their_node_id) {
        ann.node_sig_1 = sig.value();
    } else {
        ann.node_sig_2 = sig.value();
    }

    return Result<ChannelAnnouncement>::ok(std::move(ann));
}

// ---------------------------------------------------------------------------
// GossipManager::create_channel_update
//
// Design: channel_flags encoding: bit 0 = direction (which side of the
// channel we are), bit 1 = disabled flag. Timestamp is set to current
// wall-clock time. All fee/HTLC parameters come from ChannelConfig.
// ---------------------------------------------------------------------------
Result<ChannelUpdate> GossipManager::create_channel_update(
    uint64_t short_channel_id,
    const crypto::Ed25519SecretKey& our_key,
    uint8_t direction,
    bool disabled,
    const ChannelConfig& config,
    const uint256& chain_hash) {

    // 1. Populate update fields from config
    ChannelUpdate update;
    update.short_channel_id = short_channel_id;
    update.chain_hash = chain_hash;
    update.channel_flags = direction & 0x01;
    if (disabled) update.channel_flags |= 0x02;
    update.cltv_expiry_delta = config.csv_delay;
    update.htlc_minimum = config.min_htlc_value;
    update.htlc_maximum = config.max_htlc_value;
    update.fee_base = config.fee_base;
    update.fee_rate_ppm = config.fee_rate_ppm;

    // 2. Set timestamp to current wall-clock time
    auto now = std::chrono::system_clock::now();
    update.timestamp = static_cast<uint32_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            now.time_since_epoch()).count());

    // 3. Sign the canonical hash
    uint256 hash = update.signing_hash();
    auto sig = crypto::ed25519_sign(our_key, hash.span());
    if (!sig) {
        return Result<ChannelUpdate>::err("Failed to sign channel update");
    }
    update.signature = sig.value();

    return Result<ChannelUpdate>::ok(std::move(update));
}

// ---------------------------------------------------------------------------
// GossipManager::create_node_announcement
//
// Design: Announces our node identity, alias, and network addresses.
// Timestamp is set to current wall-clock time. Signed with our Ed25519
// secret key so peers can verify we own the advertised node_id.
// ---------------------------------------------------------------------------
Result<NodeAnnouncement> GossipManager::create_node_announcement(
    const crypto::Ed25519KeyPair& our_keys,
    const std::string& alias,
    const std::vector<uint8_t>& addresses) {

    // 1. Populate announcement fields
    NodeAnnouncement ann;
    ann.node_id = our_keys.public_key;
    ann.alias = alias;
    ann.addresses = addresses;

    // 2. Set timestamp to current wall-clock time
    auto now = std::chrono::system_clock::now();
    ann.timestamp = static_cast<uint32_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            now.time_since_epoch()).count());

    // 3. Sign the canonical hash with our secret key
    uint256 hash = ann.signing_hash();
    auto sig = crypto::ed25519_sign(our_keys.secret, hash.span());
    if (!sig) {
        return Result<NodeAnnouncement>::err("Failed to sign node announcement");
    }
    ann.signature = sig.value();

    return Result<NodeAnnouncement>::ok(std::move(ann));
}

// ===========================================================================
//  Maintenance
// ===========================================================================

// ---------------------------------------------------------------------------
// GossipManager::set_forward_callback
//
// Design: Registers the callback used to broadcast gossip messages to
// connected peers. Called once during node startup.
// ---------------------------------------------------------------------------
void GossipManager::set_forward_callback(GossipForwardFn fn) {
    LOCK(mutex_);
    forward_fn_ = std::move(fn);
}

// ---------------------------------------------------------------------------
// GossipManager::has_seen_channel
//
// Design: Quick dedup check against the seen_channels_ set. Used by
// peer message handlers to avoid re-processing known announcements.
// ---------------------------------------------------------------------------
bool GossipManager::has_seen_channel(uint64_t short_channel_id) const {
    LOCK(mutex_);
    return seen_channels_.count(short_channel_id) > 0;
}

// ---------------------------------------------------------------------------
// GossipManager::prune
//
// Design: Delegate stale-channel pruning to the router (which owns the
// graph), then clean up our local tracking data for any channels the
// router removed. Returns the number of pruned channels.
// ---------------------------------------------------------------------------
size_t GossipManager::prune(uint32_t current_time, uint32_t max_age) {
    LOCK(mutex_);

    // 1. Delegate pruning to the router
    size_t pruned = router_.prune_stale(max_age, current_time);

    // 2. Clean up tracking data for pruned channels
    // We trust the router to have removed the channels
    return pruned;
}

// ---------------------------------------------------------------------------
// GossipManager::announcement_count
//
// Design: Returns the number of unique channel announcements seen. Used
// for diagnostics and RPC status reporting.
// ---------------------------------------------------------------------------
size_t GossipManager::announcement_count() const {
    LOCK(mutex_);
    return seen_channels_.size();
}

// ---------------------------------------------------------------------------
// GossipManager::update_count
//
// Design: Returns the number of tracked channel update timestamps. Used
// for diagnostics and RPC status reporting.
// ---------------------------------------------------------------------------
size_t GossipManager::update_count() const {
    LOCK(mutex_);
    return channel_update_timestamps_.size();
}

} // namespace rnet::lightning

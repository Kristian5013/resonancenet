#include "lightning/gossip.h"

#include "core/logging.h"
#include "core/stream.h"
#include "core/serialize.h"
#include "crypto/keccak.h"

namespace rnet::lightning {

// ── ChannelAnnouncement ─────────────────────────────────────────────

uint256 ChannelAnnouncement::signing_hash() const {
    crypto::KeccakHasher hasher;
    core::DataStream ss;
    core::ser_write_u64(ss, short_channel_id);
    ss.write(node_id_1.data.data(), 32);
    ss.write(node_id_2.data.data(), 32);
    ss.write(chain_hash.data(), 32);
    core::ser_write_i64(ss, capacity);
    hasher.write(ss.span());
    return hasher.finalize_double();
}

bool ChannelAnnouncement::verify() const {
    uint256 hash = signing_hash();
    return crypto::ed25519_verify(node_id_1, hash.span(), node_sig_1) &&
           crypto::ed25519_verify(node_id_2, hash.span(), node_sig_2);
}

// ── ChannelUpdate ───────────────────────────────────────────────────

uint256 ChannelUpdate::signing_hash() const {
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
    hasher.write(ss.span());
    return hasher.finalize_double();
}

bool ChannelUpdate::verify(const crypto::Ed25519PublicKey& node_id) const {
    uint256 hash = signing_hash();
    return crypto::ed25519_verify(node_id, hash.span(), signature);
}

// ── NodeAnnouncement ────────────────────────────────────────────────

uint256 NodeAnnouncement::signing_hash() const {
    crypto::KeccakHasher hasher;
    core::DataStream ss;
    ss.write(node_id.data.data(), 32);
    core::ser_write_u32(ss, timestamp);
    core::Serialize(ss, alias);
    core::Serialize(ss, addresses);
    hasher.write(ss.span());
    return hasher.finalize_double();
}

bool NodeAnnouncement::verify() const {
    uint256 hash = signing_hash();
    return crypto::ed25519_verify(node_id, hash.span(), signature);
}

// ── GossipManager ───────────────────────────────────────────────────

GossipManager::GossipManager(Router& router)
    : router_(router) {}

Result<void> GossipManager::process_channel_announcement(
    const ChannelAnnouncement& ann) {
    LOCK(mutex_);

    if (seen_channels_.count(ann.short_channel_id)) {
        return Result<void>::ok();  // Already known, not an error
    }

    if (!ann.verify()) {
        return Result<void>::err("Channel announcement signature verification failed");
    }

    // Add to the graph
    ChannelEdge edge;
    edge.short_channel_id = ann.short_channel_id;
    edge.node1 = ann.node_id_1;
    edge.node2 = ann.node_id_2;
    edge.capacity = ann.capacity;

    router_.add_channel(std::move(edge));
    seen_channels_.insert(ann.short_channel_id);

    // Ensure nodes exist in the graph
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

    // Forward to peers
    if (forward_fn_) {
        core::DataStream ss;
        core::ser_write_u16(ss, static_cast<uint16_t>(GossipType::CHANNEL_ANNOUNCEMENT));
        ann.serialize(ss);
        forward_fn_(ss.vch());
    }

    return Result<void>::ok();
}

Result<void> GossipManager::process_channel_update(
    const ChannelUpdate& update) {
    LOCK(mutex_);

    // Check timestamp freshness
    uint64_t key = (static_cast<uint64_t>(update.short_channel_id) << 1) |
                   update.direction();
    auto it = channel_update_timestamps_.find(key);
    if (it != channel_update_timestamps_.end() &&
        update.timestamp <= it->second) {
        return Result<void>::ok();  // Stale update
    }

    // Verify signature against the appropriate node
    const auto* edge = router_.get_channel(update.short_channel_id);
    if (!edge) {
        return Result<void>::err("Channel update for unknown channel: " +
                                  std::to_string(update.short_channel_id));
    }

    const auto& signing_node = (update.direction() == 0) ? edge->node1 : edge->node2;
    if (!update.verify(signing_node)) {
        return Result<void>::err("Channel update signature verification failed");
    }

    // Update the channel in the router
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

    // Forward
    if (forward_fn_) {
        core::DataStream ss;
        core::ser_write_u16(ss, static_cast<uint16_t>(GossipType::CHANNEL_UPDATE));
        update.serialize(ss);
        forward_fn_(ss.vch());
    }

    return Result<void>::ok();
}

Result<void> GossipManager::process_node_announcement(
    const NodeAnnouncement& ann) {
    LOCK(mutex_);

    auto it = node_update_timestamps_.find(ann.node_id);
    if (it != node_update_timestamps_.end() &&
        ann.timestamp <= it->second) {
        return Result<void>::ok();  // Stale
    }

    if (!ann.verify()) {
        return Result<void>::err("Node announcement signature verification failed");
    }

    GraphNode node;
    node.node_id = ann.node_id;
    node.alias = ann.alias;
    node.last_update = ann.timestamp;

    // Preserve existing channel IDs
    const auto* existing = router_.get_node(ann.node_id);
    if (existing) {
        node.channel_ids = existing->channel_ids;
    }

    router_.add_node(std::move(node));
    node_update_timestamps_[ann.node_id] = ann.timestamp;

    LogPrint(LIGHTNING, "Processed node announcement: %s (%s)",
             ann.node_id.to_hex().c_str(), ann.alias.c_str());

    // Forward
    if (forward_fn_) {
        core::DataStream ss;
        core::ser_write_u16(ss, static_cast<uint16_t>(GossipType::NODE_ANNOUNCEMENT));
        ann.serialize(ss);
        forward_fn_(ss.vch());
    }

    return Result<void>::ok();
}

Result<ChannelAnnouncement> GossipManager::create_channel_announcement(
    uint64_t short_channel_id,
    const crypto::Ed25519KeyPair& our_keys,
    const crypto::Ed25519PublicKey& their_node_id,
    int64_t capacity,
    const uint256& chain_hash) {

    ChannelAnnouncement ann;
    ann.short_channel_id = short_channel_id;
    ann.capacity = capacity;
    ann.chain_hash = chain_hash;

    // Ensure node1 < node2 for canonical ordering
    if (our_keys.public_key < their_node_id) {
        ann.node_id_1 = our_keys.public_key;
        ann.node_id_2 = their_node_id;
    } else {
        ann.node_id_1 = their_node_id;
        ann.node_id_2 = our_keys.public_key;
    }

    uint256 hash = ann.signing_hash();
    auto sig = crypto::ed25519_sign(our_keys.secret, hash.span());
    if (!sig) {
        return Result<ChannelAnnouncement>::err("Failed to sign channel announcement");
    }

    if (our_keys.public_key < their_node_id) {
        ann.node_sig_1 = sig.value();
    } else {
        ann.node_sig_2 = sig.value();
    }

    return Result<ChannelAnnouncement>::ok(std::move(ann));
}

Result<ChannelUpdate> GossipManager::create_channel_update(
    uint64_t short_channel_id,
    const crypto::Ed25519SecretKey& our_key,
    uint8_t direction,
    bool disabled,
    const ChannelConfig& config,
    const uint256& chain_hash) {

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

    // Set timestamp
    auto now = std::chrono::system_clock::now();
    update.timestamp = static_cast<uint32_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            now.time_since_epoch()).count());

    uint256 hash = update.signing_hash();
    auto sig = crypto::ed25519_sign(our_key, hash.span());
    if (!sig) {
        return Result<ChannelUpdate>::err("Failed to sign channel update");
    }
    update.signature = sig.value();

    return Result<ChannelUpdate>::ok(std::move(update));
}

Result<NodeAnnouncement> GossipManager::create_node_announcement(
    const crypto::Ed25519KeyPair& our_keys,
    const std::string& alias,
    const std::vector<uint8_t>& addresses) {

    NodeAnnouncement ann;
    ann.node_id = our_keys.public_key;
    ann.alias = alias;
    ann.addresses = addresses;

    auto now = std::chrono::system_clock::now();
    ann.timestamp = static_cast<uint32_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            now.time_since_epoch()).count());

    uint256 hash = ann.signing_hash();
    auto sig = crypto::ed25519_sign(our_keys.secret, hash.span());
    if (!sig) {
        return Result<NodeAnnouncement>::err("Failed to sign node announcement");
    }
    ann.signature = sig.value();

    return Result<NodeAnnouncement>::ok(std::move(ann));
}

void GossipManager::set_forward_callback(GossipForwardFn fn) {
    LOCK(mutex_);
    forward_fn_ = std::move(fn);
}

bool GossipManager::has_seen_channel(uint64_t short_channel_id) const {
    LOCK(mutex_);
    return seen_channels_.count(short_channel_id) > 0;
}

size_t GossipManager::prune(uint32_t current_time, uint32_t max_age) {
    LOCK(mutex_);
    size_t pruned = router_.prune_stale(max_age, current_time);

    // Clean up our tracking data for pruned channels
    // We trust the router to have removed the channels
    return pruned;
}

size_t GossipManager::announcement_count() const {
    LOCK(mutex_);
    return seen_channels_.size();
}

size_t GossipManager::update_count() const {
    LOCK(mutex_);
    return channel_update_timestamps_.size();
}

}  // namespace rnet::lightning

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/types.h"
#include "lightning/channel_state.h"
#include "core/error.h"
#include "core/sync.h"
#include "crypto/ed25519.h"
#include "lightning/router.h"

namespace rnet::lightning {

// ── Gossip message types ────────────────────────────────────────────

enum class GossipType : uint16_t {
    CHANNEL_ANNOUNCEMENT    = 256,
    CHANNEL_UPDATE          = 258,
    NODE_ANNOUNCEMENT       = 257,
};

// ── Channel announcement ────────────────────────────────────────────

struct ChannelAnnouncement {
    uint64_t                     short_channel_id = 0;
    crypto::Ed25519PublicKey     node_id_1;
    crypto::Ed25519PublicKey     node_id_2;
    crypto::Ed25519Signature    node_sig_1;
    crypto::Ed25519Signature    node_sig_2;
    uint256                      chain_hash;       // Genesis block hash
    int64_t                      capacity = 0;

    /// Verify both signatures
    bool verify() const;

    /// Compute the message hash for signing
    uint256 signing_hash() const;

    template<typename Stream>
    void serialize(Stream& s) const {
        core::Serialize(s, short_channel_id);
        s.write(node_id_1.data.data(), 32);
        s.write(node_id_2.data.data(), 32);
        s.write(node_sig_1.data.data(), 64);
        s.write(node_sig_2.data.data(), 64);
        core::Serialize(s, chain_hash);
        core::Serialize(s, capacity);
    }

    template<typename Stream>
    void unserialize(Stream& s) {
        core::Unserialize(s, short_channel_id);
        s.read(node_id_1.data.data(), 32);
        s.read(node_id_2.data.data(), 32);
        s.read(node_sig_1.data.data(), 64);
        s.read(node_sig_2.data.data(), 64);
        core::Unserialize(s, chain_hash);
        core::Unserialize(s, capacity);
    }
};

// ── Channel update ──────────────────────────────────────────────────

struct ChannelUpdate {
    uint64_t                     short_channel_id = 0;
    uint32_t                     timestamp = 0;
    uint8_t                      channel_flags = 0;  // bit 0: direction, bit 1: disable
    uint32_t                     cltv_expiry_delta = DEFAULT_CLTV_EXPIRY_DELTA;
    int64_t                      htlc_minimum = MIN_HTLC_VALUE;
    int64_t                      htlc_maximum = MAX_HTLC_VALUE;
    int64_t                      fee_base = DEFAULT_FEE_BASE_MSAT;
    int64_t                      fee_rate_ppm = DEFAULT_FEE_RATE_PPM;
    crypto::Ed25519Signature    signature;
    uint256                      chain_hash;

    /// Which direction (node1->node2 = 0, node2->node1 = 1)
    uint8_t direction() const { return channel_flags & 0x01; }

    /// Is this channel disabled?
    bool is_disabled() const { return (channel_flags & 0x02) != 0; }

    /// Verify the signature
    bool verify(const crypto::Ed25519PublicKey& node_id) const;

    /// Compute message hash for signing
    uint256 signing_hash() const;

    template<typename Stream>
    void serialize(Stream& s) const {
        core::Serialize(s, short_channel_id);
        core::Serialize(s, timestamp);
        core::Serialize(s, channel_flags);
        core::Serialize(s, cltv_expiry_delta);
        core::Serialize(s, htlc_minimum);
        core::Serialize(s, htlc_maximum);
        core::Serialize(s, fee_base);
        core::Serialize(s, fee_rate_ppm);
        s.write(signature.data.data(), 64);
        core::Serialize(s, chain_hash);
    }

    template<typename Stream>
    void unserialize(Stream& s) {
        core::Unserialize(s, short_channel_id);
        core::Unserialize(s, timestamp);
        core::Unserialize(s, channel_flags);
        core::Unserialize(s, cltv_expiry_delta);
        core::Unserialize(s, htlc_minimum);
        core::Unserialize(s, htlc_maximum);
        core::Unserialize(s, fee_base);
        core::Unserialize(s, fee_rate_ppm);
        s.read(signature.data.data(), 64);
        core::Unserialize(s, chain_hash);
    }
};

// ── Node announcement ───────────────────────────────────────────────

struct NodeAnnouncement {
    crypto::Ed25519PublicKey     node_id;
    uint32_t                     timestamp = 0;
    std::string                  alias;
    std::vector<uint8_t>         addresses;  // Encoded network addresses
    crypto::Ed25519Signature    signature;

    /// Verify the signature
    bool verify() const;

    /// Compute message hash for signing
    uint256 signing_hash() const;

    template<typename Stream>
    void serialize(Stream& s) const {
        s.write(node_id.data.data(), 32);
        core::Serialize(s, timestamp);
        core::Serialize(s, alias);
        core::Serialize(s, addresses);
        s.write(signature.data.data(), 64);
    }

    template<typename Stream>
    void unserialize(Stream& s) {
        s.read(node_id.data.data(), 32);
        core::Unserialize(s, timestamp);
        core::Unserialize(s, alias);
        core::Unserialize(s, addresses);
        s.read(signature.data.data(), 64);
    }
};

// ── Gossip manager ──────────────────────────────────────────────────

/// Callback for forwarding gossip to peers
using GossipForwardFn = std::function<void(const std::vector<uint8_t>& msg)>;

/// Manages gossip message processing and graph updates
class GossipManager {
public:
    explicit GossipManager(Router& router);

    /// Process a channel announcement
    Result<void> process_channel_announcement(const ChannelAnnouncement& ann);

    /// Process a channel update
    Result<void> process_channel_update(const ChannelUpdate& update);

    /// Process a node announcement
    Result<void> process_node_announcement(const NodeAnnouncement& ann);

    /// Create a channel announcement for a new channel
    Result<ChannelAnnouncement> create_channel_announcement(
        uint64_t short_channel_id,
        const crypto::Ed25519KeyPair& our_keys,
        const crypto::Ed25519PublicKey& their_node_id,
        int64_t capacity,
        const uint256& chain_hash);

    /// Create a channel update for our side
    Result<ChannelUpdate> create_channel_update(
        uint64_t short_channel_id,
        const crypto::Ed25519SecretKey& our_key,
        uint8_t direction,
        bool disabled,
        const ChannelConfig& config,
        const uint256& chain_hash);

    /// Create a node announcement
    Result<NodeAnnouncement> create_node_announcement(
        const crypto::Ed25519KeyPair& our_keys,
        const std::string& alias,
        const std::vector<uint8_t>& addresses);

    /// Set callback for forwarding gossip
    void set_forward_callback(GossipForwardFn fn);

    /// Check if a channel announcement has already been seen
    bool has_seen_channel(uint64_t short_channel_id) const;

    /// Prune old gossip data
    size_t prune(uint32_t current_time, uint32_t max_age);

    /// Get statistics
    size_t announcement_count() const;
    size_t update_count() const;

private:
    Router&                     router_;
    mutable core::Mutex         mutex_;
    GossipForwardFn             forward_fn_;

    // Track seen messages to prevent duplicate processing
    std::unordered_set<uint64_t>                      seen_channels_;
    std::unordered_map<uint64_t, uint32_t>            channel_update_timestamps_;
    std::unordered_map<crypto::Ed25519PublicKey, uint32_t> node_update_timestamps_;
};

}  // namespace rnet::lightning

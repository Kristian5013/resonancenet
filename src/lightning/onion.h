#pragma once

#include <cstdint>
#include <vector>

#include "core/types.h"
#include "core/error.h"
#include "crypto/ed25519.h"

namespace rnet::lightning {

// ── Onion routing constants ─────────────────────────────────────────

inline constexpr size_t ONION_PACKET_SIZE       = 1366;   // Fixed packet size
inline constexpr size_t ONION_HOP_DATA_SIZE     = 65;     // Per-hop payload
inline constexpr size_t ONION_HMAC_SIZE         = 32;     // HMAC per hop
inline constexpr size_t ONION_VERSION           = 0;

// ── Per-hop payload ─────────────────────────────────────────────────

/// Hop data included in each onion layer
struct HopData {
    uint64_t    short_channel_id = 0;  // Channel to forward on
    int64_t     amount = 0;            // Amount to forward (resonances)
    uint32_t    cltv_expiry = 0;       // CLTV for this hop
    uint8_t     padding[12] = {};      // Reserved/padding

    std::vector<uint8_t> serialize() const;
    static Result<HopData> deserialize(const std::vector<uint8_t>& data);
};

// ── Onion packet ────────────────────────────────────────────────────

/// Sphinx onion routing packet
struct OnionPacket {
    uint8_t                      version = ONION_VERSION;
    crypto::Ed25519PublicKey      ephemeral_key;          // 32 bytes
    std::vector<uint8_t>          encrypted_data;         // Encrypted hop payloads
    uint256                       hmac;                   // Integrity check

    /// Total serialized size
    size_t serialized_size() const;

    std::vector<uint8_t> serialize() const;
    static Result<OnionPacket> deserialize(const std::vector<uint8_t>& data);
};

// ── Onion routing ───────────────────────────────────────────────────

/// Build a Sphinx onion packet for the given route
/// session_key: ephemeral private key for this payment
/// hops: per-hop payloads (first = next hop, last = final recipient)
/// associated_data: payment hash for binding
Result<OnionPacket> create_onion_packet(
    const crypto::Ed25519KeyPair& session_key,
    const std::vector<crypto::Ed25519PublicKey>& hop_pubkeys,
    const std::vector<HopData>& hops,
    const uint256& associated_data);

/// Process (peel one layer of) an onion packet at this hop
/// Returns the hop data for us and the packet for the next hop
struct OnionPeelResult {
    HopData       hop_data;       // Forwarding instructions for us
    OnionPacket   next_packet;    // Packet for the next hop
    bool          is_final;       // True if we are the final recipient
};

Result<OnionPeelResult> peel_onion_packet(
    const crypto::Ed25519SecretKey& our_secret,
    const OnionPacket& packet,
    const uint256& associated_data);

/// Create an error packet to send back along the route
std::vector<uint8_t> create_onion_error(
    const uint256& shared_secret,
    uint16_t failure_code,
    const std::vector<uint8_t>& failure_data);

/// Decode an error packet received from a downstream hop
struct OnionError {
    uint16_t failure_code = 0;
    std::vector<uint8_t> failure_data;
    uint32_t failing_hop = 0;     // Index of the hop that failed
};

Result<OnionError> decode_onion_error(
    const std::vector<uint256>& shared_secrets,
    const std::vector<uint8_t>& error_packet);

}  // namespace rnet::lightning

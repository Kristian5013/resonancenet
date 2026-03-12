#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "core/types.h"
#include "core/error.h"
#include "core/sync.h"
#include "crypto/ed25519.h"
#include "lightning/channel_state.h"

namespace rnet::lightning {

// ── Lightning message types ─────────────────────────────────────────

enum class LightningMsgType : uint16_t {
    // Setup & control
    INIT                  = 16,
    ERROR_MSG             = 17,
    PING                  = 18,
    PONG                  = 19,

    // Channel establishment
    OPEN_CHANNEL          = 32,
    ACCEPT_CHANNEL        = 33,
    FUNDING_CREATED       = 34,
    FUNDING_SIGNED        = 35,
    FUNDING_LOCKED        = 36,

    // Channel operation
    SHUTDOWN              = 38,
    CLOSING_SIGNED        = 39,

    // HTLC
    UPDATE_ADD_HTLC       = 128,
    UPDATE_FULFILL_HTLC   = 130,
    UPDATE_FAIL_HTLC      = 131,
    COMMITMENT_SIGNED     = 132,
    REVOKE_AND_ACK        = 133,
    UPDATE_FEE            = 134,

    // Gossip
    CHANNEL_ANNOUNCEMENT  = 256,
    NODE_ANNOUNCEMENT     = 257,
    CHANNEL_UPDATE        = 258,

    // Queries
    QUERY_SHORT_CHANNEL_IDS = 261,
    REPLY_SHORT_CHANNEL_IDS = 262,
    QUERY_CHANNEL_RANGE     = 263,
    REPLY_CHANNEL_RANGE     = 264,
};

std::string_view lightning_msg_type_name(LightningMsgType type);

// ── Lightning message ───────────────────────────────────────────────

struct LightningMessage {
    LightningMsgType     type;
    std::vector<uint8_t> payload;

    /// Serialize with 2-byte type prefix + payload
    std::vector<uint8_t> serialize() const;

    /// Deserialize from raw bytes
    static Result<LightningMessage> deserialize(const std::vector<uint8_t>& data);
};

// ── Peer state ──────────────────────────────────────────────────────

enum class PeerState : uint8_t {
    DISCONNECTED = 0,
    CONNECTING   = 1,
    HANDSHAKING  = 2,
    CONNECTED    = 3,
};

std::string_view peer_state_name(PeerState state);

// ── Message handler callback ────────────────────────────────────────

using MessageHandler = std::function<void(
    const crypto::Ed25519PublicKey& peer_id,
    const LightningMessage& msg)>;

// ── Lightning peer ──────────────────────────────────────────────────

/// Represents a connection to a Lightning peer.
/// Handles the Noise_XK handshake and message framing.
class LightningPeer {
public:
    LightningPeer() = default;
    LightningPeer(const crypto::Ed25519PublicKey& remote_id,
                   const std::string& host,
                   uint16_t port);

    ~LightningPeer() = default;

    // Move-only
    LightningPeer(LightningPeer&& other) noexcept;
    LightningPeer& operator=(LightningPeer&& other) noexcept;
    LightningPeer(const LightningPeer&) = delete;
    LightningPeer& operator=(const LightningPeer&) = delete;

    // ── Connection lifecycle ────────────────────────────────────────

    /// Initiate connection to the remote peer
    Result<void> connect(const crypto::Ed25519KeyPair& local_keys);

    /// Accept an inbound connection
    Result<void> accept(const crypto::Ed25519KeyPair& local_keys,
                         const std::vector<uint8_t>& handshake_data);

    /// Disconnect from the peer
    void disconnect(const std::string& reason = "");

    /// Check if connected
    bool is_connected() const;

    // ── Messaging ───────────────────────────────────────────────────

    /// Send a message to this peer
    Result<void> send(const LightningMessage& msg);

    /// Send a raw payload with a message type
    Result<void> send(LightningMsgType type,
                       const std::vector<uint8_t>& payload);

    /// Process received raw data (may produce messages)
    Result<std::vector<LightningMessage>> process_received(
        const std::vector<uint8_t>& data);

    /// Set the message handler
    void set_message_handler(MessageHandler handler);

    // ── Ping/pong ───────────────────────────────────────────────────

    /// Send a ping
    Result<void> send_ping(uint16_t num_pong_bytes = 0);

    /// Handle a received ping (auto-sends pong)
    Result<void> handle_ping(const std::vector<uint8_t>& payload);

    /// Record pong received (updates latency)
    void handle_pong();

    // ── Queries ─────────────────────────────────────────────────────

    const crypto::Ed25519PublicKey& remote_id() const { return remote_id_; }
    const std::string& host() const { return host_; }
    uint16_t port() const { return port_; }
    PeerState state() const;
    uint64_t bytes_sent() const { return bytes_sent_.load(); }
    uint64_t bytes_received() const { return bytes_received_.load(); }
    uint64_t messages_sent() const { return messages_sent_.load(); }
    uint64_t messages_received() const { return messages_received_.load(); }
    int64_t latency_ms() const { return latency_ms_; }

    /// Channel IDs associated with this peer
    const std::vector<ChannelId>& channel_ids() const { return channel_ids_; }
    void add_channel(const ChannelId& id);
    void remove_channel(const ChannelId& id);

    /// String representation
    std::string to_string() const;

private:
    /// Encrypt a message using the session keys
    std::vector<uint8_t> encrypt_message(const std::vector<uint8_t>& plaintext);

    /// Decrypt a received message
    Result<std::vector<uint8_t>> decrypt_message(const std::vector<uint8_t>& ciphertext);

    crypto::Ed25519PublicKey     remote_id_;
    std::string                  host_;
    uint16_t                     port_ = LIGHTNING_PORT_MAINNET;

    mutable core::Mutex          mutex_;
    PeerState                    state_ = PeerState::DISCONNECTED;
    MessageHandler               message_handler_;

    // Session encryption keys (derived from handshake)
    uint256                      send_key_;
    uint256                      recv_key_;
    uint64_t                     send_nonce_ = 0;
    uint64_t                     recv_nonce_ = 0;

    // Statistics
    std::atomic<uint64_t>        bytes_sent_{0};
    std::atomic<uint64_t>        bytes_received_{0};
    std::atomic<uint64_t>        messages_sent_{0};
    std::atomic<uint64_t>        messages_received_{0};
    int64_t                      latency_ms_ = -1;

    // Ping tracking
    uint64_t                     last_ping_sent_ = 0;

    // Associated channels
    std::vector<ChannelId>       channel_ids_;

    // Receive buffer for partial messages
    std::vector<uint8_t>         recv_buffer_;
};

}  // namespace rnet::lightning

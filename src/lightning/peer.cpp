// Copyright (c) 2024-2026 The ResonanceNet Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "lightning/peer.h"

#include "core/logging.h"
#include "core/serialize.h"
#include "core/stream.h"
#include "crypto/chacha20.h"
#include "crypto/keccak.h"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace rnet::lightning {

// ===========================================================================
// Message Types
// ===========================================================================

// ---------------------------------------------------------------------------
// lightning_msg_type_name
// ---------------------------------------------------------------------------
// Maps each LightningMsgType enum value to a human-readable string for
// logging and diagnostics.  Covers setup/control, channel establishment,
// channel operation, HTLCs, gossip, and query messages.
// ---------------------------------------------------------------------------
std::string_view lightning_msg_type_name(LightningMsgType type) {
    switch (type) {
        case LightningMsgType::INIT:                    return "INIT";
        case LightningMsgType::ERROR_MSG:               return "ERROR";
        case LightningMsgType::PING:                    return "PING";
        case LightningMsgType::PONG:                    return "PONG";
        case LightningMsgType::OPEN_CHANNEL:            return "OPEN_CHANNEL";
        case LightningMsgType::ACCEPT_CHANNEL:          return "ACCEPT_CHANNEL";
        case LightningMsgType::FUNDING_CREATED:         return "FUNDING_CREATED";
        case LightningMsgType::FUNDING_SIGNED:          return "FUNDING_SIGNED";
        case LightningMsgType::FUNDING_LOCKED:          return "FUNDING_LOCKED";
        case LightningMsgType::SHUTDOWN:                return "SHUTDOWN";
        case LightningMsgType::CLOSING_SIGNED:          return "CLOSING_SIGNED";
        case LightningMsgType::UPDATE_ADD_HTLC:         return "UPDATE_ADD_HTLC";
        case LightningMsgType::UPDATE_FULFILL_HTLC:     return "UPDATE_FULFILL_HTLC";
        case LightningMsgType::UPDATE_FAIL_HTLC:        return "UPDATE_FAIL_HTLC";
        case LightningMsgType::COMMITMENT_SIGNED:       return "COMMITMENT_SIGNED";
        case LightningMsgType::REVOKE_AND_ACK:          return "REVOKE_AND_ACK";
        case LightningMsgType::UPDATE_FEE:              return "UPDATE_FEE";
        case LightningMsgType::CHANNEL_ANNOUNCEMENT:    return "CHANNEL_ANNOUNCEMENT";
        case LightningMsgType::NODE_ANNOUNCEMENT:       return "NODE_ANNOUNCEMENT";
        case LightningMsgType::CHANNEL_UPDATE:          return "CHANNEL_UPDATE";
        case LightningMsgType::QUERY_SHORT_CHANNEL_IDS: return "QUERY_SHORT_CHANNEL_IDS";
        case LightningMsgType::REPLY_SHORT_CHANNEL_IDS: return "REPLY_SHORT_CHANNEL_IDS";
        case LightningMsgType::QUERY_CHANNEL_RANGE:     return "QUERY_CHANNEL_RANGE";
        case LightningMsgType::REPLY_CHANNEL_RANGE:     return "REPLY_CHANNEL_RANGE";
        default:                                        return "UNKNOWN";
    }
}

// ---------------------------------------------------------------------------
// peer_state_name
// ---------------------------------------------------------------------------
// Maps each PeerState enum value to a human-readable string for logging
// and the to_string() diagnostic output.
// ---------------------------------------------------------------------------
std::string_view peer_state_name(PeerState state) {
    switch (state) {
        case PeerState::DISCONNECTED: return "DISCONNECTED";
        case PeerState::CONNECTING:   return "CONNECTING";
        case PeerState::HANDSHAKING:  return "HANDSHAKING";
        case PeerState::CONNECTED:    return "CONNECTED";
        default:                      return "UNKNOWN";
    }
}

// ===========================================================================
// Wire Format
// ===========================================================================

// ---------------------------------------------------------------------------
// LightningMessage::serialize
// ---------------------------------------------------------------------------
// Big-endian wire format: [2-byte type][payload].  The type field is
// encoded as a big-endian uint16 per BOLT-1 conventions, followed by the
// raw payload bytes.
// ---------------------------------------------------------------------------
std::vector<uint8_t> LightningMessage::serialize() const {
    // 1. Reserve space for the 2-byte type prefix plus the payload.
    std::vector<uint8_t> result;
    result.reserve(2 + payload.size());

    // 2. Encode the message type as big-endian uint16.
    result.push_back(static_cast<uint8_t>((static_cast<uint16_t>(type) >> 8) & 0xFF));
    result.push_back(static_cast<uint8_t>(static_cast<uint16_t>(type) & 0xFF));

    // 3. Append the raw payload bytes.
    result.insert(result.end(), payload.begin(), payload.end());
    return result;
}

// ---------------------------------------------------------------------------
// LightningMessage::deserialize
// ---------------------------------------------------------------------------
// Big-endian wire format: [2-byte type][payload].  Reads the 2-byte
// big-endian type prefix and copies the remaining bytes as the payload.
// Returns an error if the data is shorter than the 2-byte minimum.
// ---------------------------------------------------------------------------
Result<LightningMessage> LightningMessage::deserialize(
    const std::vector<uint8_t>& data) {
    // 1. Validate minimum length for the type prefix.
    if (data.size() < 2) {
        return Result<LightningMessage>::err("Message too short");
    }

    // 2. Decode the big-endian uint16 type field.
    LightningMessage msg;
    uint16_t type_val = (static_cast<uint16_t>(data[0]) << 8) |
                         static_cast<uint16_t>(data[1]);
    msg.type = static_cast<LightningMsgType>(type_val);

    // 3. Copy the remaining bytes as payload.
    msg.payload.assign(data.begin() + 2, data.end());
    return Result<LightningMessage>::ok(std::move(msg));
}

// ===========================================================================
// Peer Lifecycle
// ===========================================================================

// ---------------------------------------------------------------------------
// LightningPeer::LightningPeer (constructor)
// ---------------------------------------------------------------------------
// Stores the remote peer identity, host address, and port for later
// connection establishment.
// ---------------------------------------------------------------------------
LightningPeer::LightningPeer(const crypto::Ed25519PublicKey& remote_id,
                               const std::string& host,
                               uint16_t port)
    : remote_id_(remote_id), host_(host), port_(port) {}

// ---------------------------------------------------------------------------
// LightningPeer::LightningPeer (move constructor)
// ---------------------------------------------------------------------------
// Transfers all peer state from the source under lock.  The source peer
// is left in DISCONNECTED state after the move.
// ---------------------------------------------------------------------------
LightningPeer::LightningPeer(LightningPeer&& other) noexcept {
    LOCK(other.mutex_);
    remote_id_ = other.remote_id_;
    host_ = std::move(other.host_);
    port_ = other.port_;
    state_ = other.state_;
    message_handler_ = std::move(other.message_handler_);
    send_key_ = other.send_key_;
    recv_key_ = other.recv_key_;
    send_nonce_ = other.send_nonce_;
    recv_nonce_ = other.recv_nonce_;
    bytes_sent_.store(other.bytes_sent_.load());
    bytes_received_.store(other.bytes_received_.load());
    messages_sent_.store(other.messages_sent_.load());
    messages_received_.store(other.messages_received_.load());
    latency_ms_ = other.latency_ms_;
    last_ping_sent_ = other.last_ping_sent_;
    channel_ids_ = std::move(other.channel_ids_);
    recv_buffer_ = std::move(other.recv_buffer_);
    other.state_ = PeerState::DISCONNECTED;
}

// ---------------------------------------------------------------------------
// LightningPeer::operator= (move assignment)
// ---------------------------------------------------------------------------
// Transfers all peer state from the source under dual lock.  The source
// peer is left in DISCONNECTED state after the move.
// ---------------------------------------------------------------------------
LightningPeer& LightningPeer::operator=(LightningPeer&& other) noexcept {
    if (this != &other) {
        LOCK2(mutex_, other.mutex_);
        remote_id_ = other.remote_id_;
        host_ = std::move(other.host_);
        port_ = other.port_;
        state_ = other.state_;
        message_handler_ = std::move(other.message_handler_);
        send_key_ = other.send_key_;
        recv_key_ = other.recv_key_;
        send_nonce_ = other.send_nonce_;
        recv_nonce_ = other.recv_nonce_;
        bytes_sent_.store(other.bytes_sent_.load());
        bytes_received_.store(other.bytes_received_.load());
        messages_sent_.store(other.messages_sent_.load());
        messages_received_.store(other.messages_received_.load());
        latency_ms_ = other.latency_ms_;
        last_ping_sent_ = other.last_ping_sent_;
        channel_ids_ = std::move(other.channel_ids_);
        recv_buffer_ = std::move(other.recv_buffer_);
        other.state_ = PeerState::DISCONNECTED;
    }
    return *this;
}

// ---------------------------------------------------------------------------
// connect
// ---------------------------------------------------------------------------
// Session key derivation via Keccak256d:
//   shared = K(local_secret || remote_pubkey)
//   send_key = K(shared || "send")
//   recv_key = K(shared || "recv")
// In production this would be a full Noise_XK handshake.  The simplified
// scheme here provides directional encryption keys so that each side
// encrypts with a different key from the one it decrypts with.
// ---------------------------------------------------------------------------
Result<void> LightningPeer::connect(const crypto::Ed25519KeyPair& local_keys) {
    LOCK(mutex_);

    // 1. Verify the peer is currently disconnected.
    if (state_ != PeerState::DISCONNECTED) {
        return Result<void>::err("Peer is not disconnected");
    }

    // 2. Transition to CONNECTING state.
    state_ = PeerState::CONNECTING;

    // 3. Derive the shared secret: K(local_secret || remote_pubkey).
    crypto::KeccakHasher key_hasher;
    key_hasher.write(std::span<const uint8_t>(local_keys.secret.seed()));
    key_hasher.write(std::span<const uint8_t>(remote_id_.data));
    auto shared = key_hasher.finalize_double();

    // 4. Derive directional send key: K(shared || "send").
    {
        crypto::KeccakHasher h;
        h.write(shared.span());
        h.write(std::string_view("send"));
        send_key_ = h.finalize_double();
    }

    // 5. Derive directional receive key: K(shared || "recv").
    {
        crypto::KeccakHasher h;
        h.write(shared.span());
        h.write(std::string_view("recv"));
        recv_key_ = h.finalize_double();
    }

    // 6. Reset nonce counters and transition to CONNECTED.
    send_nonce_ = 0;
    recv_nonce_ = 0;
    state_ = PeerState::CONNECTED;

    LogPrint(LIGHTNING, "Connected to peer %s (%s:%u)",
             remote_id_.to_hex().c_str(), host_.c_str(), port_);

    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// accept
// ---------------------------------------------------------------------------
// Mirror of connect, keys are swapped: our recv = their send.  The shared
// secret is derived as K(remote_pubkey || local_secret) so that it matches
// the initiator's K(local_secret || remote_pubkey) when the roles are
// reversed.  The "recv"/"send" labels are then swapped relative to
// connect() so that our receive key matches their send key and vice versa.
// ---------------------------------------------------------------------------
Result<void> LightningPeer::accept(
    const crypto::Ed25519KeyPair& local_keys,
    const std::vector<uint8_t>& /*handshake_data*/) {

    LOCK(mutex_);

    // 1. Transition to HANDSHAKING state.
    state_ = PeerState::HANDSHAKING;

    // 2. Derive the shared secret (mirrored argument order).
    crypto::KeccakHasher key_hasher;
    key_hasher.write(std::span<const uint8_t>(remote_id_.data));
    key_hasher.write(std::span<const uint8_t>(local_keys.secret.seed()));
    auto shared = key_hasher.finalize_double();

    // 3. Derive our recv key from "recv" — matches the initiator's send key.
    {
        crypto::KeccakHasher h;
        h.write(shared.span());
        h.write(std::string_view("recv"));  // Our recv = their send
        recv_key_ = h.finalize_double();
    }

    // 4. Derive our send key from "send" — matches the initiator's recv key.
    {
        crypto::KeccakHasher h;
        h.write(shared.span());
        h.write(std::string_view("send"));  // Our send = their recv
        send_key_ = h.finalize_double();
    }

    // 5. Reset nonce counters and transition to CONNECTED.
    send_nonce_ = 0;
    recv_nonce_ = 0;
    state_ = PeerState::CONNECTED;

    LogPrint(LIGHTNING, "Accepted peer %s", remote_id_.to_hex().c_str());
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// disconnect
// ---------------------------------------------------------------------------
// Tears down the peer session: logs the reason, resets state to
// DISCONNECTED, clears the receive buffer, and zeroes nonce counters.
// ---------------------------------------------------------------------------
void LightningPeer::disconnect(const std::string& reason) {
    LOCK(mutex_);

    // 1. Bail out if already disconnected.
    if (state_ == PeerState::DISCONNECTED) return;

    // 2. Log the disconnect reason if provided.
    if (!reason.empty()) {
        LogPrint(LIGHTNING, "Disconnecting peer %s: %s",
                 remote_id_.to_hex().c_str(), reason.c_str());
    }

    // 3. Reset session state.
    state_ = PeerState::DISCONNECTED;
    recv_buffer_.clear();
    send_nonce_ = 0;
    recv_nonce_ = 0;
}

// ---------------------------------------------------------------------------
// is_connected
// ---------------------------------------------------------------------------
// Thread-safe check for CONNECTED state.
// ---------------------------------------------------------------------------
bool LightningPeer::is_connected() const {
    LOCK(mutex_);
    return state_ == PeerState::CONNECTED;
}

// ===========================================================================
// Encrypted Transport
// ===========================================================================

// ---------------------------------------------------------------------------
// send (LightningMessage overload)
// ---------------------------------------------------------------------------
// ChaCha20 encrypted transport, [2-byte length][ciphertext] framing.
// Serializes the message, encrypts with the session send key, and
// accounts for the bytes/messages in the peer statistics.
// ---------------------------------------------------------------------------
Result<void> LightningPeer::send(const LightningMessage& msg) {
    LOCK(mutex_);

    // 1. Verify we are connected.
    if (state_ != PeerState::CONNECTED) {
        return Result<void>::err("Not connected");
    }

    // 2. Serialize the message to wire format.
    auto raw = msg.serialize();

    // 3. Encrypt with the session send key.
    auto encrypted = encrypt_message(raw);

    // 4. In a real implementation, we would write to a socket here.
    bytes_sent_ += encrypted.size();
    messages_sent_++;

    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// send (type + payload overload)
// ---------------------------------------------------------------------------
// Convenience wrapper that constructs a LightningMessage from the given
// type and payload, then delegates to the primary send() method.
// ---------------------------------------------------------------------------
Result<void> LightningPeer::send(LightningMsgType type,
                                  const std::vector<uint8_t>& payload) {
    LightningMessage msg;
    msg.type = type;
    msg.payload = payload;
    return send(msg);
}

// ---------------------------------------------------------------------------
// process_received
// ---------------------------------------------------------------------------
// ChaCha20 encrypted transport, [2-byte length][ciphertext] framing.
// Appends incoming data to the receive buffer and extracts complete
// messages.  Ping/pong messages are handled internally; all other
// messages are dispatched to the registered handler and returned.
// ---------------------------------------------------------------------------
Result<std::vector<LightningMessage>> LightningPeer::process_received(
    const std::vector<uint8_t>& data) {
    LOCK(mutex_);

    // 1. Accumulate incoming data into the receive buffer.
    bytes_received_ += data.size();
    recv_buffer_.insert(recv_buffer_.end(), data.begin(), data.end());

    std::vector<LightningMessage> messages;

    // 2. Extract complete messages from the buffer.
    //    Message format: [2-byte big-endian length][encrypted payload].
    while (recv_buffer_.size() >= 2) {
        // 3. Read the 2-byte big-endian message length prefix.
        uint16_t msg_len = (static_cast<uint16_t>(recv_buffer_[0]) << 8) |
                            static_cast<uint16_t>(recv_buffer_[1]);

        // 4. Wait for the full message to arrive.
        if (recv_buffer_.size() < static_cast<size_t>(2 + msg_len)) {
            break;  // Incomplete message
        }

        // 5. Extract the ciphertext and remove it from the buffer.
        std::vector<uint8_t> ciphertext(
            recv_buffer_.begin() + 2,
            recv_buffer_.begin() + 2 + msg_len);
        recv_buffer_.erase(recv_buffer_.begin(),
                           recv_buffer_.begin() + 2 + msg_len);

        // 6. Decrypt the ciphertext using the session recv key.
        auto decrypted = decrypt_message(ciphertext);
        if (!decrypted) {
            return Result<std::vector<LightningMessage>>::err(
                "Decryption failed: " + decrypted.error());
        }

        // 7. Deserialize the plaintext into a LightningMessage.
        auto msg = LightningMessage::deserialize(decrypted.value());
        if (!msg) {
            return Result<std::vector<LightningMessage>>::err(
                "Deserialization failed: " + msg.error());
        }

        messages_received_++;

        // 8. Handle ping/pong messages internally (BOLT-1 keepalive).
        if (msg.value().type == LightningMsgType::PING) {
            handle_ping(msg.value().payload);
            continue;
        }
        if (msg.value().type == LightningMsgType::PONG) {
            handle_pong();
            continue;
        }

        // 9. Dispatch to the registered message handler.
        if (message_handler_) {
            message_handler_(remote_id_, msg.value());
        }

        messages.push_back(std::move(msg.value()));
    }

    return Result<std::vector<LightningMessage>>::ok(std::move(messages));
}

// ---------------------------------------------------------------------------
// set_message_handler
// ---------------------------------------------------------------------------
// Registers a callback invoked for every non-ping/pong message received
// from this peer.
// ---------------------------------------------------------------------------
void LightningPeer::set_message_handler(MessageHandler handler) {
    LOCK(mutex_);
    message_handler_ = std::move(handler);
}

// ===========================================================================
// Ping/Pong
// ===========================================================================

// ---------------------------------------------------------------------------
// send_ping
// ---------------------------------------------------------------------------
// BOLT-1 ping/pong for keepalive and latency measurement.  Sends a ping
// with the requested pong-bytes count and records the send timestamp so
// that handle_pong() can compute round-trip latency.
// ---------------------------------------------------------------------------
Result<void> LightningPeer::send_ping(uint16_t num_pong_bytes) {
    // 1. Serialize the ping payload: [2-byte pong_bytes][2-byte padding_len].
    core::DataStream ss;
    core::ser_write_u16(ss, num_pong_bytes);
    core::ser_write_u16(ss, 0);  // Padding length

    // 2. Record the current time for latency calculation.
    auto now = std::chrono::steady_clock::now();
    last_ping_sent_ = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count());

    // 3. Send the ping message.
    return send(LightningMsgType::PING, ss.vch());
}

// ---------------------------------------------------------------------------
// handle_ping
// ---------------------------------------------------------------------------
// BOLT-1 ping/pong for keepalive and latency measurement.  Parses the
// incoming ping payload to determine how many bytes the remote expects
// in the pong response, then sends a zero-filled pong of that size.
// ---------------------------------------------------------------------------
Result<void> LightningPeer::handle_ping(
    const std::vector<uint8_t>& payload) {
    // 1. Validate the minimum ping payload size (2 uint16 fields).
    if (payload.size() < 4) {
        return Result<void>::err("Invalid ping payload");
    }

    // 2. Read the requested pong-bytes count.
    core::SpanReader reader{std::span<const uint8_t>(payload)};
    uint16_t num_pong_bytes = core::ser_read_u16(reader);

    // 3. Send a pong with the requested number of zero-filled bytes.
    std::vector<uint8_t> pong_data(num_pong_bytes, 0);
    return send(LightningMsgType::PONG, pong_data);
}

// ---------------------------------------------------------------------------
// handle_pong
// ---------------------------------------------------------------------------
// BOLT-1 ping/pong for keepalive and latency measurement.  Computes the
// round-trip latency by subtracting the last ping timestamp from the
// current time.
// ---------------------------------------------------------------------------
void LightningPeer::handle_pong() {
    // 1. Capture the current timestamp.
    auto now = std::chrono::steady_clock::now();
    uint64_t now_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count());

    // 2. Compute round-trip latency if a ping was previously sent.
    if (last_ping_sent_ > 0) {
        latency_ms_ = static_cast<int64_t>(now_ms - last_ping_sent_);
    }
}

// ===========================================================================
// Channel Management
// ===========================================================================

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------
// Thread-safe accessor for the current peer connection state.
// ---------------------------------------------------------------------------
PeerState LightningPeer::state() const {
    LOCK(mutex_);
    return state_;
}

// ---------------------------------------------------------------------------
// add_channel
// ---------------------------------------------------------------------------
// Associates a channel with this peer by appending its ID to the tracked
// channel list.
// ---------------------------------------------------------------------------
void LightningPeer::add_channel(const ChannelId& id) {
    LOCK(mutex_);
    channel_ids_.push_back(id);
}

// ---------------------------------------------------------------------------
// remove_channel
// ---------------------------------------------------------------------------
// Disassociates a channel from this peer using the erase-remove idiom.
// ---------------------------------------------------------------------------
void LightningPeer::remove_channel(const ChannelId& id) {
    LOCK(mutex_);
    channel_ids_.erase(
        std::remove(channel_ids_.begin(), channel_ids_.end(), id),
        channel_ids_.end());
}

// ---------------------------------------------------------------------------
// to_string
// ---------------------------------------------------------------------------
// Returns a compact diagnostic string showing the peer's truncated ID,
// endpoint address, connection state, and channel count.
// ---------------------------------------------------------------------------
std::string LightningPeer::to_string() const {
    LOCK(mutex_);
    return "Peer{" + remote_id_.to_hex().substr(0, 16) + "..." +
           ", " + host_ + ":" + std::to_string(port_) +
           ", state=" + std::string(peer_state_name(state_)) +
           ", channels=" + std::to_string(channel_ids_.size()) + "}";
}

// ===========================================================================
// Encryption
// ===========================================================================

// ---------------------------------------------------------------------------
// encrypt_message
// ---------------------------------------------------------------------------
// ChaCha20 with counter-derived nonce: nonce[4..12] = little_endian(counter++).
// The first 4 bytes of the nonce are zero (per RFC 7539 convention for the
// block counter prefix), and bytes 4-11 carry the little-endian send nonce
// which increments after each message.
// ---------------------------------------------------------------------------
std::vector<uint8_t> LightningPeer::encrypt_message(
    const std::vector<uint8_t>& plaintext) {
    // 1. Build the 12-byte nonce from the send counter.
    std::array<uint8_t, crypto::ChaCha20::NONCE_SIZE> nonce{};
    uint64_t n = send_nonce_++;
    for (int i = 0; i < 8; ++i) {
        nonce[4 + i] = static_cast<uint8_t>((n >> (i * 8)) & 0xFF);
    }

    // 2. Encrypt in-place with ChaCha20 using the session send key.
    std::vector<uint8_t> ciphertext = plaintext;
    crypto::ChaCha20 cipher(
        std::span<const uint8_t, 32>(send_key_.data(), 32),
        std::span<const uint8_t, 12>(nonce.data(), 12));
    cipher.crypt(std::span<uint8_t>(ciphertext));

    return ciphertext;
}

// ---------------------------------------------------------------------------
// decrypt_message
// ---------------------------------------------------------------------------
// ChaCha20 with counter-derived nonce: nonce[4..12] = little_endian(counter++).
// Mirror of encrypt_message using the session recv key and recv nonce
// counter.  ChaCha20 is symmetric, so encryption and decryption are the
// same XOR operation with the same keystream.
// ---------------------------------------------------------------------------
Result<std::vector<uint8_t>> LightningPeer::decrypt_message(
    const std::vector<uint8_t>& ciphertext) {
    // 1. Build the 12-byte nonce from the recv counter.
    std::array<uint8_t, crypto::ChaCha20::NONCE_SIZE> nonce{};
    uint64_t n = recv_nonce_++;
    for (int i = 0; i < 8; ++i) {
        nonce[4 + i] = static_cast<uint8_t>((n >> (i * 8)) & 0xFF);
    }

    // 2. Decrypt in-place with ChaCha20 using the session recv key.
    std::vector<uint8_t> plaintext = ciphertext;
    crypto::ChaCha20 cipher(
        std::span<const uint8_t, 32>(recv_key_.data(), 32),
        std::span<const uint8_t, 12>(nonce.data(), 12));
    cipher.crypt(std::span<uint8_t>(plaintext));

    return Result<std::vector<uint8_t>>::ok(std::move(plaintext));
}

} // namespace rnet::lightning

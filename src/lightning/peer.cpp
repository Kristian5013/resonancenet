#include "lightning/peer.h"

#include <algorithm>
#include <chrono>
#include <cstring>

#include "core/logging.h"
#include "core/stream.h"
#include "core/serialize.h"
#include "crypto/keccak.h"
#include "crypto/chacha20.h"

namespace rnet::lightning {

// ── Message type names ──────────────────────────────────────────────

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

std::string_view peer_state_name(PeerState state) {
    switch (state) {
        case PeerState::DISCONNECTED: return "DISCONNECTED";
        case PeerState::CONNECTING:   return "CONNECTING";
        case PeerState::HANDSHAKING:  return "HANDSHAKING";
        case PeerState::CONNECTED:    return "CONNECTED";
        default:                      return "UNKNOWN";
    }
}

// ── LightningMessage ────────────────────────────────────────────────

std::vector<uint8_t> LightningMessage::serialize() const {
    std::vector<uint8_t> result;
    result.reserve(2 + payload.size());
    // Big-endian type
    result.push_back(static_cast<uint8_t>((static_cast<uint16_t>(type) >> 8) & 0xFF));
    result.push_back(static_cast<uint8_t>(static_cast<uint16_t>(type) & 0xFF));
    result.insert(result.end(), payload.begin(), payload.end());
    return result;
}

Result<LightningMessage> LightningMessage::deserialize(
    const std::vector<uint8_t>& data) {
    if (data.size() < 2) {
        return Result<LightningMessage>::err("Message too short");
    }

    LightningMessage msg;
    uint16_t type_val = (static_cast<uint16_t>(data[0]) << 8) |
                         static_cast<uint16_t>(data[1]);
    msg.type = static_cast<LightningMsgType>(type_val);
    msg.payload.assign(data.begin() + 2, data.end());
    return Result<LightningMessage>::ok(std::move(msg));
}

// ── LightningPeer ───────────────────────────────────────────────────

LightningPeer::LightningPeer(const crypto::Ed25519PublicKey& remote_id,
                               const std::string& host,
                               uint16_t port)
    : remote_id_(remote_id), host_(host), port_(port) {}

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

Result<void> LightningPeer::connect(const crypto::Ed25519KeyPair& local_keys) {
    LOCK(mutex_);

    if (state_ != PeerState::DISCONNECTED) {
        return Result<void>::err("Peer is not disconnected");
    }

    state_ = PeerState::CONNECTING;

    // Derive session keys from a simple key agreement
    // (In production: full Noise_XK handshake)
    crypto::KeccakHasher key_hasher;
    key_hasher.write(std::span<const uint8_t>(local_keys.secret.seed()));
    key_hasher.write(std::span<const uint8_t>(remote_id_.data));
    auto shared = key_hasher.finalize_double();

    // Derive send and receive keys
    {
        crypto::KeccakHasher h;
        h.write(shared.span());
        h.write(std::string_view("send"));
        send_key_ = h.finalize_double();
    }
    {
        crypto::KeccakHasher h;
        h.write(shared.span());
        h.write(std::string_view("recv"));
        recv_key_ = h.finalize_double();
    }

    send_nonce_ = 0;
    recv_nonce_ = 0;
    state_ = PeerState::CONNECTED;

    LogPrint(LIGHTNING, "Connected to peer %s (%s:%u)",
             remote_id_.to_hex().c_str(), host_.c_str(), port_);

    return Result<void>::ok();
}

Result<void> LightningPeer::accept(
    const crypto::Ed25519KeyPair& local_keys,
    const std::vector<uint8_t>& /*handshake_data*/) {

    LOCK(mutex_);
    state_ = PeerState::HANDSHAKING;

    // Derive session keys (mirror of connect)
    crypto::KeccakHasher key_hasher;
    key_hasher.write(std::span<const uint8_t>(remote_id_.data));
    key_hasher.write(std::span<const uint8_t>(local_keys.secret.seed()));
    auto shared = key_hasher.finalize_double();

    {
        crypto::KeccakHasher h;
        h.write(shared.span());
        h.write(std::string_view("recv"));  // Our recv = their send
        recv_key_ = h.finalize_double();
    }
    {
        crypto::KeccakHasher h;
        h.write(shared.span());
        h.write(std::string_view("send"));  // Our send = their recv
        send_key_ = h.finalize_double();
    }

    send_nonce_ = 0;
    recv_nonce_ = 0;
    state_ = PeerState::CONNECTED;

    LogPrint(LIGHTNING, "Accepted peer %s", remote_id_.to_hex().c_str());
    return Result<void>::ok();
}

void LightningPeer::disconnect(const std::string& reason) {
    LOCK(mutex_);
    if (state_ == PeerState::DISCONNECTED) return;

    if (!reason.empty()) {
        LogPrint(LIGHTNING, "Disconnecting peer %s: %s",
                 remote_id_.to_hex().c_str(), reason.c_str());
    }

    state_ = PeerState::DISCONNECTED;
    recv_buffer_.clear();
    send_nonce_ = 0;
    recv_nonce_ = 0;
}

bool LightningPeer::is_connected() const {
    LOCK(mutex_);
    return state_ == PeerState::CONNECTED;
}

Result<void> LightningPeer::send(const LightningMessage& msg) {
    LOCK(mutex_);

    if (state_ != PeerState::CONNECTED) {
        return Result<void>::err("Not connected");
    }

    auto raw = msg.serialize();
    auto encrypted = encrypt_message(raw);

    // In a real implementation, we would write to a socket here
    bytes_sent_ += encrypted.size();
    messages_sent_++;

    return Result<void>::ok();
}

Result<void> LightningPeer::send(LightningMsgType type,
                                  const std::vector<uint8_t>& payload) {
    LightningMessage msg;
    msg.type = type;
    msg.payload = payload;
    return send(msg);
}

Result<std::vector<LightningMessage>> LightningPeer::process_received(
    const std::vector<uint8_t>& data) {
    LOCK(mutex_);

    bytes_received_ += data.size();
    recv_buffer_.insert(recv_buffer_.end(), data.begin(), data.end());

    std::vector<LightningMessage> messages;

    // Process complete messages from the buffer
    // Message format: [2-byte length][encrypted payload]
    while (recv_buffer_.size() >= 2) {
        uint16_t msg_len = (static_cast<uint16_t>(recv_buffer_[0]) << 8) |
                            static_cast<uint16_t>(recv_buffer_[1]);

        if (recv_buffer_.size() < static_cast<size_t>(2 + msg_len)) {
            break;  // Incomplete message
        }

        std::vector<uint8_t> ciphertext(
            recv_buffer_.begin() + 2,
            recv_buffer_.begin() + 2 + msg_len);
        recv_buffer_.erase(recv_buffer_.begin(),
                           recv_buffer_.begin() + 2 + msg_len);

        auto decrypted = decrypt_message(ciphertext);
        if (!decrypted) {
            return Result<std::vector<LightningMessage>>::err(
                "Decryption failed: " + decrypted.error());
        }

        auto msg = LightningMessage::deserialize(decrypted.value());
        if (!msg) {
            return Result<std::vector<LightningMessage>>::err(
                "Deserialization failed: " + msg.error());
        }

        messages_received_++;

        // Handle pings automatically
        if (msg.value().type == LightningMsgType::PING) {
            handle_ping(msg.value().payload);
            continue;
        }
        if (msg.value().type == LightningMsgType::PONG) {
            handle_pong();
            continue;
        }

        // Dispatch to handler
        if (message_handler_) {
            message_handler_(remote_id_, msg.value());
        }

        messages.push_back(std::move(msg.value()));
    }

    return Result<std::vector<LightningMessage>>::ok(std::move(messages));
}

void LightningPeer::set_message_handler(MessageHandler handler) {
    LOCK(mutex_);
    message_handler_ = std::move(handler);
}

Result<void> LightningPeer::send_ping(uint16_t num_pong_bytes) {
    core::DataStream ss;
    core::ser_write_u16(ss, num_pong_bytes);
    core::ser_write_u16(ss, 0);  // Padding length

    auto now = std::chrono::steady_clock::now();
    last_ping_sent_ = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count());

    return send(LightningMsgType::PING, ss.vch());
}

Result<void> LightningPeer::handle_ping(
    const std::vector<uint8_t>& payload) {
    if (payload.size() < 4) {
        return Result<void>::err("Invalid ping payload");
    }

    core::SpanReader reader{std::span<const uint8_t>(payload)};
    uint16_t num_pong_bytes = core::ser_read_u16(reader);

    // Send pong with requested bytes
    std::vector<uint8_t> pong_data(num_pong_bytes, 0);
    return send(LightningMsgType::PONG, pong_data);
}

void LightningPeer::handle_pong() {
    auto now = std::chrono::steady_clock::now();
    uint64_t now_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count());

    if (last_ping_sent_ > 0) {
        latency_ms_ = static_cast<int64_t>(now_ms - last_ping_sent_);
    }
}

PeerState LightningPeer::state() const {
    LOCK(mutex_);
    return state_;
}

void LightningPeer::add_channel(const ChannelId& id) {
    LOCK(mutex_);
    channel_ids_.push_back(id);
}

void LightningPeer::remove_channel(const ChannelId& id) {
    LOCK(mutex_);
    channel_ids_.erase(
        std::remove(channel_ids_.begin(), channel_ids_.end(), id),
        channel_ids_.end());
}

std::string LightningPeer::to_string() const {
    LOCK(mutex_);
    return "Peer{" + remote_id_.to_hex().substr(0, 16) + "..." +
           ", " + host_ + ":" + std::to_string(port_) +
           ", state=" + std::string(peer_state_name(state_)) +
           ", channels=" + std::to_string(channel_ids_.size()) + "}";
}

// ── Encryption ──────────────────────────────────────────────────────

std::vector<uint8_t> LightningPeer::encrypt_message(
    const std::vector<uint8_t>& plaintext) {
    // Derive nonce from counter
    std::array<uint8_t, crypto::ChaCha20::NONCE_SIZE> nonce{};
    uint64_t n = send_nonce_++;
    for (int i = 0; i < 8; ++i) {
        nonce[4 + i] = static_cast<uint8_t>((n >> (i * 8)) & 0xFF);
    }

    // Encrypt with ChaCha20
    std::vector<uint8_t> ciphertext = plaintext;
    crypto::ChaCha20 cipher(
        std::span<const uint8_t, 32>(send_key_.data(), 32),
        std::span<const uint8_t, 12>(nonce.data(), 12));
    cipher.crypt(std::span<uint8_t>(ciphertext));

    return ciphertext;
}

Result<std::vector<uint8_t>> LightningPeer::decrypt_message(
    const std::vector<uint8_t>& ciphertext) {
    // Derive nonce from counter
    std::array<uint8_t, crypto::ChaCha20::NONCE_SIZE> nonce{};
    uint64_t n = recv_nonce_++;
    for (int i = 0; i < 8; ++i) {
        nonce[4 + i] = static_cast<uint8_t>((n >> (i * 8)) & 0xFF);
    }

    // Decrypt with ChaCha20
    std::vector<uint8_t> plaintext = ciphertext;
    crypto::ChaCha20 cipher(
        std::span<const uint8_t, 32>(recv_key_.data(), 32),
        std::span<const uint8_t, 12>(nonce.data(), 12));
    cipher.crypt(std::span<uint8_t>(plaintext));

    return Result<std::vector<uint8_t>>::ok(std::move(plaintext));
}

}  // namespace rnet::lightning

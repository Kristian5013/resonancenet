#include "net/transport.h"

#include <algorithm>
#include <cstring>

#include "core/logging.h"
#include "core/random.h"
#include "crypto/chacha20.h"
#include "crypto/keccak.h"

namespace rnet::net {

// ---------------------------------------------------------------------------
// NetMessage
// ---------------------------------------------------------------------------

bool NetMessage::verify_checksum() const {
    uint32_t expected = Transport::compute_checksum(
        std::span<const uint8_t>(payload.data(), payload.size()));
    return checksum == expected;
}

// ---------------------------------------------------------------------------
// Transport — static helpers
// ---------------------------------------------------------------------------

uint32_t Transport::compute_checksum(std::span<const uint8_t> payload) {
    auto hash = crypto::keccak256d(payload);
    uint32_t cksum = 0;
    std::memcpy(&cksum, hash.data(), 4);
    return cksum;
}

std::vector<uint8_t> Transport::serialize_message(
    std::string_view command,
    std::span<const uint8_t> payload)
{
    MessageHeader hdr;
    hdr.magic = NETWORK_MAGIC;
    hdr.set_command(command);
    hdr.payload_size = static_cast<uint32_t>(payload.size());
    hdr.checksum = compute_checksum(payload);

    core::DataStream header_stream;
    header_stream.reserve(MessageHeader::HEADER_SIZE + payload.size());
    hdr.serialize(header_stream);

    std::vector<uint8_t> result = std::move(header_stream.vch());
    result.insert(result.end(), payload.begin(), payload.end());
    return result;
}

std::vector<uint8_t> Transport::serialize_message(std::string_view command) {
    return serialize_message(command, std::span<const uint8_t>{});
}

// ---------------------------------------------------------------------------
// Transport — instance methods
// ---------------------------------------------------------------------------

Transport::Transport() = default;

void Transport::feed(std::span<const uint8_t> data) {
    if (error_) return;
    recv_buf_.insert(recv_buf_.end(), data.begin(), data.end());
}

std::optional<NetMessage> Transport::next_message() {
    if (error_) return std::nullopt;

    for (;;) {
        switch (state_) {
        case ParseState::READING_HEADER: {
            if (!try_parse_header()) {
                return std::nullopt;
            }
            state_ = ParseState::READING_PAYLOAD;
            break;
        }

        case ParseState::READING_PAYLOAD: {
            if (recv_buf_.size() < current_header_.payload_size) {
                return std::nullopt;
            }

            NetMessage msg;
            msg.command = current_header_.get_command();
            msg.checksum = current_header_.checksum;

            if (current_header_.payload_size > 0) {
                msg.payload.assign(
                    recv_buf_.begin(),
                    recv_buf_.begin() +
                        static_cast<ptrdiff_t>(current_header_.payload_size));
                recv_buf_.erase(
                    recv_buf_.begin(),
                    recv_buf_.begin() +
                        static_cast<ptrdiff_t>(current_header_.payload_size));
            }

            if (!msg.verify_checksum()) {
                set_error("Bad checksum for command '" + msg.command + "'");
                return std::nullopt;
            }

            state_ = ParseState::READING_HEADER;
            return msg;
        }
        }
    }
}

bool Transport::try_parse_header() {
    if (recv_buf_.size() < MessageHeader::HEADER_SIZE) {
        return false;
    }

    while (recv_buf_.size() >= MessageHeader::HEADER_SIZE) {
        bool magic_ok = (recv_buf_[0] == NETWORK_MAGIC[0] &&
                         recv_buf_[1] == NETWORK_MAGIC[1] &&
                         recv_buf_[2] == NETWORK_MAGIC[2] &&
                         recv_buf_[3] == NETWORK_MAGIC[3]);
        if (magic_ok) break;
        recv_buf_.erase(recv_buf_.begin());
    }

    if (recv_buf_.size() < MessageHeader::HEADER_SIZE) {
        return false;
    }

    core::DataStream hdr_stream(
        std::span<const uint8_t>(recv_buf_.data(), MessageHeader::HEADER_SIZE));
    current_header_.unserialize(hdr_stream);

    recv_buf_.erase(recv_buf_.begin(),
                    recv_buf_.begin() +
                        static_cast<ptrdiff_t>(MessageHeader::HEADER_SIZE));

    if (!current_header_.valid_magic()) {
        set_error("Invalid magic bytes in header");
        return false;
    }

    if (current_header_.payload_size > MAX_MESSAGE_SIZE) {
        set_error("Payload size " +
                  std::to_string(current_header_.payload_size) +
                  " exceeds maximum " +
                  std::to_string(MAX_MESSAGE_SIZE));
        return false;
    }

    std::string cmd = current_header_.get_command();
    if (cmd.empty()) {
        set_error("Empty command in message header");
        return false;
    }

    return true;
}

void Transport::reset() {
    state_ = ParseState::READING_HEADER;
    recv_buf_.clear();
    current_header_ = MessageHeader{};
    error_ = false;
    error_msg_.clear();
}

void Transport::set_error(const std::string& msg) {
    error_ = true;
    error_msg_ = msg;
    LogPrint(NET, "Transport error: %s", msg.c_str());
}

// ---------------------------------------------------------------------------
// EncryptedTransport
// ---------------------------------------------------------------------------

EncryptedTransport::EncryptedTransport() = default;
EncryptedTransport::~EncryptedTransport() = default;

void EncryptedTransport::set_session_key(const rnet::uint256& key) {
    auto send_data = key.to_hex() + ":send";
    send_key_ = crypto::keccak256d(send_data);
    auto recv_data = key.to_hex() + ":recv";
    recv_key_ = crypto::keccak256d(recv_data);
    session_id_ = crypto::keccak256d(key.span());
    established_ = true;
}

Result<std::vector<uint8_t>> EncryptedTransport::initiate_handshake() {
    is_initiator_ = true;

    local_privkey_.resize(32);
    core::get_rand_bytes(std::span<uint8_t>(local_privkey_));

    local_pubkey_.resize(32);
    auto pub_hash = crypto::keccak256(
        std::span<const uint8_t>(local_privkey_));
    std::memcpy(local_pubkey_.data(), pub_hash.data(), 32);

    return Result<std::vector<uint8_t>>::ok(local_pubkey_);
}

Result<std::vector<uint8_t>> EncryptedTransport::process_handshake(
    std::span<const uint8_t> data)
{
    if (data.size() != 32) {
        return Result<std::vector<uint8_t>>::err(
            "Invalid handshake data size");
    }

    if (!is_initiator_ && local_privkey_.empty()) {
        local_privkey_.resize(32);
        core::get_rand_bytes(std::span<uint8_t>(local_privkey_));

        local_pubkey_.resize(32);
        auto pub_hash = crypto::keccak256(
            std::span<const uint8_t>(local_privkey_));
        std::memcpy(local_pubkey_.data(), pub_hash.data(), 32);
    }

    std::vector<uint8_t> shared_input(64);
    std::memcpy(shared_input.data(), local_privkey_.data(), 32);
    std::memcpy(shared_input.data() + 32, data.data(), 32);

    auto shared_secret = crypto::keccak256d(
        std::span<const uint8_t>(shared_input));

    if (is_initiator_) {
        auto sk_data = shared_secret.to_hex() + ":init_send";
        send_key_ = crypto::keccak256d(sk_data);
        auto rk_data = shared_secret.to_hex() + ":init_recv";
        recv_key_ = crypto::keccak256d(rk_data);
    } else {
        auto sk_data = shared_secret.to_hex() + ":init_recv";
        send_key_ = crypto::keccak256d(sk_data);
        auto rk_data = shared_secret.to_hex() + ":init_send";
        recv_key_ = crypto::keccak256d(rk_data);
    }

    session_id_ = crypto::keccak256d(shared_secret.span());
    established_ = true;

    std::memset(local_privkey_.data(), 0, local_privkey_.size());
    local_privkey_.clear();

    if (!is_initiator_) {
        return Result<std::vector<uint8_t>>::ok(local_pubkey_);
    }

    return Result<std::vector<uint8_t>>::ok(std::vector<uint8_t>{});
}

Result<std::vector<uint8_t>> EncryptedTransport::encrypt(
    std::span<const uint8_t> plaintext)
{
    if (!established_) {
        return Result<std::vector<uint8_t>>::err(
            "Transport not established");
    }

    auto nonce = make_nonce(send_nonce_++);

    return crypto::ChaCha20Poly1305::encrypt(
        std::span<const uint8_t>(send_key_.data(), 32),
        std::span<const uint8_t>(nonce.data(), nonce.size()),
        std::span<const uint8_t>{},
        plaintext);
}

Result<std::vector<uint8_t>> EncryptedTransport::decrypt(
    std::span<const uint8_t> ciphertext)
{
    if (!established_) {
        return Result<std::vector<uint8_t>>::err(
            "Transport not established");
    }

    auto nonce = make_nonce(recv_nonce_++);

    return crypto::ChaCha20Poly1305::decrypt(
        std::span<const uint8_t>(recv_key_.data(), 32),
        std::span<const uint8_t>(nonce.data(), nonce.size()),
        std::span<const uint8_t>{},
        ciphertext);
}

std::array<uint8_t, 12> EncryptedTransport::make_nonce(
    uint64_t counter) const
{
    std::array<uint8_t, 12> nonce{};
    for (int i = 0; i < 8; ++i) {
        nonce[static_cast<size_t>(4 + i)] =
            static_cast<uint8_t>((counter >> (i * 8)) & 0xFF);
    }
    return nonce;
}

}  // namespace rnet::net

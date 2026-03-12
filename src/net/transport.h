#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/error.h"
#include "core/stream.h"
#include "core/types.h"
#include "net/protocol.h"

namespace rnet::net {

/// NetMessage — a parsed protocol message (command + payload).
struct NetMessage {
    std::string command;
    std::vector<uint8_t> payload;
    uint32_t checksum = 0;

    /// Verify the checksum of the payload
    bool verify_checksum() const;
};

/// Transport — wire-level message framing (parse bytes into messages).
/// Implements the [magic][command][size][checksum][payload] protocol.
class Transport {
public:
    Transport();

    /// Feed raw bytes from the socket into the parser.
    void feed(std::span<const uint8_t> data);

    /// Try to extract the next complete message.
    /// Returns nullopt if not enough data is available.
    std::optional<NetMessage> next_message();

    /// Serialize a command + payload into wire format.
    static std::vector<uint8_t> serialize_message(
        std::string_view command,
        std::span<const uint8_t> payload);

    /// Serialize a command with no payload.
    static std::vector<uint8_t> serialize_message(std::string_view command);

    /// Compute the 4-byte checksum for a payload.
    static uint32_t compute_checksum(std::span<const uint8_t> payload);

    /// Check if the transport is in an error state.
    bool has_error() const { return error_; }
    const std::string& error_message() const { return error_msg_; }

    /// Reset the transport (clear buffers, reset state).
    void reset();

private:
    enum class ParseState { READING_HEADER, READING_PAYLOAD };

    ParseState state_ = ParseState::READING_HEADER;
    std::vector<uint8_t> recv_buf_;
    MessageHeader current_header_;
    bool error_ = false;
    std::string error_msg_;

    bool try_parse_header();
    void set_error(const std::string& msg);
};

/// EncryptedTransport — encrypted P2P transport using ChaCha20-Poly1305.
///
/// After handshake, all messages are encrypted and authenticated.
class EncryptedTransport {
public:
    EncryptedTransport();
    ~EncryptedTransport();

    // Non-copyable
    EncryptedTransport(const EncryptedTransport&) = delete;
    EncryptedTransport& operator=(const EncryptedTransport&) = delete;

    /// Initialize the transport with a session key (for testing)
    void set_session_key(const rnet::uint256& key);

    /// Perform the handshake as the initiator
    Result<std::vector<uint8_t>> initiate_handshake();

    /// Process handshake bytes from the remote side
    Result<std::vector<uint8_t>> process_handshake(
        std::span<const uint8_t> data);

    /// Check if the handshake is complete
    bool is_established() const { return established_; }

    /// Encrypt a message for sending
    Result<std::vector<uint8_t>> encrypt(
        std::span<const uint8_t> plaintext);

    /// Decrypt a received message
    Result<std::vector<uint8_t>> decrypt(
        std::span<const uint8_t> ciphertext);

    /// Get the session ID (for display/logging)
    rnet::uint256 session_id() const { return session_id_; }

private:
    bool established_ = false;
    bool is_initiator_ = false;

    rnet::uint256 send_key_;
    rnet::uint256 recv_key_;
    rnet::uint256 session_id_;

    uint64_t send_nonce_ = 0;
    uint64_t recv_nonce_ = 0;

    std::vector<uint8_t> local_privkey_;
    std::vector<uint8_t> local_pubkey_;

    std::array<uint8_t, 12> make_nonce(uint64_t counter) const;
};

}  // namespace rnet::net

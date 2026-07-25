// One peer connection: a byte stream on one side, whole messages on the other.
//
// TCP delivers bytes, not messages. A message may arrive split across three
// reads, or three messages may arrive in one — so the parser must be a state
// machine over a buffer, and every state must be safe to be interrupted in.
//
// The buffers are bounded, and that bound is a security property rather than a
// tuning choice:
//
//   * an unbounded RECEIVE buffer lets a peer stream bytes forever and exhaust
//     the node's memory,
//   * an unbounded SEND queue lets a peer stop reading and make the node buffer
//     on its behalf until it falls over.
//
// Nothing but the handshake is processed before the handshake completes. A peer
// that sends inventory before identifying itself is misbehaving, not merely
// early: accepting it would let an unauthenticated stranger drive the node's
// object-fetching logic.
#pragma once

#include <cstdint>
#include <deque>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "net/message.h"
#include "net/protocol.h"
#include "net/socket.h"
#include "util/result.h"

namespace rnet::net {

// Receive buffer ceiling: one maximum message plus a header, so a complete
// message always fits while an endless stream does not.
inline constexpr size_t kMaxReceiveBuffer = kMaxMessageSize + kHeaderSize;

// Send queue ceiling. A peer that stops reading gets disconnected rather than
// allowed to consume the node's memory.
inline constexpr size_t kMaxSendQueue = 8u * 1024 * 1024;

// A bulk feeder stops here rather than at the hard ceiling. Filling the queue to
// the brim would make the very next message — a ping, a verdict, anything — the
// one that trips the overflow and disconnects a peer that did nothing wrong.
inline constexpr size_t kSendQueueHighWater = 4u * 1024 * 1024;

// Misbehaviour past this score ends the connection. Individual increments are
// sized so that a single protocol slip is forgiven and a pattern is not.
inline constexpr int kBanThreshold = 100;

enum class PeerState {
    Connecting,      // TCP handshake outstanding
    AwaitingVersion, // connected; nothing but `version` is acceptable
    AwaitingVerack,  // version exchanged, waiting for acknowledgement
    Ready,           // fully established
    Disconnecting,   // scheduled for teardown
};

std::string PeerStateName(PeerState state);

struct ReceivedMessage {
    std::string command;
    std::vector<uint8_t> payload;
};

class Peer {
public:
    Peer(uint64_t id, Socket socket, NetAddress address, bool inbound,
         std::array<uint8_t, 4> magic);

    uint64_t id() const { return id_; }
    const NetAddress& address() const { return address_; }
    bool inbound() const { return inbound_; }
    PeerState state() const { return state_; }
    Socket& socket() { return socket_; }
    int misbehaviour() const { return misbehaviour_; }
    bool should_disconnect() const { return state_ == PeerState::Disconnecting; }
    const std::string& disconnect_reason() const { return disconnect_reason_; }

    // The peer's own claims, valid once the handshake completes.
    const VersionMessage& version() const { return version_; }

    // Feeds raw bytes from the socket and extracts whole messages. Returns an
    // error only for conditions that end the connection; a partial message is a
    // normal outcome and simply leaves bytes buffered.
    Result<std::vector<ReceivedMessage>> ReceiveBytes(std::span<const uint8_t> bytes);

    // Queues a message for sending. Fails if the queue is full, which is the
    // caller's signal that this peer is not keeping up.
    Status Send(std::string_view command, std::span<const uint8_t> payload);

    bool HasPendingSend() const { return !send_queue_.empty(); }
    size_t pending_send_bytes() const { return send_queue_.size(); }
    // Hands the next bytes to write; `consumed` reports how many the socket took.
    std::span<const uint8_t> PendingSend() const;
    void ConsumeSent(size_t consumed);

    // Whether this command may be processed right now. Called at dispatch time,
    // not at parse time — see ReceiveBytes for why the distinction matters.
    Status CheckCommandAllowed(std::string_view command);

    // Handshake driving. `local` is this node's version message.
    Status BeginHandshake(const VersionMessage& local);
    Status HandleVersion(const VersionMessage& remote, const VersionMessage& local,
                         uint64_t local_nonce);
    Status HandleVerack();

    // Records misbehaviour. Once the threshold is passed the peer is marked for
    // disconnection; the caller decides whether to also ban the address.
    void Misbehaving(int howmuch, std::string_view reason);
    void Disconnect(std::string_view reason);

    void MarkConnected();   // TCP connect finished

    int64_t last_receive() const { return last_receive_; }
    void RecordActivity(int64_t now) { last_receive_ = now; }

private:
    uint64_t id_;
    Socket socket_;
    NetAddress address_;
    bool inbound_;
    std::array<uint8_t, 4> magic_;

    PeerState state_;
    VersionMessage version_;
    bool sent_version_{false};
    int misbehaviour_{0};
    std::string disconnect_reason_;
    int64_t last_receive_{0};

    std::vector<uint8_t> receive_buffer_;
    std::deque<uint8_t> send_queue_;
    std::vector<uint8_t> send_window_;
};

// Which commands a peer may send in a given state. Enforcing this is what stops
// an unidentified stranger from driving the node's logic.
bool CommandAllowedInState(std::string_view command, PeerState state);

}  // namespace rnet::net

#pragma once

#include <atomic>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
using socket_t = SOCKET;
static constexpr socket_t INVALID_SOCK = INVALID_SOCKET;
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
using socket_t = int;
static constexpr socket_t INVALID_SOCK = -1;
#endif

#include "core/error.h"
#include "core/logging.h"
#include "core/sync.h"
#include "core/time.h"
#include "core/types.h"
#include "net/protocol.h"
#include "net/transport.h"

namespace rnet::net {

/// Connection state machine
enum class ConnState : uint8_t {
    CONNECTING,
    CONNECTED,
    DISCONNECTING,
    DISCONNECTED,
};

/// Statistics for a single connection
struct ConnectionStats {
    uint64_t id = 0;
    std::string addr_str;
    bool inbound = false;
    ConnState state = ConnState::DISCONNECTED;
    int32_t version = 0;
    uint64_t services = NODE_NONE;
    std::string user_agent;
    int32_t start_height = 0;
    int64_t connect_time = 0;
    int64_t last_send = 0;
    int64_t last_recv = 0;
    uint64_t bytes_sent = 0;
    uint64_t bytes_recv = 0;
    uint64_t messages_sent = 0;
    uint64_t messages_recv = 0;
    int64_t ping_time_ms = 0;
    bool handshake_complete = false;
};

/// CConnection — represents a single TCP peer connection.
///
/// Each connection wraps a platform socket and provides:
///   - Thread-safe send queue and receive buffer
///   - Wire-level message framing via Transport
///   - Version/verack handshake state tracking
///   - Ping/pong latency measurement
///   - Rate tracking (bytes sent/received)
///
/// CConnection is managed by ConnManager and should not be
/// created directly by application code.
class CConnection {
public:
    /// Construct a connection with an already-connected socket
    CConnection(uint64_t id, const CNetAddr& addr, socket_t sock, bool inbound);

    /// Destructor — closes socket if still open
    ~CConnection();

    // Non-copyable, non-movable (socket ownership)
    CConnection(const CConnection&) = delete;
    CConnection& operator=(const CConnection&) = delete;
    CConnection(CConnection&&) = delete;
    CConnection& operator=(CConnection&&) = delete;

    // ── Identity ────────────────────────────────────────────────────

    /// Unique connection ID assigned by ConnManager
    uint64_t id() const { return id_; }

    /// Remote address
    const CNetAddr& addr() const { return addr_; }

    /// Whether this is an inbound (accepted) connection
    bool is_inbound() const { return inbound_; }

    // ── State ───────────────────────────────────────────────────────

    /// Current connection state
    ConnState state() const { return state_.load(); }

    /// Whether the connection is active (CONNECTING or CONNECTED)
    bool is_connected() const;

    /// Request disconnection (sets state to DISCONNECTING)
    void disconnect();

    // ── Sending ─────────────────────────────────────────────────────

    /// Queue a message for sending. Thread-safe.
    /// The message is serialized with a wire header and added to the send queue.
    void send_message(std::string_view command,
                      std::span<const uint8_t> payload);

    /// Convenience: send a message with empty payload
    void send_message(std::string_view command);

    /// Flush the send queue to the socket.
    /// Returns the number of bytes actually sent, or -1 on error.
    /// Called by ConnManager from the network thread.
    int64_t flush_send();

    /// Check if there is data waiting to be sent
    bool has_send_data() const;

    // ── Receiving ───────────────────────────────────────────────────

    /// Read available data from the socket into the receive buffer.
    /// Returns bytes read, 0 on graceful close, -1 on error.
    /// Called by ConnManager from the network thread.
    int64_t recv_data();

    /// Try to extract the next complete message from the receive buffer.
    /// Returns nullopt if no complete message is available.
    std::optional<NetMessage> recv_message();

    // ── Version handshake ───────────────────────────────────────────

    /// Protocol version advertised by this peer
    int32_t version() const { return version_; }
    void set_version(int32_t v) { version_ = v; }

    /// Service flags advertised by this peer
    uint64_t services() const { return services_; }
    void set_services(uint64_t s) { services_ = s; }

    /// User agent string
    const std::string& user_agent() const { return user_agent_; }
    void set_user_agent(const std::string& ua) { user_agent_ = ua; }

    /// Best block height at connection time
    int32_t start_height() const { return start_height_; }
    void set_start_height(int32_t h) { start_height_ = h; }

    /// Relay flag (whether peer wants tx relay)
    bool relay() const { return relay_; }
    void set_relay(bool r) { relay_ = r; }

    /// Version message tracking
    bool version_sent() const { return version_sent_; }
    void set_version_sent(bool s) { version_sent_ = s; }
    bool version_received() const { return version_received_; }
    void set_version_received(bool r) { version_received_ = r; }

    /// Mark handshake as complete
    void complete_handshake();
    bool handshake_complete() const { return handshake_complete_; }

    /// Prefer headers announcements (sendheaders)
    bool prefer_headers() const { return prefer_headers_; }
    void set_prefer_headers(bool p) { prefer_headers_ = p; }

    /// Best known height/hash for this peer
    int32_t best_known_height() const { return best_known_height_; }
    void set_best_known_height(int32_t h) { best_known_height_ = h; }
    const rnet::uint256& best_known_hash() const { return best_known_hash_; }
    void set_best_known_hash(const rnet::uint256& h) { best_known_hash_ = h; }

    // ── Ping tracking ───────────────────────────────────────────────

    /// Set the nonce for an outgoing ping
    void set_ping_nonce(uint64_t nonce);
    uint64_t ping_nonce() const { return ping_nonce_; }
    bool ping_outstanding() const { return ping_outstanding_; }

    /// Record a pong response; returns latency in ms or -1 if no ping was outstanding
    int64_t record_pong(uint64_t nonce);

    /// Last measured ping time in milliseconds
    int64_t ping_time_ms() const { return ping_time_ms_; }

    // ── Stats & rate tracking ───────────────────────────────────────

    int64_t connect_time() const { return connect_time_; }
    int64_t last_send_time() const { return last_send_; }
    int64_t last_recv_time() const { return last_recv_; }
    uint64_t bytes_sent() const { return bytes_sent_.load(); }
    uint64_t bytes_recv() const { return bytes_recv_.load(); }
    uint64_t messages_sent() const { return messages_sent_.load(); }
    uint64_t messages_recv() const { return messages_recv_.load(); }

    /// Get a snapshot of connection stats
    ConnectionStats get_stats() const;

    // ── Misbehavior ─────────────────────────────────────────────────

    /// Add misbehavior points. Returns true if the peer should be banned.
    bool misbehaving(int points, const std::string& reason);

    /// Current ban score
    int ban_score() const { return ban_score_.load(); }

    /// Ban score threshold
    static constexpr int BAN_THRESHOLD = 100;

    // ── Socket access (for ConnManager polling) ─────────────────────

    /// Get the underlying socket (for select/poll)
    socket_t socket_fd() const { return socket_; }

    /// Check if a service flag is set
    bool has_service(ServiceFlags flag) const;

private:
    const uint64_t id_;
    const CNetAddr addr_;
    const bool inbound_;
    socket_t socket_;

    std::atomic<ConnState> state_{ConnState::CONNECTING};

    // Version handshake state
    int32_t version_ = 0;
    uint64_t services_ = NODE_NONE;
    std::string user_agent_;
    int32_t start_height_ = 0;
    bool relay_ = true;
    bool version_sent_ = false;
    bool version_received_ = false;
    bool handshake_complete_ = false;
    bool prefer_headers_ = false;
    int32_t best_known_height_ = 0;
    rnet::uint256 best_known_hash_;

    // Ping state
    uint64_t ping_nonce_ = 0;
    bool ping_outstanding_ = false;
    int64_t ping_send_time_ = 0;
    int64_t ping_time_ms_ = 0;

    // Timing
    int64_t connect_time_ = 0;
    int64_t last_send_ = 0;
    int64_t last_recv_ = 0;

    // Rate tracking
    std::atomic<uint64_t> bytes_sent_{0};
    std::atomic<uint64_t> bytes_recv_{0};
    std::atomic<uint64_t> messages_sent_{0};
    std::atomic<uint64_t> messages_recv_{0};

    // Misbehavior
    std::atomic<int> ban_score_{0};

    // Send queue (serialized wire bytes ready to send)
    mutable core::Mutex cs_send_;
    std::vector<uint8_t> send_buf_;

    // Receive transport (frame parser)
    Transport transport_;

    /// Close the socket
    void close_socket();
};

}  // namespace rnet::net

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/error.h"
#include "core/signal.h"
#include "core/sync.h"
#include "core/thread.h"
#include "net/connection.h"
#include "net/protocol.h"

namespace rnet::net {

/// ConnManager — manages all P2P connections.
///
/// Responsibilities:
///   - Listen for inbound connections on the P2P port
///   - Initiate outbound connections to known peers
///   - Route received messages to registered handlers
///   - Handle version/verack handshake for new peers
///   - Manage ban list
///   - Poll sockets and dispatch I/O on background threads
class ConnManager {
public:
    /// Default connection limits
    static constexpr int MAX_OUTBOUND = 8;
    static constexpr int MAX_INBOUND = 117;  // 125 total - 8 outbound
    static constexpr int MAX_CONNECTIONS = MAX_OUTBOUND + MAX_INBOUND;

    /// Ping interval (seconds)
    static constexpr int64_t PING_INTERVAL = 120;

    /// Inactivity timeout (seconds) — disconnect if no recv in this time
    static constexpr int64_t INACTIVITY_TIMEOUT = 600;

    /// Message handler callback type
    using MessageHandler = std::function<void(
        CConnection& conn, const std::string& command,
        core::DataStream& payload)>;

    ConnManager();
    ~ConnManager();

    // Non-copyable
    ConnManager(const ConnManager&) = delete;
    ConnManager& operator=(const ConnManager&) = delete;

    // ── Lifecycle ───────────────────────────────────────────────────

    /// Start the connection manager: bind listen socket, start threads
    Result<void> start(uint16_t port = DEFAULT_PORT);

    /// Stop the connection manager: close all connections, join threads
    void stop();

    /// Check if running
    bool is_running() const { return running_.load(); }

    // ── Connection management ───────────────────────────────────────

    /// Connect to a peer at the given address (outbound)
    Result<uint64_t> connect_to(const CNetAddr& addr);

    /// Disconnect a peer by connection ID
    void disconnect(uint64_t conn_id);

    /// Get a connection by ID (returns nullptr if not found)
    std::shared_ptr<CConnection> get_connection(uint64_t conn_id) const;

    /// Get all active connections
    std::vector<std::shared_ptr<CConnection>> get_connections() const;

    /// Total number of connections
    size_t connection_count() const;

    /// Number of outbound connections
    size_t outbound_count() const;

    /// Number of inbound connections
    size_t inbound_count() const;

    /// Check if we are connected to a given address
    bool is_connected(const CNetAddr& addr) const;

    /// Iterate over all connections with a callback
    void for_each_connection(
        const std::function<void(CConnection&)>& callback);

    // ── Messaging ───────────────────────────────────────────────────

    /// Send a message to a specific connection
    void send_to(uint64_t conn_id, std::string_view command,
                 std::span<const uint8_t> payload);

    /// Send a message to a specific connection (empty payload)
    void send_to(uint64_t conn_id, std::string_view command);

    /// Broadcast a message to all handshake-complete connections
    void broadcast(std::string_view command,
                   std::span<const uint8_t> payload);

    /// Broadcast to all except one connection
    void broadcast_except(uint64_t exclude_id, std::string_view command,
                          std::span<const uint8_t> payload);

    /// Register a handler for a specific command
    void register_handler(const std::string& command,
                          MessageHandler handler);

    // ── Node properties ─────────────────────────────────────────────

    /// Set/get the local services we advertise
    void set_local_services(uint64_t services) { local_services_ = services; }
    uint64_t local_services() const { return local_services_; }

    /// Set/get the user agent string
    void set_user_agent(const std::string& ua) { user_agent_ = ua; }
    const std::string& user_agent() const { return user_agent_; }

    /// Set current best block height (for version messages)
    void set_best_height(int32_t h) { best_height_ = h; }
    int32_t best_height() const { return best_height_.load(); }

    // ── Ban management ──────────────────────────────────────────────

    /// Ban a peer address for a duration (default 24 hours)
    void ban(const CNetAddr& addr, int64_t duration_seconds = 86400);

    /// Check if an address is banned
    bool is_banned(const CNetAddr& addr) const;

    /// Unban an address
    void unban(const CNetAddr& addr);

    /// Clear all bans
    void clear_bans();

    // ── Signals ─────────────────────────────────────────────────────

    /// Emitted when a new connection is fully established (after handshake)
    core::Signal<CConnection&> on_connected;

    /// Emitted when a connection is about to be removed
    core::Signal<uint64_t> on_disconnected;

    /// Emitted when any message is received (after dispatch)
    core::Signal<CConnection&, const std::string&> on_message_received;

private:
    mutable core::Mutex cs_conns_;

    /// Next connection ID counter
    std::atomic<uint64_t> next_conn_id_{1};

    /// Active connections: id -> connection
    std::unordered_map<uint64_t, std::shared_ptr<CConnection>> connections_;

    /// Message handlers: command -> handler
    std::unordered_map<std::string, MessageHandler> handlers_;

    /// Running state
    std::atomic<bool> running_{false};

    /// Listen socket and port
    socket_t listen_socket_ = INVALID_SOCK;
    uint16_t listen_port_ = DEFAULT_PORT;

    /// Local node properties
    uint64_t local_services_ = NODE_NETWORK;
    std::string user_agent_ = "/ResonanceNet:2.0.0/";
    std::atomic<int32_t> best_height_{0};

    /// Thread group for network threads
    core::ThreadGroup threads_;

    /// Ban list: address string -> ban expiry time (unix timestamp)
    mutable core::Mutex cs_ban_;
    std::unordered_map<std::string, int64_t> banned_;

    // ── Thread functions ────────────────────────────────────────────

    /// Accept loop: listen for inbound connections
    void accept_loop();

    /// Socket I/O loop: poll sockets, send/recv, dispatch messages
    void socket_loop();

    /// Maintenance loop: ping, timeout, cleanup
    void maintenance_loop();

    // ── Internal helpers ────────────────────────────────────────────

    /// Create the listening socket
    Result<void> create_listen_socket();

    /// Set a socket to non-blocking mode
    static bool set_nonblocking(socket_t sock);

    /// Process a received message from a connection
    void process_message(CConnection& conn, NetMessage& msg);

    /// Handle version handshake
    void handle_version(CConnection& conn, core::DataStream& payload);
    void handle_verack(CConnection& conn);

    /// Remove a connection (must hold cs_conns_)
    void remove_connection_locked(uint64_t conn_id);

    /// Send a version message to a new outbound connection
    void send_version(CConnection& conn);
};

}  // namespace rnet::net

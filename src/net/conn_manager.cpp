// Copyright (c) 2024-2026 The ResonanceNet Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "net/conn_manager.h"

#include "core/logging.h"
#include "core/random.h"
#include "core/time.h"
#include "crypto/keccak.h"
#include "net/addr_man.h"

#include <algorithm>
#include <cstring>

namespace rnet::net {

// ===========================================================================
//  Design note — Connection lifecycle
// ---------------------------------------------------------------------------
//  ConnManager owns three background threads that together form the P2P I/O
//  engine:
//
//    rnet-accept   — blocks on accept(), creates inbound CConnection objects
//    rnet-socket   — polls every 10 ms, calls recv/send, dispatches messages
//    rnet-maint    — runs once per second: pings, timeouts, auto-connect
//
//  Every new peer goes through a VERSION/VERACK handshake before any
//  application messages are accepted.  Misbehaviour scores accumulate
//  and can result in automatic banning.
// ===========================================================================

// ===========================================================================
//  Design note — Peer management
// ---------------------------------------------------------------------------
//  Peers are stored in connections_ (guarded by cs_conns_).  The map is
//  keyed by a monotonically increasing uint64 conn_id.  Outbound slots are
//  capped at MAX_OUTBOUND (8), inbound at MAX_INBOUND (117), giving a total
//  budget of 125 connections — matching Bitcoin Core's default.
//
//  The ban list is a separate map (guarded by cs_ban_) mapping address
//  strings to unix-timestamp expiry times.  Expired entries are lazily
//  pruned on lookup.
// ===========================================================================

// ---------------------------------------------------------------------------
// WSA initialisation helper (Windows only)
// ---------------------------------------------------------------------------

#ifdef _WIN32
namespace {
struct WinsockInit {
    WinsockInit() {
        WSADATA wsa_data;
        int result = WSAStartup(MAKEWORD(2, 2), &wsa_data);
        if (result != 0) {
            LogPrintf("WSAStartup failed: %d", result);
        }
    }
    ~WinsockInit() {
        WSACleanup();
    }
};
static WinsockInit g_winsock_init;
} // anonymous namespace
#endif

// ---------------------------------------------------------------------------
// ConnManager (constructor)
// ---------------------------------------------------------------------------

ConnManager::ConnManager() = default;

// ---------------------------------------------------------------------------
// ~ConnManager (destructor)
// ---------------------------------------------------------------------------

ConnManager::~ConnManager() {
    stop();
}

// ---------------------------------------------------------------------------
// start
// ---------------------------------------------------------------------------

Result<void> ConnManager::start(uint16_t port) {
    // 1. Reject duplicate start.
    if (running_.load()) {
        return Result<void>::err("ConnManager already running");
    }

    // 2. Bind the listen socket.
    listen_port_ = port;

    auto result = create_listen_socket();
    if (result.is_err()) {
        return result;
    }

    // 3. Flip the running flag and launch network threads.
    running_.store(true);

    threads_.create_thread("rnet-accept", [this]() { accept_loop(); });
    threads_.create_thread("rnet-socket", [this]() { socket_loop(); });
    threads_.create_thread("rnet-maint",  [this]() { maintenance_loop(); });

    LogPrintf("P2P network started on port %d", static_cast<int>(port));
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// stop
// ---------------------------------------------------------------------------

void ConnManager::stop() {
    if (!running_.load()) return;

    // 1. Signal all loops to exit.
    running_.store(false);

    // 2. Close listen socket to unblock accept().
    if (listen_socket_ != INVALID_SOCK) {
#ifdef _WIN32
        ::closesocket(listen_socket_);
#else
        ::close(listen_socket_);
#endif
        listen_socket_ = INVALID_SOCK;
    }

    // 3. Disconnect every active connection.
    {
        LOCK(cs_conns_);
        for (auto& [id, conn] : connections_) {
            conn->disconnect();
        }
        connections_.clear();
    }

    // 4. Wait for all threads to finish.
    threads_.join_all();
    LogPrintf("P2P network stopped");
}

// ---------------------------------------------------------------------------
// connect_to
// ---------------------------------------------------------------------------

Result<uint64_t> ConnManager::connect_to(const CNetAddr& addr) {
    // 1. Pre-flight checks.
    if (is_banned(addr)) {
        return Result<uint64_t>::err("Address is banned");
    }

    if (is_connected(addr)) {
        return Result<uint64_t>::err("Already connected to " + addr.to_string());
    }

    if (outbound_count() >= static_cast<size_t>(MAX_OUTBOUND)) {
        return Result<uint64_t>::err("Max outbound connections reached");
    }

    // 2. Create a TCP socket matching the address family.
    int af = addr.is_ipv4() ? AF_INET : AF_INET6;
    socket_t sock = ::socket(af, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCK) {
        return Result<uint64_t>::err("Failed to create socket");
    }

    // 3. Build sockaddr from CNetAddr.
    struct sockaddr_storage ss;
    std::memset(&ss, 0, sizeof(ss));
    socklen_t ss_len = 0;

    if (addr.is_ipv4()) {
        auto* sin = reinterpret_cast<struct sockaddr_in*>(&ss);
        sin->sin_family = AF_INET;
        sin->sin_port = htons(addr.port);
        std::memcpy(&sin->sin_addr, &addr.ip[12], 4);
        ss_len = sizeof(struct sockaddr_in);
    } else {
        auto* sin6 = reinterpret_cast<struct sockaddr_in6*>(&ss);
        sin6->sin6_family = AF_INET6;
        sin6->sin6_port = htons(addr.port);
        std::memcpy(&sin6->sin6_addr, addr.ip.data(), 16);
        ss_len = sizeof(struct sockaddr_in6);
    }

    // 4. Blocking connect (real code would use non-blocking + poll).
    int ret = ::connect(sock, reinterpret_cast<struct sockaddr*>(&ss), ss_len);
    if (ret != 0) {
#ifdef _WIN32
        ::closesocket(sock);
#else
        ::close(sock);
#endif
        return Result<uint64_t>::err(
            "Failed to connect to " + addr.to_string());
    }

    // 5. Switch to non-blocking for ongoing I/O.
    set_nonblocking(sock);

    // 6. Register the new connection.
    uint64_t conn_id = next_conn_id_.fetch_add(1);
    auto conn = std::make_shared<CConnection>(conn_id, addr, sock, false);

    {
        LOCK(cs_conns_);
        connections_[conn_id] = conn;
    }

    // 7. Initiate the VERSION handshake.
    send_version(*conn);

    LogPrint(NET, "Connected to peer %llu at %s",
             static_cast<unsigned long long>(conn_id),
             addr.to_string().c_str());

    return Result<uint64_t>::ok(conn_id);
}

// ---------------------------------------------------------------------------
// disconnect
// ---------------------------------------------------------------------------

void ConnManager::disconnect(uint64_t conn_id) {
    LOCK(cs_conns_);
    auto it = connections_.find(conn_id);
    if (it != connections_.end()) {
        it->second->disconnect();
        on_disconnected.emit(conn_id);
        connections_.erase(it);

        LogPrint(NET, "Disconnected connection %llu",
                 static_cast<unsigned long long>(conn_id));
    }
}

// ---------------------------------------------------------------------------
// get_connection
// ---------------------------------------------------------------------------

std::shared_ptr<CConnection> ConnManager::get_connection(
    uint64_t conn_id) const {
    LOCK(cs_conns_);
    auto it = connections_.find(conn_id);
    if (it != connections_.end()) return it->second;
    return nullptr;
}

// ---------------------------------------------------------------------------
// get_connections
// ---------------------------------------------------------------------------

std::vector<std::shared_ptr<CConnection>> ConnManager::get_connections() const {
    LOCK(cs_conns_);
    std::vector<std::shared_ptr<CConnection>> result;
    result.reserve(connections_.size());
    for (const auto& [id, conn] : connections_) {
        result.push_back(conn);
    }
    return result;
}

// ---------------------------------------------------------------------------
// connection_count
// ---------------------------------------------------------------------------

size_t ConnManager::connection_count() const {
    LOCK(cs_conns_);
    return connections_.size();
}

// ---------------------------------------------------------------------------
// outbound_count
// ---------------------------------------------------------------------------

size_t ConnManager::outbound_count() const {
    LOCK(cs_conns_);
    size_t count = 0;
    for (const auto& [id, conn] : connections_) {
        if (!conn->is_inbound()) ++count;
    }
    return count;
}

// ---------------------------------------------------------------------------
// inbound_count
// ---------------------------------------------------------------------------

size_t ConnManager::inbound_count() const {
    LOCK(cs_conns_);
    size_t count = 0;
    for (const auto& [id, conn] : connections_) {
        if (conn->is_inbound()) ++count;
    }
    return count;
}

// ---------------------------------------------------------------------------
// is_connected
// ---------------------------------------------------------------------------

bool ConnManager::is_connected(const CNetAddr& addr) const {
    LOCK(cs_conns_);
    for (const auto& [id, conn] : connections_) {
        if (conn->addr() == addr) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// for_each_connection
// ---------------------------------------------------------------------------

void ConnManager::for_each_connection(
    const std::function<void(CConnection&)>& callback) {
    LOCK(cs_conns_);
    for (auto& [id, conn] : connections_) {
        callback(*conn);
    }
}

// ---------------------------------------------------------------------------
// send_to  (with payload)
// ---------------------------------------------------------------------------

void ConnManager::send_to(uint64_t conn_id, std::string_view command,
                          std::span<const uint8_t> payload) {
    LOCK(cs_conns_);
    auto it = connections_.find(conn_id);
    if (it != connections_.end()) {
        it->second->send_message(command, payload);
    }
}

// ---------------------------------------------------------------------------
// send_to  (empty payload)
// ---------------------------------------------------------------------------

void ConnManager::send_to(uint64_t conn_id, std::string_view command) {
    send_to(conn_id, command, std::span<const uint8_t>{});
}

// ---------------------------------------------------------------------------
// broadcast
// ---------------------------------------------------------------------------

void ConnManager::broadcast(std::string_view command,
                            std::span<const uint8_t> payload) {
    LOCK(cs_conns_);
    for (auto& [id, conn] : connections_) {
        if (conn->handshake_complete() && conn->is_connected()) {
            conn->send_message(command, payload);
        }
    }
}

// ---------------------------------------------------------------------------
// broadcast_except
// ---------------------------------------------------------------------------

void ConnManager::broadcast_except(uint64_t exclude_id,
                                   std::string_view command,
                                   std::span<const uint8_t> payload) {
    LOCK(cs_conns_);
    for (auto& [id, conn] : connections_) {
        if (id != exclude_id && conn->handshake_complete() &&
            conn->is_connected()) {
            conn->send_message(command, payload);
        }
    }
}

// ---------------------------------------------------------------------------
// register_handler
// ---------------------------------------------------------------------------

void ConnManager::register_handler(const std::string& command,
                                   MessageHandler handler) {
    handlers_[command] = std::move(handler);
}

// ---------------------------------------------------------------------------
// ban
// ---------------------------------------------------------------------------

void ConnManager::ban(const CNetAddr& addr, int64_t duration_seconds) {
    LOCK(cs_ban_);
    auto key = addr.to_string();
    banned_[key] = core::get_time() + duration_seconds;

    LogPrint(NET, "Banned %s for %lld seconds",
             key.c_str(),
             static_cast<long long>(duration_seconds));
}

// ---------------------------------------------------------------------------
// is_banned
// ---------------------------------------------------------------------------

bool ConnManager::is_banned(const CNetAddr& addr) const {
    LOCK(cs_ban_);
    auto key = addr.to_string();
    auto it = banned_.find(key);
    if (it == banned_.end()) return false;

    if (it->second < core::get_time()) {
        // Ban expired — lazily prune the entry.
        const_cast<ConnManager*>(this)->banned_.erase(key);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// unban
// ---------------------------------------------------------------------------

void ConnManager::unban(const CNetAddr& addr) {
    LOCK(cs_ban_);
    banned_.erase(addr.to_string());
}

// ---------------------------------------------------------------------------
// clear_bans
// ---------------------------------------------------------------------------

void ConnManager::clear_bans() {
    LOCK(cs_ban_);
    banned_.clear();
}

// ---------------------------------------------------------------------------
// accept_loop
// ---------------------------------------------------------------------------

void ConnManager::accept_loop() {
    core::set_thread_name("rnet-accept");

    while (running_.load()) {
        // 1. Wait until the listen socket is ready.
        if (listen_socket_ == INVALID_SOCK) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 100 ms poll
            continue;
        }

        // 2. Block on accept().
        struct sockaddr_storage client_addr;
        socklen_t addr_len = sizeof(client_addr);

        socket_t client_sock = ::accept(listen_socket_,
            reinterpret_cast<struct sockaddr*>(&client_addr),
            &addr_len);

        if (client_sock == INVALID_SOCK) {
            if (!running_.load()) break; // shutting down
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); // 10 ms backoff
            continue;
        }

        // 3. Enforce inbound connection limit.
        if (inbound_count() >= static_cast<size_t>(MAX_INBOUND)) {
#ifdef _WIN32
            ::closesocket(client_sock);
#else
            ::close(client_sock);
#endif
            continue;
        }

        // 4. Extract the peer address.
        CNetAddr net_addr;
        if (client_addr.ss_family == AF_INET) {
            auto* sin = reinterpret_cast<struct sockaddr_in*>(&client_addr);
            auto* ip_bytes = reinterpret_cast<uint8_t*>(&sin->sin_addr);
            net_addr.set_ipv4(ip_bytes[0], ip_bytes[1],
                              ip_bytes[2], ip_bytes[3]);
            net_addr.port = ntohs(sin->sin_port);
        } else if (client_addr.ss_family == AF_INET6) {
            auto* sin6 = reinterpret_cast<struct sockaddr_in6*>(&client_addr);
            std::memcpy(net_addr.ip.data(), &sin6->sin6_addr, 16);
            net_addr.port = ntohs(sin6->sin6_port);
        }

        // 5. Reject banned addresses immediately.
        if (is_banned(net_addr)) {
#ifdef _WIN32
            ::closesocket(client_sock);
#else
            ::close(client_sock);
#endif
            continue;
        }

        // 6. Register the inbound connection.
        set_nonblocking(client_sock);

        uint64_t conn_id = next_conn_id_.fetch_add(1);
        auto conn = std::make_shared<CConnection>(
            conn_id, net_addr, client_sock, true);

        {
            LOCK(cs_conns_);
            connections_[conn_id] = conn;
        }

        LogPrint(NET, "Accepted inbound connection %llu from %s",
                 static_cast<unsigned long long>(conn_id),
                 net_addr.to_string().c_str());
    }
}

// ---------------------------------------------------------------------------
// socket_loop
// ---------------------------------------------------------------------------

void ConnManager::socket_loop() {
    core::set_thread_name("rnet-socket");

    while (running_.load()) {
        // 1. Snapshot the connection list under the lock.
        std::vector<std::shared_ptr<CConnection>> conns;
        {
            LOCK(cs_conns_);
            conns.reserve(connections_.size());
            for (auto& [id, conn] : connections_) {
                conns.push_back(conn);
            }
        }

        if (conns.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50)); // 50 ms idle
            continue;
        }

        // 2. Process each connection: recv, dispatch, send.
        std::vector<uint64_t> to_remove;

        for (auto& conn : conns) {
            if (!conn->is_connected()) {
                to_remove.push_back(conn->id());
                continue;
            }

            // 2a. Receive data from the socket.
            int64_t recv_result = conn->recv_data();
            if (recv_result < 0) {
                to_remove.push_back(conn->id());
                continue;
            }
            if (recv_result == 0 && !conn->is_connected()) {
                to_remove.push_back(conn->id());
                continue;
            }

            // 2b. Dispatch all complete messages.
            while (auto msg = conn->recv_message()) {
                process_message(*conn, *msg);
            }

            // 2c. Flush the outbound send queue.
            int64_t send_result = conn->flush_send();
            if (send_result < 0) {
                to_remove.push_back(conn->id());
                continue;
            }
        }

        // 3. Remove dead connections.
        if (!to_remove.empty()) {
            LOCK(cs_conns_);
            for (auto id : to_remove) {
                remove_connection_locked(id);
            }
        }

        // 4. Brief sleep to avoid busy-spinning (10 ms).
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

// ---------------------------------------------------------------------------
// maintenance_loop
// ---------------------------------------------------------------------------

void ConnManager::maintenance_loop() {
    core::set_thread_name("rnet-maint");

    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1)); // 1 s tick
        if (!running_.load()) break;

        int64_t now = core::get_time();
        std::vector<uint64_t> to_disconnect;

        // 1. Scan for timed-out or dead connections and send pings.
        {
            LOCK(cs_conns_);
            for (auto& [id, conn] : connections_) {
                if (!conn->is_connected()) {
                    to_disconnect.push_back(id);
                    continue;
                }

                // 1a. Inactivity timeout — INACTIVITY_TIMEOUT seconds (600 s).
                if (conn->handshake_complete() &&
                    (now - conn->last_recv_time()) > INACTIVITY_TIMEOUT) {
                    LogPrint(NET, "Peer %llu timed out (no recv for %llds)",
                             static_cast<unsigned long long>(id),
                             static_cast<long long>(
                                 now - conn->last_recv_time()));
                    to_disconnect.push_back(id);
                    continue;
                }

                // 1b. Send ping if idle for PING_INTERVAL seconds (120 s).
                if (conn->handshake_complete() &&
                    !conn->ping_outstanding() &&
                    (now - conn->last_send_time()) > PING_INTERVAL) {
                    uint64_t nonce = core::get_rand_u64();
                    conn->set_ping_nonce(nonce);

                    core::DataStream ss;
                    core::ser_write_u64(ss, nonce);
                    conn->send_message(msg::PING, ss.span());
                }
            }
        }

        // 2. Disconnect timed-out peers (outside the lock).
        for (auto id : to_disconnect) {
            disconnect(id);
        }

        // 3. Periodic addr self-advertisement (~every 30 min).
        if (addr_broadcast_fn_ &&
            (now - last_addr_broadcast_) > ADDR_BROADCAST_INTERVAL) {
            last_addr_broadcast_ = now;
            addr_broadcast_fn_();
        }

        // 4. Auto-connect: fill outbound slots from AddrManager.
        if (addrman_ && outbound_count() < static_cast<size_t>(MAX_OUTBOUND)) {
            CNetAddr addr = addrman_->select();
            if (addr.port != 0 && !is_connected(addr) && !is_banned(addr)) {
                addrman_->mark_attempt(addr);
                auto res = connect_to(addr);
                if (res.is_ok()) {
                    LogPrintf("Auto-connected to peer %s",
                              addr.to_string().c_str());
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// create_listen_socket
// ---------------------------------------------------------------------------

Result<void> ConnManager::create_listen_socket() {
    // 1. Try IPv6 dual-stack first, fall back to IPv4-only.
    listen_socket_ = ::socket(AF_INET6, SOCK_STREAM, IPPROTO_TCP);
    if (listen_socket_ == INVALID_SOCK) {
        // 1a. IPv6 unavailable — fall back to IPv4.
        listen_socket_ = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (listen_socket_ == INVALID_SOCK) {
            return Result<void>::err("Failed to create listen socket");
        }

        struct sockaddr_in sin;
        std::memset(&sin, 0, sizeof(sin));
        sin.sin_family = AF_INET;
        sin.sin_port = htons(listen_port_);
        sin.sin_addr.s_addr = INADDR_ANY;

        // Allow address reuse.
        int opt = 1;
#ifdef _WIN32
        ::setsockopt(listen_socket_, SOL_SOCKET, SO_REUSEADDR,
                     reinterpret_cast<const char*>(&opt), sizeof(opt));
#else
        ::setsockopt(listen_socket_, SOL_SOCKET, SO_REUSEADDR,
                     &opt, sizeof(opt));
#endif

        if (::bind(listen_socket_,
                   reinterpret_cast<struct sockaddr*>(&sin),
                   sizeof(sin)) != 0) {
            return Result<void>::err("Failed to bind to port " +
                                    std::to_string(listen_port_));
        }
    } else {
        // 1b. IPv6 dual-stack — accept both v4 and v6.
        int opt_off = 0;
#ifdef _WIN32
        ::setsockopt(listen_socket_, IPPROTO_IPV6, IPV6_V6ONLY,
                     reinterpret_cast<const char*>(&opt_off), sizeof(opt_off));
#else
        ::setsockopt(listen_socket_, IPPROTO_IPV6, IPV6_V6ONLY,
                     &opt_off, sizeof(opt_off));
#endif

        int opt = 1;
#ifdef _WIN32
        ::setsockopt(listen_socket_, SOL_SOCKET, SO_REUSEADDR,
                     reinterpret_cast<const char*>(&opt), sizeof(opt));
#else
        ::setsockopt(listen_socket_, SOL_SOCKET, SO_REUSEADDR,
                     &opt, sizeof(opt));
#endif

        struct sockaddr_in6 sin6;
        std::memset(&sin6, 0, sizeof(sin6));
        sin6.sin6_family = AF_INET6;
        sin6.sin6_port = htons(listen_port_);
        sin6.sin6_addr = in6addr_any;

        if (::bind(listen_socket_,
                   reinterpret_cast<struct sockaddr*>(&sin6),
                   sizeof(sin6)) != 0) {
            return Result<void>::err("Failed to bind to port " +
                                    std::to_string(listen_port_));
        }
    }

    // 2. Start listening.
    if (::listen(listen_socket_, SOMAXCONN) != 0) {
        return Result<void>::err("Failed to listen on port " +
                                std::to_string(listen_port_));
    }

    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// set_nonblocking
// ---------------------------------------------------------------------------

bool ConnManager::set_nonblocking(socket_t sock) {
#ifdef _WIN32
    u_long mode = 1;
    return ::ioctlsocket(sock, FIONBIO, &mode) == 0;
#else
    int flags = ::fcntl(sock, F_GETFL, 0);
    if (flags == -1) return false;
    return ::fcntl(sock, F_SETFL, flags | O_NONBLOCK) == 0;
#endif
}

// ---------------------------------------------------------------------------
// process_message
// ---------------------------------------------------------------------------

void ConnManager::process_message(CConnection& conn, NetMessage& msg) {
    // 1. Handle VERSION/VERACK handshake internally.
    if (msg.command == msg::VERSION) {
        core::DataStream payload(std::move(msg.payload));
        handle_version(conn, payload);
        on_message_received.emit(conn, msg.command);
        return;
    }

    if (msg.command == msg::VERACK) {
        handle_verack(conn);
        on_message_received.emit(conn, msg.command);
        return;
    }

    // 2. Reject all messages before handshake completion.
    if (!conn.handshake_complete()) {
        conn.misbehaving(10, "Message before handshake: " + msg.command);
        return;
    }

    // 3. Dispatch to the registered handler, if any.
    auto it = handlers_.find(msg.command);
    if (it != handlers_.end()) {
        core::DataStream payload(std::move(msg.payload));
        it->second(conn, msg.command, payload);
    }

    on_message_received.emit(conn, msg.command);
}

// ---------------------------------------------------------------------------
// handle_version
// ---------------------------------------------------------------------------

void ConnManager::handle_version(CConnection& conn,
                                 core::DataStream& payload) {
    VersionMessage ver;
    ver.unserialize(payload);

    // 1. Reject peers running obsolete protocol versions.
    if (ver.version < MIN_PROTOCOL_VERSION) {
        conn.misbehaving(100, "Obsolete protocol version " +
                              std::to_string(ver.version));
        return;
    }

    // 2. Record the peer's advertised properties.
    conn.set_version(ver.version);
    conn.set_services(ver.services);
    conn.set_user_agent(ver.user_agent);
    conn.set_start_height(ver.start_height);
    conn.set_relay(ver.relay);
    conn.set_version_received(true);

    // 3. Learn our external address from the peer's addr_recv field.
    //    The remote node tells us what IP+port it connected to, which
    //    is how we discover our externally-reachable address.
    if (ver.addr_recv.is_routable() && external_addr_fn_) {
        CNetAddr discovered = ver.addr_recv;
        discovered.port = listen_port_;
        discovered.services = local_services_;
        external_addr_fn_(discovered);
    }

    // 4. If this is an inbound peer, send our own VERSION first.
    if (conn.is_inbound() && !conn.version_sent()) {
        send_version(conn);
    }

    // 5. Acknowledge with VERACK.
    conn.send_message(msg::VERACK);

    LogPrint(NET, "Received version from peer %llu: %s v%d height=%d",
             static_cast<unsigned long long>(conn.id()),
             ver.user_agent.c_str(), ver.version, ver.start_height);
}

// ---------------------------------------------------------------------------
// handle_verack
// ---------------------------------------------------------------------------

void ConnManager::handle_verack(CConnection& conn) {
    if (conn.version_received()) {
        conn.complete_handshake();
        on_connected.emit(conn);
    }
}

// ---------------------------------------------------------------------------
// send_version
// ---------------------------------------------------------------------------

void ConnManager::send_version(CConnection& conn) {
    VersionMessage ver;
    ver.version      = PROTOCOL_VERSION;
    ver.services     = local_services_;
    ver.timestamp    = core::get_time();
    ver.nonce        = core::get_rand_u64();
    ver.user_agent   = user_agent_;
    ver.start_height = best_height_.load();
    ver.relay        = true;

    core::DataStream ss;
    ver.serialize(ss);
    conn.send_message(msg::VERSION, ss.span());
    conn.set_version_sent(true);
}

// ---------------------------------------------------------------------------
// remove_connection_locked
// ---------------------------------------------------------------------------

void ConnManager::remove_connection_locked(uint64_t conn_id) {
    auto it = connections_.find(conn_id);
    if (it != connections_.end()) {
        connections_.erase(it);
        on_disconnected.emit(conn_id);
    }
}

} // namespace rnet::net

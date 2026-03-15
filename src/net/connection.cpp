#include "net/connection.h"

#include <algorithm>
#include <cstring>

#include "core/logging.h"
#include "core/time.h"
#include "crypto/keccak.h"

namespace rnet::net {

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

CConnection::CConnection(uint64_t id, const CNetAddr& addr,
                         socket_t sock, bool inbound)
    : id_(id)
    , addr_(addr)
    , inbound_(inbound)
    , socket_(sock)
    , connect_time_(core::get_time())
{
    if (socket_ != INVALID_SOCK) {
        state_.store(ConnState::CONNECTED);
    }

    last_recv_ = connect_time_;
    last_send_ = connect_time_;
}

CConnection::~CConnection() {
    close_socket();
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

bool CConnection::is_connected() const {
    auto s = state_.load();
    return s == ConnState::CONNECTING || s == ConnState::CONNECTED;
}

void CConnection::disconnect() {
    auto expected = ConnState::CONNECTED;
    if (state_.compare_exchange_strong(expected, ConnState::DISCONNECTING)) {
        LogPrint(NET, "Disconnecting peer %llu (%s)",
                 static_cast<unsigned long long>(id_),
                 addr_.to_string().c_str());
    } else {
        // Also handle CONNECTING state
        expected = ConnState::CONNECTING;
        state_.compare_exchange_strong(expected, ConnState::DISCONNECTING);
    }
}

// ---------------------------------------------------------------------------
// Sending
// ---------------------------------------------------------------------------

void CConnection::send_message(std::string_view command,
                               std::span<const uint8_t> payload) {
    if (!is_connected()) return;

    auto wire_bytes = Transport::serialize_message(command, payload);

    {
        LOCK(cs_send_);
        send_buf_.insert(send_buf_.end(),
                         wire_bytes.begin(), wire_bytes.end());
    }

    messages_sent_.fetch_add(1);
    last_send_ = core::get_time();

    LogDebug(NET, "Queued '%s' (%zu bytes payload) to peer %llu",
             std::string(command).c_str(),
             payload.size(),
             static_cast<unsigned long long>(id_));
}

void CConnection::send_message(std::string_view command) {
    send_message(command, std::span<const uint8_t>{});
}

int64_t CConnection::flush_send() {
    if (socket_ == INVALID_SOCK) return -1;

    std::vector<uint8_t> to_send;
    {
        LOCK(cs_send_);
        if (send_buf_.empty()) return 0;
        to_send.swap(send_buf_);
    }

    size_t total_sent = 0;
    while (total_sent < to_send.size()) {
        auto remaining = to_send.size() - total_sent;

#ifdef _WIN32
        int chunk = static_cast<int>(
            (std::min)(remaining, static_cast<size_t>(INT_MAX)));
        int sent = ::send(socket_,
                          reinterpret_cast<const char*>(
                              to_send.data() + total_sent),
                          chunk, 0);
#else
        ssize_t sent = ::send(socket_,
                              to_send.data() + total_sent,
                              remaining, MSG_NOSIGNAL);
#endif

        if (sent <= 0) {
            // Put unsent data back into the buffer
            {
                LOCK(cs_send_);
                send_buf_.insert(send_buf_.begin(),
                                 to_send.begin() +
                                     static_cast<ptrdiff_t>(total_sent),
                                 to_send.end());
            }

#ifdef _WIN32
            int err = WSAGetLastError();
            if (err == WSAEWOULDBLOCK) {
                break;  // Try again later
            }
#else
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                break;
            }
#endif
            LogPrint(NET, "Send error on peer %llu",
                     static_cast<unsigned long long>(id_));
            return -1;
        }

        total_sent += static_cast<size_t>(sent);
    }

    bytes_sent_.fetch_add(total_sent);
    return static_cast<int64_t>(total_sent);
}

bool CConnection::has_send_data() const {
    LOCK(cs_send_);
    return !send_buf_.empty();
}

// ---------------------------------------------------------------------------
// Receiving
// ---------------------------------------------------------------------------

int64_t CConnection::recv_data() {
    if (socket_ == INVALID_SOCK) return -1;

    uint8_t buf[8192];

#ifdef _WIN32
    int n = ::recv(socket_, reinterpret_cast<char*>(buf),
                   static_cast<int>(sizeof(buf)), 0);
#else
    ssize_t n = ::recv(socket_, buf, sizeof(buf), 0);
#endif

    if (n > 0) {
        bytes_recv_.fetch_add(static_cast<uint64_t>(n));
        last_recv_ = core::get_time();
        transport_.feed(std::span<const uint8_t>(buf, static_cast<size_t>(n)));
        return static_cast<int64_t>(n);
    }

    if (n == 0) {
        // Graceful close — log once and mark disconnected.
        if (!disconnected_) {
            disconnected_ = true;
            LogPrint(NET, "Peer %llu closed connection",
                     static_cast<unsigned long long>(id_));
        }
        return 0;
    }

    // Error
#ifdef _WIN32
    int err = WSAGetLastError();
    if (err == WSAEWOULDBLOCK) {
        return 0;  // No data available, not an error
    }
#else
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
        return 0;
    }
#endif

    LogPrint(NET, "Recv error on peer %llu",
             static_cast<unsigned long long>(id_));
    return -1;
}

std::optional<NetMessage> CConnection::recv_message() {
    auto msg = transport_.next_message();
    if (msg.has_value()) {
        messages_recv_.fetch_add(1);
    }
    return msg;
}

// ---------------------------------------------------------------------------
// Version handshake
// ---------------------------------------------------------------------------

void CConnection::complete_handshake() {
    handshake_complete_ = true;
    LogPrint(NET, "Handshake complete with peer %llu (%s v%d, height=%d)",
             static_cast<unsigned long long>(id_),
             user_agent_.c_str(), version_, start_height_);
}

// ---------------------------------------------------------------------------
// Ping tracking
// ---------------------------------------------------------------------------

void CConnection::set_ping_nonce(uint64_t nonce) {
    ping_nonce_ = nonce;
    ping_outstanding_ = true;
    ping_send_time_ = core::get_time_millis();
}

int64_t CConnection::record_pong(uint64_t nonce) {
    if (!ping_outstanding_ || nonce != ping_nonce_) {
        return -1;
    }

    ping_outstanding_ = false;
    ping_time_ms_ = core::get_time_millis() - ping_send_time_;
    return ping_time_ms_;
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

ConnectionStats CConnection::get_stats() const {
    ConnectionStats stats;
    stats.id = id_;
    stats.addr_str = addr_.to_string();
    stats.inbound = inbound_;
    stats.state = state_.load();
    stats.version = version_;
    stats.services = services_;
    stats.user_agent = user_agent_;
    stats.start_height = start_height_;
    stats.connect_time = connect_time_;
    stats.last_send = last_send_;
    stats.last_recv = last_recv_;
    stats.bytes_sent = bytes_sent_.load();
    stats.bytes_recv = bytes_recv_.load();
    stats.messages_sent = messages_sent_.load();
    stats.messages_recv = messages_recv_.load();
    stats.ping_time_ms = ping_time_ms_;
    stats.handshake_complete = handshake_complete_;
    return stats;
}

// ---------------------------------------------------------------------------
// Misbehavior
// ---------------------------------------------------------------------------

bool CConnection::misbehaving(int points, const std::string& reason) {
    int new_score = ban_score_.fetch_add(points) + points;
    if (new_score >= BAN_THRESHOLD) {
        LogPrint(NET, "Peer %llu misbehaving (score=%d): %s — banning",
                 static_cast<unsigned long long>(id_), new_score,
                 reason.c_str());
        disconnect();
        return true;
    }
    LogPrint(NET, "Peer %llu misbehaving (score=%d): %s",
             static_cast<unsigned long long>(id_), new_score,
             reason.c_str());
    return false;
}

bool CConnection::has_service(ServiceFlags flag) const {
    return (services_ & static_cast<uint64_t>(flag)) != 0;
}

// ---------------------------------------------------------------------------
// Socket management
// ---------------------------------------------------------------------------

void CConnection::close_socket() {
    if (socket_ != INVALID_SOCK) {
#ifdef _WIN32
        ::closesocket(socket_);
#else
        ::close(socket_);
#endif
        socket_ = INVALID_SOCK;
    }
    state_.store(ConnState::DISCONNECTED);
}

}  // namespace rnet::net

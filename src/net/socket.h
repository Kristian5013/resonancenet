// RAII sockets.
//
// Every descriptor is owned by exactly one object and closed exactly once. A
// leaked descriptor is not a tidiness problem: a node that leaks them stops being
// able to accept connections at all, which is a denial of service an attacker can
// trigger deliberately by opening and dropping connections.
//
// Everything is non-blocking. A blocking read on a peer socket hands that peer
// control of the node's progress — it simply stops sending and the node waits
// forever.
#pragma once

#include <cstdint>
#include <span>
#include <string>

#include "net/protocol.h"
#include "util/result.h"

namespace rnet::net {

// Result of a partial read or write on a non-blocking socket.
struct IoResult {
    size_t bytes{0};
    bool would_block{false};   // nothing more available right now; not an error
    bool closed{false};        // peer closed the connection cleanly
};

class Socket {
public:
    Socket() = default;
    explicit Socket(int fd) : fd_(fd) {}
    ~Socket();

    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
    Socket(Socket&& other) noexcept;
    Socket& operator=(Socket&& other) noexcept;

    bool valid() const { return fd_ >= 0; }
    int fd() const { return fd_; }
    void Close();

    // Non-blocking read/write. Neither treats "would block" as failure — that is
    // the normal state of an idle connection.
    Result<IoResult> Read(std::span<uint8_t> buffer);
    Result<IoResult> Write(std::span<const uint8_t> data);

    Status SetNonBlocking();
    Status SetNoDelay();       // small control messages should not wait on Nagle
    Status SetReuseAddr();

    // Starts a non-blocking connect. Completion is observed later via poll();
    // `in_progress` is the expected outcome, not an error.
    static Result<Socket> StartConnect(const NetAddress& address, bool& in_progress);

    // Checks whether a connect that was in progress has finished.
    Status ConnectResult() const;

    static Result<Socket> Listen(const NetAddress& bind_address, int backlog = 64);

    // Accepts one pending connection. Returns would_block when there are none.
    Result<Socket> Accept(NetAddress& peer_address, bool& would_block) const;

    Result<NetAddress> LocalAddress() const;

private:
    int fd_{-1};
};

// Readiness state for one descriptor, as reported by the poll loop.
struct PollEntry {
    int fd{-1};
    bool want_read{false};
    bool want_write{false};
    bool readable{false};
    bool writable{false};
    bool error{false};
};

// Waits for readiness. `timeout_ms` of -1 blocks until something happens; 0 polls.
Result<size_t> PollSockets(std::vector<PollEntry>& entries, int timeout_ms);

}  // namespace rnet::net

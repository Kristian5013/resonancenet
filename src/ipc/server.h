// The listening end of the local channel.
//
// Access control here is filesystem permissions, and getting that right is a
// sequence rather than a setting:
//
//   * The FILESYSTEM namespace, not the abstract one. An abstract socket
//     (leading NUL) has no inode and therefore no permissions at all — every
//     process on the machine can connect to it. It looks tidier and protects
//     nothing.
//
//   * The DIRECTORY is 0700 and is what actually enforces access. A mode set on
//     the socket after bind() leaves a window in which the socket exists with
//     umask-derived permissions and a racing process can connect. Putting the
//     socket inside a directory that was already 0700 closes that window before
//     the socket exists, rather than narrowing it afterwards.
//
//   * BIND, THEN CHMOD, is still done — belt and braces, since the directory is
//     the real control and the mode is defence in depth.
//
//   * A STALE SOCKET is not simply unlinked. A path with a live daemon behind it
//     and a path left by one that crashed look identical on disk; the difference
//     is whether a connect() succeeds. Unlinking blindly would let a second
//     instance silently steal the channel from a running one.
//
// SO_PEERCRED gives the connecting process's uid. It is checked, and it is
// defence in depth only: the honest threat model is that a process running as the
// SAME USER can do anything this daemon can do, because it can read the daemon's
// own files. What the check buys is that a different user cannot, even if the
// directory permissions were ever loosened by accident.
#pragma once

#include <sys/types.h>

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <vector>

#include "ipc/frame.h"
#include "util/result.h"

namespace rnet::ipc {

// A worker that connects and says nothing holds a slot for free.
inline constexpr int64_t kHandshakeTimeoutMs = 15'000;

// Local clients are few by nature: one worker per GPU, plus a status query. A
// bound exists so a runaway script cannot exhaust the daemon's descriptors.
inline constexpr size_t kMaxClients = 16;

// An RAII descriptor. Every path out of this file closes what it opened,
// including the error paths — a leaked descriptor from a rejected connection is a
// resource an attacker gets to spend by being wrong repeatedly.
class Fd {
public:
    Fd() = default;
    explicit Fd(int fd) : fd_(fd) {}
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    Fd(Fd&& other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
    Fd& operator=(Fd&& other) noexcept;
    ~Fd() { Close(); }

    int get() const { return fd_; }
    bool valid() const { return fd_ >= 0; }
    int release();
    void Close();

private:
    int fd_{-1};
};

// What arrived, and from whom.
struct ReceivedFrame {
    Frame frame;
    Fd descriptor;      // present only for messages that require one
};

// One connected worker.
class Client {
public:
    Client(uint64_t id, Fd socket, uid_t peer_uid, pid_t peer_pid, int64_t connected_at);

    uint64_t id() const { return id_; }
    uid_t peer_uid() const { return peer_uid_; }
    // Recorded for the log only. A pid is never an authorization input: it is
    // reused, and the process behind it can be replaced between the check and the
    // use.
    pid_t peer_pid() const { return peer_pid_; }
    int fd() const { return socket_.get(); }
    int64_t connected_at() const { return connected_at_; }

    bool handshake_done() const { return handshake_done_; }
    void MarkHandshakeDone() { handshake_done_ = true; }

    bool should_close() const { return should_close_; }
    const std::string& close_reason() const { return close_reason_; }
    void Close(std::string reason);

    // Reads one record. Returns nullopt-shaped success (an empty frame type) when
    // the socket would block, and an error when the connection must end.
    Result<bool> Receive(ReceivedFrame& out);

    // Sends one frame. Local sockets do not benefit from queuing: if the worker is
    // not reading, waiting is the right answer and buffering is not.
    Status Send(const Frame& frame);

    // Sends a frame together with a descriptor. Used for bulk data in the
    // daemon-to-worker direction, which is otherwise identical.
    Status SendWithDescriptor(const Frame& frame, int fd);

private:
    uint64_t id_;
    Fd socket_;
    uid_t peer_uid_;
    pid_t peer_pid_;
    int64_t connected_at_;
    bool handshake_done_{false};
    bool should_close_{false};
    std::string close_reason_;
};

class Server {
public:
    Server() = default;
    ~Server();

    // Creates the directory 0700, refuses a socket that is already live, and binds.
    Status Listen(const std::filesystem::path& socket_path);
    void Stop();

    bool listening() const { return listener_.valid(); }
    const std::filesystem::path& path() const { return path_; }

    // Accepts pending connections and collects whatever arrived. Non-blocking
    // throughout, so the daemon's single loop cannot be stalled by a worker.
    struct Delivery {
        uint64_t client_id{0};
        ReceivedFrame received;
    };
    Result<std::vector<Delivery>> Poll(int64_t now_ms);

    Client* Find(uint64_t client_id);
    size_t client_count() const { return clients_.size(); }

    // Drops clients marked for closing and those that never finished a handshake.
    void Sweep(int64_t now_ms);

private:
    Status AcceptPending(int64_t now_ms);

    Fd listener_;
    std::filesystem::path path_;
    std::map<uint64_t, std::unique_ptr<Client>> clients_;
    uint64_t next_client_id_{1};
};

// True if something is already listening on this path. Used instead of unlinking
// blindly, because a live socket and a stale one look identical on disk and only
// a connect() tells them apart.
bool SocketIsLive(const std::filesystem::path& path);

// Receives a sealed memfd and returns its size, refusing anything that is not a
// regular file sealed against modification. A descriptor the sender can still
// write to would let the bytes change between being hashed and being served.
Result<uint64_t> SealedSizeOf(int fd);

// Builds a memfd holding `bytes` and seals it against write, shrink and grow
// before returning. Sealing happens before the descriptor leaves this function,
// so there is no window in which a receiver could be handed a writable view.
Result<Fd> MakeSealedMemfd(std::span<const uint8_t> bytes, const char* name);

}  // namespace rnet::ipc

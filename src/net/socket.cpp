#include "net/socket.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <format>

namespace rnet::net {

namespace {

std::string Errno() { return std::strerror(errno); }

// Fills a sockaddr_in6 from a NetAddress. IPv4 travels as a v4-mapped v6 address
// so one code path serves both families.
void ToSockAddr(const NetAddress& address, sockaddr_in6& out) {
    std::memset(&out, 0, sizeof(out));
    out.sin6_family = AF_INET6;
    out.sin6_port = htons(address.port);
    std::memcpy(&out.sin6_addr, address.ip.data(), 16);
}

NetAddress FromSockAddr(const sockaddr_in6& addr) {
    NetAddress out;
    std::memcpy(out.ip.data(), &addr.sin6_addr, 16);
    out.port = ntohs(addr.sin6_port);
    return out;
}

}  // namespace

Socket::~Socket() { Close(); }

Socket::Socket(Socket&& other) noexcept : fd_(other.fd_) { other.fd_ = -1; }

Socket& Socket::operator=(Socket&& other) noexcept {
    if (this != &other) {
        Close();
        fd_ = other.fd_;
        other.fd_ = -1;
    }
    return *this;
}

void Socket::Close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

Status Socket::SetNonBlocking() {
    if (!valid()) return Err("socket: not open");
    const int flags = ::fcntl(fd_, F_GETFL, 0);
    if (flags < 0) return Err("socket: fcntl(F_GETFL): " + Errno());
    if (::fcntl(fd_, F_SETFL, flags | O_NONBLOCK) < 0) {
        return Err("socket: fcntl(F_SETFL): " + Errno());
    }
    return Status::Ok();
}

Status Socket::SetNoDelay() {
    if (!valid()) return Err("socket: not open");
    int one = 1;
    if (::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) < 0) {
        return Err("socket: TCP_NODELAY: " + Errno());
    }
    return Status::Ok();
}

Status Socket::SetReuseAddr() {
    if (!valid()) return Err("socket: not open");
    int one = 1;
    if (::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) < 0) {
        return Err("socket: SO_REUSEADDR: " + Errno());
    }
    return Status::Ok();
}

Result<IoResult> Socket::Read(std::span<uint8_t> buffer) {
    if (!valid()) return Err("socket: not open");
    if (buffer.empty()) return IoResult{};

    const ssize_t n = ::recv(fd_, buffer.data(), buffer.size(), 0);
    if (n > 0) return IoResult{static_cast<size_t>(n), false, false};
    if (n == 0) return IoResult{0, false, true};   // orderly shutdown

    if (errno == EAGAIN || errno == EWOULDBLOCK) return IoResult{0, true, false};
    if (errno == EINTR) return IoResult{0, true, false};   // retry next poll
    return Err("socket: recv: " + Errno());
}

Result<IoResult> Socket::Write(std::span<const uint8_t> data) {
    if (!valid()) return Err("socket: not open");
    if (data.empty()) return IoResult{};

    // MSG_NOSIGNAL: writing to a closed peer must return an error, not raise
    // SIGPIPE and kill the node.
    const ssize_t n = ::send(fd_, data.data(), data.size(), MSG_NOSIGNAL);
    if (n >= 0) return IoResult{static_cast<size_t>(n), false, false};

    if (errno == EAGAIN || errno == EWOULDBLOCK) return IoResult{0, true, false};
    if (errno == EINTR) return IoResult{0, true, false};
    if (errno == EPIPE || errno == ECONNRESET) return IoResult{0, false, true};
    return Err("socket: send: " + Errno());
}

Result<Socket> Socket::StartConnect(const NetAddress& address, bool& in_progress) {
    in_progress = false;

    const int fd = ::socket(AF_INET6, SOCK_STREAM, 0);
    if (fd < 0) return Err("socket: create: " + Errno());
    Socket sock(fd);

    // Accept v4-mapped addresses on the same socket.
    int off = 0;
    ::setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &off, sizeof(off));

    if (auto st = sock.SetNonBlocking(); !st) return Err(st.error());
    if (auto st = sock.SetNoDelay(); !st) return Err(st.error());

    sockaddr_in6 target{};
    ToSockAddr(address, target);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&target), sizeof(target)) == 0) {
        return sock;   // connected immediately (loopback usually does)
    }
    if (errno == EINPROGRESS) {
        in_progress = true;   // expected: completion is reported by poll()
        return sock;
    }
    return Err("socket: connect " + address.ToString() + ": " + Errno());
}

Status Socket::ConnectResult() const {
    if (!valid()) return Err("socket: not open");
    int error = 0;
    socklen_t len = sizeof(error);
    if (::getsockopt(fd_, SOL_SOCKET, SO_ERROR, &error, &len) < 0) {
        return Err("socket: getsockopt(SO_ERROR): " + Errno());
    }
    if (error != 0) return Err(std::string("socket: connect failed: ") + std::strerror(error));
    return Status::Ok();
}

Result<Socket> Socket::Listen(const NetAddress& bind_address, int backlog) {
    const int fd = ::socket(AF_INET6, SOCK_STREAM, 0);
    if (fd < 0) return Err("socket: create: " + Errno());
    Socket sock(fd);

    int off = 0;
    ::setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &off, sizeof(off));
    if (auto st = sock.SetReuseAddr(); !st) return Err(st.error());
    if (auto st = sock.SetNonBlocking(); !st) return Err(st.error());

    sockaddr_in6 addr{};
    ToSockAddr(bind_address, addr);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        return Err("socket: bind " + bind_address.ToString() + ": " + Errno());
    }
    if (::listen(fd, backlog) < 0) return Err("socket: listen: " + Errno());
    return sock;
}

Result<Socket> Socket::Accept(NetAddress& peer_address, bool& would_block) const {
    would_block = false;
    if (!valid()) return Err("socket: not open");

    sockaddr_in6 addr{};
    socklen_t len = sizeof(addr);
    const int fd = ::accept(fd_, reinterpret_cast<sockaddr*>(&addr), &len);
    if (fd < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
            would_block = true;
            return Socket();
        }
        return Err("socket: accept: " + Errno());
    }

    Socket peer(fd);
    if (auto st = peer.SetNonBlocking(); !st) return Err(st.error());
    if (auto st = peer.SetNoDelay(); !st) return Err(st.error());
    peer_address = FromSockAddr(addr);
    return peer;
}

Result<NetAddress> Socket::LocalAddress() const {
    if (!valid()) return Err("socket: not open");
    sockaddr_in6 addr{};
    socklen_t len = sizeof(addr);
    if (::getsockname(fd_, reinterpret_cast<sockaddr*>(&addr), &len) < 0) {
        return Err("socket: getsockname: " + Errno());
    }
    return FromSockAddr(addr);
}

Result<size_t> PollSockets(std::vector<PollEntry>& entries, int timeout_ms) {
    if (entries.empty()) return size_t{0};

    std::vector<pollfd> fds;
    fds.reserve(entries.size());
    for (const auto& e : entries) {
        pollfd p{};
        p.fd = e.fd;
        p.events = static_cast<short>((e.want_read ? POLLIN : 0) | (e.want_write ? POLLOUT : 0));
        fds.push_back(p);
    }

    const int ready = ::poll(fds.data(), fds.size(), timeout_ms);
    if (ready < 0) {
        // A signal during poll is normal; the caller simply polls again.
        if (errno == EINTR) return size_t{0};
        return Err("socket: poll: " + Errno());
    }

    for (size_t i = 0; i < entries.size(); ++i) {
        entries[i].readable = (fds[i].revents & POLLIN) != 0;
        entries[i].writable = (fds[i].revents & POLLOUT) != 0;
        entries[i].error = (fds[i].revents & (POLLERR | POLLHUP | POLLNVAL)) != 0;
    }
    return static_cast<size_t>(ready);
}

}  // namespace rnet::net

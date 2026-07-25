#include "ipc/server.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <format>

#include "util/logging.h"

namespace rnet::ipc {

namespace {

// A record can be at most one maximum frame. Sized once, on the stack of the
// receive path, so a hostile sender cannot make the daemon allocate per message.
constexpr size_t kReceiveBuffer = kMaxFrameSize;

Status SetNonBlocking(int fd) {
    const int flags = ::fcntl(fd, F_GETFL, 0);
    if (flags < 0) return Err(std::string("fcntl(F_GETFL): ") + std::strerror(errno));
    if (::fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
        return Err(std::string("fcntl(F_SETFL): ") + std::strerror(errno));
    }
    return Status::Ok();
}

// Fills a sockaddr_un, refusing a path that would be truncated. A truncated path
// binds somewhere other than where the caller asked, which is worse than failing.
Result<sockaddr_un> AddressFor(const std::filesystem::path& path) {
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    const std::string text = path.string();
    if (text.size() >= sizeof(address.sun_path)) {
        return Err(std::format("ipc: socket path is {} bytes, the limit is {}", text.size(),
                               sizeof(address.sun_path) - 1));
    }
    std::memcpy(address.sun_path, text.c_str(), text.size());
    return address;
}

}  // namespace

Fd& Fd::operator=(Fd&& other) noexcept {
    if (this != &other) {
        Close();
        fd_ = other.fd_;
        other.fd_ = -1;
    }
    return *this;
}

int Fd::release() {
    const int fd = fd_;
    fd_ = -1;
    return fd;
}

void Fd::Close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

bool SocketIsLive(const std::filesystem::path& path) {
    auto address = AddressFor(path);
    if (!address) return false;

    Fd probe(::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0));
    if (!probe.valid()) return false;
    const int rc = ::connect(probe.get(), reinterpret_cast<sockaddr*>(&address.value()),
                             sizeof(sockaddr_un));
    // Connecting succeeds only if something is listening. ECONNREFUSED means the
    // inode is there and nobody is behind it — the definition of stale.
    return rc == 0;
}

Result<uint64_t> SealedSizeOf(int fd) {
    struct stat info {};
    if (::fstat(fd, &info) != 0) {
        return Err(std::string("ipc: fstat on the received descriptor: ") + std::strerror(errno));
    }
    // A pipe, a socket or a device would let the sender feed bytes as they are
    // read — so the bytes hashed and the bytes stored could differ.
    if (!S_ISREG(info.st_mode)) {
        return Err("ipc: the received descriptor is not a regular file");
    }

    const int seals = ::fcntl(fd, F_GET_SEALS);
    if (seals < 0) {
        // Only memfds support sealing. A file on disk cannot be sealed, so this
        // also discriminates: a path the sender can still write to is refused
        // here rather than trusted.
        return Err(std::string("ipc: the received descriptor cannot be sealed (") +
                   std::strerror(errno) + ") — it must be a sealed memfd");
    }
    const int required = F_SEAL_WRITE | F_SEAL_SHRINK | F_SEAL_GROW;
    if ((seals & required) != required) {
        // Without all three the sender can still change the contents or the length
        // between the hash and the store, and the daemon would serve peers bytes
        // it never verified.
        return Err(std::format(
            "ipc: descriptor is not sealed against modification (seals 0x{:x}, need write, shrink "
            "and grow)",
            seals));
    }
    if (info.st_size < 0) return Err("ipc: descriptor reports a negative size");
    return static_cast<uint64_t>(info.st_size);
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

Client::Client(uint64_t id, Fd socket, uid_t peer_uid, pid_t peer_pid, int64_t connected_at)
    : id_(id),
      socket_(std::move(socket)),
      peer_uid_(peer_uid),
      peer_pid_(peer_pid),
      connected_at_(connected_at) {}

void Client::Close(std::string reason) {
    if (should_close_) return;
    should_close_ = true;
    close_reason_ = std::move(reason);
}

Result<bool> Client::Receive(ReceivedFrame& out) {
    std::vector<uint8_t> buffer(kReceiveBuffer);
    // Room for one descriptor. A sender offering more is malformed, and the
    // control buffer is sized so the extras are truncated rather than accepted —
    // MSG_CTRUNC then tells us it happened.
    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))];

    iovec io{};
    io.iov_base = buffer.data();
    io.iov_len = buffer.size();

    msghdr message{};
    message.msg_iov = &io;
    message.msg_iovlen = 1;
    message.msg_control = control;
    message.msg_controllen = sizeof(control);

    const ssize_t got = ::recvmsg(socket_.get(), &message, MSG_CMSG_CLOEXEC | MSG_DONTWAIT);
    if (got < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) return false;
        if (errno == EINTR) return false;
        Close(std::string("recvmsg: ") + std::strerror(errno));
        return Err(std::string("ipc: recvmsg: ") + std::strerror(errno));
    }
    if (got == 0) {
        Close("worker closed the connection");
        return false;
    }

    // Collect descriptors first, whatever else is wrong, so none is leaked by an
    // early return below.
    std::vector<Fd> descriptors;
    for (cmsghdr* header = CMSG_FIRSTHDR(&message); header != nullptr;
         header = CMSG_NXTHDR(&message, header)) {
        if (header->cmsg_level != SOL_SOCKET || header->cmsg_type != SCM_RIGHTS) continue;
        const size_t count = (header->cmsg_len - CMSG_LEN(0)) / sizeof(int);
        const int* fds = reinterpret_cast<const int*>(CMSG_DATA(header));
        for (size_t i = 0; i < count; ++i) descriptors.emplace_back(fds[i]);
    }

    if ((message.msg_flags & MSG_CTRUNC) != 0) {
        // More descriptors were offered than fit. Whatever the intent, the daemon
        // cannot see all of them, so it must not act on the ones it got.
        Close("descriptors were truncated");
        return Err("ipc: the sender offered more descriptors than the protocol allows");
    }
    if ((message.msg_flags & MSG_TRUNC) != 0) {
        // The record was larger than the buffer. Since the buffer is one maximum
        // frame, this is a message that could never be legal.
        Close("record larger than the frame limit");
        return Err("ipc: record exceeds the maximum frame size");
    }

    auto frame = DecodeFrame(std::span<const uint8_t>(buffer.data(), static_cast<size_t>(got)));
    if (!frame) {
        Close(frame.error());
        return Err(frame.error());
    }
    // Each type is legal in one direction only, so a reply cannot be replayed at
    // the daemon and made to look like a request.
    if (!IsWorkerToDaemon(frame.value().type)) {
        Close(std::format("'{}' is not a message a worker may send",
                          MessageTypeName(frame.value().type)));
        return Err("ipc: message travelling in the wrong direction");
    }

    const uint32_t required = RequiredDescriptors(frame.value().type);
    if (descriptors.size() != required) {
        Close(std::format("'{}' requires {} descriptor(s), {} arrived",
                          MessageTypeName(frame.value().type), required, descriptors.size()));
        return Err("ipc: wrong number of descriptors");
    }

    out.frame = std::move(frame.value());
    out.descriptor = required == 1 ? std::move(descriptors[0]) : Fd{};
    return true;
}

Status Client::Send(const Frame& frame) {
    auto encoded = EncodeFrame(frame);
    if (!encoded) return Err(encoded.error());

    // One record, one write. SOCK_SEQPACKET does not do partial records: it
    // either takes the whole thing or refuses it, so there is no partial-write
    // state to carry.
    const ssize_t sent = ::send(socket_.get(), encoded.value().data(), encoded.value().size(),
                                MSG_NOSIGNAL | MSG_DONTWAIT);
    if (sent < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // The worker is not reading. Waiting for it is the right answer;
            // buffering on its behalf is how one stuck client consumes the
            // daemon's memory.
            Close("worker is not reading its socket");
            return Err("ipc: send would block");
        }
        Close(std::string("send: ") + std::strerror(errno));
        return Err(std::string("ipc: send: ") + std::strerror(errno));
    }
    if (static_cast<size_t>(sent) != encoded.value().size()) {
        Close("partial record");
        return Err("ipc: the transport truncated a record");
    }
    return Status::Ok();
}

Status Client::SendWithDescriptor(const Frame& frame, int fd) {
    auto encoded = EncodeFrame(frame);
    if (!encoded) return Err(encoded.error());

    iovec io{};
    io.iov_base = encoded.value().data();
    io.iov_len = encoded.value().size();

    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))];
    msghdr message{};
    message.msg_iov = &io;
    message.msg_iovlen = 1;
    message.msg_control = control;
    message.msg_controllen = sizeof(control);
    cmsghdr* header = CMSG_FIRSTHDR(&message);
    header->cmsg_level = SOL_SOCKET;
    header->cmsg_type = SCM_RIGHTS;
    header->cmsg_len = CMSG_LEN(sizeof(int));
    std::memcpy(CMSG_DATA(header), &fd, sizeof(int));

    const ssize_t sent = ::sendmsg(socket_.get(), &message, MSG_NOSIGNAL | MSG_DONTWAIT);
    if (sent < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            Close("worker is not reading its socket");
            return Err("ipc: send would block");
        }
        Close(std::string("sendmsg: ") + std::strerror(errno));
        return Err(std::string("ipc: sendmsg: ") + std::strerror(errno));
    }
    if (static_cast<size_t>(sent) != encoded.value().size()) {
        Close("partial record");
        return Err("ipc: the transport truncated a record");
    }
    return Status::Ok();
}

Result<Fd> MakeSealedMemfd(std::span<const uint8_t> bytes, const char* name) {
    Fd fd(::memfd_create(name, MFD_CLOEXEC | MFD_ALLOW_SEALING));
    if (!fd.valid()) return Err(std::string("ipc: memfd_create: ") + std::strerror(errno));

    size_t written = 0;
    while (written < bytes.size()) {
        const ssize_t n = ::write(fd.get(), bytes.data() + written, bytes.size() - written);
        if (n < 0) {
            if (errno == EINTR) continue;
            return Err(std::string("ipc: writing the memfd: ") + std::strerror(errno));
        }
        if (n == 0) return Err("ipc: the memfd accepted no bytes");
        written += static_cast<size_t>(n);
    }
    // Sealed before it leaves this function, so no receiver is ever handed a
    // descriptor it or anyone else could still modify.
    if (::fcntl(fd.get(), F_ADD_SEALS, F_SEAL_WRITE | F_SEAL_SHRINK | F_SEAL_GROW) != 0) {
        return Err(std::string("ipc: sealing the memfd: ") + std::strerror(errno));
    }
    return fd;
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

Server::~Server() { Stop(); }

Status Server::Listen(const std::filesystem::path& socket_path) {
    if (listener_.valid()) return Err("ipc: already listening");

    const auto directory = socket_path.parent_path();
    if (!directory.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(directory, ec);
        if (ec) {
            return Err(std::format("ipc: cannot create {}: {}", directory.string(), ec.message()));
        }
        // The directory is the real access control, and it is set BEFORE the
        // socket exists. Setting a mode on the socket after bind() leaves a window
        // in which it is reachable with umask-derived permissions.
        if (::chmod(directory.c_str(), S_IRWXU) != 0) {
            return Err(std::format("ipc: cannot restrict {}: {}", directory.string(),
                                   std::strerror(errno)));
        }
    }

    // A live socket and a stale one look identical on disk. Only a connect() tells
    // them apart, and unlinking blindly would let a second instance quietly steal
    // the channel from a running one.
    if (std::filesystem::exists(socket_path)) {
        if (SocketIsLive(socket_path)) {
            return Err(std::format(
                "ipc: {} already has a daemon listening on it — refusing to take over",
                socket_path.string()));
        }
        std::error_code ec;
        std::filesystem::remove(socket_path, ec);
        if (ec) {
            return Err(std::format("ipc: cannot remove the stale socket {}: {}",
                                   socket_path.string(), ec.message()));
        }
        RNET_LOG_INFO("ipc: removed a stale socket at {}", socket_path.string());
    }

    auto address = AddressFor(socket_path);
    if (!address) return Err(address.error());

    // SEQPACKET rather than STREAM: records are preserved, so a message cannot be
    // split across reads and there is no reassembly buffer to get wrong.
    Fd socket(::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0));
    if (!socket.valid()) {
        return Err(std::string("ipc: socket: ") + std::strerror(errno));
    }
    if (::bind(socket.get(), reinterpret_cast<sockaddr*>(&address.value()), sizeof(sockaddr_un)) !=
        0) {
        return Err(std::format("ipc: bind {}: {}", socket_path.string(), std::strerror(errno)));
    }
    // Defence in depth behind the 0700 directory.
    if (::chmod(socket_path.c_str(), S_IRUSR | S_IWUSR) != 0) {
        std::error_code ec;
        std::filesystem::remove(socket_path, ec);
        return Err(std::format("ipc: cannot restrict the socket: {}", std::strerror(errno)));
    }
    if (::listen(socket.get(), static_cast<int>(kMaxClients)) != 0) {
        std::error_code ec;
        std::filesystem::remove(socket_path, ec);
        return Err(std::string("ipc: listen: ") + std::strerror(errno));
    }

    listener_ = std::move(socket);
    path_ = socket_path;
    RNET_LOG_INFO("ipc: listening on {}", socket_path.string());
    return Status::Ok();
}

void Server::Stop() {
    clients_.clear();
    if (listener_.valid()) {
        listener_.Close();
        // Removed on the way out so the next start does not have to decide whether
        // a leftover is stale.
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }
}

Status Server::AcceptPending(int64_t now_ms) {
    while (clients_.size() < kMaxClients) {
        Fd connection(::accept4(listener_.get(), nullptr, nullptr, SOCK_CLOEXEC | SOCK_NONBLOCK));
        if (!connection.valid()) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) return Status::Ok();
            if (errno == EINTR) continue;
            return Err(std::string("ipc: accept: ") + std::strerror(errno));
        }

        ucred credentials{};
        socklen_t length = sizeof(credentials);
        if (::getsockopt(connection.get(), SOL_SOCKET, SO_PEERCRED, &credentials, &length) != 0) {
            RNET_LOG_WARN("ipc: refusing a connection whose credentials could not be read: {}",
                          std::strerror(errno));
            continue;   // the descriptor closes here
        }

        // Defence in depth. The directory mode is what actually enforces this; the
        // check catches a permissions mistake rather than a determined attacker,
        // and a process running as the same user is not constrained by either.
        if (credentials.uid != ::getuid()) {
            RNET_LOG_WARN("ipc: refusing a connection from uid {} (this daemon runs as {})",
                          credentials.uid, ::getuid());
            continue;
        }

        const uint64_t id = next_client_id_++;
        clients_[id] = std::make_unique<Client>(id, std::move(connection), credentials.uid,
                                                credentials.pid, now_ms);
        RNET_LOG_INFO("ipc: worker {} connected (uid {}, pid {})", id, credentials.uid,
                      credentials.pid);
    }
    return Status::Ok();
}

Result<std::vector<Server::Delivery>> Server::Poll(int64_t now_ms) {
    std::vector<Delivery> delivered;
    if (!listener_.valid()) return delivered;

    if (auto st = AcceptPending(now_ms); !st) return Err(st.error());

    for (auto& [id, client] : clients_) {
        if (client->should_close()) continue;
        // Drained until it would block, so a worker that sent several messages in
        // one tick is not made to wait a tick per message.
        while (true) {
            ReceivedFrame received;
            auto got = client->Receive(received);
            if (!got) break;             // the client is already marked for closing
            if (!got.value()) break;     // nothing more right now
            delivered.push_back(Delivery{id, std::move(received)});
            if (client->should_close()) break;
        }
    }
    return delivered;
}

Client* Server::Find(uint64_t client_id) {
    const auto it = clients_.find(client_id);
    return it == clients_.end() ? nullptr : it->second.get();
}

void Server::Sweep(int64_t now_ms) {
    for (auto it = clients_.begin(); it != clients_.end();) {
        auto& client = *it->second;
        // A worker that connects and says nothing costs it nothing and costs the
        // daemon a slot.
        if (!client.handshake_done() && now_ms - client.connected_at() > kHandshakeTimeoutMs) {
            client.Close("handshake timeout");
        }
        if (!client.should_close()) {
            ++it;
            continue;
        }
        RNET_LOG_INFO("ipc: worker {} disconnected: {}", it->first, client.close_reason());
        it = clients_.erase(it);
    }
}

}  // namespace rnet::ipc

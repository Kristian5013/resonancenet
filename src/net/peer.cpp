#include "net/peer.h"

#include <algorithm>
#include <format>

#include "util/logging.h"

namespace rnet::net {

std::string PeerStateName(PeerState state) {
    switch (state) {
        case PeerState::Connecting: return "connecting";
        case PeerState::AwaitingVersion: return "awaiting-version";
        case PeerState::AwaitingVerack: return "awaiting-verack";
        case PeerState::Ready: return "ready";
        case PeerState::Disconnecting: return "disconnecting";
    }
    return "unknown";
}

bool CommandAllowedInState(std::string_view command, PeerState state) {
    if (state == PeerState::Ready) return true;
    if (state == PeerState::AwaitingVersion) return command == cmd::kVersion;
    if (state == PeerState::AwaitingVerack) {
        // `reject` is allowed here so a peer can explain a refused handshake
        // rather than dropping the connection silently.
        return command == cmd::kVerack || command == cmd::kReject;
    }
    return false;
}

Peer::Peer(uint64_t id, Socket socket, NetAddress address, bool inbound,
           std::array<uint8_t, 4> magic)
    : id_(id),
      socket_(std::move(socket)),
      address_(address),
      inbound_(inbound),
      magic_(magic),
      state_(inbound ? PeerState::AwaitingVersion : PeerState::Connecting) {}

void Peer::MarkConnected() {
    if (state_ == PeerState::Connecting) state_ = PeerState::AwaitingVersion;
}

Result<std::vector<ReceivedMessage>> Peer::ReceiveBytes(std::span<const uint8_t> bytes) {
    if (receive_buffer_.size() + bytes.size() > kMaxReceiveBuffer) {
        // A peer streaming faster than the node can frame is either broken or
        // deliberately exhausting memory. Either way the connection ends.
        Disconnect("receive buffer overflow");
        return Err("peer: receive buffer would exceed the limit");
    }
    receive_buffer_.insert(receive_buffer_.end(), bytes.begin(), bytes.end());

    std::vector<ReceivedMessage> messages;
    size_t offset = 0;

    while (true) {
        // A header may be split across reads; wait for the whole thing.
        if (receive_buffer_.size() - offset < kHeaderSize) break;

        auto header = ParseHeader(
            std::span<const uint8_t>(receive_buffer_).subspan(offset, kHeaderSize), magic_);
        if (!header) {
            // Bad magic or an impossible length: the stream is not recoverable,
            // because there is no way to know where the next message starts.
            Disconnect(header.error());
            return Err(header.error());
        }

        const size_t total = kHeaderSize + header.value().payload_size;
        if (receive_buffer_.size() - offset < total) break;   // payload still arriving

        const auto payload =
            std::span<const uint8_t>(receive_buffer_).subspan(offset + kHeaderSize,
                                                              header.value().payload_size);
        if (auto st = VerifyPayload(header.value(), payload); !st) {
            Disconnect(st.error());
            return Err(st.error());
        }

        // Framing only. Whether a command is allowed is a question about the
        // state at the moment it is HANDLED, not the moment it was parsed: a
        // single read can carry `version` and `verack` together, and the second
        // becomes legal only because the first was processed. Checking here would
        // judge both against the pre-handshake state and reject the verack.
        messages.push_back(ReceivedMessage{header.value().command,
                                           std::vector<uint8_t>(payload.begin(), payload.end())});
        offset += total;
    }

    if (offset > 0) {
        receive_buffer_.erase(receive_buffer_.begin(),
                              receive_buffer_.begin() + static_cast<ptrdiff_t>(offset));
    }
    return messages;
}

Status Peer::Send(std::string_view command, std::span<const uint8_t> payload) {
    auto framed = SerializeMessage(magic_, command, payload);
    if (!framed) return Err(framed.error());

    if (send_queue_.size() + framed.value().size() > kMaxSendQueue) {
        // The peer has stopped reading. Buffering on its behalf is how one slow
        // peer takes down a node.
        Disconnect("send queue overflow");
        return Err("peer: send queue full");
    }
    send_queue_.insert(send_queue_.end(), framed.value().begin(), framed.value().end());
    return Status::Ok();
}

std::span<const uint8_t> Peer::PendingSend() const {
    const_cast<Peer*>(this)->send_window_.assign(
        send_queue_.begin(),
        send_queue_.begin() + static_cast<ptrdiff_t>(std::min<size_t>(send_queue_.size(), 64 * 1024)));
    return send_window_;
}

void Peer::ConsumeSent(size_t consumed) {
    const size_t n = std::min(consumed, send_queue_.size());
    send_queue_.erase(send_queue_.begin(), send_queue_.begin() + static_cast<ptrdiff_t>(n));
}

Status Peer::CheckCommandAllowed(std::string_view command) {
    if (CommandAllowedInState(command, state_)) return Status::Ok();
    // Driving the node's object logic before identifying yourself is not an
    // ordering mistake to be tolerated.
    Misbehaving(20, std::format("'{}' before handshake completed", command));
    return Err(std::format("peer: '{}' not allowed in state {}", command, PeerStateName(state_)));
}

Status Peer::BeginHandshake(const VersionMessage& local) {
    if (sent_version_) return Status::Ok();
    if (auto st = Send(cmd::kVersion, local.Serialize()); !st) return st;
    sent_version_ = true;
    return Status::Ok();
}

Status Peer::HandleVersion(const VersionMessage& remote, const VersionMessage& local,
                           uint64_t local_nonce) {
    if (state_ != PeerState::AwaitingVersion) {
        Misbehaving(20, "duplicate version");
        return Err("peer: version received twice");
    }
    if (auto st = remote.Validate(); !st) {
        Disconnect(st.error());
        return st;
    }
    // Connecting to yourself wastes a slot and pollutes the address database.
    //
    // Answer before hanging up. Only the side that RECEIVES a version can see the
    // matching nonce, and for an inbound connection that side holds an ephemeral
    // source port — forgetting it achieves nothing, while the dialer, which holds
    // the address that is actually in the database, learns only that the
    // connection closed. Sending our version first lets it draw the same
    // conclusion and drop the address. Without this the node dials itself on
    // every tick forever; measured on the seed at twenty-one attempts a minute.
    if (remote.nonce == local_nonce) {
        is_self_ = true;
        if (!sent_version_) {
            (void)Send(cmd::kVersion, local.Serialize());
            sent_version_ = true;
        }
        Disconnect("self-connection");
        return Err("peer: connected to self");
    }
    // A node on a different round, or under different rules, is not a peer: its
    // contributions would be incompatible with everything this node does.
    if (remote.genesis_hash != local.genesis_hash) {
        Disconnect("different genesis");
        return Err("peer: different genesis");
    }
    if (remote.policy_hash != local.policy_hash) {
        Disconnect("different policy");
        return Err("peer: different policy");
    }
    if (remote.protocol_version < kMinProtocolVersion) {
        Disconnect("obsolete protocol version");
        return Err("peer: obsolete protocol version");
    }

    version_ = remote;
    state_ = PeerState::AwaitingVerack;

    // An inbound peer spoke first; reply with our own version before verack.
    if (!sent_version_) {
        if (auto st = Send(cmd::kVersion, local.Serialize()); !st) return st;
        sent_version_ = true;
    }
    return Send(cmd::kVerack, {});
}

Status Peer::HandleVerack() {
    if (state_ != PeerState::AwaitingVerack) {
        Misbehaving(10, "unexpected verack");
        return Err("peer: unexpected verack");
    }
    state_ = PeerState::Ready;
    return Status::Ok();
}

void Peer::Misbehaving(int howmuch, std::string_view reason) {
    misbehaviour_ += howmuch;
    RNET_LOG_DEBUG("peer {} misbehaving (+{} = {}): {}", id_, howmuch, misbehaviour_, reason);
    if (misbehaviour_ >= kBanThreshold) {
        Disconnect(std::format("misbehaviour threshold reached: {}", reason));
    }
}

void Peer::Disconnect(std::string_view reason) {
    if (state_ == PeerState::Disconnecting) return;
    state_ = PeerState::Disconnecting;
    disconnect_reason_ = reason;
    RNET_LOG_INFO("peer {} ({}) disconnecting: {}", id_, address_.ToString(), disconnect_reason_);
}

}  // namespace rnet::net

#include "net/node.h"

#include <algorithm>
#include <format>

#include "consensus/genesis.h"
#include "util/hex.h"
#include "util/logging.h"

namespace rnet::net {

namespace {
constexpr size_t kReadChunk = 64 * 1024;
}

Status NodeConfig::Validate() const {
    if (max_inbound == 0 && max_outbound == 0) {
        return Err("node: at least one connection direction must be allowed");
    }
    if (handshake_timeout_ms <= 0) return Err("node: handshake timeout must be positive");
    if (peer_timeout_ms <= 0) return Err("node: peer timeout must be positive");
    return Status::Ok();
}

Node::Node(NodeConfig config, crypto::Hash256 addrman_key, uint64_t nonce)
    : config_(std::move(config)),
      params_(consensus::Params(config_.network)),
      magic_(params_.message_magic),
      nonce_(nonce),
      addresses_(addrman_key) {}

Node::~Node() { Stop(); }

Status Node::Start() {
    if (auto st = config_.Validate(); !st) return st;
    if (running_) return Status::Ok();

    if (config_.bind_address.port != 0) {
        auto listener = Socket::Listen(config_.bind_address);
        if (!listener) return Err(listener.error());
        listener_ = std::move(listener.value());
        auto local = listener_.LocalAddress();
        RNET_LOG_INFO("listening on {}", local.ok() ? local.value().ToString()
                                                    : config_.bind_address.ToString());
    }
    running_ = true;
    return Status::Ok();
}

void Node::Stop() {
    peers_.clear();
    connected_at_.clear();
    listener_.Close();
    running_ = false;
}

VersionMessage Node::LocalVersion(int64_t now_ms) const {
    VersionMessage v;
    v.protocol_version = kProtocolVersion;
    v.services = config_.services;
    v.timestamp = now_ms / 1000;
    v.nonce = nonce_;
    v.user_agent = config_.user_agent;
    v.worker_id = config_.worker_id;
    v.determinism_class = config_.determinism_class;
    // Both anchors travel in the handshake: a peer on a different round or under
    // different rules is not a peer at all.
    auto genesis = util::FromHexFixed<32>(params_.genesis_hash_hex);
    auto policy = util::FromHexFixed<32>(params_.policy_hash_hex);
    if (genesis) v.genesis_hash = genesis.value();
    if (policy) v.policy_hash = policy.value();
    return v;
}

bool Node::IsBanned(const NetAddress& address, int64_t now_ms) const {
    const auto it = banned_until_.find(address);
    return it != banned_until_.end() && it->second > now_ms;
}

void Node::Ban(const NetAddress& address, int64_t now_ms) {
    // Bans expire: a permanent list is itself something an attacker can grow, and
    // addresses change hands.
    banned_until_[address] = now_ms + config_.ban_duration_ms;
    RNET_LOG_INFO("banned {} until {}", address.ToString(), banned_until_[address]);
}

size_t Node::InboundCount() const {
    return static_cast<size_t>(std::count_if(peers_.begin(), peers_.end(),
                                             [](const auto& kv) { return kv.second->inbound(); }));
}

size_t Node::OutboundCount() const { return peers_.size() - InboundCount(); }

size_t Node::ReadyCount() const {
    return static_cast<size_t>(
        std::count_if(peers_.begin(), peers_.end(),
                      [](const auto& kv) { return kv.second->state() == PeerState::Ready; }));
}

std::vector<uint64_t> Node::ReadyPeerIds() const {
    std::vector<uint64_t> out;
    for (const auto& [id, peer] : peers_) {
        if (peer->state() == PeerState::Ready) out.push_back(id);
    }
    return out;
}

void Node::SetHandler(std::string command, MessageHandler handler) {
    handlers_[std::move(command)] = std::move(handler);
}

Status Node::ConnectTo(const NetAddress& address, int64_t now_ms) {
    if (OutboundCount() >= config_.max_outbound) return Err("node: outbound slots full");
    if (IsBanned(address, now_ms)) return Err("node: address is banned");

    for (const auto& [id, peer] : peers_) {
        if (peer->address() == address) return Err("node: already connected");
    }

    bool in_progress = false;
    auto socket = Socket::StartConnect(address, in_progress);
    if (!socket) {
        addresses_.MarkFailed(address, now_ms / 1000);
        return Err(socket.error());
    }

    const uint64_t id = next_peer_id_++;
    auto peer = std::make_unique<Peer>(id, std::move(socket.value()), address, /*inbound=*/false,
                                       magic_);
    if (!in_progress) {
        // Loopback usually connects immediately; speak first as the initiator.
        peer->MarkConnected();
        if (auto st = peer->BeginHandshake(LocalVersion(now_ms)); !st) return st;
    }
    connected_at_[id] = now_ms;
    peers_[id] = std::move(peer);
    return Status::Ok();
}

void Node::AcceptPending(int64_t now_ms) {
    if (!listener_.valid()) return;

    while (InboundCount() < config_.max_inbound) {
        NetAddress peer_address;
        bool would_block = false;
        auto socket = listener_.Accept(peer_address, would_block);
        if (would_block) return;
        if (!socket) {
            RNET_LOG_WARN("accept failed: {}", socket.error());
            return;
        }
        if (IsBanned(peer_address, now_ms)) {
            RNET_LOG_DEBUG("refused banned peer {}", peer_address.ToString());
            continue;   // socket closes as it goes out of scope
        }

        const uint64_t id = next_peer_id_++;
        auto peer = std::make_unique<Peer>(id, std::move(socket.value()), peer_address,
                                           /*inbound=*/true, magic_);
        RNET_LOG_INFO("inbound peer {} from {}", id, peer_address.ToString());
        connected_at_[id] = now_ms;
        peers_[id] = std::move(peer);
    }
}

Status Node::DispatchMessage(Peer& peer, const ReceivedMessage& message, int64_t now_ms) {
    // Checked here, where the peer's state reflects every message already handled
    // in this batch.
    if (auto st = peer.CheckCommandAllowed(message.command); !st) return st;

    // Handshake messages are handled here rather than by installed handlers: the
    // connection's own state must not depend on what a caller happened to register.
    if (message.command == cmd::kVersion) {
        auto remote = VersionMessage::Deserialize(message.payload);
        if (!remote) {
            peer.Misbehaving(20, remote.error());
            return Err(remote.error());
        }
        return peer.HandleVersion(remote.value(), LocalVersion(now_ms), nonce_);
    }
    if (message.command == cmd::kVerack) {
        if (auto st = peer.HandleVerack(); !st) return st;
        RNET_LOG_INFO("peer {} ready ({}, class {})", peer.id(), peer.version().user_agent,
                      peer.version().determinism_class);
        // A peer that just proved itself is worth remembering and worth asking
        // for more addresses.
        addresses_.MarkGood(peer.address(), now_ms / 1000);
        return peer.Send(cmd::kGetAddr, {});
    }
    if (message.command == cmd::kPing) {
        return peer.Send(cmd::kPong, message.payload);
    }
    if (message.command == cmd::kPong) return Status::Ok();

    if (message.command == cmd::kGetAddr) {
        AddrMessage reply;
        reply.addresses = addresses_.GetAddresses(kMaxAddrPerMessage, static_cast<uint64_t>(now_ms));
        return peer.Send(cmd::kAddr, reply.Serialize());
    }
    // Object exchange is the node's own business — storage, hash verification and
    // relay must not depend on what a caller happened to register. A registered
    // handler still runs, but afterwards and as a notification: by then the bytes
    // have been verified, so the application layer never sees unchecked data.
    if (message.command == cmd::kInv) {
        if (auto st = HandleInv(peer, message, now_ms); !st) return st;
        return NotifyHandler(peer, message, now_ms);
    }
    if (message.command == cmd::kGetData) {
        if (auto st = HandleGetData(peer, message); !st) return st;
        return NotifyHandler(peer, message, now_ms);
    }
    if (message.command == cmd::kObject) {
        if (auto st = HandleObject(peer, message, now_ms); !st) return st;
        return NotifyHandler(peer, message, now_ms);
    }
    if (message.command == cmd::kGetCorpus) {
        if (auto st = HandleGetCorpus(peer, message); !st) return st;
        return NotifyHandler(peer, message, now_ms);
    }
    if (message.command == cmd::kCorpus) {
        if (auto st = HandleCorpus(peer, message); !st) return st;
        return NotifyHandler(peer, message, now_ms);
    }
    if (message.command == cmd::kGetChunks) {
        if (auto st = HandleGetChunks(peer, message); !st) return st;
        return NotifyHandler(peer, message, now_ms);
    }
    if (message.command == cmd::kChunk) {
        if (auto st = HandleChunk(peer, message, now_ms); !st) return st;
        return NotifyHandler(peer, message, now_ms);
    }

    if (message.command == cmd::kAddr) {
        auto addr = AddrMessage::Deserialize(message.payload);
        if (!addr) {
            peer.Misbehaving(20, addr.error());
            return Err(addr.error());
        }
        // Bucketed by the SOURCE as well as the address, so a peer gossiping
        // thousands of entries cannot displace the rest of the database.
        addresses_.AddMany(addr.value().addresses, peer.address(), now_ms / 1000);
        return Status::Ok();
    }

    const auto handler = handlers_.find(message.command);
    if (handler == handlers_.end()) {
        // Unknown commands are ignored rather than fatal: that is what lets the
        // protocol gain messages without splitting the network.
        RNET_LOG_DEBUG("peer {} sent unhandled '{}'", peer.id(), message.command);
        return Status::Ok();
    }
    return handler->second(peer, message, now_ms);
}

Status Node::ServicePeer(Peer& peer, PollEntry& entry, int64_t now_ms) {
    if (entry.error) {
        peer.Disconnect("socket error");
        return Status::Ok();
    }

    if (peer.state() == PeerState::Connecting) {
        if (!entry.writable) return Status::Ok();
        if (auto st = peer.socket().ConnectResult(); !st) {
            peer.Disconnect(st.error());
            addresses_.MarkFailed(peer.address(), now_ms / 1000);
            return Status::Ok();
        }
        peer.MarkConnected();
        if (auto st = peer.BeginHandshake(LocalVersion(now_ms)); !st) return st;
    }

    if (entry.writable && peer.HasPendingSend()) {
        const auto pending = peer.PendingSend();
        auto written = peer.socket().Write(pending);
        if (!written) {
            peer.Disconnect(written.error());
            return Status::Ok();
        }
        if (written.value().closed) {
            peer.Disconnect("peer closed while writing");
            return Status::Ok();
        }
        peer.ConsumeSent(written.value().bytes);
    }

    if (entry.readable) {
        std::vector<uint8_t> buffer(kReadChunk);
        auto read = peer.socket().Read(buffer);
        if (!read) {
            peer.Disconnect(read.error());
            return Status::Ok();
        }
        if (read.value().closed) {
            peer.Disconnect("peer closed the connection");
            return Status::Ok();
        }
        if (read.value().bytes > 0) {
            peer.RecordActivity(now_ms);
            auto messages = peer.ReceiveBytes(
                std::span<const uint8_t>(buffer.data(), read.value().bytes));
            if (!messages) return Status::Ok();   // peer already marked for teardown

            for (const auto& message : messages.value()) {
                if (auto st = DispatchMessage(peer, message, now_ms); !st) {
                    // A handler rejecting a message is misbehaviour by the sender,
                    // not a failure of the node.
                    peer.Misbehaving(10, st.error());
                    if (peer.should_disconnect()) break;
                }
            }
        }
    }
    return Status::Ok();
}

void Node::Maintenance(int64_t now_ms) {
    for (auto& [id, peer] : peers_) {
        const int64_t since_connect = now_ms - connected_at_[id];

        // A peer that connects and says nothing costs it nothing and costs us a
        // slot.
        if (peer->state() != PeerState::Ready && since_connect > config_.handshake_timeout_ms) {
            peer->Disconnect("handshake timeout");
            continue;
        }
        if (peer->state() == PeerState::Ready && peer->last_receive() > 0 &&
            now_ms - peer->last_receive() > config_.peer_timeout_ms) {
            peer->Disconnect("peer timeout");
        }
    }
}

void Node::DropDisconnected(int64_t now_ms) {
    for (auto it = peers_.begin(); it != peers_.end();) {
        if (!it->second->should_disconnect()) {
            ++it;
            continue;
        }
        // Only a peer that actually earned a ban gets one; an ordinary
        // disconnection is not evidence of hostility.
        if (it->second->misbehaviour() >= kBanThreshold) Ban(it->second->address(), now_ms);
        // A node's own address is not a peer and never will be. Left in the
        // database it is dialled again on the next tick, forever.
        if (it->second->is_self()) addresses_.Forget(it->second->address());
        // A departed peer cannot deliver what it owed; release those requests and
        // abandon any half-assembled transfer, which is worthless without its
        // remaining chunks.
        objects_.ForgetPeer(it->first);
        for (auto b = bulk_.begin(); b != bulk_.end();) {
            b = (b->second.peer_id == it->first) ? bulk_.erase(b) : std::next(b);
        }
        chunk_feeds_.erase(it->first);
        corpus_feeds_.erase(it->first);
        connected_at_.erase(it->first);
        it = peers_.erase(it);
    }
}

Status Node::Tick(int64_t now_ms, int timeout_ms) {
    if (!running_) return Err("node: not started");

    AcceptPending(now_ms);

    std::vector<PollEntry> entries;
    std::vector<uint64_t> ids;
    entries.reserve(peers_.size() + 1);

    for (auto& [id, peer] : peers_) {
        if (peer->should_disconnect()) continue;
        PollEntry entry;
        entry.fd = peer->socket().fd();
        entry.want_read = true;
        entry.want_write = peer->HasPendingSend() || peer->state() == PeerState::Connecting;
        entries.push_back(entry);
        ids.push_back(id);
    }

    if (!entries.empty()) {
        auto ready = PollSockets(entries, timeout_ms);
        if (!ready) return Err(ready.error());

        for (size_t i = 0; i < entries.size(); ++i) {
            const auto it = peers_.find(ids[i]);
            if (it == peers_.end()) continue;
            if (auto st = ServicePeer(*it->second, entries[i], now_ms); !st) {
                RNET_LOG_WARN("peer {}: {}", ids[i], st.error());
            }
        }
    }

    Maintenance(now_ms);
    ServeChunkFeeds();
    ServeCorpusFeeds();
    RetryExpiredRequests(now_ms);
    DropDisconnected(now_ms);
    return Status::Ok();
}

Status Node::HandleInv(Peer& peer, const ReceivedMessage& message, int64_t now_ms) {
    auto inv = InvMessage::Deserialize(message.payload);
    if (!inv) {
        peer.Misbehaving(20, inv.error());
        return Err(inv.error());
    }

    // Announcing is free, so it is bounded here and causes no storage on its own:
    // only the entries this node lacks and is not already fetching are requested.
    const auto wanted = objects_.NoteAnnouncement(peer.id(), inv.value().entries, now_ms);
    if (wanted.empty()) return Status::Ok();

    InvMessage request;
    for (const auto& entry : wanted) {
        if (!objects_.NoteRequested(peer.id(), entry, now_ms)) continue;   // owes too much already
        request.entries.push_back(entry);
    }
    if (request.entries.empty()) return Status::Ok();
    return peer.Send(cmd::kGetData, request.Serialize());
}

Status Node::HandleGetData(Peer& peer, const ReceivedMessage& message) {
    auto request = InvMessage::Deserialize(message.payload);
    if (!request) {
        peer.Misbehaving(20, request.error());
        return Err(request.error());
    }

    InvMessage missing;
    for (const auto& entry : request.value().entries) {
        const auto* object = objects_.Peek(entry.hash);
        if (object == nullptr) {
            missing.entries.push_back(entry);
            continue;
        }
        // Objects beyond one message travel as chunks. Saying nothing would leave
        // the requester waiting out its timeout for something that was never
        // coming in this form; `notfound` would be a lie. It is refused, with the
        // reason, so an honest peer switches to `getchunks` at once.
        if (object->data.size() > kMaxMessageSize - kChunkMetadataAllowance) {
            RejectMessage reject;
            reject.command = cmd::kGetData;
            reject.code = RejectCode::NotServed;
            // The hash goes in the reason: `reject` carries no object field, and a
            // requester with several fetches outstanding must know which one this
            // refers to.
            reject.reason = "use getchunks for " + util::ToHex(entry.hash);
            if (auto st = peer.Send(cmd::kReject, reject.Serialize()); !st) return st;
            continue;
        }

        canon::Writer w;
        w.U16(static_cast<uint16_t>(object->type));
        w.Hash(object->hash);
        w.VarBytes(object->data);
        if (auto st = peer.Send(cmd::kObject, w.data()); !st) return st;
    }
    if (!missing.entries.empty()) {
        (void)peer.Send(cmd::kNotFound, missing.Serialize());
    }
    return Status::Ok();
}

Status Node::HandleObject(Peer& peer, const ReceivedMessage& message, int64_t now_ms) {
    canon::Reader r(message.payload);
    auto type = r.U16();
    if (!type) { peer.Misbehaving(20, type.error()); return Err(type.error()); }
    if (type.value() < 1 || type.value() > 6) {
        peer.Misbehaving(20, "object: unknown type");
        return Err("object: unknown type");
    }
    auto hash = r.Hash();
    if (!hash) { peer.Misbehaving(20, hash.error()); return Err(hash.error()); }
    auto data = r.VarBytes();
    if (!data) { peer.Misbehaving(20, data.error()); return Err(data.error()); }
    if (auto st = r.ExpectExhausted(); !st) {
        peer.Misbehaving(20, st.error());
        return Err(st.error());
    }

    // Admitted only if the bytes hash to the identity they were requested under.
    // Delivering something else is substitution, not a mistake.
    auto stored = objects_.Put(static_cast<InvType>(type.value()), hash.value(),
                               std::move(data.value()), now_ms);
    if (!stored) {
        peer.Misbehaving(50, stored.error());
        return Err(stored.error());
    }
    if (!stored.value()) return Status::Ok();   // already held

    // Relay onward so the object spreads without every node fetching from the
    // origin.
    AnnounceToOthers(peer.id(), static_cast<InvType>(type.value()), hash.value());
    return Status::Ok();
}

Status Node::HandleGetChunks(Peer& peer, const ReceivedMessage& message) {
    auto request = GetChunksMessage::Deserialize(message.payload);
    if (!request) {
        peer.Misbehaving(20, request.error());
        return Err(request.error());
    }
    if (auto st = request.value().Validate(); !st) {
        peer.Misbehaving(20, st.error());
        return Err(st.error());
    }

    auto size = ObjectSize(request.value().hash);
    if (!size) {
        InvMessage missing;
        missing.entries.push_back(InvEntry{request.value().type, request.value().hash});
        return peer.Send(cmd::kNotFound, missing.Serialize());
    }
    const uint64_t total_chunks = (size.value() + kBulkChunkSize - 1) / kBulkChunkSize;

    // A peer asking for chunk 10^9 of a two-chunk object is not making a mistake
    // it can be helped with.
    if (request.value().first_chunk >= total_chunks) {
        peer.Misbehaving(20, "getchunks beyond the end of the object");
        return Err("getchunks: first_chunk past the end");
    }

    // Recorded, not served here. A 983 MB payload is 1,877 chunks and the send
    // queue holds 8 MiB — writing them all into it would overflow the queue after
    // sixteen, and the overflow rule disconnects the peer this node is in the
    // middle of answering. The feed is drained across ticks instead, as the
    // socket accepts bytes.
    ChunkFeed feed;
    feed.type = request.value().type;
    feed.hash = request.value().hash;
    feed.next_index = request.value().first_chunk;
    feed.last_index =
        std::min(total_chunks, request.value().first_chunk + request.value().chunk_count);
    chunk_feeds_[peer.id()] = feed;
    return Status::Ok();
}

Result<uint64_t> Node::ObjectSize(const crypto::Hash256& hash) const {
    if (const auto* stored = objects_.Peek(hash); stored != nullptr) {
        return static_cast<uint64_t>(stored->data.size());
    }
    if (weights_ != nullptr && weights_->Has(hash)) return weights_->SizeOf(hash);
    return Err("node: object is not held");
}

Result<size_t> Node::ReadObjectRange(const crypto::Hash256& hash, uint64_t offset,
                                     std::span<uint8_t> out) const {
    if (const auto* stored = objects_.Peek(hash); stored != nullptr) {
        if (offset >= stored->data.size()) return size_t{0};
        const size_t length =
            std::min<size_t>(out.size(), stored->data.size() - static_cast<size_t>(offset));
        std::copy(stored->data.begin() + static_cast<ptrdiff_t>(offset),
                  stored->data.begin() + static_cast<ptrdiff_t>(offset) +
                      static_cast<ptrdiff_t>(length),
                  out.begin());
        return length;
    }
    if (weights_ != nullptr && weights_->Has(hash)) return weights_->ReadRange(hash, offset, out);
    return Err("node: object is not held");
}

Status Node::RequestCorpusChunk(uint64_t peer_id, uint64_t chunk_index,
                                const crypto::Hash256& dataset_root) {
    if (corpus_ready_.count(chunk_index) != 0) return Status::Ok();   // already here
    if (corpus_fetches_.count(chunk_index) != 0) return Status::Ok(); // already asked

    auto peer = peers_.find(peer_id);
    if (peer == peers_.end()) return Err("corpus: no such peer");

    GetCorpusMessage request;
    request.first_chunk = chunk_index;
    request.chunk_count = 1;
    if (auto st = peer->second->Send(cmd::kGetCorpus, request.Serialize()); !st) return st;

    CorpusFetch fetch;
    fetch.dataset_root = dataset_root;
    corpus_fetches_.emplace(chunk_index, std::move(fetch));
    return Status::Ok();
}

// A part of a corpus chunk arrived.
//
// The assembler is created from part zero, because that is the part carrying the
// inclusion proof, and a chunk assembled without one could not be checked against
// anything. Parts for a chunk nobody asked for are dropped rather than buffered:
// otherwise a peer could fill this node's memory by sending corpus it was never
// asked for, which is the same shape as the unsolicited-object problem.
Status Node::HandleCorpus(Peer& peer, const ReceivedMessage& message) {
    auto part = CorpusMessage::Deserialize(message.payload);
    if (!part) {
        peer.Misbehaving(20, part.error());
        return Err(part.error());
    }
    const uint64_t index = part.value().chunk_index;

    auto fetch = corpus_fetches_.find(index);
    if (fetch == corpus_fetches_.end()) {
        peer.Misbehaving(10, "corpus part for a chunk this node did not request");
        return Status::Ok();
    }

    if (fetch->second.assembler == nullptr) {
        if (part.value().part != 0) return Status::Ok();   // waiting for the proof
        fetch->second.assembler = std::make_unique<CorpusAssembler>(
            index, part.value().chunk_bytes, part.value().proof);
    }
    if (auto st = fetch->second.assembler->Accept(part.value().part, part.value().data); !st) {
        peer.Misbehaving(20, st.error());
        corpus_fetches_.erase(fetch);
        return Err(st.error());
    }
    if (!fetch->second.assembler->Complete()) return Status::Ok();

    auto bytes = fetch->second.assembler->Finish(fetch->second.dataset_root);
    if (!bytes) {
        // Served bytes that do not prove membership in the pinned root. That is
        // not a mistake a healthy node makes.
        peer.Misbehaving(50, bytes.error());
        corpus_fetches_.erase(fetch);
        return Err(bytes.error());
    }
    corpus_ready_[index] = std::move(bytes.value());
    corpus_fetches_.erase(fetch);
    RNET_LOG_DEBUG("corpus: chunk {} verified against the pinned root", index);
    return Status::Ok();
}

const std::vector<uint8_t>* Node::TakeCorpusChunk(uint64_t chunk_index) {
    auto it = corpus_ready_.find(chunk_index);
    return it == corpus_ready_.end() ? nullptr : &it->second;
}

bool Node::CorpusChunkInFlight(uint64_t chunk_index) const {
    return corpus_fetches_.count(chunk_index) != 0;
}

Status Node::HandleGetCorpus(Peer& peer, const ReceivedMessage& message) {
    auto request = GetCorpusMessage::Deserialize(message.payload);
    if (!request) {
        peer.Misbehaving(20, request.error());
        return Err(request.error());
    }
    if (corpus_ == nullptr) {
        // Said rather than ignored: a worker told "I do not serve the corpus" asks
        // someone else, while a worker told nothing waits out its timeout.
        return peer.Send(cmd::kNotFound, message.payload);
    }
    if (request.value().first_chunk >= corpus_->n_chunks()) {
        peer.Misbehaving(20, "getcorpus beyond the end of the corpus");
        return Err("getcorpus: first_chunk past the end");
    }

    CorpusFeed feed;
    feed.next_chunk = request.value().first_chunk;
    feed.last_chunk = std::min<uint64_t>(corpus_->n_chunks(),
                                         request.value().first_chunk + request.value().chunk_count);
    feed.next_part = 0;
    corpus_feeds_[peer.id()] = feed;
    return Status::Ok();
}

void Node::ServeCorpusFeeds() {
    if (corpus_ == nullptr) {
        corpus_feeds_.clear();
        return;
    }
    for (auto it = corpus_feeds_.begin(); it != corpus_feeds_.end();) {
        const auto peer = peers_.find(it->first);
        if (peer == peers_.end() || peer->second->state() != PeerState::Ready) {
            it = corpus_feeds_.erase(it);
            continue;
        }
        auto& feed = it->second;

        while (feed.next_chunk < feed.last_chunk &&
               peer->second->pending_send_bytes() < kSendQueueHighWater) {
            auto chunk_bytes = corpus_->ChunkBytes(feed.next_chunk);
            if (!chunk_bytes) break;
            const uint32_t total_parts = static_cast<uint32_t>(
                (chunk_bytes.value() + kCorpusPartBytes - 1) / kCorpusPartBytes);

            CorpusMessage part;
            part.chunk_index = feed.next_chunk;
            part.chunk_bytes = chunk_bytes.value();
            part.part = feed.next_part;
            // The proof rides on part zero. Sending it once makes it unambiguous
            // which proof the assembled chunk was checked against.
            if (feed.next_part == 0) {
                auto proof = corpus_->ProofForChunk(feed.next_chunk);
                if (!proof) break;
                part.proof = std::move(proof.value());
            }

            const uint64_t offset = static_cast<uint64_t>(feed.next_part) * kCorpusPartBytes;
            const size_t length = static_cast<size_t>(
                std::min<uint64_t>(kCorpusPartBytes, chunk_bytes.value() - offset));
            part.data.resize(length);
            // Read a slice at a time from disk: a four-megabyte chunk is never
            // resident just because a peer asked for it, and neither is the corpus.
            auto got = corpus_->ReadChunkRange(feed.next_chunk, offset, part.data);
            if (!got || got.value() != length) break;

            if (auto st = peer->second->Send(cmd::kCorpus, part.Serialize()); !st) break;
            if (++feed.next_part >= total_parts) {
                feed.next_part = 0;
                ++feed.next_chunk;
            }
        }
        it = (feed.next_chunk >= feed.last_chunk) ? corpus_feeds_.erase(it) : std::next(it);
    }
}

void Node::ServeChunkFeeds() {
    for (auto it = chunk_feeds_.begin(); it != chunk_feeds_.end();) {
        const auto peer = peers_.find(it->first);
        if (peer == peers_.end() || peer->second->state() != PeerState::Ready) {
            it = chunk_feeds_.erase(it);
            continue;
        }
        auto& feed = it->second;

        auto size = ObjectSize(feed.hash);
        if (!size) {
            // Evicted or pruned mid-transfer. The requester's own request will time
            // out and go elsewhere, which is the honest outcome — this node no
            // longer has what it promised.
            it = chunk_feeds_.erase(it);
            continue;
        }
        const uint64_t total_chunks = (size.value() + kBulkChunkSize - 1) / kBulkChunkSize;

        while (feed.next_index < feed.last_index &&
               peer->second->pending_send_bytes() < kSendQueueHighWater) {
            const uint64_t offset = feed.next_index * kBulkChunkSize;
            if (offset >= size.value()) break;
            const size_t length =
                static_cast<size_t>(std::min<uint64_t>(kBulkChunkSize, size.value() - offset));

            ChunkMessage chunk;
            chunk.type = feed.type;
            chunk.hash = feed.hash;
            chunk.index = feed.next_index;
            chunk.total_chunks = total_chunks;
            chunk.data.resize(length);
            // Read a slice at a time, from memory or from disk. A two-gigabyte model
            // is never resident just because a peer asked for it.
            auto got = ReadObjectRange(feed.hash, offset, chunk.data);
            if (!got || got.value() != length) break;
            if (auto st = peer->second->Send(cmd::kChunk, chunk.Serialize()); !st) break;
            ++feed.next_index;
        }

        it = (feed.next_index >= feed.last_index) ? chunk_feeds_.erase(it) : std::next(it);
    }
}

Status Node::HandleChunk(Peer& peer, const ReceivedMessage& message, int64_t now_ms) {
    auto chunk = ChunkMessage::Deserialize(message.payload);
    if (!chunk) {
        peer.Misbehaving(20, chunk.error());
        return Err(chunk.error());
    }
    if (auto st = chunk.value().Validate(); !st) {
        peer.Misbehaving(20, st.error());
        return Err(st.error());
    }
    const crypto::Hash256 hash = chunk.value().hash;

    // Unsolicited chunks are how a peer makes the node allocate on demand.
    const auto transfer = bulk_.find(hash);
    if (transfer == bulk_.end()) {
        peer.Misbehaving(10, "chunk for an object that was never requested");
        return Err("chunk: unsolicited");
    }
    // One transfer, one source. Otherwise a second peer can poison an assembly it
    // was never asked to contribute to, and the whole-object hash check at the end
    // would have no way to say which of them did it.
    if (transfer->second.peer_id != peer.id()) {
        peer.Misbehaving(10, "chunk from a peer this transfer was not requested from");
        return Err("chunk: wrong source");
    }
    const InvType type = transfer->second.type;

    if (auto st = transfer->second.assembler->Accept(chunk.value()); !st) {
        peer.Misbehaving(20, st.error());
        return Err(st.error());
    }
    if (!transfer->second.assembler->Complete()) return Status::Ok();

    // Verified as a whole, never chunk by chunk: a chunk carries no proof of its
    // own, only the finished object does.
    auto assembled = transfer->second.assembler->Finish();
    bulk_.erase(transfer);
    if (!assembled) {
        // Every chunk arrived and the object still does not hash correctly, so
        // this peer substituted content somewhere in the middle.
        peer.Misbehaving(50, assembled.error());
        objects_.DropRequest(hash);
        return Err(assembled.error());
    }

    // Checkpoint weights go to disk. Putting a two-gigabyte model into the
    // in-memory store would defeat the reason the weights store exists.
    if (type == InvType::Checkpoint && weights_ != nullptr) {
        if (auto st = weights_->Put(hash, assembled.value()); !st) return Err(st.error());
        AnnounceToOthers(peer.id(), type, hash);
        return Status::Ok();
    }

    auto stored = objects_.Put(type, hash, std::move(assembled.value()), now_ms);
    if (!stored) return Err(stored.error());
    if (stored.value()) AnnounceToOthers(peer.id(), type, hash);
    return Status::Ok();
}

Status Node::RequestBulk(uint64_t peer_id, InvType type, const crypto::Hash256& hash,
                         uint64_t total_bytes, int64_t now_ms) {
    if (objects_.Has(hash)) return Err("bulk: already held");
    if (bulk_.find(hash) != bulk_.end()) return Err("bulk: already in flight");
    if (total_bytes == 0) return Err("bulk: refusing a zero-length transfer");

    const auto peer = peers_.find(peer_id);
    if (peer == peers_.end() || peer->second->state() != PeerState::Ready) {
        return Err("bulk: no such established peer");
    }

    const uint64_t total_chunks = (total_bytes + kBulkChunkSize - 1) / kBulkChunkSize;

    BulkTransfer transfer;
    transfer.peer_id = peer_id;
    transfer.type = type;
    transfer.started_at = now_ms;
    transfer.assembler = std::make_unique<ChunkAssembler>(type, hash, total_bytes);
    bulk_[hash] = std::move(transfer);

    GetChunksMessage request;
    request.type = type;
    request.hash = hash;
    request.first_chunk = 0;
    request.chunk_count = total_chunks;
    if (auto st = peer->second->Send(cmd::kGetChunks, request.Serialize()); !st) {
        bulk_.erase(hash);
        return st;
    }
    return Status::Ok();
}

bool Node::BulkInFlight(const crypto::Hash256& hash) const {
    return bulk_.find(hash) != bulk_.end();
}

Status Node::NotifyHandler(Peer& peer, const ReceivedMessage& message, int64_t now_ms) {
    const auto handler = handlers_.find(message.command);
    if (handler == handlers_.end()) return Status::Ok();
    return handler->second(peer, message, now_ms);
}

void Node::AnnounceToOthers(uint64_t origin_peer, InvType type, const crypto::Hash256& hash) {
    InvMessage announcement;
    announcement.entries.push_back(InvEntry{type, hash});
    const auto serialized = announcement.Serialize();
    for (auto& [id, other] : peers_) {
        if (id == origin_peer || other->state() != PeerState::Ready) continue;
        (void)other->Send(cmd::kInv, serialized);
    }
}

void Node::RetryExpiredRequests(int64_t now_ms) {
    for (const auto& request : objects_.ExpiredRequests(now_ms)) {
        // A peer that announces and never delivers stalls the node silently. The
        // request goes elsewhere and the peer that failed is scored.
        const auto failed = peers_.find(request.peer_id);
        if (failed != peers_.end()) failed->second->Misbehaving(5, "announced but did not deliver");
        objects_.DropRequest(request.hash);

        for (uint64_t candidate : objects_.PeersAnnouncing(request.hash)) {
            if (candidate == request.peer_id) continue;
            const auto peer = peers_.find(candidate);
            if (peer == peers_.end() || peer->second->state() != PeerState::Ready) continue;

            InvEntry entry{request.type, request.hash};
            if (!objects_.NoteRequested(candidate, entry, now_ms)) continue;
            InvMessage retry;
            retry.entries.push_back(entry);
            (void)peer->second->Send(cmd::kGetData, retry.Serialize());
            break;
        }
    }
}

Result<crypto::Hash256> Node::Publish(InvType type, std::vector<uint8_t> data, int64_t now_ms) {
    const crypto::Hash256 hash = crypto::Sha3_256(data);
    auto stored = objects_.Put(type, hash, std::move(data), now_ms);
    if (!stored) return Err(stored.error());

    InvMessage announcement;
    announcement.entries.push_back(InvEntry{type, hash});
    Broadcast(cmd::kInv, announcement.Serialize());
    return hash;
}

size_t Node::Broadcast(std::string_view command, std::span<const uint8_t> payload) {
    size_t sent = 0;
    for (auto& [id, peer] : peers_) {
        if (peer->state() != PeerState::Ready) continue;
        if (peer->Send(command, payload)) ++sent;
    }
    return sent;
}

}  // namespace rnet::net

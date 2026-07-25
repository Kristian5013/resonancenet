#include "protocol/workerservice.h"

#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <format>

#include "consensus/genesis.h"
#include "ipc/cbor.h"
#include "util/hex.h"
#include <set>

#include "util/logging.h"

namespace rnet::protocol {

std::string IpcErrorName(IpcError error) {
    switch (error) {
        case IpcError::BadSchema:      return "bad-schema";
        case IpcError::OutOfSequence:  return "out-of-sequence";
        case IpcError::AnchorMismatch: return "anchor-mismatch";
        case IpcError::NotAssigned:    return "not-assigned";
        case IpcError::Rejected:       return "rejected";
        case IpcError::Unavailable:    return "unavailable";
    }
    return "unknown";
}

namespace {

// Hashing a gigabyte through a mapping, a slice at a time. The whole point of the
// sealed descriptor is that these bytes cannot change while this runs.
constexpr size_t kHashSlice = 4u * 1024 * 1024;

std::span<const uint8_t> AsBytes(const crypto::Hash256& hash) {
    return std::span<const uint8_t>(hash.data(), hash.size());
}

}  // namespace

WorkerService::WorkerService(ipc::Server& server, net::Node& node, Participant& participant)
    : server_(server), node_(node), participant_(participant) {}

Status WorkerService::Refuse(ipc::Client& client, uint64_t request_id, IpcError error,
                             std::string_view reason) {
    ++refused_;
    RNET_LOG_INFO("ipc: worker {} refused ({}): {}", client.id(), IpcErrorName(error), reason);

    auto payload = ipc::CborWriter()
                       .BeginMap(2)
                       .Key("code").Uint(static_cast<uint64_t>(error))
                       .Key("reason").Text(reason.substr(0, 256))
                       .Take();
    if (!payload) return Err(payload.error());

    ipc::Frame frame;
    frame.type = ipc::MessageType::Error;
    frame.request_id = request_id;
    frame.payload = std::move(payload.value());
    return client.Send(frame);
}

Status WorkerService::Poll(int64_t now_ms) {
    auto delivered = server_.Poll(now_ms);
    if (!delivered) return Err(delivered.error());

    for (auto& delivery : delivered.value()) {
        auto* client = server_.Find(delivery.client_id);
        if (client == nullptr || client->should_close()) continue;
        if (auto st = Dispatch(*client, delivery.received, now_ms); !st) {
            // Reaching here means the request could not even be answered — a
            // refusal returns Ok, because a refusal IS an answer. Leaving the
            // connection open would make the worker wait out its handshake
            // timeout for a reply that is never coming, turning a legible fault
            // into fifteen seconds of silence.
            RNET_LOG_WARN("ipc: worker {} could not be answered: {}", delivery.client_id,
                          st.error());
            client->Close(st.error());
        }
    }

    // Assignments belonging to departed workers are dropped, so a reconnecting
    // process cannot inherit one it never received.
    for (auto it = assignments_.begin(); it != assignments_.end();) {
        it = server_.Find(it->first) == nullptr ? assignments_.erase(it) : std::next(it);
    }
    server_.Sweep(now_ms);
    return Status::Ok();
}

Status WorkerService::Dispatch(ipc::Client& client, ipc::ReceivedFrame& received, int64_t now_ms) {
    const auto& frame = received.frame;

    // Nothing but the handshake is processed before the handshake completes, for
    // the same reason as on the network side: an unidentified peer must not be
    // able to drive the node's logic.
    if (!client.handshake_done() && frame.type != ipc::MessageType::Hello) {
        (void)Refuse(client, frame.request_id, IpcError::OutOfSequence,
                     "the first message must be hello");
        client.Close("spoke before the handshake");
        return Err("ipc: message before handshake");
    }

    switch (frame.type) {
        case ipc::MessageType::Hello:
            return OnHello(client, frame);
        case ipc::MessageType::GetAssignment:
            return OnGetAssignment(client, frame, now_ms);
        case ipc::MessageType::SubmitContribution:
            return OnSubmitContribution(client, received, now_ms);
        case ipc::MessageType::GetStatus:
            return OnGetStatus(client, frame);
        case ipc::MessageType::SubmitWeights:
            return OnSubmitWeights(client, received, now_ms);
        case ipc::MessageType::GetWeights:
            return OnGetWeights(client, frame);
        default:
            return Refuse(client, frame.request_id, IpcError::BadSchema,
                          "this daemon does not implement that message yet");
    }
}

Status WorkerService::OnHello(ipc::Client& client, const ipc::Frame& frame) {
    if (client.handshake_done()) {
        client.Close("duplicate hello");
        return Err("ipc: hello received twice");
    }

    auto message = ipc::CborParse(frame.payload);
    if (!message) return Refuse(client, frame.request_id, IpcError::BadSchema, message.error());

    auto genesis_hash = message.value().GetHash("genesis_hash");
    if (!genesis_hash) {
        return Refuse(client, frame.request_id, IpcError::BadSchema, genesis_hash.error());
    }
    auto policy_hash = message.value().GetHash("policy_hash");
    if (!policy_hash) {
        return Refuse(client, frame.request_id, IpcError::BadSchema, policy_hash.error());
    }
    auto user_agent = message.value().GetText("user_agent");
    if (!user_agent) {
        return Refuse(client, frame.request_id, IpcError::BadSchema, user_agent.error());
    }

    // Note what is NOT in this message: the worker does not state its own
    // worker_id. That value belongs to the node's configuration, and a worker that
    // could choose it could submit under someone else's name.
    const auto& params = participant_.params();
    const auto genesis_hex = util::ToHex(genesis_hash.value());
    const auto policy_hex = util::ToHex(policy_hash.value());

    // Compared, never adopted. A worker built against a different round would
    // train a different model and call the result a contribution; there is no
    // negotiation and no downgrade.
    if (genesis_hex != params.genesis_hash_hex) {
        (void)Refuse(client, frame.request_id, IpcError::AnchorMismatch,
                     std::format("worker genesis {} != node genesis {}", genesis_hex,
                                 params.genesis_hash_hex));
        client.Close("different genesis");
        return Err("ipc: worker is on a different round");
    }
    if (policy_hex != params.policy_hash_hex) {
        (void)Refuse(client, frame.request_id, IpcError::AnchorMismatch,
                     std::format("worker policy {} != node policy {}", policy_hex,
                                 params.policy_hash_hex));
        client.Close("different policy");
        return Err("ipc: worker is under different rules");
    }

    auto genesis_weights = util::FromHexFixed<32>(params.genesis_weights_hash_hex);
    if (!genesis_weights) {
        return Refuse(client, frame.request_id, IpcError::Unavailable,
                      "this network has no pinned genesis weights, so there is no agreed state to "
                      "start training from");
    }

    // The reply carries the round and policy CONTAINERS, not their fields. The
    // worker re-derives everything from bytes it can hash against its own anchors,
    // so nothing it uses arrives as a bare number this daemon could get wrong.
    const auto round_container = consensus::GenesisContainer(params.network);
    const auto policy_container = consensus::PolicyContainer(params.network);

    // Canonical key order: shorter encoded keys first, then bytewise. Sorting by
    // eye is exactly what the writer exists to catch, and it caught this.
    auto payload = ipc::CborWriter()
                       .BeginMap(6)
                       .Key("round").Bytes(round_container)
                       .Key("policy").Bytes(policy_container)
                       .Key("n_params").Uint(params.round.model.ParameterCount())
                       .Key("worker_id").Uint(participant_.worker_id())
                       .Key("node_time_ms").Uint(static_cast<uint64_t>(client.connected_at()))
                       .Key("genesis_weights_hash").Bytes(genesis_weights.value())
                       .Take();
    if (!payload) return Err(payload.error());

    client.MarkHandshakeDone();
    RNET_LOG_INFO("ipc: worker {} ready ({}), submitting as worker id {}", client.id(),
                  user_agent.value().substr(0, 64), participant_.worker_id());

    ipc::Frame reply;
    reply.type = ipc::MessageType::HelloOk;
    reply.request_id = frame.request_id;
    reply.payload = std::move(payload.value());
    return client.Send(reply);
}

// The node aggregated an update but cannot apply it: the weights live in the
// worker's GPU memory. Offered as an ANSWER to a poll rather than pushed, so the
// worker stays the only initiator and neither side grows a second state machine.
Status WorkerService::OfferPendingApply(ipc::Client& client, const ipc::Frame& frame,
                                        int64_t now_ms) {
    const auto* pending = participant_.Pending();
    if (pending == nullptr) return Err("ipc: nothing staged");

    // One worker at a time. Two applying the same update would each report a hash
    // and the node would have to choose between them — a choice it must never have
    // to make, because both would look equally valid.
    if (applying_client_ != 0 && applying_client_ != client.id()) {
        return Err("ipc: another worker is already applying this update");
    }

    // Narrowed to int32 with a checked cast rather than assumed to fit. The
    // aggregate is a mean of int8 values, so with the policy's momentum and
    // learning rate the update stays around three digits — int32 leaves six orders
    // of magnitude of headroom, and a value that still does not fit means something
    // is wrong rather than that the format is tight.
    std::vector<uint8_t> narrowed(pending->update.size() * 4);
    for (size_t i = 0; i < pending->update.size(); ++i) {
        const int64_t v = pending->update[i];
        if (v > INT32_MAX || v < INT32_MIN) {
            return Err(std::format("ipc: update value {} at index {} does not fit in int32", v, i));
        }
        const auto u = static_cast<uint32_t>(static_cast<int32_t>(v));
        narrowed[i * 4 + 0] = static_cast<uint8_t>(u >> 24);
        narrowed[i * 4 + 1] = static_cast<uint8_t>(u >> 16);
        narrowed[i * 4 + 2] = static_cast<uint8_t>(u >> 8);
        narrowed[i * 4 + 3] = static_cast<uint8_t>(u);
    }
    const auto update_hash = crypto::Sha3_256(narrowed);

    auto sealed = ipc::MakeSealedMemfd(narrowed, "rnet-update");
    if (!sealed) return Err(sealed.error());

    const uint64_t assignment_id = next_assignment_id_++;
    auto payload = ipc::CborWriter()
                       .BeginMap(6)
                       .Key("scale_exp").Int(pending->scale_exp)
                       .Key("outer_step").Uint(pending->outer_step)
                       .Key("update_hash").Bytes(AsBytes(update_hash))
                       .Key("assignment_id").Uint(assignment_id)
                       .Key("base_weights_hash").Bytes(AsBytes(pending->base_weights))
                       .Key("contributor_count").Uint(pending->contributor_count)
                       .Take();
    if (!payload) return Err(payload.error());

    ipc::Frame reply;
    reply.type = ipc::MessageType::AssignApply;
    reply.request_id = frame.request_id;
    reply.payload = std::move(payload.value());
    if (auto st = client.SendWithDescriptor(reply, sealed.value().get()); !st) return st;

    applying_client_ = client.id();
    applying_assignment_ = assignment_id;
    RNET_LOG_INFO("ipc: worker {} asked to apply the update for step {} ({} contributions)",
                  client.id(), pending->outer_step, pending->contributor_count);
    (void)now_ms;
    return Status::Ok();
}

// A joining worker needs the state everyone else is at. Deriving genesis is not
// enough once the network has moved on: catching up needs either every historical
// payload — which the retention window deliberately does not keep — or the weights
// themselves.
Status WorkerService::OnGetWeights(ipc::Client& client, const ipc::Frame& frame) {
    auto message = ipc::CborParse(frame.payload);
    if (!message) return Refuse(client, frame.request_id, IpcError::BadSchema, message.error());

    auto wanted = message.value().GetHash("weights_hash");
    if (!wanted) return Refuse(client, frame.request_id, IpcError::BadSchema, wanted.error());

    auto* store = node_.weights();
    if (store == nullptr) {
        return Refuse(client, frame.request_id, IpcError::Unavailable,
                      "this node keeps no weights, so it cannot start a worker from a checkpoint");
    }
    if (!store->Has(wanted.value())) {
        // Said plainly. A worker told "not yet" can wait for the network to supply
        // them; a worker told nothing waits forever.
        return Refuse(client, frame.request_id, IpcError::Unavailable,
                      std::format("this node does not hold the weights for {}",
                                  util::ToHex(wanted.value()).substr(0, 16)));
    }
    auto size = store->SizeOf(wanted.value());
    if (!size) return Refuse(client, frame.request_id, IpcError::Unavailable, size.error());

    auto fd = store->OpenForRead(wanted.value());
    if (!fd) return Refuse(client, frame.request_id, IpcError::Unavailable, fd.error());
    ipc::Fd owned(fd.value());

    // An ordinary read-only descriptor rather than a sealed memfd: in this
    // direction the RECEIVER verifies, because it has the checkpoint header naming
    // the hash. A sealed copy would cost a second two gigabytes to prove something
    // the worker checks anyway.
    auto payload = ipc::CborWriter()
                       .BeginMap(2)
                       .Key("size").Uint(size.value())
                       .Key("weights_hash").Bytes(AsBytes(wanted.value()))
                       .Take();
    if (!payload) return Err(payload.error());

    ipc::Frame reply;
    reply.type = ipc::MessageType::Weights;
    reply.request_id = frame.request_id;
    reply.payload = std::move(payload.value());
    return client.SendWithDescriptor(reply, owned.get());
}

Status WorkerService::OnSubmitWeights(ipc::Client& client, ipc::ReceivedFrame& received,
                                      int64_t now_ms) {
    const auto& frame = received.frame;
    auto message = ipc::CborParse(frame.payload);
    if (!message) return Refuse(client, frame.request_id, IpcError::BadSchema, message.error());

    auto assignment_id = message.value().GetUint("assignment_id");
    if (!assignment_id) {
        return Refuse(client, frame.request_id, IpcError::BadSchema, assignment_id.error());
    }
    auto weights_hash = message.value().GetHash("weights_hash");
    if (!weights_hash) {
        return Refuse(client, frame.request_id, IpcError::BadSchema, weights_hash.error());
    }

    // A hash from a worker that was never asked to apply anything is not an answer
    // to a question — accepting it would let a bare 32 bytes manufacture a
    // checkpoint.
    if (applying_client_ != client.id() || applying_assignment_ != assignment_id.value()) {
        return Refuse(client, frame.request_id, IpcError::NotAssigned,
                      "this node did not ask this worker to apply that update");
    }
    applying_client_ = 0;
    applying_assignment_ = 0;

    // The bytes arrive with the hash. A checkpoint whose weights nobody holds is a
    // checkpoint no new node can start from, so producing one without storing them
    // would publish a state the network cannot reach.
    auto* store = node_.weights();
    if (store == nullptr) {
        return Refuse(client, frame.request_id, IpcError::Unavailable,
                      "this node keeps no weights, so it cannot publish a checkpoint");
    }
    const uint64_t expected_bytes = participant_.params().round.model.ParameterCount() * 2;
    // Verified against the hash the worker reported, which is in turn the value the
    // checkpoint will commit to. Bytes that do not match are refused before the
    // checkpoint exists, rather than discovered by a peer afterwards.
    if (auto st = store->PutFromFd(weights_hash.value(), received.descriptor.get(), expected_bytes);
        !st) {
        return Refuse(client, frame.request_id, IpcError::Rejected, st.error());
    }

    if (auto st = participant_.CompleteProduction(weights_hash.value(),
                                                  static_cast<uint64_t>(now_ms));
        !st) {
        return Refuse(client, frame.request_id, IpcError::Rejected, st.error());
    }

    // Only what the retention window needs is kept. The rule lives in the chain, so
    // this asks rather than deciding.
    std::set<crypto::Hash256> keep;
    for (const auto& id : participant_.chain().RequiredFullCheckpoints()) {
        auto entry = participant_.chain().Get(id);
        if (entry) keep.insert(entry.value().header.weights_hash);
    }
    if (auto pruned = store->RetainOnly(keep); pruned && pruned.value() > 0) {
        RNET_LOG_INFO("weights: pruned {} checkpoint(s) outside the retention window",
                      pruned.value());
    }

    auto tip = participant_.chain().Tip();
    auto payload = ipc::CborWriter()
                       .BeginMap(2)
                       .Key("outer_step").Uint(participant_.outer_step())
                       .Key("checkpoint_id")
                       .Bytes(AsBytes(tip.ok() ? tip.value().header.Id() : crypto::Hash256{}))
                       .Take();
    if (!payload) return Err(payload.error());

    ipc::Frame reply;
    reply.type = ipc::MessageType::WeightsAccepted;
    reply.request_id = frame.request_id;
    reply.payload = std::move(payload.value());
    return client.Send(reply);
}

Status WorkerService::OnGetAssignment(ipc::Client& client, const ipc::Frame& frame,
                                      int64_t now_ms) {
    // An aggregated update waiting to be applied takes priority over new training:
    // until the step advances, anything trained now would be against a base that is
    // about to be superseded.
    if (participant_.Pending() != nullptr &&
        (applying_client_ == 0 || applying_client_ == client.id())) {
        return OfferPendingApply(client, frame, now_ms);
    }

    // Everything the worker needs that it cannot derive: which step, and which
    // weights to start from. Deliberately absent: batch offsets and token windows.
    // If the node chose the data, a colluding node and worker would regain exactly
    // the freedom the deterministic derivation exists to remove — so the worker
    // derives its own windows from values it can verify.
    Assignment assignment;
    assignment.assignment_id = next_assignment_id_++;
    assignment.outer_step = participant_.outer_step();
    assignment.base_checkpoint = participant_.base_checkpoint();
    assignment.issued_at = now_ms;
    assignments_[client.id()] = assignment;

    auto tip = participant_.chain().Tip();
    if (!tip) {
        return Refuse(client, frame.request_id, IpcError::Unavailable,
                      "this node holds no checkpoint to train from");
    }

    auto payload = ipc::CborWriter()
                       .BeginMap(5)
                       .Key("round_id").Uint(participant_.params().round.round_id)
                       .Key("outer_step").Uint(assignment.outer_step)
                       .Key("assignment_id").Uint(assignment.assignment_id)
                       .Key("base_checkpoint").Bytes(AsBytes(assignment.base_checkpoint))
                       .Key("base_weights_hash").Bytes(AsBytes(tip.value().header.weights_hash))
                       .Take();
    if (!payload) return Err(payload.error());

    ipc::Frame reply;
    reply.type = ipc::MessageType::AssignTrain;
    reply.request_id = frame.request_id;
    reply.payload = std::move(payload.value());
    return client.Send(reply);
}

Result<std::vector<uint8_t>> WorkerService::ReadSealedPayload(int fd, uint64_t expected_bytes,
                                                             crypto::Hash256& hash_out) {
    // The size is the model's, not the sender's. There is no length field in any
    // IPC message precisely so there is one authority for this number.
    auto actual = ipc::SealedSizeOf(fd);
    if (!actual) return Err(actual.error());
    if (actual.value() != expected_bytes) {
        return Err(std::format(
            "ipc: payload is {} bytes, the model has {} parameters and an int8 payload is exactly "
            "one byte each",
            actual.value(), expected_bytes));
    }
    if (expected_bytes == 0) return Err("ipc: refusing an empty payload");

    void* mapping = ::mmap(nullptr, static_cast<size_t>(expected_bytes), PROT_READ, MAP_PRIVATE, fd,
                           0);
    if (mapping == MAP_FAILED) {
        return Err(std::string("ipc: mmap: ") + std::strerror(errno));
    }
    const auto* bytes = static_cast<const uint8_t*>(mapping);

    crypto::Sha3Hasher hasher;
    for (uint64_t offset = 0; offset < expected_bytes; offset += kHashSlice) {
        const size_t length =
            static_cast<size_t>(std::min<uint64_t>(kHashSlice, expected_bytes - offset));
        hasher.Update(std::span<const uint8_t>(bytes + offset, length));
    }
    hash_out = hasher.Finalize();

    std::vector<uint8_t> copy(bytes, bytes + expected_bytes);
    ::munmap(mapping, static_cast<size_t>(expected_bytes));
    return copy;
}

Status WorkerService::OnSubmitContribution(ipc::Client& client, ipc::ReceivedFrame& received,
                                           int64_t now_ms) {
    const auto& frame = received.frame;

    auto message = ipc::CborParse(frame.payload);
    if (!message) return Refuse(client, frame.request_id, IpcError::BadSchema, message.error());

    auto assignment_id = message.value().GetUint("assignment_id");
    if (!assignment_id) {
        return Refuse(client, frame.request_id, IpcError::BadSchema, assignment_id.error());
    }
    auto scale_exp = message.value().GetInt("scale_exp");
    if (!scale_exp) {
        return Refuse(client, frame.request_id, IpcError::BadSchema, scale_exp.error());
    }
    auto claimed_payload_hash = message.value().GetHash("payload_hash");
    if (!claimed_payload_hash) {
        return Refuse(client, frame.request_id, IpcError::BadSchema, claimed_payload_hash.error());
    }

    const auto held = assignments_.find(client.id());
    if (held == assignments_.end() || held->second.assignment_id != assignment_id.value()) {
        return Refuse(client, frame.request_id, IpcError::NotAssigned,
                      "this submission names an assignment this worker was not given");
    }
    // The tip may have moved while the worker was training. Its work is against a
    // base that is no longer current — late, not dishonest, and it must be told
    // which rather than left to guess.
    if (held->second.base_checkpoint != participant_.base_checkpoint()) {
        assignments_.erase(held);
        return Refuse(client, frame.request_id, IpcError::NotAssigned,
                      "the checkpoint this work was based on is no longer the tip");
    }

    const uint64_t expected_bytes = participant_.params().round.model.ParameterCount();
    crypto::Hash256 payload_hash{};
    auto payload = ReadSealedPayload(received.descriptor.get(), expected_bytes, payload_hash);
    if (!payload) return Refuse(client, frame.request_id, IpcError::Rejected, payload.error());

    // The worker's claim about its own bytes is checked against the bytes. It is
    // the one value here the worker genuinely produced, and it is still only a
    // claim until the daemon has hashed what arrived.
    if (payload_hash != claimed_payload_hash.value()) {
        return Refuse(client, frame.request_id, IpcError::Rejected,
                      "the payload does not hash to the value the header claims");
    }

    // The header is rebuilt from the daemon's own state. The worker contributes
    // exactly two values it is entitled to: the scale exponent it chose and the
    // hash of the work it did.
    canon::ContributionHeader header;
    header.round_id = participant_.params().round.round_id;
    header.worker_id = participant_.worker_id();
    header.outer_step = participant_.outer_step();
    header.base_checkpoint = participant_.base_checkpoint();
    header.payload_hash = payload_hash;
    header.n_params = expected_bytes;
    header.inner_steps = participant_.params().policy.inner_steps;
    header.scale_exp = static_cast<int32_t>(scale_exp.value());
    if (auto st = header.Validate(); !st) {
        return Refuse(client, frame.request_id, IpcError::Rejected, st.error());
    }

    // Published before the header, so that by the time peers learn a contribution
    // exists, the bytes it refers to are already fetchable from this node.
    auto published = node_.Publish(net::InvType::Contribution, std::move(payload.value()), now_ms);
    if (!published) {
        return Refuse(client, frame.request_id, IpcError::Unavailable, published.error());
    }

    // Routed through the same validation a peer's contribution goes through.
    if (auto st = participant_.SubmitOwn(header, static_cast<uint64_t>(now_ms)); !st) {
        return Refuse(client, frame.request_id, IpcError::Rejected, st.error());
    }
    assignments_.erase(held);
    ++accepted_;

    const auto id = header.Id();
    RNET_LOG_INFO("ipc: worker {} contributed {} at step {} (payload {})", client.id(),
                  util::ToHex(id).substr(0, 16), header.outer_step,
                  util::ToHex(payload_hash).substr(0, 16));

    auto reply_payload = ipc::CborWriter()
                             .BeginMap(2)
                             .Key("outer_step").Uint(header.outer_step)
                             .Key("contribution_id").Bytes(AsBytes(id))
                             .Take();
    if (!reply_payload) return Err(reply_payload.error());

    ipc::Frame reply;
    reply.type = ipc::MessageType::SubmitAccepted;
    reply.request_id = frame.request_id;
    reply.payload = std::move(reply_payload.value());
    return client.Send(reply);
}

Status WorkerService::OnGetStatus(ipc::Client& client, const ipc::Frame& frame) {
    auto payload = ipc::CborWriter()
                       .BeginMap(4)
                       .Key("peers").Uint(node_.ReadyCount())
                       .Key("objects").Uint(node_.objects().Size())
                       .Key("outer_step").Uint(participant_.outer_step())
                       .Key("contributions").Uint(participant_.submission_count())
                       .Take();
    if (!payload) return Err(payload.error());

    ipc::Frame reply;
    reply.type = ipc::MessageType::Status;
    reply.request_id = frame.request_id;
    reply.payload = std::move(payload.value());
    return client.Send(reply);
}

}  // namespace rnet::protocol

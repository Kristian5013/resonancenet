#include "protocol/service.h"

#include "canon/canon.h"
#include "util/hex.h"
#include "util/logging.h"

namespace rnet::protocol {

Service::Service(net::Node& node, Participant& participant)
    : node_(node), participant_(participant) {}

void Service::Attach() {
    // Aggregation needs the payload bytes, which live in the node's object store.
    // Borrowed rather than copied: each is one byte per parameter, so a copy per
    // contribution would be a gigabyte per contributor.
    participant_.SetPayloadProvider(
        [this](const crypto::Hash256& hash) -> const std::vector<uint8_t>* {
            const auto* stored = node_.objects().Peek(hash);
            return stored == nullptr ? nullptr : &stored->data;
        });

    // Registered as a notification: the node has already stored and hash-verified
    // the object by the time this runs, so consensus never sees unchecked bytes.
    const auto route = [this](net::Peer& peer, const net::ReceivedMessage& message,
                              int64_t now_ms) -> Status {
        return OnObjectMessage(peer, message, now_ms);
    };
    node_.SetHandler(net::cmd::kObject, route);
    node_.SetHandler(net::cmd::kChunk, route);
}

Status Service::OnObjectMessage(net::Peer& peer, const net::ReceivedMessage& message,
                                int64_t now_ms) {
    // Both `object` and `chunk` messages start with a type and a hash; what
    // matters here is which object landed, and the node's store already holds it.
    canon::Reader r(message.payload);
    auto type = r.U16();
    if (!type) return Err(type.error());
    auto hash = r.Hash();
    if (!hash) return Err(hash.error());

    auto stored = node_.objects().Get(hash.value());
    if (!stored) {
        // A chunk message for a transfer still in progress: nothing to do yet.
        return Status::Ok();
    }

    // A payload is bulk data, not a canon object — it is checked by the hash the
    // header named, which the store already enforced.
    const auto fetch = fetching_.find(hash.value());
    if (fetch != fetching_.end()) {
        fetching_.erase(fetch);
        if (auto st = participant_.NotePayload(hash.value()); !st) {
            // The bytes were honest but nobody was waiting for them. Not a
            // consensus fault, but not something to record either.
            RNET_LOG_DEBUG("payload {} arrived unwanted: {}",
                           util::ToHex(hash.value()).substr(0, 16), st.error());
        }
        return Status::Ok();
    }

    const auto verdict = participant_.OnObject(static_cast<net::InvType>(type.value()),
                                               hash.value(), stored.value().data, now_ms);
    if (verdict.accepted()) {
        ++accepted_;
        return Status::Ok();
    }
    if (verdict.admission == Admission::Duplicate ||
        verdict.admission == Admission::NotForUs) {
        return Status::Ok();
    }

    ++rejected_;
    // Scored against the peer that sent it. A peer that keeps offering
    // contributions for the wrong model is not making mistakes.
    //
    // Returning Ok afterwards is deliberate. The node charges a flat ten points for
    // any handler that returns an error, which is right for handlers with no
    // judgement of their own — but here the participant has already made a graded
    // one, from "malformed, twenty" to "forged identity, fifty" to "refused, but
    // not your fault, zero". Reporting the rejection upward silently added ten to
    // every one of those and turned every deliberate zero into a ten.
    //   proven by: protocol_tests.cpp ANodeBehindTheTipDoesNotBanThePeersThatAreOnIt
    if (verdict.misbehaviour > 0) peer.Misbehaving(verdict.misbehaviour, verdict.reason);
    return Status::Ok();
}

void Service::PublishOutgoing(int64_t now_ms) {
    for (auto& out : participant_.TakeOutgoing()) {
        auto published = node_.Publish(out.type, std::move(out.container), now_ms);
        if (!published) {
            RNET_LOG_ERROR("cannot publish {}: {}", util::ToHex(out.hash).substr(0, 16),
                           published.error());
            continue;
        }
        RNET_LOG_INFO("published {} {}", static_cast<int>(out.type),
                      util::ToHex(published.value()).substr(0, 16));
    }
}

void Service::FetchMissingPayloads(int64_t now_ms) {
    // A worker that announces a contribution and never delivers its bytes holds
    // up the round for everyone, so a stalled fetch is reassigned rather than
    // waited on.
    for (auto it = fetching_.begin(); it != fetching_.end();) {
        if (now_ms - it->second.started_at <= kPayloadFetchTimeoutMs) {
            ++it;
            continue;
        }
        RNET_LOG_DEBUG("payload {} not delivered by peer {}; will ask elsewhere",
                       util::ToHex(it->first).substr(0, 16), it->second.peer_id);
        it = fetching_.erase(it);
    }

    for (const auto& missing : participant_.MissingPayloads()) {
        if (node_.objects().Has(missing.payload_hash)) {
            (void)participant_.NotePayload(missing.payload_hash);
            continue;
        }
        if (fetching_.find(missing.payload_hash) != fetching_.end()) continue;
        if (node_.BulkInFlight(missing.payload_hash)) continue;

        // Only peers that claimed to have it. Asking everyone would multiply this
        // node's own bandwidth cost by its peer count.
        for (uint64_t peer_id : node_.objects().PeersAnnouncing(missing.payload_hash)) {
            // The size comes from the contribution header, not from the peer: an
            // int8 payload is exactly one byte per parameter.
            if (auto st = node_.RequestBulk(peer_id, net::InvType::Contribution,
                                            missing.payload_hash, missing.expected_bytes, now_ms);
                !st) {
                continue;
            }
            fetching_[missing.payload_hash] =
                PayloadFetch{peer_id, now_ms, missing.expected_bytes};
            break;
        }
    }
}

Status Service::Tick(int64_t now_ms) {
    if (auto st = participant_.Tick(static_cast<uint64_t>(now_ms)); !st) return st;
    PublishOutgoing(now_ms);
    FetchMissingPayloads(now_ms);
    return Status::Ok();
}

}  // namespace rnet::protocol

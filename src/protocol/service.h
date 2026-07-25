// The seam between the wire and consensus.
//
// Two layers meet here and neither is allowed to reach into the other:
//
//   * The NODE knows about sockets, peers and hashes. It can prove that bytes
//     arrived intact and that a peer sent them. It cannot know whether an object
//     means anything.
//
//   * The PARTICIPANT knows about rounds, checkpoints and challenges. It decides
//     what an object means. It has no idea what a socket is, which is what keeps
//     its decisions testable without one.
//
// This class carries messages between them, and its whole job is to make sure
// nothing crosses in the wrong direction:
//
//   * An object reaches the participant only after the node has verified its
//     bytes hash to the identity it was requested under.
//
//   * A rejection by the participant is scored against the PEER THAT SENT IT.
//     Rejections are evidence about a peer, not just local outcomes — a peer that
//     keeps sending contributions for the wrong model is not making mistakes.
//
//   * Bulk payloads are fetched at the size the header fixes, never at a size the
//     sender proposes.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <set>

#include "net/node.h"
#include "protocol/participant.h"

namespace rnet::protocol {

// How long to wait for a payload before asking someone else for it. A worker that
// announces a contribution and never delivers its bytes holds up the round for
// everyone, so the wait is bounded.
inline constexpr int64_t kPayloadFetchTimeoutMs = 120'000;

class Service {
public:
    Service(net::Node& node, Participant& participant);

    // Installs the handlers. Called once, before the node starts ticking.
    void Attach();

    // Drives one iteration of the consensus side: publishes what the participant
    // produced, and fetches payloads it is waiting for.
    Status Tick(int64_t now_ms);

    uint64_t objects_accepted() const { return accepted_; }
    uint64_t objects_rejected() const { return rejected_; }

private:
    Status OnObjectMessage(net::Peer& peer, const net::ReceivedMessage& message, int64_t now_ms);
    void PublishOutgoing(int64_t now_ms);
    void FetchMissingPayloads(int64_t now_ms);

    net::Node& node_;
    Participant& participant_;

    // Payload fetches in flight, so one is not requested from every peer at once.
    struct PayloadFetch {
        uint64_t peer_id{0};
        int64_t started_at{0};
        uint64_t expected_bytes{0};
    };
    std::map<crypto::Hash256, PayloadFetch> fetching_;

    uint64_t accepted_{0};
    uint64_t rejected_{0};
};

}  // namespace rnet::protocol

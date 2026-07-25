// What the daemon does with what a local worker says.
//
// The worker is a separate process doing the expensive part, and it is NOT
// trusted more than a peer on the network is. It is trusted less in one respect
// and more in another, and both are deliberate:
//
//   * LESS: its submission is routed through `Participant::OnObject`, the same
//     path a stranger's bytes take. A node that accepts from its own worker what
//     it would refuse from a peer is a node whose bugs reach the network before
//     anyone notices them.
//
//   * MORE: it is allowed to submit under this node's worker identity, which no
//     peer may do. That is the whole privilege the channel grants, and it is why
//     the socket is 0700 rather than a port.
//
// The rule that generates the rest: THE WORKER MAY AUTHOR BYTES, NEVER
// JUDGEMENTS, AND NEVER A VALUE CONSENSUS ALREADY FIXES. It computes a
// pseudo-gradient — genuinely its own work — and states a scale exponent and a
// payload hash. Everything else in the header the daemon rebuilds from its own
// state and requires to match byte for byte, so the worker's authorship of the
// container is a statement of agreement rather than a source of truth.
#pragma once

#include <cstdint>
#include <map>
#include <string>

#include "ipc/server.h"
#include "net/node.h"
#include "protocol/participant.h"

namespace rnet::protocol {

// Why a request was refused. Sent back so an honest worker can correct itself
// instead of guessing, and so a mismatch is legible in a log rather than
// appearing as a silent stall.
enum class IpcError : uint16_t {
    BadSchema = 1,        // the message does not parse under the strict profile
    OutOfSequence = 2,    // legal message, wrong moment
    AnchorMismatch = 3,   // the worker is running different consensus rules
    NotAssigned = 4,      // submitting work that was never assigned
    Rejected = 5,         // the consensus layer refused it
    Unavailable = 6,      // the daemon cannot answer right now
};

std::string IpcErrorName(IpcError error);

// Bridges the IPC server and the consensus layer. Holds no consensus state of its
// own — every decision is delegated, so there is nowhere for a second, divergent
// opinion to live.
class WorkerService {
public:
    WorkerService(ipc::Server& server, net::Node& node, Participant& participant);

    // Handles everything that arrived. `now_ms` is supplied rather than read from
    // a clock, like the rest of the daemon.
    Status Poll(int64_t now_ms);

    uint64_t contributions_accepted() const { return accepted_; }
    uint64_t requests_refused() const { return refused_; }

private:
    Status Dispatch(ipc::Client& client, ipc::ReceivedFrame& received, int64_t now_ms);
    Status OnHello(ipc::Client& client, const ipc::Frame& frame);
    Status OnGetAssignment(ipc::Client& client, const ipc::Frame& frame, int64_t now_ms);
    Status OnSubmitContribution(ipc::Client& client, ipc::ReceivedFrame& received, int64_t now_ms);
    Status OnGetStatus(ipc::Client& client, const ipc::Frame& frame);
    Status OnSubmitWeights(ipc::Client& client, ipc::ReceivedFrame& received, int64_t now_ms);
    Status OnGetWeights(ipc::Client& client, const ipc::Frame& frame);
    Status OfferPendingApply(ipc::Client& client, const ipc::Frame& frame, int64_t now_ms);

    Status Refuse(ipc::Client& client, uint64_t request_id, IpcError error, std::string_view reason);

    // Reads a sealed descriptor into memory, hashing as it goes. The expected size
    // comes from the model, never from the sender, and a descriptor of any other
    // size is refused before a byte is read.
    Result<std::vector<uint8_t>> ReadSealedPayload(int fd, uint64_t expected_bytes,
                                                   crypto::Hash256& hash_out);

    ipc::Server& server_;
    net::Node& node_;
    Participant& participant_;

    // Assignments handed out, so a submission can be tied to one. A worker that
    // submits without asking is not following the protocol, and the assignment is
    // also where the base checkpoint it was told about is recorded.
    struct Assignment {
        uint64_t assignment_id{0};
        uint64_t outer_step{0};
        crypto::Hash256 base_checkpoint{};
        int64_t issued_at{0};
    };
    std::map<uint64_t, Assignment> assignments_;   // by client id
    uint64_t next_assignment_id_{1};

    // The client currently applying a staged update, and under which assignment.
    // One at a time: two workers applying the same update would each report a hash
    // and the node would have to pick, which is a choice it must never have to make.
    uint64_t applying_client_{0};
    uint64_t applying_assignment_{0};

    uint64_t accepted_{0};
    uint64_t refused_{0};
};

}  // namespace rnet::protocol

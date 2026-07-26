// The participant: what a node does with the objects that arrive on its wire.
//
// The net layer moves bytes and proves they hash to what they claim. That is all
// it can prove. Everything else — that a contribution belongs to the round in
// progress, that it descends from a checkpoint this node recognises, that a
// verdict came from the verifier the beacon actually chose — is a consensus
// question, and this is where it is asked.
//
// The rule is that nothing is accepted on a peer's word:
//
//   * A CONTRIBUTION is refused unless it names the open round, the current outer
//     step, a base checkpoint this node holds, and the parameter count the genesis
//     model fixes. A worker that computed against different weights is not
//     contributing to this model, whatever its arithmetic was.
//
//   * A CHECKPOINT is refused unless it extends the chain this node already has.
//     Accepting a checkpoint with an unknown parent would mean training onward
//     from a state nobody proved, which is how a fork becomes silently canonical.
//
//   * A CHALLENGE is refused unless the beacon really does select that
//     contribution and really does assign that verifier. Otherwise anyone could
//     order anyone to burn a GPU-hour, or arrange to verify their own work.
//
//   * A VERDICT is refused unless it answers a challenge this node knows about.
//     Verdicts about nothing are how a ledger gets filled with claims that have no
//     challenge to check them against.
//
// Slashing runs in SHADOW MODE: evidence is built and recorded, and nothing is
// settled. The economic consequence of a false accusation is permanent, so it
// waits until the evidence has been produced by real hardware for long enough to
// be believed.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <vector>

#include "canon/objects.h"
#include "consensus/params.h"
#include <functional>

#include "diloco/chain.h"
#include "diloco/outer_optimizer.h"
#include "diloco/producer.h"
#include "diloco/round.h"
#include "net/protocol.h"
#include "util/result.h"
#include "verification/challenge.h"
#include "verification/evidence.h"

namespace rnet::protocol {

// What arrived and what the participant did with it. Returned rather than logged
// so the caller can score the peer that sent it: a rejection here is evidence
// about a peer, not just a local outcome.
enum class Admission {
    Accepted,        // valid and now part of this node's state
    Duplicate,       // already known; not a fault
    NotForUs,        // valid but irrelevant (another worker's challenge, say)
    Rejected,        // inconsistent with consensus — the sender is at fault
};

std::string AdmissionName(Admission admission);

struct ObjectVerdict {
    Admission admission{Admission::Rejected};
    std::string reason;              // filled whenever admission != Accepted
    int misbehaviour{0};             // what the sender earned, 0 for honest mistakes

    bool accepted() const { return admission == Admission::Accepted; }
};

// Something the participant wants published. The caller announces it; the
// participant does not touch the network itself, so its logic stays testable
// without sockets and its decisions cannot depend on transport state.
struct Outgoing {
    net::InvType type{net::InvType::Contribution};
    crypto::Hash256 hash{};
    std::vector<uint8_t> container;
};

struct ParticipantConfig {
    consensus::Network network{consensus::Network::Regtest};
    uint64_t worker_id{0};
    bool verifier{false};            // whether this node answers challenges

    Status Validate() const;
};

class Participant {
public:
    Participant(ParticipantConfig config, canon::CheckpointHeader genesis_checkpoint);

    // Borrows a contribution's payload bytes. Supplied by the caller because the
    // participant deliberately knows nothing about storage or transport; returning
    // a pointer rather than a copy matters when a payload is a gigabyte.
    using PayloadProvider =
        std::function<const std::vector<uint8_t>*(const crypto::Hash256& payload_hash)>;
    void SetPayloadProvider(PayloadProvider provider) { payloads_ = std::move(provider); }

    // Producing a checkpoint happens in TWO PHASES, because the second one is not
    // this process's to perform.
    //
    // The node can aggregate: it holds the payloads. It cannot apply the result to
    // the weights, because the weights live in the worker's GPU memory and copying
    // a multi-gigabyte model into the daemon to compute one hash would put it in
    // the wrong place. So the node stages an update, the worker applies it and
    // reports what the result hashes to, and the node finishes the checkpoint.
    //
    // What the worker does here is a COMPUTATION, not a judgement: anyone holding
    // the same base weights and the same update reaches the same hash. That is
    // exactly what makes a published checkpoint checkable by everyone else.
    // True while this node cannot reproduce the optimizer state the chain is at.
    //
    // It happens legitimately: a node that joined late, or whose payloads never
    // arrived, has no way to replay the aggregates that built the momentum. Such a
    // node can still FOLLOW the chain — it has the weights hash — but it must not
    // PRODUCE, because its update would be computed from momentum nobody else has.
    //
    // Said out loud rather than tracked silently, because the failure it prevents
    // is precisely the one that leaves every test green.
    bool optimizer_desynced() const { return optimizer_desynced_; }
    const crypto::Hash256& optimizer_state() const { return optimizer_state_; }

    struct PendingProduction {
        uint64_t outer_step{0};              // the step being produced (parent + 1)
        crypto::Hash256 base_weights{};      // what the update applies to
        int32_t scale_exp{0};                // update units are 2^-scale_exp
        std::vector<int64_t> update;
        crypto::Hash256 evidence_hash{};
        uint32_t contributor_count{0};
        uint64_t staged_ms{0};               // when it was staged; bounds the wait
    };

    // The update waiting to be applied, or nullptr. Borrowed: for the launch model
    // this is nearly a gigabyte of int32 once narrowed, and copying it to answer a
    // poll would be absurd.
    const PendingProduction* Pending() const { return pending_ ? &*pending_ : nullptr; }

    // Finishes the checkpoint with the hash the worker reported. Fails if nothing
    // is pending, so a stray reply cannot manufacture a checkpoint.
    Status CompleteProduction(const crypto::Hash256& weights_hash, uint64_t now_ms);

    // --- inbound: objects the net layer has already hash-verified ---

    // Routes by type and validates against consensus. `container` is the canonical
    // container bytes, exactly as they hashed.
    ObjectVerdict OnObject(net::InvType type, const crypto::Hash256& hash,
                           std::span<const uint8_t> container, uint64_t now_ms);

    // --- outbound: what this node has to say ---

    // Objects produced since the last call. Drained, so nothing is announced twice.
    std::vector<Outgoing> TakeOutgoing();

    // Advances the clock: closes a round whose deadline has passed and whose
    // participation is sufficient, and issues challenges for the step just closed.
    Status Tick(uint64_t now_ms);

    // Submits this node's own contribution to the round in progress. Validated on
    // the same terms as anyone else's — a node must not be able to accept from
    // itself what it would refuse from a peer.
    Status SubmitOwn(const canon::ContributionHeader& header, uint64_t now_ms);

    // --- what this node is waiting for ---

    // Payload hashes named by accepted contributions whose bytes have not arrived.
    // These are what the caller fetches in bulk; the size is derived from the
    // header, never asked of the sender.
    struct MissingPayload {
        crypto::Hash256 payload_hash{};
        crypto::Hash256 contribution_id{};
        uint64_t expected_bytes{0};
    };
    std::vector<MissingPayload> MissingPayloads() const;

    // Records that a payload's bytes arrived and hashed correctly.
    Status NotePayload(const crypto::Hash256& payload_hash);

    // Challenges assigned to this node that it has not yet answered.
    std::vector<canon::ChallengeOrder> PendingChallenges() const;

    // Files the result of recomputing a challenged contribution. The verdict is
    // published; whether it settles anything is the ledger's business, and in
    // shadow mode it settles nothing.
    Status SubmitVerdict(const canon::VerificationVerdict& verdict, uint64_t now_ms);

    // Whether this node is the elected producer for the step in progress. Derived,
    // never volunteered: see diloco/producer.h.
    bool IsProducerForThisStep() const;

    // The election slot the wall clock is in. Not a count of local retries: rounds
    // open at different moments on different nodes, and a counter anchored to a
    // local event would give each node a different answer. Absolute time divided
    // by a consensus-fixed slot length gives every node with a synced clock the
    // same value at the same instant.
    //   proven by: protocol_tests.cpp TwoNodesWhoseRoundsOpenedApartStillAgree
    uint64_t ProducerSlotAt(uint64_t now_ms) const;

    // Discards everything this node computed on a branch it is leaving. Called
    // before adopting a competing checkpoint for a height already filled.
    void AbandonWorkOnTheOldBranch(const diloco::ChainEntry& new_parent);

    // --- state ---
    const diloco::CheckpointChain& chain() const { return chain_; }
    const verification::EvidenceLedger& ledger() const { return ledger_; }
    uint64_t outer_step() const { return outer_step_; }
    // The weights a contribution must be computed from right now. Exposed because
    // a local worker has to be told; it is never a value the worker may choose.
    const crypto::Hash256& base_checkpoint() const { return base_checkpoint_; }
    uint64_t worker_id() const { return config_.worker_id; }
    uint32_t submission_count() const { return round_ ? round_->submission_count() : 0; }
    const consensus::NetworkParams& params() const { return params_; }

private:
    ObjectVerdict OnContribution(const crypto::Hash256& id, std::span<const uint8_t> container,
                                 uint64_t now_ms);
    ObjectVerdict OnCheckpoint(const crypto::Hash256& id, std::span<const uint8_t> container);
    ObjectVerdict OnChallenge(const crypto::Hash256& id, std::span<const uint8_t> container);
    ObjectVerdict OnVerdict(const crypto::Hash256& id, std::span<const uint8_t> container);
    ObjectVerdict OnSlashEvidence(const crypto::Hash256& id, std::span<const uint8_t> container);

    Status OpenRound(uint64_t now_ms);
    Status IssueChallenges(uint64_t closed_step);
    Status ProduceCheckpoint(uint64_t now_ms);

    // Aggregates the closed round and advances the optimizer. Run by EVERY node,
    // producer or not: momentum is state the whole network shares, and a node that
    // only steps when it happens to be elected arrives at its turn with momentum
    // nobody else has.
    Status AdvanceOptimizer();
    crypto::Hash256 BeaconFor(uint64_t outer_step) const;

    ParticipantConfig config_;
    const consensus::NetworkParams& params_;

    diloco::CheckpointChain chain_;
    verification::EvidenceLedger ledger_;
    std::unique_ptr<diloco::Round> round_;
    uint64_t outer_step_{0};
    crypto::Hash256 base_checkpoint_{};

    // Contributions accepted this step, and which of their payloads have landed.
    std::map<crypto::Hash256, canon::ContributionHeader> accepted_;
    std::set<crypto::Hash256> payloads_held_;

    // Challenges this node knows about, and the ones it owes an answer to.
    std::map<crypto::Hash256, canon::ChallengeOrder> challenges_;
    std::set<crypto::Hash256> answered_;

    // Who may publish the checkpoint for the step in progress: the workers whose
    // contributions were aggregated into the PREVIOUS checkpoint. Settled before
    // this round opened, so the current producer cannot shape it.
    std::vector<uint64_t> producer_pool_;

    PayloadProvider payloads_;
    std::unique_ptr<diloco::OuterOptimizer> optimizer_;
    std::optional<PendingProduction> pending_;

    // The optimizer advances exactly once per outer step. Ticking twice while a
    // round sits Closed must not step twice — that would be the same divergence
    // this whole change exists to remove, arriving by a different door.
    uint64_t optimizer_stepped_through_{0};
    crypto::Hash256 optimizer_state_{};
    // WHICH contributions optimizer_state_ was reached from. Without it the state
    // is a number with no stated inputs, and comparing it against a checkpoint's is
    // comparing two answers to two different questions.
    crypto::Hash256 optimizer_evidence_{};
    bool optimizer_desynced_{false};
    // The slot the last election used. Stored so IsProducerForThisStep(), which has
    // no clock, answers the same question Tick() asked.
    mutable uint64_t producer_slot_{0};
    // The update this node computed for the step in progress. The producer
    // publishes it; everyone else keeps it only to prove they reached the same
    // place.
    std::vector<int64_t> pending_update_;
    int32_t pending_scale_exp_{0};
    crypto::Hash256 pending_evidence_{};
    uint32_t pending_contributors_{0};

    std::vector<Outgoing> outgoing_;
};

}  // namespace rnet::protocol

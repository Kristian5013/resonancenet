// The outer round: the unit in which independent machines become one trainer.
//
// A round is opened against a base checkpoint. Workers train locally for
// `inner_steps`, submit one contribution each, and when the round closes their
// contributions are averaged into a single outer step. Everything about joining
// and leaving is round-atomic, which is what lets machines come and go without
// corrupting the shared model.
//
// Three policies make this survivable on a volunteer network:
//
//   * QUORUM. A round closes only if at least `min_contributors` submitted.
//     Averaging one straggler's update as if it were the network's consensus
//     would let a single machine steer the model.
//
//   * DEADLINE. Waiting for the slowest machine would make the network as slow as
//     its worst member. Past the deadline the round closes with whoever made it —
//     late work is dropped for that round, not merged half-applied.
//
//   * ROUND-ATOMIC MEMBERSHIP. A partial local run is never accepted. A worker
//     that leaves mid-round simply does not appear in it, and one that joins
//     starts at the next boundary from the published checkpoint.
//
// Time is supplied by the caller rather than read from a clock, so the state
// machine is deterministic and testable, and so nodes on skewed clocks cannot
// reach different conclusions about the same evidence.
#pragma once

#include <cstdint>
#include <map>
#include <span>
#include <vector>

#include "canon/objects.h"
#include "diloco/aggregation.h"
#include "util/result.h"

namespace rnet::diloco {

enum class RoundState {
    Open,        // accepting contributions
    Closed,      // quorum reached and deadline passed; ready to aggregate
    Abandoned,   // deadline passed without quorum
};

struct RoundConfig {
    uint32_t inner_steps{250};
    uint32_t min_contributors{1};
    uint64_t deadline_ms{0};        // 0 disables the deadline (tests, single machine)
    // A worker may submit once per round. Re-submission is rejected rather than
    // overwriting, so a worker cannot revise its contribution after seeing others.
    bool allow_resubmit{false};

    Status Validate() const;
};

struct SubmissionRecord {
    canon::ContributionHeader header;
    uint64_t received_ms{0};
};

class Round {
public:
    Round(uint64_t round_id, uint64_t outer_step, crypto::Hash256 base_checkpoint,
          uint64_t opened_ms, RoundConfig config);

    // Accepts a contribution header. The payload is verified elsewhere; this
    // enforces the round's rules: right base, right round, right step, one per
    // worker, before the deadline.
    Status Submit(const canon::ContributionHeader& header, uint64_t now_ms);

    // Recomputes state from the evidence held. Idempotent and pure — two nodes
    // with the same submissions and the same `now_ms` always agree.
    RoundState StateAt(uint64_t now_ms) const;

    // True when there is no reason to wait any longer: either the deadline has
    // passed, or every expected participant has already submitted.
    bool CanClose(uint64_t now_ms, uint32_t expected_participants) const;

    // Deterministic ordering: contributions are aggregated in ascending worker id
    // regardless of arrival order. Integer aggregation is order-independent, so
    // this is belt and braces — it also makes logs and hashes comparable.
    std::vector<canon::ContributionHeader> OrderedSubmissions() const;

    uint64_t round_id() const { return round_id_; }
    uint64_t outer_step() const { return outer_step_; }
    const crypto::Hash256& base_checkpoint() const { return base_checkpoint_; }
    uint32_t submission_count() const { return static_cast<uint32_t>(submissions_.size()); }
    const RoundConfig& config() const { return config_; }
    uint64_t opened_ms() const { return opened_ms_; }

    // Identity of the evidence set: the hash of all accepted contribution ids in
    // canonical order. Two nodes that closed the same round produce the same
    // value, which is what a checkpoint commits to.
    crypto::Hash256 EvidenceHash() const;

private:
    uint64_t round_id_;
    uint64_t outer_step_;
    crypto::Hash256 base_checkpoint_;
    uint64_t opened_ms_;
    RoundConfig config_;
    std::map<uint64_t, SubmissionRecord> submissions_;   // worker_id -> record
};

// Tracks which workers are currently expected to participate. Membership changes
// take effect at round boundaries only.
class Membership {
public:
    void Join(uint64_t worker_id);
    void Leave(uint64_t worker_id);

    // Applies pending joins and leaves. Call between rounds, never inside one.
    void AdvanceRound();

    bool IsActive(uint64_t worker_id) const;
    uint32_t ActiveCount() const { return static_cast<uint32_t>(active_.size()); }
    std::vector<uint64_t> ActiveWorkers() const;

    // A worker that misses rounds stops counting toward quorum, so the network
    // does not stall waiting for machines that have quietly disappeared.
    void RecordParticipation(uint64_t worker_id, bool participated);
    uint32_t ConsecutiveMisses(uint64_t worker_id) const;
    void DropInactive(uint32_t max_consecutive_misses);

private:
    std::map<uint64_t, uint32_t> active_;        // worker_id -> consecutive misses
    std::vector<uint64_t> pending_joins_;
    std::vector<uint64_t> pending_leaves_;
};

}  // namespace rnet::diloco

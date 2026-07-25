#include "diloco/round.h"

#include <algorithm>
#include <format>

#include "canon/canon.h"

namespace rnet::diloco {

Status RoundConfig::Validate() const {
    if (inner_steps == 0) return Err("round: inner_steps must be non-zero");
    if (min_contributors == 0) return Err("round: min_contributors must be non-zero");
    return Status::Ok();
}

Round::Round(uint64_t round_id, uint64_t outer_step, crypto::Hash256 base_checkpoint,
             uint64_t opened_ms, RoundConfig config)
    : round_id_(round_id),
      outer_step_(outer_step),
      base_checkpoint_(base_checkpoint),
      opened_ms_(opened_ms),
      config_(config) {}

Status Round::Submit(const canon::ContributionHeader& header, uint64_t now_ms) {
    if (auto st = config_.Validate(); !st) return st;

    if (StateAt(now_ms) != RoundState::Open) {
        return Err(std::format("round {}: closed to submissions", round_id_));
    }
    if (header.round_id != round_id_) {
        return Err(std::format("round {}: contribution is for round {}", round_id_,
                               header.round_id));
    }
    if (header.outer_step != outer_step_) {
        return Err(std::format("round {}: contribution is for outer step {}, expected {}",
                               round_id_, header.outer_step, outer_step_));
    }
    // The base checkpoint is what makes a contribution meaningful: a delta computed
    // from different starting weights is not comparable and must not be averaged in.
    if (header.base_checkpoint != base_checkpoint_) {
        return Err(std::format("round {}: worker {} started from a different checkpoint", round_id_,
                               header.worker_id));
    }
    if (header.inner_steps != config_.inner_steps) {
        return Err(std::format("round {}: worker {} did {} inner steps, round requires {}",
                               round_id_, header.worker_id, header.inner_steps,
                               config_.inner_steps));
    }
    if (auto st = header.Validate(); !st) return st;

    const auto existing = submissions_.find(header.worker_id);
    if (existing != submissions_.end() && !config_.allow_resubmit) {
        return Err(std::format("round {}: worker {} already submitted", round_id_,
                               header.worker_id));
    }

    submissions_[header.worker_id] = SubmissionRecord{header, now_ms};
    return Status::Ok();
}

RoundState Round::StateAt(uint64_t now_ms) const {
    const bool past_deadline =
        config_.deadline_ms != 0 && now_ms >= opened_ms_ + config_.deadline_ms;
    const bool has_quorum = submission_count() >= config_.min_contributors;

    if (!past_deadline) return RoundState::Open;
    return has_quorum ? RoundState::Closed : RoundState::Abandoned;
}

bool Round::CanClose(uint64_t now_ms, uint32_t expected_participants) const {
    if (submission_count() < config_.min_contributors) return false;
    // Everyone expected has already reported: no reason to hold the round open for
    // the remainder of its deadline.
    if (expected_participants > 0 && submission_count() >= expected_participants) return true;
    return config_.deadline_ms != 0 && now_ms >= opened_ms_ + config_.deadline_ms;
}

std::vector<canon::ContributionHeader> Round::OrderedSubmissions() const {
    std::vector<canon::ContributionHeader> out;
    out.reserve(submissions_.size());
    for (const auto& [worker_id, record] : submissions_) out.push_back(record.header);
    return out;   // std::map iterates in ascending worker id
}

crypto::Hash256 Round::EvidenceHash() const {
    canon::Writer w;
    w.U64(round_id_);
    w.U64(outer_step_);
    w.Hash(base_checkpoint_);
    w.U32(submission_count());
    for (const auto& header : OrderedSubmissions()) {
        w.U64(header.worker_id);
        w.Hash(header.Id());
    }
    return crypto::Sha3_256(w.data());
}

// ---------------------------------------------------------------------------
// Membership
// ---------------------------------------------------------------------------
void Membership::Join(uint64_t worker_id) { pending_joins_.push_back(worker_id); }

void Membership::Leave(uint64_t worker_id) { pending_leaves_.push_back(worker_id); }

void Membership::AdvanceRound() {
    for (uint64_t id : pending_joins_) active_.try_emplace(id, 0);
    for (uint64_t id : pending_leaves_) active_.erase(id);
    pending_joins_.clear();
    pending_leaves_.clear();
}

bool Membership::IsActive(uint64_t worker_id) const {
    return active_.find(worker_id) != active_.end();
}

std::vector<uint64_t> Membership::ActiveWorkers() const {
    std::vector<uint64_t> out;
    out.reserve(active_.size());
    for (const auto& [worker_id, misses] : active_) out.push_back(worker_id);
    return out;
}

void Membership::RecordParticipation(uint64_t worker_id, bool participated) {
    const auto it = active_.find(worker_id);
    if (it == active_.end()) return;
    it->second = participated ? 0 : it->second + 1;
}

void Membership::DropInactive(uint32_t max_consecutive_misses) {
    for (auto it = active_.begin(); it != active_.end();) {
        it = it->second > max_consecutive_misses ? active_.erase(it) : std::next(it);
    }
}

uint32_t Membership::ConsecutiveMisses(uint64_t worker_id) const {
    const auto it = active_.find(worker_id);
    return it == active_.end() ? 0 : it->second;
}

}  // namespace rnet::diloco

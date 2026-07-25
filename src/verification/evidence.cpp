#include "verification/evidence.h"

#include <algorithm>
#include <format>

#include "util/hex.h"
#include "verification/replay.h"

namespace rnet::verification {

std::string OutcomeName(EvidenceOutcome outcome) {
    switch (outcome) {
        case EvidenceOutcome::Pending: return "pending";
        case EvidenceOutcome::Cleared: return "cleared";
        case EvidenceOutcome::Slashable: return "slashable";
        case EvidenceOutcome::Advisory: return "advisory";
        case EvidenceOutcome::Inconclusive: return "inconclusive";
    }
    return "unknown";
}

EvidenceLedger::EvidenceLedger(uint32_t slash_quorum, bool shadow_mode)
    : slash_quorum_(slash_quorum == 0 ? 1 : slash_quorum), shadow_mode_(shadow_mode) {}

Status EvidenceLedger::Record(const canon::VerificationVerdict& verdict,
                              const canon::ChallengeOrder& challenge) {
    if (auto st = verdict.Validate(); !st) return st;
    if (verdict.contribution_id != challenge.contribution_id) {
        return Err("evidence: verdict and challenge name different contributions");
    }
    if (verdict.challenge_id != challenge.Id()) {
        return Err("evidence: verdict does not answer this challenge");
    }

    ContributionRecord& record = records_[verdict.contribution_id];
    if (record.by_verifier.count(verdict.verifier_id) != 0) {
        return Err(std::format("evidence: verifier {} already reported on this contribution",
                               verdict.verifier_id));
    }
    record.by_verifier[verdict.verifier_id] = verdict;

    switch (verdict.kind) {
        case canon::VerdictKind::Match: ++record.matches; break;
        case canon::VerdictKind::Mismatch:
            ++record.mismatches;
            if (IsSlashable(verdict, challenge)) ++record.slashable;
            break;
        case canon::VerdictKind::Unavailable: ++record.unavailable; break;
        case canon::VerdictKind::OutOfClass: ++record.out_of_class; break;
    }
    return Status::Ok();
}

EvidenceOutcome EvidenceLedger::OutcomeFor(const crypto::Hash256& contribution_id) const {
    const auto it = records_.find(contribution_id);
    if (it == records_.end()) return EvidenceOutcome::Pending;
    const ContributionRecord& record = it->second;

    // Out-of-class and unavailable verdicts carry no judgement either way.
    const uint32_t judging = record.matches + record.mismatches;

    if (record.slashable >= slash_quorum_) {
        // Mismatches carry weight only if nobody reproduced the work successfully;
        // a split verdict means the evidence does not establish anything.
        return record.matches == 0 ? EvidenceOutcome::Slashable : EvidenceOutcome::Inconclusive;
    }
    if (record.mismatches >= slash_quorum_ && record.matches == 0) {
        // Enough mismatches, but the determinism class is unassigned, so this may
        // be a hardware difference rather than dishonesty. Recorded, never settled.
        return EvidenceOutcome::Advisory;
    }
    if (record.matches >= slash_quorum_ && record.mismatches == 0) return EvidenceOutcome::Cleared;
    if (judging == 0) return EvidenceOutcome::Pending;
    if (record.matches > 0 && record.mismatches > 0) return EvidenceOutcome::Inconclusive;
    return EvidenceOutcome::Pending;
}

Result<canon::SlashEvidence> EvidenceLedger::BuildEvidence(const crypto::Hash256& contribution_id,
                                                           uint64_t worker_id, uint64_t round_id,
                                                           uint64_t finalized_at_step) const {
    const auto it = records_.find(contribution_id);
    if (it == records_.end()) return Err("evidence: no verdicts for this contribution");

    const EvidenceOutcome outcome = OutcomeFor(contribution_id);
    if (outcome != EvidenceOutcome::Slashable && outcome != EvidenceOutcome::Advisory) {
        return Err("evidence: outcome is " + OutcomeName(outcome) + ", not actionable");
    }

    canon::SlashEvidence evidence;
    evidence.contribution_id = contribution_id;
    evidence.worker_id = worker_id;
    evidence.round_id = round_id;
    evidence.finalized_at_step = finalized_at_step;

    uint16_t determinism_class = 0;
    for (const auto& [verifier_id, verdict] : it->second.by_verifier) {
        if (verdict.kind != canon::VerdictKind::Mismatch) continue;
        evidence.verdict_ids.push_back(verdict.Id());
        determinism_class = verdict.determinism_class;
    }
    evidence.determinism_class = determinism_class;

    // Sorted so two nodes building this independently produce identical bytes.
    std::sort(evidence.verdict_ids.begin(), evidence.verdict_ids.end());
    evidence.verdict_ids.erase(std::unique(evidence.verdict_ids.begin(), evidence.verdict_ids.end()),
                               evidence.verdict_ids.end());

    if (auto st = evidence.Validate(); !st) return Err(st.error());
    return evidence;
}

bool EvidenceLedger::WouldSettle(EvidenceOutcome outcome) const {
    if (shadow_mode_) return false;   // evidence recorded, nothing settled
    return outcome == EvidenceOutcome::Slashable;
}

uint32_t EvidenceLedger::VerdictCount(const crypto::Hash256& contribution_id) const {
    const auto it = records_.find(contribution_id);
    return it == records_.end() ? 0 : static_cast<uint32_t>(it->second.by_verifier.size());
}

uint32_t EvidenceLedger::MismatchCount(const crypto::Hash256& contribution_id) const {
    const auto it = records_.find(contribution_id);
    return it == records_.end() ? 0 : it->second.mismatches;
}

uint32_t VerificationOverheadPercent(const consensus::NetworkParams& params) {
    // Replaying H inner steps costs what the worker spent on them. There is no
    // shortcut that makes checking cheaper than doing, so the overhead is exactly
    // the fraction of work that gets rechecked.
    return params.policy.challenge_percent;
}

}  // namespace rnet::verification

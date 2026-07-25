#include "verification/challenge.h"

#include <format>

#include "canon/canon.h"

namespace rnet::verification {

namespace {

// Domain-separated draws from the same beacon: selection and verifier assignment
// must not be derivable from one another.
constexpr uint8_t kDomainSelect = 0x01;
constexpr uint8_t kDomainAssign = 0x02;

crypto::Hash256 Draw(uint8_t domain, const crypto::Hash256& beacon,
                     const crypto::Hash256& contribution_id) {
    canon::Writer w;
    w.U8(domain);
    w.Hash(beacon);
    w.Hash(contribution_id);
    return crypto::Sha3_256(w.data());
}

uint64_t Be64(const crypto::Hash256& h) {
    uint64_t v = 0;
    for (size_t i = 0; i < 8; ++i) v = (v << 8) | h[i];
    return v;
}

}  // namespace

bool IsSelected(const crypto::Hash256& beacon, const crypto::Hash256& contribution_id,
                uint32_t challenge_percent) {
    if (challenge_percent == 0) return false;
    if (challenge_percent >= 100) return true;
    const uint64_t draw = Be64(Draw(kDomainSelect, beacon, contribution_id)) % kSelectionDenominator;
    return draw < static_cast<uint64_t>(challenge_percent) * 100;
}

Result<uint64_t> AssignVerifier(const crypto::Hash256& beacon,
                                const crypto::Hash256& contribution_id,
                                std::span<const uint64_t> eligible_verifiers) {
    if (eligible_verifiers.empty()) {
        return Err("challenge: no eligible verifier in this determinism class");
    }
    const uint64_t draw = Be64(Draw(kDomainAssign, beacon, contribution_id));
    return eligible_verifiers[draw % eligible_verifiers.size()];
}

Result<std::vector<canon::ChallengeOrder>> SelectChallenges(
    std::span<const canon::ContributionHeader> contributions, const crypto::Hash256& beacon,
    uint64_t beacon_height, uint64_t issued_at_step, uint16_t determinism_class,
    std::span<const uint64_t> eligible_verifiers, uint32_t challenge_percent) {
    if (beacon == crypto::Hash256{}) {
        return Err("challenge: beacon must be external randomness, not zero");
    }
    if (challenge_percent > 100) return Err("challenge: challenge_percent must be <= 100");

    std::vector<canon::ChallengeOrder> orders;
    for (const auto& contribution : contributions) {
        const crypto::Hash256 id = contribution.Id();
        if (!IsSelected(beacon, id, challenge_percent)) continue;

        // A worker must not audit its own work.
        std::vector<uint64_t> candidates;
        candidates.reserve(eligible_verifiers.size());
        for (uint64_t verifier : eligible_verifiers) {
            if (verifier != contribution.worker_id) candidates.push_back(verifier);
        }
        auto verifier = AssignVerifier(beacon, id, candidates);
        if (!verifier) return Err(verifier.error());

        canon::ChallengeOrder order;
        order.round_id = contribution.round_id;
        order.outer_step = contribution.outer_step;
        order.contribution_id = id;
        order.beacon = beacon;
        order.beacon_height = beacon_height;
        order.verifier_id = verifier.value();
        order.issued_at_step = issued_at_step;
        order.determinism_class = determinism_class;
        if (auto st = order.Validate(); !st) return Err(st.error());
        orders.push_back(order);
    }
    return orders;
}

uint64_t ChallengeExpiryStep(const canon::ChallengeOrder& challenge,
                             uint32_t challenge_deadline_steps) {
    return challenge.issued_at_step + challenge_deadline_steps;
}

bool IsChallengeOpen(const canon::ChallengeOrder& challenge, uint64_t current_step,
                     uint32_t challenge_deadline_steps) {
    return current_step <= ChallengeExpiryStep(challenge, challenge_deadline_steps);
}

Status ValidateChallengeWindow(const consensus::NetworkParams& params) {
    if (params.policy.challenge_deadline_steps >= params.policy.retained_checkpoints) {
        return Err(std::format(
            "verification: challenge window ({} steps) reaches past the retention window ({} "
            "checkpoints) — those challenges could never be answered",
            params.policy.challenge_deadline_steps, params.policy.retained_checkpoints));
    }
    return Status::Ok();
}

}  // namespace rnet::verification

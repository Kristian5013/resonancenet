#include "verification/replay.h"

#include <format>

#include "canon/canon.h"

namespace rnet::verification {

crypto::Hash256 ReplayInputsHash(const ReplayInputs& inputs) {
    canon::Writer w;
    w.Hash(inputs.base_checkpoint);
    w.Hash(inputs.dataset_root);
    w.U64(inputs.round_id);
    w.U64(inputs.worker_id);
    w.U64(inputs.outer_step);
    w.U32(inputs.inner_steps);
    w.U32(inputs.micro_batch);
    w.U64(inputs.n_tokens);
    w.U32(inputs.seq_len);
    return crypto::Sha3_256(w.data());
}

Result<canon::VerificationVerdict> MakeVerdict(const canon::ChallengeOrder& challenge,
                                               const canon::ContributionHeader& contribution,
                                               const crypto::Hash256& recomputed,
                                               uint16_t verifier_class, uint64_t current_step) {
    if (challenge.contribution_id != contribution.Id()) {
        return Err("replay: verdict does not correspond to the challenged contribution");
    }

    canon::VerificationVerdict verdict;
    verdict.challenge_id = challenge.Id();
    verdict.contribution_id = challenge.contribution_id;
    verdict.verifier_id = challenge.verifier_id;
    verdict.claimed_payload_hash = contribution.payload_hash;
    verdict.recomputed_payload_hash = recomputed;
    verdict.determinism_class = verifier_class;
    verdict.reported_at_step = current_step;

    if (verifier_class != challenge.determinism_class) {
        // Different arithmetic: this verifier cannot make a bit-exact statement
        // about that worker's output, and pretending otherwise would manufacture
        // false mismatches.
        verdict.kind = canon::VerdictKind::OutOfClass;
        return verdict;
    }

    verdict.kind = (recomputed == contribution.payload_hash) ? canon::VerdictKind::Match
                                                             : canon::VerdictKind::Mismatch;
    if (auto st = verdict.Validate(); !st) return Err(st.error());
    return verdict;
}

canon::VerificationVerdict MakeUnavailableVerdict(const canon::ChallengeOrder& challenge,
                                                  const canon::ContributionHeader& contribution,
                                                  uint64_t verifier_id, uint16_t verifier_class,
                                                  uint64_t current_step) {
    canon::VerificationVerdict verdict;
    verdict.challenge_id = challenge.Id();
    verdict.contribution_id = challenge.contribution_id;
    verdict.verifier_id = verifier_id;
    verdict.kind = canon::VerdictKind::Unavailable;
    verdict.claimed_payload_hash = contribution.payload_hash;
    verdict.recomputed_payload_hash = crypto::Hash256{};
    verdict.determinism_class = verifier_class;
    verdict.reported_at_step = current_step;
    return verdict;
}

bool IsSlashable(const canon::VerificationVerdict& verdict, const canon::ChallengeOrder& challenge) {
    if (verdict.kind != canon::VerdictKind::Mismatch) return false;
    if (verdict.challenge_id != challenge.Id()) return false;
    if (verdict.determinism_class != challenge.determinism_class) return false;
    // Until determinism classes are measured and assigned, a mismatch is not
    // evidence of dishonesty — it may simply be a different GPU architecture.
    if (verdict.determinism_class == kDeterminismClassPending) return false;
    return true;
}

Result<uint64_t> MinimumStakeMultiple(uint32_t challenge_percent) {
    if (challenge_percent == 0) {
        return Err("verification: with no challenges, no bond makes cheating unprofitable");
    }
    if (challenge_percent > 100) return Err("verification: challenge_percent must be <= 100");
    if (challenge_percent == 100) return uint64_t{1};   // always caught

    // stake > benefit * (1 - p) / p, with p = challenge_percent/100. Integer
    // arithmetic, rounded up, so the published requirement is never too lenient.
    const uint64_t numerator = 100 - challenge_percent;
    const uint64_t multiple = (numerator + challenge_percent - 1) / challenge_percent;
    return multiple + 1;   // strictly greater than break-even
}

}  // namespace rnet::verification

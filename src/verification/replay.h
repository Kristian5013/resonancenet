// Answering a challenge by recomputation, and what the answer is worth.
//
// The verifier holds the same inputs the worker had — base weights named by the
// contribution, batches fixed by the deterministic schedule, the round's inner
// step count — and recomputes the pseudo-gradient. If the payload hashes match
// bit for bit, the work was done as claimed.
//
// There is no asymmetry in the verifier's favour. Replaying H inner steps costs
// the same GPU time the worker spent, so the network's verification overhead is
// challenge_percent of its total compute, and no parameter choice can escape
// that. What challenge_percent actually buys is the stake multiple required to
// make cheating unprofitable — see MinimumStakeMultiple().
//
// COMPARISON IS EXACT OR IT IS NOTHING. A tolerance would reintroduce the hiding
// place this project already measured: a targeted backdoor is a small, coherent
// perturbation, exactly the shape that fits under a loose threshold. So the only
// verdicts are match, mismatch, or "I could not judge this".
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "canon/objects.h"
#include "util/result.h"

namespace rnet::verification {

// A determinism class of 0 means "not yet assigned": cross-architecture
// bit-exactness has not been established. Until it is, a mismatch may mean
// "different GPU" rather than "dishonest", so verdicts in class 0 are advisory.
inline constexpr uint16_t kDeterminismClassPending = 0;

struct ReplayInputs {
    crypto::Hash256 base_checkpoint{};   // weights the round started from
    crypto::Hash256 dataset_root{};      // corpus the schedule draws from
    uint64_t round_id{};
    uint64_t worker_id{};
    uint64_t outer_step{};
    uint32_t inner_steps{};
    uint32_t micro_batch{};
    uint64_t n_tokens{};
    uint32_t seq_len{};
};

// Everything a verifier must reproduce, hashed. If a verifier and a worker
// disagree about any of it they were not running the same experiment, and a
// mismatch would say nothing about honesty.
crypto::Hash256 ReplayInputsHash(const ReplayInputs& inputs);

// Builds the verdict for a completed replay. `recomputed` is the hash of the
// payload the verifier produced.
Result<canon::VerificationVerdict> MakeVerdict(const canon::ChallengeOrder& challenge,
                                               const canon::ContributionHeader& contribution,
                                               const crypto::Hash256& recomputed,
                                               uint16_t verifier_class, uint64_t current_step);

// The verdict a verifier must report when it cannot obtain the inputs — pruned
// weights, missing payload, corpus unavailable. Recorded rather than silently
// dropped, because a pattern of Unavailable is itself information.
canon::VerificationVerdict MakeUnavailableVerdict(const canon::ChallengeOrder& challenge,
                                                  const canon::ContributionHeader& contribution,
                                                  uint64_t verifier_id, uint16_t verifier_class,
                                                  uint64_t current_step);

// Whether a verdict can support slashing at all. Mismatches from an unassigned
// determinism class, or from a class other than the challenge's, prove nothing
// about honesty and must never cost anyone their stake.
bool IsSlashable(const canon::VerificationVerdict& verdict, const canon::ChallengeOrder& challenge);

// Minimum stake, as a multiple of the benefit of cheating, for cheating to be
// negative expected value.
//
//   EV(cheat) = (1 - p) * benefit - p * stake < 0   =>   stake > benefit * (1 - p) / p
//
// with p = challenge_percent / 100. This is the real meaning of challenge_percent:
// not "how much we check" but "how large a bond a participant must post". Lower
// verification overhead is paid for with a larger bond.
Result<uint64_t> MinimumStakeMultiple(uint32_t challenge_percent);

}  // namespace rnet::verification

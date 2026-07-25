// Deciding what gets recomputed, and by whom.
//
// Selection must satisfy two opposing requirements:
//
//   * UNPREDICTABLE BEFORE. If a worker can tell in advance that its contribution
//     will not be checked, it cheats exactly there. So selection is driven by a
//     beacon fixed only after the round closed — external randomness, never a
//     value any participant here can grind.
//
//   * VERIFIABLE AFTER. Anyone must be able to re-derive the same selection from
//     public inputs, or an accuser could invent challenges to harass a worker.
//     Selection is therefore a pure function of (beacon, contribution id).
//
// The challenge window is bounded by storage, not by policy. Replaying a round
// requires the weights it started from, and those live only inside the retention
// window — so `challenge_deadline_steps < retained_checkpoints` is a consensus
// invariant, enforced in NetworkParams::Validate().
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "canon/objects.h"
#include "consensus/params.h"
#include "util/result.h"

namespace rnet::verification {

// Selection probability is expressed in basis points so the comparison is exact
// integer arithmetic; a percentage in floating point could round differently on
// two nodes and make them disagree about who was challenged.
inline constexpr uint64_t kSelectionDenominator = 10'000;

// True if this contribution is selected under the given beacon. Pure, integer,
// reproducible by anyone holding the same public inputs.
bool IsSelected(const crypto::Hash256& beacon, const crypto::Hash256& contribution_id,
                uint32_t challenge_percent);

// Picks the verifier for a challenge from the eligible set. Eligibility is by
// determinism class: a verifier whose arithmetic belongs to a different class
// cannot produce a meaningful bit-exact comparison, so assigning it would only
// manufacture false mismatches.
Result<uint64_t> AssignVerifier(const crypto::Hash256& beacon,
                                const crypto::Hash256& contribution_id,
                                std::span<const uint64_t> eligible_verifiers);

// Builds the challenges for one closed round.
Result<std::vector<canon::ChallengeOrder>> SelectChallenges(
    std::span<const canon::ContributionHeader> contributions, const crypto::Hash256& beacon,
    uint64_t beacon_height, uint64_t issued_at_step, uint16_t determinism_class,
    std::span<const uint64_t> eligible_verifiers, uint32_t challenge_percent);

// A challenge is answerable only while the base weights still exist.
bool IsChallengeOpen(const canon::ChallengeOrder& challenge, uint64_t current_step,
                     uint32_t challenge_deadline_steps);

// The step at which a challenge expires.
uint64_t ChallengeExpiryStep(const canon::ChallengeOrder& challenge,
                             uint32_t challenge_deadline_steps);

// Confirms the network's storage window can actually answer its own challenges.
// Called on startup so a misconfiguration surfaces before the network relies on
// verification that cannot physically happen.
Status ValidateChallengeWindow(const consensus::NetworkParams& params);

}  // namespace rnet::verification

// Collecting verdicts into evidence, and deciding what happens next.
//
// Slashing needs a quorum. A single verifier — colluding, buggy, or simply on
// hardware whose class was assigned wrongly — must never be able to burn an
// honest worker's stake.
//
// SHADOW MODE. While the network is in shadow mode, evidence is collected,
// hashed and recorded exactly as it would be in production, but nothing is
// settled. Two independent reasons, and either alone is sufficient:
//
//   * No token exists until convergence is demonstrated, by explicit design.
//   * Cross-architecture bit-exactness is an open measurement. Until determinism
//     classes are assigned from real data, a mismatch may mean "different GPU"
//     rather than "dishonest", and acting on it would punish honest volunteers.
//
// The mechanics are fully exercised in shadow mode, so enabling settlement later
// is a parameter change rather than new, untested code on a live network.
#pragma once

#include <cstdint>
#include <map>
#include <span>
#include <vector>

#include "canon/objects.h"
#include "consensus/params.h"
#include "util/result.h"

namespace rnet::verification {

enum class EvidenceOutcome {
    Pending,          // not enough verdicts yet
    Cleared,          // quorum of matches: the contribution stands
    Slashable,        // quorum of mismatches, and the class is assigned
    Advisory,         // quorum of mismatches, but the class is unassigned
    Inconclusive,     // verdicts disagree, or too many were unanswerable
};

std::string OutcomeName(EvidenceOutcome outcome);

// Accumulates verdicts for one contribution and reports what they add up to.
class EvidenceLedger {
public:
    EvidenceLedger(uint32_t slash_quorum, bool shadow_mode);

    // Records a verdict. Rejects a second verdict from the same verifier: without
    // that, one verifier could manufacture a quorum by itself.
    Status Record(const canon::VerificationVerdict& verdict, const canon::ChallengeOrder& challenge);

    EvidenceOutcome OutcomeFor(const crypto::Hash256& contribution_id) const;

    // Builds the evidence object for a Slashable or Advisory outcome. The verdict
    // ids are sorted so two nodes independently building it produce identical
    // bytes and therefore the same evidence id.
    Result<canon::SlashEvidence> BuildEvidence(const crypto::Hash256& contribution_id,
                                               uint64_t worker_id, uint64_t round_id,
                                               uint64_t finalized_at_step) const;

    // Whether this outcome causes an actual penalty on this network. Always false
    // in shadow mode — evidence is still recorded.
    bool WouldSettle(EvidenceOutcome outcome) const;

    uint32_t VerdictCount(const crypto::Hash256& contribution_id) const;
    uint32_t MismatchCount(const crypto::Hash256& contribution_id) const;
    bool shadow_mode() const { return shadow_mode_; }
    uint32_t slash_quorum() const { return slash_quorum_; }

private:
    struct ContributionRecord {
        std::map<uint64_t, canon::VerificationVerdict> by_verifier;   // verifier_id -> verdict
        uint32_t matches{0};
        uint32_t mismatches{0};
        uint32_t unavailable{0};
        uint32_t out_of_class{0};
        uint32_t slashable{0};      // mismatches that are actually actionable
    };

    uint32_t slash_quorum_;
    bool shadow_mode_;
    std::map<crypto::Hash256, ContributionRecord> records_;
};

// Verification overhead as a fraction of the network's compute. Replay costs
// exactly what the original work cost, so this is simply challenge_percent —
// stated as a function to keep the honesty of that fact in the code rather than
// in a comment someone can forget.
uint32_t VerificationOverheadPercent(const consensus::NetworkParams& params);

}  // namespace rnet::verification

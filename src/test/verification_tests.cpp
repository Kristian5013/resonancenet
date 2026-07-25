#include <vector>

#include "consensus/params.h"
#include "test/framework.h"
#include "verification/challenge.h"
#include "verification/evidence.h"
#include "verification/replay.h"

using namespace rnet;
using namespace rnet::verification;

namespace {

const crypto::Hash256 kBase = crypto::Sha3_256(std::string_view("base-weights"));
const crypto::Hash256 kBeacon = crypto::Sha3_256(std::string_view("external-beacon-height-1000"));

canon::ContributionHeader MakeContribution(uint64_t worker, uint64_t step = 1) {
    canon::ContributionHeader h;
    h.round_id = 0;
    h.worker_id = worker;
    h.outer_step = step;
    h.base_checkpoint = kBase;
    h.payload_hash = crypto::Sha3_256(std::string_view("payload-" + std::to_string(worker)));
    h.n_params = 1000;
    h.inner_steps = 250;
    h.scale_exp = 10;
    return h;
}

canon::ChallengeOrder MakeChallenge(const canon::ContributionHeader& contribution,
                                    uint64_t verifier, uint16_t determinism_class = 1) {
    canon::ChallengeOrder c;
    c.round_id = contribution.round_id;
    c.outer_step = contribution.outer_step;
    c.contribution_id = contribution.Id();
    c.beacon = kBeacon;
    c.beacon_height = 1000;
    c.verifier_id = verifier;
    c.issued_at_step = contribution.outer_step;
    c.determinism_class = determinism_class;
    return c;
}

}  // namespace

// ---------------------------------------------------------------------------
// The invariant that ties verification to storage
// ---------------------------------------------------------------------------

// Replay needs the weights the round started from. Those exist only inside the
// retention window, so a challenge that outlives them is unanswerable by
// construction — the deadline must expire first. This is a consensus parameter,
// not a tunable.
RNET_TEST(ChallengeWindow, MustFitInsideTheStorageWindow) {
    for (auto net : {consensus::Network::Main, consensus::Network::Test,
                     consensus::Network::Regtest}) {
        const auto& params = consensus::Params(net);
        RNET_CHECK_OK(ValidateChallengeWindow(params));
        if (params.policy.challenge_deadline_steps >= params.policy.retained_checkpoints) {
            RNET_FAIL("{}: challenge window {} >= retention {}", params.name,
                      params.policy.challenge_deadline_steps, params.policy.retained_checkpoints);
        }
    }
}

RNET_TEST(ChallengeWindow, ParamsRejectAnUnanswerableConfiguration) {
    auto params = consensus::Params(consensus::Network::Regtest);
    params.policy.challenge_deadline_steps = params.policy.retained_checkpoints;   // one step too far
    RNET_CHECK_ERR(params.Validate());
    RNET_CHECK_ERR(ValidateChallengeWindow(params));
}

RNET_TEST(ChallengeWindow, ExpiresAtTheDeadline) {
    const auto contribution = MakeContribution(1, 10);
    const auto challenge = MakeChallenge(contribution, 2);

    RNET_CHECK(IsChallengeOpen(challenge, 10, 3));
    RNET_CHECK(IsChallengeOpen(challenge, 13, 3));
    RNET_CHECK(!IsChallengeOpen(challenge, 14, 3));
    RNET_CHECK_EQ(ChallengeExpiryStep(challenge, 3), 13ull);
}

// ---------------------------------------------------------------------------
// Selection: unpredictable before, verifiable after
// ---------------------------------------------------------------------------
RNET_TEST(Selection, IsReproducibleFromPublicInputs) {
    const auto contribution = MakeContribution(7);
    const crypto::Hash256 id = contribution.Id();
    const bool first = IsSelected(kBeacon, id, 50);
    for (int i = 0; i < 10; ++i) RNET_CHECK_EQ(IsSelected(kBeacon, id, 50), first);
}

// A worker must not be able to predict its fate from anything it controls: change
// the beacon and the selection changes.
RNET_TEST(Selection, DependsOnTheBeacon) {
    const auto contribution = MakeContribution(7);
    const crypto::Hash256 id = contribution.Id();

    uint32_t differing = 0;
    for (int i = 0; i < 200; ++i) {
        const crypto::Hash256 beacon =
            crypto::Sha3_256(std::string_view("beacon-" + std::to_string(i)));
        if (IsSelected(beacon, id, 50) != IsSelected(kBeacon, id, 50)) ++differing;
    }
    RNET_CHECK(differing > 50);   // clearly beacon-driven, not fixed per contribution
}

RNET_TEST(Selection, HitsTheConfiguredRateApproximately) {
    uint32_t selected = 0;
    const uint32_t trials = 2000;
    for (uint32_t i = 0; i < trials; ++i) {
        const crypto::Hash256 id = crypto::Sha3_256(std::string_view("c-" + std::to_string(i)));
        if (IsSelected(kBeacon, id, 10)) ++selected;
    }
    // 10% of 2000 = 200; allow generous slack for a 2000-sample draw.
    RNET_CHECK(selected > 130 && selected < 280);
}

RNET_TEST(Selection, ExtremesBehaveExactly) {
    const crypto::Hash256 id = crypto::Sha3_256(std::string_view("any"));
    RNET_CHECK(!IsSelected(kBeacon, id, 0));
    RNET_CHECK(IsSelected(kBeacon, id, 100));
}

RNET_TEST(Selection, RejectsAZeroBeacon) {
    const auto contribution = MakeContribution(1);
    const std::vector<canon::ContributionHeader> contributions{contribution};
    const std::vector<uint64_t> verifiers{2, 3};
    // A zero beacon would mean selection was not driven by external randomness —
    // exactly the grinding hole the beacon exists to close.
    RNET_CHECK_ERR(SelectChallenges(contributions, crypto::Hash256{}, 1000, 1, 1, verifiers, 100));
}

RNET_TEST(Assignment, NeverAssignsAWorkerToAuditItself) {
    std::vector<canon::ContributionHeader> contributions;
    for (uint64_t w = 1; w <= 8; ++w) contributions.push_back(MakeContribution(w));
    const std::vector<uint64_t> verifiers{1, 2, 3, 4, 5, 6, 7, 8};

    auto challenges = SelectChallenges(contributions, kBeacon, 1000, 1, 1, verifiers, 100);
    RNET_CHECK_OK(challenges);
    RNET_CHECK_EQ(challenges.value().size(), size_t{8});

    for (const auto& challenge : challenges.value()) {
        for (const auto& contribution : contributions) {
            if (contribution.Id() == challenge.contribution_id) {
                RNET_CHECK(challenge.verifier_id != contribution.worker_id);
            }
        }
    }
}

RNET_TEST(Assignment, IsDeterministicAndSpreadsWork) {
    const std::vector<uint64_t> verifiers{10, 11, 12, 13};
    std::map<uint64_t, uint32_t> counts;
    for (uint32_t i = 0; i < 200; ++i) {
        const crypto::Hash256 id = crypto::Sha3_256(std::string_view("c-" + std::to_string(i)));
        auto a = AssignVerifier(kBeacon, id, verifiers);
        auto b = AssignVerifier(kBeacon, id, verifiers);
        RNET_CHECK_OK(a);
        RNET_CHECK_OK(b);
        RNET_CHECK_EQ(a.value(), b.value());
        ++counts[a.value()];
    }
    RNET_CHECK_EQ(counts.size(), size_t{4});   // every verifier gets work
}

RNET_TEST(Assignment, FailsWhenNoVerifierSharesTheClass) {
    const crypto::Hash256 id = crypto::Sha3_256(std::string_view("c"));
    RNET_CHECK_ERR(AssignVerifier(kBeacon, id, {}));
}

// ---------------------------------------------------------------------------
// Verdicts
// ---------------------------------------------------------------------------
RNET_TEST(Verdict, MatchOnExactReproduction) {
    const auto contribution = MakeContribution(1);
    const auto challenge = MakeChallenge(contribution, 2);

    auto verdict = MakeVerdict(challenge, contribution, contribution.payload_hash, 1, 5);
    RNET_CHECK_OK(verdict);
    RNET_CHECK(verdict.value().kind == canon::VerdictKind::Match);
    RNET_CHECK(IsSlashable(verdict.value(), challenge) == false);
}

RNET_TEST(Verdict, MismatchOnAnyDifference) {
    const auto contribution = MakeContribution(1);
    const auto challenge = MakeChallenge(contribution, 2);
    const crypto::Hash256 different = crypto::Sha3_256(std::string_view("not-what-was-claimed"));

    auto verdict = MakeVerdict(challenge, contribution, different, 1, 5);
    RNET_CHECK_OK(verdict);
    RNET_CHECK(verdict.value().kind == canon::VerdictKind::Mismatch);
    RNET_CHECK(IsSlashable(verdict.value(), challenge));
}

// Different arithmetic cannot produce a meaningful bit-exact statement, so a
// verifier in another class must say so rather than manufacture a mismatch.
RNET_TEST(Verdict, OutOfClassVerifierDoesNotAccuse) {
    const auto contribution = MakeContribution(1);
    const auto challenge = MakeChallenge(contribution, 2, /*class=*/1);
    const crypto::Hash256 different = crypto::Sha3_256(std::string_view("different-hardware"));

    auto verdict = MakeVerdict(challenge, contribution, different, /*verifier_class=*/2, 5);
    RNET_CHECK_OK(verdict);
    RNET_CHECK(verdict.value().kind == canon::VerdictKind::OutOfClass);
    RNET_CHECK(!IsSlashable(verdict.value(), challenge));
}

// The safety property that protects honest volunteers today: while the
// determinism class is unassigned, a mismatch may mean "different GPU", so it
// must never be actionable.
RNET_TEST(Verdict, PendingClassIsNeverSlashable) {
    const auto contribution = MakeContribution(1);
    const auto challenge = MakeChallenge(contribution, 2, kDeterminismClassPending);
    const crypto::Hash256 different = crypto::Sha3_256(std::string_view("maybe-just-a-4090"));

    auto verdict = MakeVerdict(challenge, contribution, different, kDeterminismClassPending, 5);
    RNET_CHECK_OK(verdict);
    RNET_CHECK(verdict.value().kind == canon::VerdictKind::Mismatch);
    RNET_CHECK(!IsSlashable(verdict.value(), challenge));
}

RNET_TEST(Verdict, MalformedVerdictsAreRejected) {
    canon::VerificationVerdict verdict;
    verdict.challenge_id = crypto::Sha3_256(std::string_view("c"));
    verdict.contribution_id = crypto::Sha3_256(std::string_view("x"));
    verdict.kind = canon::VerdictKind::Match;
    verdict.claimed_payload_hash = crypto::Sha3_256(std::string_view("a"));
    verdict.recomputed_payload_hash = crypto::Sha3_256(std::string_view("b"));
    // Vouching for work whose hashes differ would let a verifier approve anything.
    RNET_CHECK_ERR(verdict.Validate());

    verdict.kind = canon::VerdictKind::Mismatch;
    verdict.recomputed_payload_hash = verdict.claimed_payload_hash;
    RNET_CHECK_ERR(verdict.Validate());
}

RNET_TEST(Verdict, UnavailableIsRecordedNotDropped) {
    const auto contribution = MakeContribution(1);
    const auto challenge = MakeChallenge(contribution, 2);
    const auto verdict = MakeUnavailableVerdict(challenge, contribution, 2, 1, 7);
    RNET_CHECK(verdict.kind == canon::VerdictKind::Unavailable);
    RNET_CHECK_OK(verdict.Validate());
    RNET_CHECK(!IsSlashable(verdict, challenge));
}

RNET_TEST(Replay, InputsHashCoversEverythingReproduced) {
    ReplayInputs a{kBase, crypto::Sha3_256(std::string_view("corpus")), 0, 1, 1, 250, 1, 1'000'000, 8192};
    const crypto::Hash256 base_hash = ReplayInputsHash(a);

    auto b = a; b.inner_steps = 251;
    RNET_CHECK(ReplayInputsHash(b) != base_hash);
    b = a; b.worker_id = 2;
    RNET_CHECK(ReplayInputsHash(b) != base_hash);
    b = a; b.dataset_root[0] ^= 1;
    RNET_CHECK(ReplayInputsHash(b) != base_hash);
    b = a; b.seq_len = 4096;
    RNET_CHECK(ReplayInputsHash(b) != base_hash);
}

// ---------------------------------------------------------------------------
// Evidence and shadow mode
// ---------------------------------------------------------------------------
RNET_TEST(Evidence, RequiresAQuorumBeforeSlashing) {
    const auto contribution = MakeContribution(1);
    const crypto::Hash256 id = contribution.Id();
    const crypto::Hash256 wrong = crypto::Sha3_256(std::string_view("forged"));
    EvidenceLedger ledger(3, /*shadow_mode=*/false);

    for (uint64_t verifier = 2; verifier <= 3; ++verifier) {
        const auto challenge = MakeChallenge(contribution, verifier);
        auto verdict = MakeVerdict(challenge, contribution, wrong, 1, 5);
        RNET_CHECK_OK(verdict);
        RNET_CHECK_OK(ledger.Record(verdict.value(), challenge));
    }
    // Two mismatches, quorum is three: one or two verifiers must not be able to
    // burn an honest worker's stake.
    RNET_CHECK(ledger.OutcomeFor(id) == EvidenceOutcome::Pending);

    const auto challenge = MakeChallenge(contribution, 4);
    auto verdict = MakeVerdict(challenge, contribution, wrong, 1, 5);
    RNET_CHECK_OK(verdict);
    RNET_CHECK_OK(ledger.Record(verdict.value(), challenge));
    RNET_CHECK(ledger.OutcomeFor(id) == EvidenceOutcome::Slashable);
}

RNET_TEST(Evidence, OneVerifierCannotManufactureAQuorum) {
    const auto contribution = MakeContribution(1);
    const crypto::Hash256 wrong = crypto::Sha3_256(std::string_view("forged"));
    EvidenceLedger ledger(2, false);

    const auto challenge = MakeChallenge(contribution, 2);
    auto verdict = MakeVerdict(challenge, contribution, wrong, 1, 5);
    RNET_CHECK_OK(verdict);
    RNET_CHECK_OK(ledger.Record(verdict.value(), challenge));
    // The same verifier reporting twice must not count twice.
    RNET_CHECK_ERR(ledger.Record(verdict.value(), challenge));
    RNET_CHECK_EQ(ledger.VerdictCount(contribution.Id()), 1u);
}

RNET_TEST(Evidence, DisagreementIsInconclusiveNotPunitive) {
    const auto contribution = MakeContribution(1);
    const crypto::Hash256 wrong = crypto::Sha3_256(std::string_view("forged"));
    EvidenceLedger ledger(1, false);

    const auto c1 = MakeChallenge(contribution, 2);
    auto mismatch = MakeVerdict(c1, contribution, wrong, 1, 5);
    RNET_CHECK_OK(mismatch);
    RNET_CHECK_OK(ledger.Record(mismatch.value(), c1));

    const auto c2 = MakeChallenge(contribution, 3);
    auto match = MakeVerdict(c2, contribution, contribution.payload_hash, 1, 5);
    RNET_CHECK_OK(match);
    RNET_CHECK_OK(ledger.Record(match.value(), c2));

    // Someone reproduced the work successfully: the evidence no longer establishes
    // dishonesty, whatever the other verifier reported.
    RNET_CHECK(ledger.OutcomeFor(contribution.Id()) == EvidenceOutcome::Inconclusive);
}

RNET_TEST(Evidence, PendingClassYieldsAdvisoryNotSlashable) {
    const auto contribution = MakeContribution(1);
    const crypto::Hash256 wrong = crypto::Sha3_256(std::string_view("maybe-different-gpu"));
    EvidenceLedger ledger(2, false);

    for (uint64_t verifier = 2; verifier <= 3; ++verifier) {
        const auto challenge = MakeChallenge(contribution, verifier, kDeterminismClassPending);
        auto verdict = MakeVerdict(challenge, contribution, wrong, kDeterminismClassPending, 5);
        RNET_CHECK_OK(verdict);
        RNET_CHECK_OK(ledger.Record(verdict.value(), challenge));
    }
    // Until determinism classes are measured, this is information, not proof.
    RNET_CHECK(ledger.OutcomeFor(contribution.Id()) == EvidenceOutcome::Advisory);
    RNET_CHECK(!ledger.WouldSettle(EvidenceOutcome::Advisory));
}

// Shadow mode exercises the entire mechanism and settles nothing, so enabling
// economics later is a parameter change rather than new code on a live network.
RNET_TEST(Evidence, ShadowModeRecordsButNeverSettles) {
    const auto contribution = MakeContribution(1);
    const crypto::Hash256 wrong = crypto::Sha3_256(std::string_view("forged"));
    EvidenceLedger shadow(2, /*shadow_mode=*/true);

    for (uint64_t verifier = 2; verifier <= 3; ++verifier) {
        const auto challenge = MakeChallenge(contribution, verifier);
        auto verdict = MakeVerdict(challenge, contribution, wrong, 1, 5);
        RNET_CHECK_OK(verdict);
        RNET_CHECK_OK(shadow.Record(verdict.value(), challenge));
    }

    const auto outcome = shadow.OutcomeFor(contribution.Id());
    RNET_CHECK(outcome == EvidenceOutcome::Slashable);   // fully determined
    RNET_CHECK(!shadow.WouldSettle(outcome));            // and deliberately not acted on

    // The evidence object is still built and hashed, exactly as in production.
    auto evidence = shadow.BuildEvidence(contribution.Id(), 1, 0, 9);
    RNET_CHECK_OK(evidence);
    RNET_CHECK_EQ(evidence.value().verdict_ids.size(), size_t{2});
}

RNET_TEST(Evidence, IsIdenticalWhenBuiltIndependently) {
    const auto contribution = MakeContribution(1);
    const crypto::Hash256 wrong = crypto::Sha3_256(std::string_view("forged"));

    auto build = [&](bool reverse) {
        EvidenceLedger ledger(2, false);
        std::vector<uint64_t> verifiers{2, 3, 4};
        if (reverse) std::reverse(verifiers.begin(), verifiers.end());
        for (uint64_t verifier : verifiers) {
            const auto challenge = MakeChallenge(contribution, verifier);
            auto verdict = MakeVerdict(challenge, contribution, wrong, 1, 5);
            (void)ledger.Record(verdict.value(), challenge);
        }
        return ledger.BuildEvidence(contribution.Id(), 1, 0, 9);
    };

    auto forward = build(false);
    auto backward = build(true);
    RNET_CHECK_OK(forward);
    RNET_CHECK_OK(backward);
    // Two nodes that saw the same verdicts in different orders must publish the
    // same evidence id.
    RNET_CHECK_EQ(forward.value().Id(), backward.value().Id());
}

RNET_TEST(Evidence, MainnetShipsInShadowMode) {
    const auto& params = consensus::Params(consensus::Network::Main);
    // No settlement before convergence is demonstrated and determinism classes
    // are measured — both are still open.
    RNET_CHECK(params.policy.shadow_mode != 0);
}

// ---------------------------------------------------------------------------
// Economics
// ---------------------------------------------------------------------------

// What challenge_percent actually controls is the bond, not merely "how much we
// check": lower verification overhead is paid for with a larger stake.
RNET_TEST(Economics, StakeRequirementFollowsFromChallengeRate) {
    auto five = MinimumStakeMultiple(5);
    RNET_CHECK_OK(five);
    RNET_CHECK(five.value() >= 20);      // 5% checked -> bond above 19x the benefit

    auto ten = MinimumStakeMultiple(10);
    RNET_CHECK_OK(ten);
    RNET_CHECK(ten.value() >= 10 && ten.value() < five.value());

    auto all = MinimumStakeMultiple(100);
    RNET_CHECK_OK(all);
    RNET_CHECK_EQ(all.value(), 1ull);    // always caught: any bond suffices

    // With nothing checked, no bond makes cheating unprofitable.
    RNET_CHECK_ERR(MinimumStakeMultiple(0));
}

// Replay costs what the original work cost. There is no asymmetry favouring the
// verifier, and no parameter choice escapes it.
RNET_TEST(Economics, OverheadEqualsTheChallengeRate) {
    for (auto net : {consensus::Network::Main, consensus::Network::Test,
                     consensus::Network::Regtest}) {
        const auto& params = consensus::Params(net);
        RNET_CHECK_EQ(VerificationOverheadPercent(params), params.policy.challenge_percent);
    }
}

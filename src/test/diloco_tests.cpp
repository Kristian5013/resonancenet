#include <algorithm>
#include <random>

#include "diloco/aggregation.h"
#include "diloco/chain.h"
#include "diloco/outer_optimizer.h"
#include "diloco/quantize.h"
#include "diloco/round.h"
#include "test/framework.h"

using namespace rnet;
using namespace rnet::diloco;

namespace {

canon::ContributionHeader MakeHeader(uint64_t worker, uint64_t round_id, uint64_t step,
                                     const crypto::Hash256& base, uint32_t inner_steps = 250) {
    canon::ContributionHeader h;
    h.round_id = round_id;
    h.worker_id = worker;
    h.outer_step = step;
    h.base_checkpoint = base;
    h.payload_hash = crypto::Sha3_256(std::string_view("payload"));
    h.n_params = 1000;
    h.inner_steps = inner_steps;
    h.scale_exp = 10;
    return h;
}

canon::CheckpointHeader MakeCheckpoint(const crypto::Hash256& parent, uint64_t step,
                                       uint32_t contributors = 4) {
    canon::CheckpointHeader c;
    c.parent = parent;
    c.round_id = 0;
    c.outer_step = step;
    c.weights_hash = crypto::Sha3_256(std::string_view("weights-" + std::to_string(step)));
    c.n_params = 1'000'000;
    c.contributor_count = contributors;
    // A non-genesis checkpoint must name what it aggregated, or nobody could
    // recompute it. Any non-zero value satisfies the invariant here; the real
    // derivation is tested where checkpoints are produced.
    if (contributors > 0) {
        c.evidence_hash = crypto::Sha3_256(std::string_view("test-evidence"));
    }
    c.weight_dtype = 1;   // bf16
    return c;
}

const crypto::Hash256 kBase = crypto::Sha3_256(std::string_view("base-checkpoint"));

}  // namespace

// ---------------------------------------------------------------------------
// Quantisation
// ---------------------------------------------------------------------------
RNET_TEST(Quantize, RoundTripsWithinResolution) {
    const std::vector<float> deltas{0.5f, -0.25f, 0.125f, -1.0f, 0.0f};
    auto exp = ChooseScaleExp(deltas);
    RNET_CHECK_OK(exp);

    auto q = Quantize(deltas, exp.value());
    RNET_CHECK_OK(q);
    RNET_CHECK_EQ(q.value().clamped, 0ull);

    auto back = Dequantize(q.value().values, exp.value());
    RNET_CHECK_OK(back);
    // Every input here is a dyadic rational, so a power-of-two scale reproduces
    // them exactly — this is precisely why the scale is a power of two.
    for (size_t i = 0; i < deltas.size(); ++i) {
        if (back.value()[i] != deltas[i]) {
            RNET_FAIL("element {}: {} != {}", i, back.value()[i], deltas[i]);
        }
    }
}

RNET_TEST(Quantize, ScaleUsesTheFullRangeWithoutClipping) {
    const std::vector<float> deltas{0.003f, -0.001f, 0.0025f};
    auto exp = ChooseScaleExp(deltas);
    RNET_CHECK_OK(exp);

    auto q = Quantize(deltas, exp.value());
    RNET_CHECK_OK(q);
    RNET_CHECK_EQ(q.value().clamped, 0ull);

    int8_t peak = 0;
    for (int8_t v : q.value().values) peak = std::max<int8_t>(peak, static_cast<int8_t>(std::abs(v)));
    // A power-of-two scale can leave up to a factor of two on the table, so the
    // largest magnitude must land in the top half of the range.
    RNET_CHECK(peak > 63);
}

RNET_TEST(Quantize, NonFiniteInputIsRejected) {
    const std::vector<float> nan_delta{0.5f, std::numeric_limits<float>::quiet_NaN()};
    const std::vector<float> inf_delta{0.5f, std::numeric_limits<float>::infinity()};
    // A diverged local run must not be quantised into plausible-looking integers.
    RNET_CHECK_ERR(ChooseScaleExp(nan_delta));
    RNET_CHECK_ERR(ChooseScaleExp(inf_delta));
    RNET_CHECK_ERR(Quantize(nan_delta, 10));
}

RNET_TEST(Quantize, ClampingIsReported) {
    const std::vector<float> deltas{1000.0f, -1000.0f, 0.0f};
    auto q = Quantize(deltas, 10);   // deliberately far too fine
    RNET_CHECK_OK(q);
    RNET_CHECK_EQ(q.value().clamped, 2ull);   // never silent
}

RNET_TEST(Quantize, RangeIsSymmetric) {
    const std::vector<float> deltas{1.0f, -1.0f};
    auto q = Quantize(deltas, 7);   // 1.0 * 128 -> clamps to 127
    RNET_CHECK_OK(q);
    RNET_CHECK_EQ(q.value().values[0], int8_t{127});
    // -128 is excluded so the representable range is symmetric; an asymmetric
    // range would bias the mean of many contributions.
    RNET_CHECK_EQ(q.value().values[1], int8_t{-127});
}

RNET_TEST(Quantize, RoundingIsTiesAwayFromZero) {
    RNET_CHECK_EQ(RoundTiesAwayFromZero(0.5f), 1);
    RNET_CHECK_EQ(RoundTiesAwayFromZero(-0.5f), -1);
    RNET_CHECK_EQ(RoundTiesAwayFromZero(1.5f), 2);
    RNET_CHECK_EQ(RoundTiesAwayFromZero(-1.5f), -2);
    RNET_CHECK_EQ(RoundTiesAwayFromZero(2.4f), 2);
}

// ---------------------------------------------------------------------------
// Aggregation — every contribution counts, order cannot matter
// ---------------------------------------------------------------------------
RNET_TEST(Aggregate, AveragesAllContributions) {
    const std::vector<int8_t> a{10, 20, 30};
    const std::vector<int8_t> b{20, 40, 60};
    const std::vector<Contribution> contributions{{1, 10, a}, {2, 10, b}};

    auto agg = AverageContributions(contributions, 3);
    RNET_CHECK_OK(agg);
    RNET_CHECK_EQ(agg.value().contributor_count, 2u);
    RNET_CHECK_EQ(agg.value().values, (std::vector<int64_t>{15, 30, 45}));
}

// The decisive property for a network: contributions arrive in arbitrary order,
// and every node must still compute the identical aggregate.
RNET_TEST(Aggregate, ResultIsIndependentOfArrivalOrder) {
    std::mt19937 rng(42);
    std::vector<std::vector<int8_t>> payloads;
    for (int w = 0; w < 16; ++w) {
        std::vector<int8_t> values(64);
        for (auto& v : values) v = static_cast<int8_t>(static_cast<int>(rng() % 255) - 127);
        payloads.push_back(std::move(values));
    }

    std::vector<Contribution> contributions;
    for (size_t i = 0; i < payloads.size(); ++i) {
        contributions.push_back({static_cast<uint64_t>(i), 10, payloads[i]});
    }

    auto reference = AverageContributions(contributions, 64);
    RNET_CHECK_OK(reference);

    for (int trial = 0; trial < 8; ++trial) {
        auto shuffled = contributions;
        std::shuffle(shuffled.begin(), shuffled.end(), rng);
        auto result = AverageContributions(shuffled, 64);
        RNET_CHECK_OK(result);
        RNET_CHECK_EQ(result.value().values, reference.value().values);
    }
}

RNET_TEST(Aggregate, AlignsDifferentScalesExactly) {
    const std::vector<int8_t> coarse{1};   // at 2^-8  -> 1/256
    const std::vector<int8_t> fine{4};     // at 2^-10 -> 4/1024 = 1/256, the same value
    const std::vector<Contribution> contributions{{1, 8, coarse}, {2, 10, fine}};

    auto agg = SumContributions(contributions, 1);
    RNET_CHECK_OK(agg);
    RNET_CHECK_EQ(agg.value().scale_exp, 10);
    RNET_CHECK_EQ(agg.value().values[0], int64_t{8});   // 4 + 4, exactly
}

RNET_TEST(Aggregate, DivergentScaleIsRejected) {
    const std::vector<int8_t> a{1};
    const std::vector<int8_t> b{1};
    // A worker 2^40 away from its peers has diverged, not merely rounded
    // differently; averaging it in would let it dominate the update.
    const std::vector<Contribution> contributions{{1, -30, a}, {2, 30, b}};
    RNET_CHECK_ERR(SumContributions(contributions, 1));
}

RNET_TEST(Aggregate, MismatchedLengthIsRejected) {
    const std::vector<int8_t> a{1, 2, 3};
    const std::vector<int8_t> b{1, 2};
    const std::vector<Contribution> contributions{{1, 10, a}, {2, 10, b}};
    RNET_CHECK_ERR(SumContributions(contributions, 3));
}

RNET_TEST(Aggregate, RoundingIsSymmetricAboutZero) {
    RNET_CHECK_EQ(RoundedDiv(5, 2), int64_t{3});
    RNET_CHECK_EQ(RoundedDiv(-5, 2), int64_t{-3});
    RNET_CHECK_EQ(RoundedDiv(7, 3), int64_t{2});
    RNET_CHECK_EQ(RoundedDiv(-7, 3), int64_t{-2});
    RNET_CHECK_EQ(RoundedDiv(1, 2), int64_t{1});
    RNET_CHECK_EQ(RoundedDiv(-1, 2), int64_t{-1});
}

RNET_TEST(Aggregate, EmptyRoundIsAnError) {
    RNET_CHECK_ERR(SumContributions({}, 10));
}

// ---------------------------------------------------------------------------
// Outer optimizer
// ---------------------------------------------------------------------------
RNET_TEST(OuterOptimizer, IsDeterministicAcrossInstances) {
    Aggregate agg;
    agg.values = {100, -200, 300};
    agg.scale_exp = 10;
    agg.contributor_count = 4;

    OuterOptimizer a(3, 10, {});
    OuterOptimizer b(3, 10, {});
    for (int step = 0; step < 20; ++step) {
        auto ua = a.Step(agg);
        auto ub = b.Step(agg);
        RNET_CHECK_OK(ua);
        RNET_CHECK_OK(ub);
        RNET_CHECK_EQ(ua.value(), ub.value());
    }
    // Momentum accumulates across rounds, so identical state is what lets two
    // nodes agree on the next checkpoint.
    RNET_CHECK_EQ(a.StateHash(), b.StateHash());
}

RNET_TEST(OuterOptimizer, MomentumAccumulates) {
    Aggregate agg;
    agg.values = {1000};
    agg.scale_exp = 10;
    agg.contributor_count = 1;

    OuterOptimizer opt(1, 10, {});
    auto first = opt.Step(agg);
    RNET_CHECK_OK(first);
    auto second = opt.Step(agg);
    RNET_CHECK_OK(second);
    // With a constant pseudo-gradient the update grows as momentum builds.
    RNET_CHECK(std::abs(second.value()[0]) > std::abs(first.value()[0]));
}

RNET_TEST(OuterOptimizer, RescalesAggregateExactly) {
    Aggregate coarse;
    coarse.values = {4};
    coarse.scale_exp = 8;
    coarse.contributor_count = 1;

    Aggregate fine;
    fine.values = {16};       // same magnitude at 2^-10
    fine.scale_exp = 10;
    fine.contributor_count = 1;

    OuterOptimizer a(1, 10, {});
    OuterOptimizer b(1, 10, {});
    auto ua = a.Step(coarse);
    auto ub = b.Step(fine);
    RNET_CHECK_OK(ua);
    RNET_CHECK_OK(ub);
    RNET_CHECK_EQ(ua.value(), ub.value());
}

RNET_TEST(OuterOptimizer, RejectsWrongLength) {
    Aggregate agg;
    agg.values = {1, 2};
    agg.scale_exp = 10;
    agg.contributor_count = 1;
    OuterOptimizer opt(3, 10, {});
    RNET_CHECK_ERR(opt.Step(agg));
}

RNET_TEST(OuterOptimizer, RejectsInvalidCoefficients) {
    RNET_CHECK_ERR(OuterOptimizerConfig{kQ16One, kDefaultOuterLrQ16, true}.Validate());
    RNET_CHECK_ERR(OuterOptimizerConfig{kDefaultMomentumQ16, 0, true}.Validate());
}

RNET_TEST(OuterOptimizer, ParsesDecimalCoefficients) {
    auto nine = ParseQ16("0.9");
    RNET_CHECK_OK(nine);
    RNET_CHECK_EQ(nine.value(), kDefaultMomentumQ16);

    auto seven = ParseQ16("0.7");
    RNET_CHECK_OK(seven);
    RNET_CHECK_EQ(seven.value(), kDefaultOuterLrQ16);

    auto one = ParseQ16("1.0");
    RNET_CHECK_OK(one);
    RNET_CHECK_EQ(one.value(), kQ16One);

    RNET_CHECK_ERR(ParseQ16("abc"));
    RNET_CHECK_ERR(ParseQ16(""));
}

RNET_TEST(OuterOptimizer, Q16MultiplyRoundsSymmetrically) {
    RNET_CHECK_EQ(MulQ16(100, kQ16One), int64_t{100});
    RNET_CHECK_EQ(MulQ16(-100, kQ16One), int64_t{-100});
    RNET_CHECK_EQ(MulQ16(100, kQ16One / 2), int64_t{50});
    RNET_CHECK_EQ(MulQ16(-100, kQ16One / 2), int64_t{-50});
}

// ---------------------------------------------------------------------------
// Round: joining machines into one trainer
// ---------------------------------------------------------------------------
RNET_TEST(Round, AcceptsContributionsFromManyWorkers) {
    Round round(0, 1, kBase, 0, {250, 2, 60'000, false});
    for (uint64_t worker = 1; worker <= 4; ++worker) {
        RNET_CHECK_OK(round.Submit(MakeHeader(worker, 0, 1, kBase), 1000));
    }
    RNET_CHECK_EQ(round.submission_count(), 4u);
}

RNET_TEST(Round, RejectsDuplicateSubmission) {
    Round round(0, 1, kBase, 0, {250, 1, 60'000, false});
    RNET_CHECK_OK(round.Submit(MakeHeader(1, 0, 1, kBase), 100));
    // Re-submitting after seeing others would let a worker revise its contribution.
    RNET_CHECK_ERR(round.Submit(MakeHeader(1, 0, 1, kBase), 200));
}

RNET_TEST(Round, RejectsWrongBaseCheckpoint) {
    Round round(0, 1, kBase, 0, {250, 1, 60'000, false});
    const crypto::Hash256 other = crypto::Sha3_256(std::string_view("different-weights"));
    // A delta computed from different starting weights is not comparable.
    RNET_CHECK_ERR(round.Submit(MakeHeader(1, 0, 1, other), 100));
}

RNET_TEST(Round, RejectsWrongInnerStepCount) {
    Round round(0, 1, kBase, 0, {250, 1, 60'000, false});
    RNET_CHECK_ERR(round.Submit(MakeHeader(1, 0, 1, kBase, 100), 100));
}

RNET_TEST(Round, RejectsWrongRoundOrStep) {
    Round round(0, 5, kBase, 0, {250, 1, 60'000, false});
    RNET_CHECK_ERR(round.Submit(MakeHeader(1, 1, 5, kBase), 100));   // wrong round
    RNET_CHECK_ERR(round.Submit(MakeHeader(1, 0, 4, kBase), 100));   // wrong outer step
}

RNET_TEST(Round, ClosesOnQuorumAndDeadline) {
    Round round(0, 1, kBase, 1000, {250, 3, 10'000, false});
    RNET_CHECK(round.StateAt(1000) == RoundState::Open);

    RNET_CHECK_OK(round.Submit(MakeHeader(1, 0, 1, kBase), 2000));
    RNET_CHECK_OK(round.Submit(MakeHeader(2, 0, 1, kBase), 3000));
    RNET_CHECK(round.StateAt(5000) == RoundState::Open);          // quorum not met

    RNET_CHECK_OK(round.Submit(MakeHeader(3, 0, 1, kBase), 4000));
    RNET_CHECK(round.StateAt(5000) == RoundState::Open);          // deadline not reached
    RNET_CHECK(round.StateAt(11'000) == RoundState::Closed);
}

// Waiting for the slowest machine would make the network as slow as its worst
// member; past the deadline the round proceeds with whoever arrived.
RNET_TEST(Round, StragglersDoNotStallTheNetwork) {
    Round round(0, 1, kBase, 0, {250, 2, 10'000, false});
    RNET_CHECK_OK(round.Submit(MakeHeader(1, 0, 1, kBase), 100));
    RNET_CHECK_OK(round.Submit(MakeHeader(2, 0, 1, kBase), 200));

    RNET_CHECK(round.StateAt(10'000) == RoundState::Closed);
    // The straggler's work is dropped for this round, never merged half-applied.
    RNET_CHECK_ERR(round.Submit(MakeHeader(3, 0, 1, kBase), 10'500));
    RNET_CHECK_EQ(round.submission_count(), 2u);
}

RNET_TEST(Round, WithoutQuorumTheRoundIsAbandoned) {
    Round round(0, 1, kBase, 0, {250, 5, 10'000, false});
    RNET_CHECK_OK(round.Submit(MakeHeader(1, 0, 1, kBase), 100));
    // Averaging one machine's update as if it were the network's would let a
    // single participant steer the model.
    RNET_CHECK(round.StateAt(11'000) == RoundState::Abandoned);
}

RNET_TEST(Round, ClosesEarlyWhenEveryoneHasReported) {
    Round round(0, 1, kBase, 0, {250, 2, 600'000, false});
    RNET_CHECK_OK(round.Submit(MakeHeader(1, 0, 1, kBase), 100));
    RNET_CHECK_OK(round.Submit(MakeHeader(2, 0, 1, kBase), 200));
    RNET_CHECK(round.CanClose(300, 2));    // no reason to wait out the deadline
    RNET_CHECK(!round.CanClose(300, 5));   // three participants still expected
}

RNET_TEST(Round, EvidenceHashIsOrderIndependent) {
    Round a(0, 1, kBase, 0, {250, 1, 0, false});
    RNET_CHECK_OK(a.Submit(MakeHeader(3, 0, 1, kBase), 100));
    RNET_CHECK_OK(a.Submit(MakeHeader(1, 0, 1, kBase), 200));
    RNET_CHECK_OK(a.Submit(MakeHeader(2, 0, 1, kBase), 300));

    Round b(0, 1, kBase, 0, {250, 1, 0, false});
    RNET_CHECK_OK(b.Submit(MakeHeader(1, 0, 1, kBase), 150));
    RNET_CHECK_OK(b.Submit(MakeHeader(2, 0, 1, kBase), 250));
    RNET_CHECK_OK(b.Submit(MakeHeader(3, 0, 1, kBase), 350));

    RNET_CHECK_EQ(a.EvidenceHash(), b.EvidenceHash());
}

RNET_TEST(Membership, ChangesTakeEffectAtRoundBoundaries) {
    Membership members;
    members.Join(1);
    members.Join(2);
    RNET_CHECK_EQ(members.ActiveCount(), 0u);   // not yet: mid-round joins are unsafe

    members.AdvanceRound();
    RNET_CHECK_EQ(members.ActiveCount(), 2u);

    members.Leave(1);
    RNET_CHECK(members.IsActive(1));            // still counted for the current round
    members.AdvanceRound();
    RNET_CHECK(!members.IsActive(1));
}

RNET_TEST(Membership, DisappearedWorkersStopBlockingQuorum) {
    Membership members;
    members.Join(1);
    members.Join(2);
    members.AdvanceRound();

    for (int round = 0; round < 4; ++round) {
        members.RecordParticipation(1, true);
        members.RecordParticipation(2, false);
    }
    RNET_CHECK_EQ(members.ConsecutiveMisses(2), 4u);

    members.DropInactive(3);
    RNET_CHECK(members.IsActive(1));
    RNET_CHECK(!members.IsActive(2));
}

// ---------------------------------------------------------------------------
// Checkpoint chain
// ---------------------------------------------------------------------------
RNET_TEST(Chain, LinksCheckpointsAndProvesAncestry) {
    CheckpointChain chain;
    const auto genesis = MakeCheckpoint(crypto::Hash256{}, 0, 0);
    RNET_CHECK_OK(chain.SetGenesis(genesis));

    crypto::Hash256 parent = genesis.Id();
    for (uint64_t step = 1; step <= 5; ++step) {
        const auto cp = MakeCheckpoint(parent, step);
        RNET_CHECK_OK(chain.Append(cp));
        parent = cp.Id();
    }

    RNET_CHECK_EQ(chain.Height(), 5ull);
    RNET_CHECK(chain.DescendsFromGenesis(parent));

    auto ancestry = chain.AncestryOf(parent);
    RNET_CHECK_OK(ancestry);
    // Six 32-byte links prove provenance — no intermediate weights required.
    RNET_CHECK_EQ(ancestry.value().size(), size_t{6});
    RNET_CHECK_EQ(ancestry.value().front(), genesis.Id());
}

RNET_TEST(Chain, RejectsGapsAndWrongParents) {
    CheckpointChain chain;
    const auto genesis = MakeCheckpoint(crypto::Hash256{}, 0, 0);
    RNET_CHECK_OK(chain.SetGenesis(genesis));

    const auto skipped = MakeCheckpoint(genesis.Id(), 2);   // gap
    RNET_CHECK_ERR(chain.Append(skipped));

    const crypto::Hash256 stranger = crypto::Sha3_256(std::string_view("not-our-chain"));
    RNET_CHECK_ERR(chain.Append(MakeCheckpoint(stranger, 1)));
}

RNET_TEST(Chain, GenesisRulesAreEnforced) {
    CheckpointChain chain;
    // A parentless checkpoint at a non-zero step, or step zero with a parent, are
    // both malformed.
    RNET_CHECK_ERR(MakeCheckpoint(crypto::Hash256{}, 3, 0).Validate());

    const auto genesis = MakeCheckpoint(crypto::Hash256{}, 0, 0);
    RNET_CHECK_OK(chain.SetGenesis(genesis));
    RNET_CHECK_ERR(chain.SetGenesis(genesis));   // only one genesis
}

RNET_TEST(Chain, RollsBackWithinTheRetentionWindow) {
    CheckpointChain chain(3);
    const auto genesis = MakeCheckpoint(crypto::Hash256{}, 0, 0);
    RNET_CHECK_OK(chain.SetGenesis(genesis));

    std::vector<crypto::Hash256> ids{genesis.Id()};
    crypto::Hash256 parent = genesis.Id();
    for (uint64_t step = 1; step <= 6; ++step) {
        const auto cp = MakeCheckpoint(parent, step);
        RNET_CHECK_OK(chain.Append(cp));
        parent = cp.Id();
        ids.push_back(parent);
    }

    RNET_CHECK_OK(chain.RollbackTo(ids[4]));
    RNET_CHECK_EQ(chain.Height(), 4ull);
    RNET_CHECK(!chain.Has(ids[6]));

    // Beyond the retention window the weights have been pruned, so the rollback
    // could not actually be carried out — refusing is more honest than pretending.
    RNET_CHECK_ERR(chain.RollbackTo(ids[0]));
}

RNET_TEST(Chain, RetainsOnlyRecentFullCheckpoints) {
    CheckpointChain chain(3);
    const auto genesis = MakeCheckpoint(crypto::Hash256{}, 0, 0);
    RNET_CHECK_OK(chain.SetGenesis(genesis));

    crypto::Hash256 parent = genesis.Id();
    for (uint64_t step = 1; step <= 10; ++step) {
        const auto cp = MakeCheckpoint(parent, step);
        RNET_CHECK_OK(chain.Append(cp));
        parent = cp.Id();
    }

    const auto required = chain.RequiredFullCheckpoints();
    RNET_CHECK_EQ(required.size(), size_t{3});
    RNET_CHECK_EQ(required.front(), parent);   // newest first
}

// Storage is a computed protocol number, not folklore: the header chain stays
// tiny while full weights are bounded by the retention window.
RNET_TEST(Chain, StorageIsBoundedByRetentionNotHistory) {
    const uint64_t params = 1'000'000'000;   // RN-1B
    const auto day = EstimateStorage(131, params, 1, 5);          // one day of rounds
    const auto year = EstimateStorage(131 * 365, params, 1, 5);   // a year of rounds

    RNET_CHECK_EQ(day.full_checkpoint_bytes, year.full_checkpoint_bytes);
    RNET_CHECK(year.header_bytes < 10'000'000);          // headers stay in megabytes
    RNET_CHECK(year.total_bytes < 20'000'000'000ull);    // ~10 GB of weights, not 95 TB
}

// ---------------------------------------------------------------------------
// Who publishes the checkpoint.
//
// Aggregation is the one step nobody currently checks, and every node holds a
// slightly different set of contributions because the network is asynchronous. If
// each aggregated its own set there would be no fact of the matter about which
// checkpoint was right. One node produces, everyone else recomputes.
// ---------------------------------------------------------------------------
#include "diloco/producer.h"

namespace {

canon::ContributionHeader ContributionBy(uint64_t worker_id, uint64_t step = 0) {
    canon::ContributionHeader h;
    h.round_id = 0;
    h.worker_id = worker_id;
    h.outer_step = step;
    h.base_checkpoint = crypto::Sha3_256(std::string_view("base"));
    h.payload_hash = crypto::Sha3_256(std::string_view("payload-" + std::to_string(worker_id)));
    h.n_params = 1000;
    h.inner_steps = 250;
    h.scale_exp = 10;
    return h;
}

}  // namespace

// Two nodes that hold the same contributions must agree on the evidence hash even
// though the network delivered them in different orders.
RNET_TEST(Producer, EvidenceHashIgnoresArrivalOrder) {
    std::vector<canon::ContributionHeader> forwards;
    for (uint64_t w = 1; w <= 5; ++w) forwards.push_back(ContributionBy(w));

    std::vector<canon::ContributionHeader> backwards(forwards.rbegin(), forwards.rend());
    std::vector<canon::ContributionHeader> shuffled{forwards[3], forwards[0], forwards[4],
                                                    forwards[1], forwards[2]};

    const auto expected = EvidenceHashOf(forwards);
    RNET_CHECK(EvidenceHashOf(backwards) == expected);
    RNET_CHECK(EvidenceHashOf(shuffled) == expected);

    // And a different set is a different hash — otherwise the commitment would
    // not commit to anything.
    std::vector<canon::ContributionHeader> fewer(forwards.begin(), forwards.end() - 1);
    RNET_CHECK(!(EvidenceHashOf(fewer) == expected));
}

// The election must be a function of values fixed before the round opened, so
// nobody can steer it.
RNET_TEST(Producer, ElectionIsDeterministicAndDependsOnTheParent) {
    const std::vector<uint64_t> eligible{1, 2, 3, 4, 5};
    const auto parent_a = crypto::Sha3_256(std::string_view("checkpoint a"));
    const auto parent_b = crypto::Sha3_256(std::string_view("checkpoint b"));

    auto first = ElectProducer(parent_a, 7, eligible);
    RNET_CHECK_OK(first);
    // Same inputs, same answer, every time and on every node.
    for (int i = 0; i < 10; ++i) {
        auto again = ElectProducer(parent_a, 7, eligible);
        RNET_CHECK_OK(again);
        RNET_CHECK_EQ(again.value(), first.value());
    }
    // The pool order must not matter: two nodes listing the same workers
    // differently are the same pool.
    const std::vector<uint64_t> reordered{5, 3, 1, 4, 2};
    auto shuffled = ElectProducer(parent_a, 7, reordered);
    RNET_CHECK_OK(shuffled);
    RNET_CHECK_EQ(shuffled.value(), first.value());

    // A different parent or a different step is a different draw, or the same
    // node would produce forever.
    bool moves_with_parent = false;
    bool moves_with_step = false;
    for (uint64_t step = 0; step < 24; ++step) {
        auto a = ElectProducer(parent_a, step, eligible);
        auto b = ElectProducer(parent_b, step, eligible);
        RNET_CHECK_OK(a);
        RNET_CHECK_OK(b);
        if (a.value() != b.value()) moves_with_parent = true;
        if (a.value() != first.value()) moves_with_step = true;
    }
    RNET_CHECK(moves_with_parent);
    RNET_CHECK(moves_with_step);
}

// Over many steps the role must actually circulate. An election that always
// returns the same worker is a single point of failure wearing the costume of a
// protocol.
RNET_TEST(Producer, TheRoleCirculatesAcrossSteps) {
    const std::vector<uint64_t> eligible{1, 2, 3, 4, 5, 6, 7, 8};
    const auto parent = crypto::Sha3_256(std::string_view("parent"));

    std::map<uint64_t, int> elected;
    for (uint64_t step = 0; step < 400; ++step) {
        auto producer = ElectProducer(parent, step, eligible);
        RNET_CHECK_OK(producer);
        ++elected[producer.value()];
    }
    RNET_CHECK_EQ(elected.size(), eligible.size());
    // Roughly even, checked loosely: the claim is that nobody is excluded and
    // nobody dominates, not that SHA3 is a perfect die.
    for (const auto& [worker, count] : elected) {
        if (count < 20 || count > 100) {
            RNET_FAIL("worker {} produced {} of 400 steps — the draw is skewed", worker, count);
        }
    }
}

// A step with nobody entitled to publish is a stalled network. Saying so beats
// quietly electing worker 0, which is the unset value.
RNET_TEST(Producer, RefusesAnEmptyOrInvalidPool) {
    const auto parent = crypto::Sha3_256(std::string_view("parent"));
    RNET_CHECK_ERR(ElectProducer(parent, 1, {}));

    const std::vector<uint64_t> only_unset{0, 0};
    RNET_CHECK_ERR(ElectProducer(parent, 1, only_unset));

    // A pool containing the unset value alongside real workers still elects a real
    // one rather than tripping over the zero.
    const std::vector<uint64_t> mixed{0, 9};
    auto elected = ElectProducer(parent, 1, mixed);
    RNET_CHECK_OK(elected);
    RNET_CHECK_EQ(elected.value(), uint64_t{9});
}

// The pool comes from the PREVIOUS checkpoint. If it were drawn from the set the
// producer publishes, a producer could omit whoever would have beaten it and elect
// itself — this test states that attack concretely so the design cannot drift back
// into it.
RNET_TEST(Producer, ExcludingARivalWouldChangeTheWinnerIfThePoolCameFromThisRound) {
    const auto parent = crypto::Sha3_256(std::string_view("parent"));
    const std::vector<uint64_t> full{1, 2, 3, 4, 5, 6, 7, 8};

    auto honest = ElectProducer(parent, 11, full);
    RNET_CHECK_OK(honest);

    // Someone who is not the winner drops the winner from the set they publish.
    std::vector<uint64_t> stuffed;
    for (uint64_t w : full) {
        if (w != honest.value()) stuffed.push_back(w);
    }
    auto after_exclusion = ElectProducer(parent, 11, stuffed);
    RNET_CHECK_OK(after_exclusion);

    // The winner changes — which is exactly why eligibility must be settled one
    // step earlier, before anyone knows this round's contributions.
    if (after_exclusion.value() == honest.value()) {
        RNET_FAIL("excluding the winner left the winner unchanged, so this test proves nothing");
    }
}

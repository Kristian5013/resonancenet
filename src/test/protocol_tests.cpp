// The participant: what a node does with the objects that arrive on its wire.
//
// The net layer proves an object's bytes hash to what they claim, and that is the
// only thing it can prove. These tests are about everything a peer can still lie
// about once the bytes are honest — the round, the base checkpoint, the parameter
// count, who was selected for a challenge, who was allowed to answer it.
//
// The refusals matter more than the acceptances here. An accepted contribution is
// arithmetic; an accepted forgery is consensus.
#include <format>

#include "consensus/genesis.h"
#include "protocol/participant.h"
#include "test/framework.h"
#include "util/hex.h"

using namespace rnet;
using namespace rnet::protocol;

namespace {

// Regtest keeps the model small while running the same code paths as main, so it
// is the default here.
//
// It is NOT the right network for everything, and that is a real distinction
// rather than a testing detail: regtest challenges 100% of contributions and sets
// its slash quorum to 1, precisely so tests never wait for chance. Under those
// settings "the beacon did not select this" and "fewer verdicts than the quorum"
// are states that cannot exist — so the tests for those refusals run on main,
// where the parameters are the ones the network will actually launch with.
const consensus::NetworkParams& P(consensus::Network net = consensus::Network::Regtest) {
    return consensus::Params(net);
}

canon::CheckpointHeader GenesisCheckpoint(consensus::Network net = consensus::Network::Regtest) {
    canon::CheckpointHeader header;
    header.parent = crypto::Hash256{};        // genesis, by definition
    header.round_id = P(net).round.round_id;
    header.outer_step = 0;
    header.weights_hash = crypto::Sha3_256(std::string_view("initial weights"));
    header.n_params = P(net).round.model.ParameterCount();
    header.contributor_count = 0;
    header.weight_dtype = 1;                  // bf16
    return header;
}

ParticipantConfig Config(uint64_t worker_id, bool verifier = true,
                         consensus::Network net = consensus::Network::Regtest) {
    ParticipantConfig config;
    config.network = net;
    config.worker_id = worker_id;
    config.verifier = verifier;
    return config;
}

canon::ContributionHeader Contribution(uint64_t worker_id, const crypto::Hash256& base,
                                       consensus::Network net = consensus::Network::Regtest) {
    canon::ContributionHeader header;
    header.round_id = P(net).round.round_id;
    header.worker_id = worker_id;
    header.outer_step = 0;
    header.base_checkpoint = base;
    header.payload_hash = crypto::Sha3_256(std::string_view(std::format("payload-{}", worker_id)));
    header.n_params = P(net).round.model.ParameterCount();
    header.inner_steps = P(net).policy.inner_steps;
    header.scale_exp = 12;
    return header;
}

ObjectVerdict Offer(Participant& participant, net::InvType type, std::vector<uint8_t> container,
                    const crypto::Hash256& id, uint64_t now_ms = 1000) {
    return participant.OnObject(type, id, container, now_ms);
}

}  // namespace

RNET_TEST(Participant, AcceptsAWellFormedContribution) {
    Participant participant(Config(1), GenesisCheckpoint());
    const auto base = GenesisCheckpoint().Id();

    const auto header = Contribution(2, base);
    const auto verdict =
        Offer(participant, net::InvType::Contribution, header.ToContainer(), header.Id());
    if (!verdict.accepted()) RNET_FAIL("honest contribution refused: {}", verdict.reason);
    RNET_CHECK_EQ(participant.submission_count(), uint32_t{1});

    // The same object twice is not a fault; peers relay, and relays overlap.
    const auto again =
        Offer(participant, net::InvType::Contribution, header.ToContainer(), header.Id());
    RNET_CHECK_EQ(static_cast<int>(again.admission), static_cast<int>(Admission::Duplicate));
    RNET_CHECK_EQ(again.misbehaviour, 0);
    RNET_CHECK_EQ(participant.submission_count(), uint32_t{1});
}

// The announced identity and the content must be the same thing. If they differ,
// one of them is a lie, and a node that stores it under the announced name has
// been made to hold something nobody can verify.
RNET_TEST(Participant, RefusesAnObjectAnnouncedUnderTheWrongIdentity) {
    Participant participant(Config(1), GenesisCheckpoint());
    const auto header = Contribution(2, GenesisCheckpoint().Id());

    const auto lied_about = crypto::Sha3_256(std::string_view("some other object"));
    const auto verdict =
        Offer(participant, net::InvType::Contribution, header.ToContainer(), lied_about);
    RNET_CHECK_EQ(static_cast<int>(verdict.admission), static_cast<int>(Admission::Rejected));
    RNET_CHECK(verdict.misbehaviour >= 50);
    RNET_CHECK_EQ(participant.submission_count(), uint32_t{0});
}

// A contribution computed against a different model is not a contribution to this
// one, however well-formed it is. This is the check that stops a node from
// aggregating a delta of the wrong shape.
RNET_TEST(Participant, RefusesAContributionForADifferentModel) {
    Participant participant(Config(1), GenesisCheckpoint());

    auto header = Contribution(2, GenesisCheckpoint().Id());
    header.n_params = P().round.model.ParameterCount() + 1;
    const auto verdict =
        Offer(participant, net::InvType::Contribution, header.ToContainer(), header.Id());
    RNET_CHECK_EQ(static_cast<int>(verdict.admission), static_cast<int>(Admission::Rejected));
    RNET_CHECK(verdict.misbehaviour >= 50);
}

// The inner-step count is fixed by policy. A worker doing fewer steps and claiming
// the reward for a full round is the cheapest possible cheat.
RNET_TEST(Participant, RefusesAContributionThatDidTheWrongAmountOfWork) {
    Participant participant(Config(1), GenesisCheckpoint());

    auto header = Contribution(2, GenesisCheckpoint().Id());
    header.inner_steps = P().policy.inner_steps / 2;
    const auto verdict =
        Offer(participant, net::InvType::Contribution, header.ToContainer(), header.Id());
    RNET_CHECK_EQ(static_cast<int>(verdict.admission), static_cast<int>(Admission::Rejected));
    if (verdict.reason.find("inner steps") == std::string::npos) {
        RNET_FAIL("the refusal should name what was wrong, got '{}'", verdict.reason);
    }
}

// A contribution against weights this node has never seen cannot be placed. It is
// refused, but as an unknown base rather than as a forgery: the sender may simply
// be ahead.
RNET_TEST(Participant, RefusesAContributionAgainstUnknownWeights) {
    Participant participant(Config(1), GenesisCheckpoint());

    auto header = Contribution(2, crypto::Sha3_256(std::string_view("weights nobody published")));
    const auto verdict =
        Offer(participant, net::InvType::Contribution, header.ToContainer(), header.Id());
    RNET_CHECK_EQ(static_cast<int>(verdict.admission), static_cast<int>(Admission::Rejected));
    RNET_CHECK_EQ(participant.submission_count(), uint32_t{0});
}

// A node must not accept from itself what it would refuse from a peer, or its own
// bugs stay invisible until they reach the network.
RNET_TEST(Participant, OwnContributionsGoThroughTheSameChecks) {
    Participant participant(Config(7), GenesisCheckpoint());

    auto bad = Contribution(7, GenesisCheckpoint().Id());
    bad.inner_steps = 1;
    RNET_CHECK_ERR(participant.SubmitOwn(bad, 1000));
    RNET_CHECK_EQ(participant.submission_count(), uint32_t{0});

    const auto good = Contribution(7, GenesisCheckpoint().Id());
    RNET_CHECK_OK(participant.SubmitOwn(good, 1000));
    RNET_CHECK_EQ(participant.submission_count(), uint32_t{1});

    // And it is announced, or nobody else would ever see it.
    const auto outgoing = participant.TakeOutgoing();
    RNET_CHECK_EQ(outgoing.size(), size_t{1});
    RNET_CHECK(outgoing[0].hash == good.Id());
}

RNET_TEST(Participant, RefusesToSubmitUnderAnotherWorkersIdentity) {
    Participant participant(Config(7), GenesisCheckpoint());
    RNET_CHECK_ERR(participant.SubmitOwn(Contribution(8, GenesisCheckpoint().Id()), 1000));
}

// A checkpoint whose parent is unknown cannot be placed in the chain. Accepting it
// would mean training onward from a state nobody proved — a fork becoming
// canonical without anyone deciding it should.
RNET_TEST(Participant, RefusesACheckpointWithAnUnknownParent) {
    Participant participant(Config(1), GenesisCheckpoint());

    canon::CheckpointHeader orphan;
    orphan.parent = crypto::Sha3_256(std::string_view("a checkpoint nobody published"));
    orphan.round_id = P().round.round_id;
    orphan.outer_step = 1;
    orphan.weights_hash = crypto::Sha3_256(std::string_view("weights"));
    orphan.n_params = P().round.model.ParameterCount();
    orphan.contributor_count = 2;
    orphan.evidence_hash = crypto::Sha3_256(std::string_view("orphan evidence"));
    orphan.optimizer_state_hash = crypto::Sha3_256(std::string_view("orphan optimizer"));
    orphan.weight_dtype = 1;

    const auto verdict =
        Offer(participant, net::InvType::Checkpoint, orphan.ToContainer(), orphan.Id());
    RNET_CHECK_EQ(static_cast<int>(verdict.admission), static_cast<int>(Admission::NotForUs));
    RNET_CHECK_EQ(participant.chain().Height(), uint64_t{0});
}

// When the tip moves, the round moves with it: work against the old base becomes
// late rather than invalid, and the new round starts empty.
RNET_TEST(Participant, AdvancingTheTipOpensAFreshRound) {
    Participant participant(Config(1), GenesisCheckpoint());
    const auto genesis_id = GenesisCheckpoint().Id();

    const auto header = Contribution(2, genesis_id);
    RNET_CHECK(Offer(participant, net::InvType::Contribution, header.ToContainer(), header.Id())
                   .accepted());
    RNET_CHECK_EQ(participant.submission_count(), uint32_t{1});

    canon::CheckpointHeader next;
    next.parent = genesis_id;
    next.round_id = P().round.round_id;
    next.outer_step = 1;
    next.weights_hash = crypto::Sha3_256(std::string_view("weights after step 1"));
    next.n_params = P().round.model.ParameterCount();
    next.contributor_count = 2;
    next.evidence_hash = crypto::Sha3_256(std::string_view("step 1 evidence"));
    next.optimizer_state_hash = crypto::Sha3_256(std::string_view("step 1 optimizer"));
    next.weight_dtype = 1;

    RNET_CHECK(
        Offer(participant, net::InvType::Checkpoint, next.ToContainer(), next.Id()).accepted());
    RNET_CHECK_EQ(participant.outer_step(), uint64_t{1});
    RNET_CHECK_EQ(participant.submission_count(), uint32_t{0});   // fresh round

    // The same contribution now arrives against a base that is no longer the tip.
    // Late, not dishonest — and scored as such.
    const auto late =
        Offer(participant, net::InvType::Contribution, header.ToContainer(), header.Id());
    RNET_CHECK_EQ(static_cast<int>(late.admission), static_cast<int>(Admission::NotForUs));
    RNET_CHECK_EQ(late.misbehaviour, 0);
}

// The size of a payload comes from the header that named it, never from whoever
// offers to send it — otherwise a peer chooses how much this node allocates.
RNET_TEST(Participant, PayloadSizeComesFromTheHeaderNotTheSender) {
    Participant participant(Config(1), GenesisCheckpoint());
    const auto header = Contribution(2, GenesisCheckpoint().Id());
    RNET_CHECK(Offer(participant, net::InvType::Contribution, header.ToContainer(), header.Id())
                   .accepted());

    const auto missing = participant.MissingPayloads();
    RNET_CHECK_EQ(missing.size(), size_t{1});
    RNET_CHECK(missing[0].payload_hash == header.payload_hash);
    // One int8 unit per parameter — fixed by the model, not negotiable.
    RNET_CHECK_EQ(missing[0].expected_bytes, P().round.model.ParameterCount());

    // A payload nobody asked for cannot be used to mark a contribution complete.
    RNET_CHECK_ERR(participant.NotePayload(crypto::Sha3_256(std::string_view("unrelated"))));
    RNET_CHECK_OK(participant.NotePayload(header.payload_hash));
    RNET_CHECK(participant.MissingPayloads().empty());
}

// The beacon decides who is challenged. Without re-deriving it, anyone could order
// anyone to burn a GPU-hour — a denial of service dressed as verification.
//
// Run on main: regtest challenges everything, so there is no such thing there as a
// contribution the beacon did not select.
RNET_TEST(Participant, RefusesAChallengeTheBeaconDidNotSelect) {
    constexpr auto kNet = consensus::Network::Main;
    RNET_CHECK(P(kNet).policy.challenge_percent < 100);

    Participant participant(Config(3, true, kNet), GenesisCheckpoint(kNet));
    const auto header = Contribution(2, GenesisCheckpoint(kNet).Id(), kNet);
    RNET_CHECK(Offer(participant, net::InvType::Contribution, header.ToContainer(), header.Id())
                   .accepted());

    canon::ChallengeOrder forged;
    forged.round_id = P(kNet).round.round_id;
    forged.outer_step = 0;
    forged.contribution_id = header.Id();
    forged.beacon_height = 1;
    forged.verifier_id = 3;
    forged.issued_at_step = 0;
    forged.determinism_class = static_cast<uint16_t>(P(kNet).round.determinism_class);

    // A beacon the attacker picked rather than one derived from the closed round.
    bool found = false;
    for (uint64_t attempt = 0; attempt < 512 && !found; ++attempt) {
        forged.beacon = crypto::Sha3_256(std::string_view(std::format("forged-{}", attempt)));
        found = !verification::IsSelected(forged.beacon, forged.contribution_id,
                                          P(kNet).policy.challenge_percent);
    }
    RNET_CHECK(found);

    const auto verdict =
        Offer(participant, net::InvType::Challenge, forged.ToContainer(), forged.Id());
    RNET_CHECK_EQ(static_cast<int>(verdict.admission), static_cast<int>(Admission::Rejected));
    RNET_CHECK(verdict.misbehaviour >= 50);
    RNET_CHECK(participant.PendingChallenges().empty());
}

// A verdict about a challenge this node has never seen has nothing to be checked
// against. Recording it would fill the ledger with unfalsifiable claims.
RNET_TEST(Participant, RefusesAVerdictWithNoChallengeBehindIt) {
    Participant participant(Config(1), GenesisCheckpoint());

    canon::VerificationVerdict verdict;
    verdict.challenge_id = crypto::Sha3_256(std::string_view("a challenge nobody issued"));
    verdict.contribution_id = crypto::Sha3_256(std::string_view("some contribution"));
    verdict.verifier_id = 9;
    verdict.kind = canon::VerdictKind::Mismatch;
    verdict.claimed_payload_hash = crypto::Sha3_256(std::string_view("claimed"));
    verdict.recomputed_payload_hash = crypto::Sha3_256(std::string_view("recomputed"));
    verdict.determinism_class = static_cast<uint16_t>(P().round.determinism_class);
    verdict.reported_at_step = 0;

    const auto admission =
        Offer(participant, net::InvType::Verdict, verdict.ToContainer(), verdict.Id());
    RNET_CHECK_EQ(static_cast<int>(admission.admission), static_cast<int>(Admission::NotForUs));
    RNET_CHECK_EQ(participant.ledger().VerdictCount(verdict.contribution_id), uint32_t{0});
}

// Evidence carrying fewer verdicts than the quorum is not evidence. One colluding
// or faulty verifier must not be able to burn an honest worker's stake.
//
// Run on main: regtest sets the quorum to 1, where "below quorum" means an empty
// verdict list — which is refused a step earlier as malformed, so the quorum check
// itself would never be reached.
RNET_TEST(Participant, RefusesEvidenceBelowTheQuorum) {
    constexpr auto kNet = consensus::Network::Main;
    RNET_CHECK(P(kNet).policy.slash_quorum > 1);

    Participant participant(Config(1, true, kNet), GenesisCheckpoint(kNet));

    canon::SlashEvidence evidence;
    evidence.contribution_id = crypto::Sha3_256(std::string_view("the accused"));
    evidence.worker_id = 2;
    evidence.round_id = P(kNet).round.round_id;
    evidence.determinism_class = static_cast<uint16_t>(P(kNet).round.determinism_class);
    evidence.finalized_at_step = 1;
    // Well-formed in every respect except the one that matters: one verdict short.
    for (uint32_t i = 0; i + 1 < P(kNet).policy.slash_quorum; ++i) {
        evidence.verdict_ids.push_back(
            crypto::Sha3_256(std::string_view(std::format("verdict-{}", i))));
    }
    std::sort(evidence.verdict_ids.begin(), evidence.verdict_ids.end());
    RNET_CHECK_OK(evidence.Validate());   // the object itself is fine

    const auto admission =
        Offer(participant, net::InvType::SlashEvidence, evidence.ToContainer(), evidence.Id());
    RNET_CHECK_EQ(static_cast<int>(admission.admission), static_cast<int>(Admission::Rejected));
    RNET_CHECK(admission.misbehaviour >= 50);
    if (admission.reason.find("quorum") == std::string::npos) {
        RNET_FAIL("the refusal should name the quorum, got '{}'", admission.reason);
    }
}

// Slashing runs in shadow mode: the evidence is built and published, and nothing
// is settled. This test exists to make that a fact about the code rather than an
// intention in a comment.
RNET_TEST(Participant, SlashingIsRecordedButNotSettled) {
    RNET_CHECK(P().policy.shadow_mode != 0);

    Participant participant(Config(1), GenesisCheckpoint());
    RNET_CHECK(participant.ledger().shadow_mode());
    // Whatever the outcome, shadow mode settles nothing.
    RNET_CHECK(!participant.ledger().WouldSettle(verification::EvidenceOutcome::Slashable));
}

// A single worker cannot be verified: there is nobody to check it who is not it.
// The protocol must produce no challenges rather than assign someone to audit
// their own work.
RNET_TEST(Participant, NoChallengesWhenThereIsNobodyToCheck) {
    Participant participant(Config(1), GenesisCheckpoint());
    const auto header = Contribution(1, GenesisCheckpoint().Id());
    RNET_CHECK_OK(participant.SubmitOwn(header, 1000));
    RNET_CHECK_OK(participant.NotePayload(header.payload_hash));
    (void)participant.TakeOutgoing();

    RNET_CHECK_OK(participant.Tick(2000));
    for (const auto& out : participant.TakeOutgoing()) {
        if (out.type == net::InvType::Challenge) {
            RNET_FAIL("a lone worker was assigned to verify itself");
        }
    }
}

// ---------------------------------------------------------------------------
// The seam, end to end over real sockets.
//
// Everything above tests the participant with objects handed to it directly. This
// tests what happens when they have to cross a TCP connection first: a
// contribution announced, requested, delivered, validated against consensus, and
// its bulk payload fetched at the size the header fixed.
//
// It is the only test here that can fail for a reason that has nothing to do with
// consensus, and that is the point — the two layers meeting is where the mistakes
// live.
// ---------------------------------------------------------------------------
#include <functional>

#include "net/node.h"
#include "protocol/service.h"

namespace {

net::NodeConfig ServiceNodeConfig(uint16_t port) {
    net::NodeConfig config;
    config.network = consensus::Network::Regtest;
    auto bind = net::NetAddress::FromString(std::format("127.0.0.1:{}", port), port);
    config.bind_address = bind.ok() ? bind.value() : net::NetAddress{};
    config.max_inbound = 8;
    config.max_outbound = 8;
    config.handshake_timeout_ms = 5000;
    return config;
}

// Drives both sides until `done`, or until the budget runs out. Time is supplied
// rather than read from a clock, so the test cannot become flaky on a busy
// machine.
bool PumpBoth(net::Node& a, protocol::Service& sa, net::Node& b, protocol::Service& sb,
              const std::function<bool()>& done, int iterations = 400) {
    for (int i = 0; i < iterations; ++i) {
        const int64_t now = 1000 + i * 10;
        (void)a.Tick(now, 2);
        (void)b.Tick(now, 2);
        (void)sa.Tick(now);
        (void)sb.Tick(now);
        if (done()) return true;
    }
    return done();
}

}  // namespace

RNET_TEST(Service, AContributionCrossesTheWireAndIsAccepted) {
    Participant producer(Config(1), GenesisCheckpoint());
    Participant consumer(Config(2), GenesisCheckpoint());

    net::Node node_a(ServiceNodeConfig(19820), crypto::Sha3_256(std::string_view("a")), 1111);
    net::Node node_b(ServiceNodeConfig(0), crypto::Sha3_256(std::string_view("b")), 2222);
    protocol::Service service_a(node_a, producer);
    protocol::Service service_b(node_b, consumer);
    service_a.Attach();
    service_b.Attach();

    RNET_CHECK_OK(node_a.Start());
    RNET_CHECK_OK(node_b.Start());
    auto target = net::NetAddress::FromString("127.0.0.1:19820", 19820);
    RNET_CHECK_OK(target);
    RNET_CHECK_OK(node_b.ConnectTo(target.value(), 1000));
    PumpBoth(node_a, service_a, node_b, service_b,
             [&] { return node_a.ReadyCount() == 1 && node_b.ReadyCount() == 1; });
    RNET_CHECK_EQ(node_a.ReadyCount(), size_t{1});

    // The producer submits its own work. Announcing it is the service's job.
    const auto header = Contribution(1, GenesisCheckpoint().Id());
    RNET_CHECK_OK(producer.SubmitOwn(header, 1000));

    const bool arrived = PumpBoth(node_a, service_a, node_b, service_b,
                                  [&] { return consumer.submission_count() == 1; });
    if (!arrived) {
        RNET_FAIL("the contribution did not reach the peer (accepted {}, refused {})",
                  service_b.objects_accepted(), service_b.objects_rejected());
    }
    RNET_CHECK_EQ(service_b.objects_rejected(), uint64_t{0});

    // Having accepted the header, the consumer knows a payload is outstanding and
    // knows exactly how large it must be — from the header, not from the sender.
    const auto missing = consumer.MissingPayloads();
    RNET_CHECK_EQ(missing.size(), size_t{1});
    RNET_CHECK_EQ(missing[0].expected_bytes, P().round.model.ParameterCount());
}

// A peer sending contributions for a different model is not making mistakes, and
// the score has to reach the connection layer or the peer never goes away.
RNET_TEST(Service, ConsensusRejectionsAreScoredAgainstThePeerThatSentThem) {
    Participant honest(Config(1), GenesisCheckpoint());
    Participant victim(Config(2), GenesisCheckpoint());

    net::Node node_a(ServiceNodeConfig(19821), crypto::Sha3_256(std::string_view("a")), 1111);
    net::Node node_b(ServiceNodeConfig(0), crypto::Sha3_256(std::string_view("b")), 2222);
    protocol::Service service_a(node_a, honest);
    protocol::Service service_b(node_b, victim);
    service_a.Attach();
    service_b.Attach();

    RNET_CHECK_OK(node_a.Start());
    RNET_CHECK_OK(node_b.Start());
    auto target = net::NetAddress::FromString("127.0.0.1:19821", 19821);
    RNET_CHECK_OK(target);
    RNET_CHECK_OK(node_b.ConnectTo(target.value(), 1000));
    PumpBoth(node_a, service_a, node_b, service_b,
             [&] { return node_a.ReadyCount() == 1 && node_b.ReadyCount() == 1; });

    // A contribution claiming a different parameter count: well-formed bytes, and
    // a model this network is not training.
    auto forged = Contribution(1, GenesisCheckpoint().Id());
    forged.n_params = P().round.model.ParameterCount() * 2;
    RNET_CHECK_OK(node_a.Publish(net::InvType::Contribution, forged.ToContainer(), 1000));

    const bool judged = PumpBoth(node_a, service_a, node_b, service_b,
                                 [&] { return service_b.objects_rejected() > 0; });
    if (!judged) RNET_FAIL("a contribution for the wrong model was not refused");
    RNET_CHECK_EQ(victim.submission_count(), uint32_t{0});
}

// Two nodes on the same network must reach the same starting state without
// exchanging any weights at all — that is the whole reason the initial state is
// derived rather than distributed.
RNET_TEST(Service, NodesAgreeOnWhereTrainingStartsWithoutTransferringWeights) {
    auto a = consensus::GenesisCheckpointFor(consensus::Network::Regtest);
    auto b = consensus::GenesisCheckpointFor(consensus::Network::Regtest);
    RNET_CHECK_OK(a);
    RNET_CHECK_OK(b);
    RNET_CHECK(a.value().Id() == b.value().Id());

    // And a node on another network does not, despite the identical architecture.
    auto other = consensus::GenesisCheckpointFor(consensus::Network::Test);
    RNET_CHECK_OK(other);
    RNET_CHECK(!(a.value().Id() == other.value().Id()));
}

// ---------------------------------------------------------------------------
// The local channel, against the real service.
//
// The transport tests in ipc_tests.cpp prove a frame crosses the socket. They do
// not prove the daemon can build a REPLY — and a daemon that fails to build one
// leaves the worker waiting out its handshake timeout, which reads as a hang
// rather than as a fault. These tests exercise the service, which is where that
// distinction lives.
// ---------------------------------------------------------------------------
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstring>
#include <filesystem>

#include "ipc/cbor.h"
#include "ipc/server.h"
#include "protocol/workerservice.h"

namespace {

// sun_path holds 107 bytes and the project's scratch paths are longer, so tests
// bind under /tmp directly.
std::filesystem::path WorkerSocketDir(std::string_view name) {
    return std::filesystem::path("/tmp") / std::format("rnet-ws-{}", name);
}

int ConnectWorker(const std::filesystem::path& path) {
    const int fd = ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (fd < 0) return -1;
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    const std::string text = path.string();
    std::memcpy(address.sun_path, text.c_str(), text.size());
    if (::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}

Status SendFrame(int fd, ipc::MessageType type, uint64_t request_id,
                 const std::vector<uint8_t>& payload, int passed_fd = -1) {
    ipc::Frame frame;
    frame.type = type;
    frame.request_id = request_id;
    frame.payload = payload;
    auto encoded = ipc::EncodeFrame(frame);
    if (!encoded) return Err(encoded.error());

    iovec io{};
    io.iov_base = encoded.value().data();
    io.iov_len = encoded.value().size();
    msghdr message{};
    message.msg_iov = &io;
    message.msg_iovlen = 1;

    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))];
    if (passed_fd >= 0) {
        message.msg_control = control;
        message.msg_controllen = sizeof(control);
        cmsghdr* header = CMSG_FIRSTHDR(&message);
        header->cmsg_level = SOL_SOCKET;
        header->cmsg_type = SCM_RIGHTS;
        header->cmsg_len = CMSG_LEN(sizeof(int));
        std::memcpy(CMSG_DATA(header), &passed_fd, sizeof(int));
    }
    if (::sendmsg(fd, &message, MSG_NOSIGNAL) < 0) return Err("sendmsg failed");
    return Status::Ok();
}

// Reads one reply, driving the service until it arrives.
Result<std::pair<ipc::MessageType, ipc::CborValue>> Exchange(protocol::WorkerService& service,
                                                             int fd, int64_t& clock) {
    for (int i = 0; i < 50; ++i) {
        clock += 10;
        if (auto st = service.Poll(clock); !st) return Err(st.error());

        std::vector<uint8_t> buffer(ipc::kMaxFrameSize);
        const ssize_t got = ::recv(fd, buffer.data(), buffer.size(), MSG_DONTWAIT);
        if (got > 0) {
            auto frame = ipc::DecodeFrame(std::span<const uint8_t>(buffer.data(),
                                                                   static_cast<size_t>(got)));
            if (!frame) return Err(frame.error());
            auto parsed = ipc::CborParse(frame.value().payload);
            if (!parsed) return Err(parsed.error());
            return std::make_pair(frame.value().type, std::move(parsed.value()));
        }
        if (got == 0) return Err("the daemon closed the connection without replying");
    }
    return Err("no reply arrived");
}

std::vector<uint8_t> HelloPayload(const std::string& genesis_hex, const std::string& policy_hex) {
    auto genesis = util::FromHexFixed<32>(genesis_hex);
    auto policy = util::FromHexFixed<32>(policy_hex);
    auto payload = ipc::CborWriter()
                       .BeginMap(3)
                       .Key("user_agent").Text("test/1")
                       .Key("policy_hash").Bytes(policy.ok() ? policy.value()
                                                             : std::array<uint8_t, 32>{})
                       .Key("genesis_hash").Bytes(genesis.ok() ? genesis.value()
                                                               : std::array<uint8_t, 32>{})
                       .Take();
    return payload.ok() ? payload.value() : std::vector<uint8_t>{};
}

// A memfd sealed against every kind of modification.
int SealedPayload(const std::vector<uint8_t>& bytes) {
    const int fd = ::memfd_create("rnet-test-payload", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (fd < 0) return -1;
    size_t written = 0;
    while (written < bytes.size()) {
        const ssize_t n = ::write(fd, bytes.data() + written, bytes.size() - written);
        if (n <= 0) { ::close(fd); return -1; }
        written += static_cast<size_t>(n);
    }
    if (::fcntl(fd, F_ADD_SEALS, F_SEAL_WRITE | F_SEAL_SHRINK | F_SEAL_GROW) != 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}

}  // namespace

// The handshake must actually complete. A reply the daemon cannot construct is
// indistinguishable, from the worker's side, from a daemon that is ignoring it.
RNET_TEST(WorkerService, TheHandshakeCompletesAndCarriesTheContainers) {
    const auto dir = WorkerSocketDir("hello");
    std::filesystem::remove_all(dir);

    ipc::Server server;
    RNET_CHECK_OK(server.Listen(dir / "rnet.sock"));

    net::NodeConfig node_config;
    node_config.network = consensus::Network::Regtest;
    net::Node node(node_config, crypto::Sha3_256(std::string_view("k")), 77);
    RNET_CHECK_OK(node.Start());

    Participant participant(Config(5), GenesisCheckpoint());
    protocol::WorkerService service(server, node, participant);

    const int client = ConnectWorker(dir / "rnet.sock");
    RNET_CHECK(client >= 0);

    int64_t clock = 1000;
    RNET_CHECK_OK(SendFrame(client, ipc::MessageType::Hello, 1,
                            HelloPayload(P().genesis_hash_hex, P().policy_hash_hex)));

    auto reply = Exchange(service, client, clock);
    RNET_CHECK_OK(reply);
    RNET_CHECK_EQ(static_cast<int>(reply.value().first),
                  static_cast<int>(ipc::MessageType::HelloOk));

    // The worker id comes from the node, never from the worker: a worker that
    // could name itself could submit under someone else's identity.
    RNET_CHECK_EQ(reply.value().second.GetUint("worker_id").value(), uint64_t{5});
    RNET_CHECK_EQ(reply.value().second.GetUint("n_params").value(),
                  P().round.model.ParameterCount());

    // The round and policy travel as containers, so the worker verifies bytes
    // against its own anchors rather than trusting fields the daemon typed out.
    auto round = reply.value().second.GetBytes("round");
    RNET_CHECK_OK(round);
    RNET_CHECK_STREQ(util::ToHex(crypto::Sha3_256(round.value())), P().genesis_hash_hex);
    auto policy = reply.value().second.GetBytes("policy");
    RNET_CHECK_OK(policy);
    RNET_CHECK_STREQ(util::ToHex(crypto::Sha3_256(policy.value())), P().policy_hash_hex);

    ::close(client);
    server.Stop();
    std::filesystem::remove_all(dir);
}

// A worker built against a different round would train a different model and call
// the result a contribution. Compared, never adopted — and the refusal says so
// rather than leaving the worker to time out.
RNET_TEST(WorkerService, RefusesAWorkerOnDifferentConsensusRules) {
    const auto dir = WorkerSocketDir("anchors");
    std::filesystem::remove_all(dir);

    ipc::Server server;
    RNET_CHECK_OK(server.Listen(dir / "rnet.sock"));
    net::NodeConfig node_config;
    node_config.network = consensus::Network::Regtest;
    net::Node node(node_config, crypto::Sha3_256(std::string_view("k")), 78);
    RNET_CHECK_OK(node.Start());
    Participant participant(Config(5), GenesisCheckpoint());
    protocol::WorkerService service(server, node, participant);

    const int client = ConnectWorker(dir / "rnet.sock");
    RNET_CHECK(client >= 0);

    int64_t clock = 1000;
    // Main's anchors offered to a regtest node.
    RNET_CHECK_OK(SendFrame(client, ipc::MessageType::Hello, 1,
                            HelloPayload(P(consensus::Network::Main).genesis_hash_hex,
                                         P(consensus::Network::Main).policy_hash_hex)));

    auto reply = Exchange(service, client, clock);
    RNET_CHECK_OK(reply);
    RNET_CHECK_EQ(static_cast<int>(reply.value().first), static_cast<int>(ipc::MessageType::Error));
    RNET_CHECK_EQ(reply.value().second.GetUint("code").value(),
                  static_cast<uint64_t>(protocol::IpcError::AnchorMismatch));

    ::close(client);
    server.Stop();
    std::filesystem::remove_all(dir);
}

// The whole point of the channel: a local process hands over work and it becomes
// a contribution, validated on exactly the terms a peer's would be.
RNET_TEST(WorkerService, AWorkerSubmitsAContributionThroughTheSocket) {
    const auto dir = WorkerSocketDir("submit");
    std::filesystem::remove_all(dir);

    ipc::Server server;
    RNET_CHECK_OK(server.Listen(dir / "rnet.sock"));
    net::NodeConfig node_config;
    node_config.network = consensus::Network::Regtest;
    net::Node node(node_config, crypto::Sha3_256(std::string_view("k")), 79);
    RNET_CHECK_OK(node.Start());
    Participant participant(Config(5), GenesisCheckpoint());
    protocol::WorkerService service(server, node, participant);

    const int client = ConnectWorker(dir / "rnet.sock");
    RNET_CHECK(client >= 0);
    int64_t clock = 1000;

    RNET_CHECK_OK(SendFrame(client, ipc::MessageType::Hello, 1,
                            HelloPayload(P().genesis_hash_hex, P().policy_hash_hex)));
    RNET_CHECK_OK(Exchange(service, client, clock));

    RNET_CHECK_OK(SendFrame(client, ipc::MessageType::GetAssignment, 2, {}));
    auto assignment = Exchange(service, client, clock);
    RNET_CHECK_OK(assignment);
    RNET_CHECK_EQ(static_cast<int>(assignment.value().first),
                  static_cast<int>(ipc::MessageType::AssignTrain));
    const uint64_t assignment_id = assignment.value().second.GetUint("assignment_id").value();

    // Deliberately absent from the assignment: batch offsets and token windows. If
    // the node chose the data, a colluding node and worker would regain exactly
    // the freedom the deterministic derivation removes.
    RNET_CHECK(assignment.value().second.Find("offsets") == nullptr);
    RNET_CHECK(assignment.value().second.Find("tokens") == nullptr);
    RNET_CHECK(assignment.value().second.Find("lr") == nullptr);

    const size_t n_params = static_cast<size_t>(P().round.model.ParameterCount());
    std::vector<uint8_t> payload(n_params);
    for (size_t i = 0; i < payload.size(); ++i) payload[i] = static_cast<uint8_t>(i * 37 + 11);
    const auto payload_hash = crypto::Sha3_256(payload);

    auto submit = ipc::CborWriter()
                      .BeginMap(3)
                      .Key("scale_exp").Int(12)
                      .Key("payload_hash").Bytes(payload_hash)
                      .Key("assignment_id").Uint(assignment_id)
                      .Take();
    RNET_CHECK_OK(submit);

    const int sealed = SealedPayload(payload);
    RNET_CHECK(sealed >= 0);
    RNET_CHECK_OK(SendFrame(client, ipc::MessageType::SubmitContribution, 3, submit.value(),
                            sealed));
    ::close(sealed);

    auto accepted = Exchange(service, client, clock);
    RNET_CHECK_OK(accepted);
    RNET_CHECK_EQ(static_cast<int>(accepted.value().first),
                  static_cast<int>(ipc::MessageType::SubmitAccepted));
    RNET_CHECK_EQ(participant.submission_count(), uint32_t{1});

    // The payload bytes must be on the node, not merely acknowledged — otherwise
    // no peer could ever fetch what the contribution refers to.
    RNET_CHECK(node.objects().Has(payload_hash));
    RNET_CHECK_EQ(node.objects().Get(payload_hash).value().data.size(), n_params);

    ::close(client);
    server.Stop();
    std::filesystem::remove_all(dir);
}

// The size is fixed by the model at one byte per parameter. A payload of any other
// length is refused before it is read, so the sender never chooses how much the
// daemon allocates.
RNET_TEST(WorkerService, RefusesAPayloadOfTheWrongSize) {
    const auto dir = WorkerSocketDir("size");
    std::filesystem::remove_all(dir);

    ipc::Server server;
    RNET_CHECK_OK(server.Listen(dir / "rnet.sock"));
    net::NodeConfig node_config;
    node_config.network = consensus::Network::Regtest;
    net::Node node(node_config, crypto::Sha3_256(std::string_view("k")), 80);
    RNET_CHECK_OK(node.Start());
    Participant participant(Config(5), GenesisCheckpoint());
    protocol::WorkerService service(server, node, participant);

    const int client = ConnectWorker(dir / "rnet.sock");
    RNET_CHECK(client >= 0);
    int64_t clock = 1000;

    RNET_CHECK_OK(SendFrame(client, ipc::MessageType::Hello, 1,
                            HelloPayload(P().genesis_hash_hex, P().policy_hash_hex)));
    RNET_CHECK_OK(Exchange(service, client, clock));
    RNET_CHECK_OK(SendFrame(client, ipc::MessageType::GetAssignment, 2, {}));
    auto assignment = Exchange(service, client, clock);
    RNET_CHECK_OK(assignment);
    const uint64_t assignment_id = assignment.value().second.GetUint("assignment_id").value();

    std::vector<uint8_t> short_payload(1024, 0x5A);
    auto submit = ipc::CborWriter()
                      .BeginMap(3)
                      .Key("scale_exp").Int(12)
                      .Key("payload_hash").Bytes(crypto::Sha3_256(short_payload))
                      .Key("assignment_id").Uint(assignment_id)
                      .Take();
    RNET_CHECK_OK(submit);

    const int sealed = SealedPayload(short_payload);
    RNET_CHECK(sealed >= 0);
    RNET_CHECK_OK(SendFrame(client, ipc::MessageType::SubmitContribution, 3, submit.value(),
                            sealed));
    ::close(sealed);

    auto refused = Exchange(service, client, clock);
    RNET_CHECK_OK(refused);
    RNET_CHECK_EQ(static_cast<int>(refused.value().first),
                  static_cast<int>(ipc::MessageType::Error));
    RNET_CHECK_EQ(participant.submission_count(), uint32_t{0});

    ::close(client);
    server.Stop();
    std::filesystem::remove_all(dir);
}

// Nothing but the handshake is processed before the handshake completes, for the
// same reason as on the network side.
RNET_TEST(WorkerService, RefusesWorkBeforeTheHandshake) {
    const auto dir = WorkerSocketDir("sequence");
    std::filesystem::remove_all(dir);

    ipc::Server server;
    RNET_CHECK_OK(server.Listen(dir / "rnet.sock"));
    net::NodeConfig node_config;
    node_config.network = consensus::Network::Regtest;
    net::Node node(node_config, crypto::Sha3_256(std::string_view("k")), 81);
    RNET_CHECK_OK(node.Start());
    Participant participant(Config(5), GenesisCheckpoint());
    protocol::WorkerService service(server, node, participant);

    const int client = ConnectWorker(dir / "rnet.sock");
    RNET_CHECK(client >= 0);
    int64_t clock = 1000;

    RNET_CHECK_OK(SendFrame(client, ipc::MessageType::GetAssignment, 1, {}));
    auto refused = Exchange(service, client, clock);
    RNET_CHECK_OK(refused);
    RNET_CHECK_EQ(static_cast<int>(refused.value().first),
                  static_cast<int>(ipc::MessageType::Error));
    RNET_CHECK_EQ(refused.value().second.GetUint("code").value(),
                  static_cast<uint64_t>(protocol::IpcError::OutOfSequence));

    ::close(client);
    server.Stop();
    std::filesystem::remove_all(dir);
}

// ---------------------------------------------------------------------------
// Rounds that close, and checkpoints that get produced.
//
// Until this existed, the round's deadline was hardcoded to zero and every round
// stayed open forever: nothing was ever aggregated, no checkpoint after genesis
// was ever built, and the challenge path was unreachable in production. A network
// in that state looks entirely healthy and produces nothing.
// ---------------------------------------------------------------------------

namespace {

// A participant wired to in-memory payloads and a fake applier, so the round can
// be driven without a GPU. The applier stands in for the worker; what it returns
// is a hash of its inputs, which is enough to prove the checkpoint commits to the
// right base and the right update.
struct DrivenParticipant {
    std::map<crypto::Hash256, std::vector<uint8_t>> payloads;
    std::unique_ptr<Participant> participant;
    int applier_calls{0};
    crypto::Hash256 last_base{};

    explicit DrivenParticipant(uint64_t worker_id) {
        participant = std::make_unique<Participant>(Config(worker_id), GenesisCheckpoint());
        participant->SetPayloadProvider(
            [this](const crypto::Hash256& hash) -> const std::vector<uint8_t>* {
                const auto it = payloads.find(hash);
                return it == payloads.end() ? nullptr : &it->second;
            });
    }

    // Stands in for the worker's second phase: applies whatever is staged and
    // reports a hash of its inputs. Enough to prove the checkpoint commits to the
    // right base and the right update without needing a GPU.
    Status ApplyPendingAsWorkerWould(uint64_t now_ms) {
        const auto* pending = participant->Pending();
        if (pending == nullptr) return Err("nothing staged");
        ++applier_calls;
        last_base = pending->base_weights;

        canon::Writer w;
        w.Hash(pending->base_weights);
        w.U32(static_cast<uint32_t>(pending->scale_exp));
        for (size_t i = 0; i < std::min<size_t>(pending->update.size(), 64); ++i) {
            w.U64(static_cast<uint64_t>(pending->update[i]));
        }
        return participant->CompleteProduction(crypto::Sha3_256(w.data()), now_ms);
    }

    // Registers a contribution and its bytes, as the network would.
    canon::ContributionHeader Contribute(uint64_t worker_id, uint64_t now_ms) {
        std::vector<uint8_t> bytes(static_cast<size_t>(P().round.model.ParameterCount()));
        for (size_t i = 0; i < bytes.size(); ++i) {
            bytes[i] = static_cast<uint8_t>((i * 7 + worker_id * 13) % 251);
        }
        auto header = Contribution(worker_id, GenesisCheckpoint().Id());
        header.payload_hash = crypto::Sha3_256(bytes);
        payloads[header.payload_hash] = std::move(bytes);

        const auto verdict = participant->OnObject(net::InvType::Contribution, header.Id(),
                                                   header.ToContainer(), now_ms);
        RNET_CHECK(verdict.accepted());
        RNET_CHECK_OK(participant->NotePayload(header.payload_hash));
        return header;
    }
};

}  // namespace

// The deadline comes from the policy, not from the node. Two nodes using different
// numbers would disagree about who was late and produce different checkpoints from
// the same network.
RNET_TEST(Round, TheDeadlineIsAConsensusValueAndIsNonZero) {
    for (auto net : {consensus::Network::Main, consensus::Network::Test,
                     consensus::Network::Regtest}) {
        const auto& policy = consensus::Params(net).policy;
        if (policy.round_deadline_ms == 0) {
            RNET_FAIL("{}: a round with no deadline never closes", consensus::Params(net).name);
        }
        RNET_CHECK_OK(policy.Validate());
    }
}

// The whole point: a round closes, its contributions are aggregated, and a
// checkpoint appears that names what went into it.
RNET_TEST(Round, ClosingProducesACheckpointThatNamesItsInputs) {
    DrivenParticipant driven(1);
    auto& participant = *driven.participant;

    const uint64_t opened = 100'000;
    const auto a = driven.Contribute(1, opened);
    const auto b = driven.Contribute(2, opened);
    RNET_CHECK_EQ(participant.submission_count(), uint32_t{2});

    // Before the deadline nothing happens, however many contributions arrived.
    RNET_CHECK_OK(participant.Tick(opened + P().policy.round_deadline_ms - 1));
    RNET_CHECK_EQ(participant.chain().Height(), uint64_t{0});
    RNET_CHECK_EQ(driven.applier_calls, 0);

    // Past it, the elected producer aggregates and stages the update. The
    // checkpoint does not exist yet: applying the update to the weights is the
    // worker's half, and it has not run.
    const uint64_t closed = opened + P().policy.round_deadline_ms + 1;
    RNET_CHECK_OK(participant.Tick(closed));
    RNET_CHECK(participant.Pending() != nullptr);
    RNET_CHECK_EQ(participant.chain().Height(), uint64_t{0});

    RNET_CHECK_OK(driven.ApplyPendingAsWorkerWould(closed));
    RNET_CHECK(participant.Pending() == nullptr);
    if (participant.chain().Height() != 1) {
        RNET_FAIL("the round closed but no checkpoint was produced (applier calls {})",
                  driven.applier_calls);
    }
    RNET_CHECK_EQ(driven.applier_calls, 1);
    // The update is applied to the weights the tip named, not to something else.
    RNET_CHECK(driven.last_base == GenesisCheckpoint().weights_hash);

    auto tip = participant.chain().Tip();
    RNET_CHECK_OK(tip);
    const auto& header = tip.value().header;
    RNET_CHECK_EQ(header.outer_step, uint64_t{1});
    RNET_CHECK(header.parent == GenesisCheckpoint().Id());
    RNET_CHECK_EQ(header.contributor_count, uint32_t{2});

    // The checkpoint must say WHICH contributions it aggregated, or nobody can
    // recompute it and a fabricated checkpoint is indistinguishable from an honest
    // one.
    std::vector<canon::ContributionHeader> expected{a, b};
    RNET_CHECK(header.evidence_hash == diloco::EvidenceHashOf(expected));
    RNET_CHECK(!(header.evidence_hash == crypto::Hash256{}));

    // And it was announced, or no peer would ever learn the step advanced.
    bool announced = false;
    for (const auto& out : participant.TakeOutgoing()) {
        if (out.type == net::InvType::Checkpoint && out.hash == header.Id()) announced = true;
    }
    RNET_CHECK(announced);
}

// A node that is not the producer must not publish. If every node aggregated its
// own view, every node would produce a different checkpoint for the same step.
RNET_TEST(Round, OnlyTheElectedProducerPublishes) {
    // Two nodes holding the same contributions and the same tip. Exactly one of
    // them must consider itself the producer.
    DrivenParticipant first(1);
    DrivenParticipant second(2);
    const uint64_t opened = 100'000;
    for (auto* driven : {&first, &second}) {
        driven->Contribute(1, opened);
        driven->Contribute(2, opened);
    }

    const bool first_produces = first.participant->IsProducerForThisStep();
    const bool second_produces = second.participant->IsProducerForThisStep();
    if (first_produces == second_produces) {
        RNET_FAIL("both nodes think they are{} the producer", first_produces ? "" : " not");
    }

    const uint64_t after = opened + P().policy.round_deadline_ms + 1;
    RNET_CHECK_OK(first.participant->Tick(after));
    RNET_CHECK_OK(second.participant->Tick(after));

    auto& producer = first_produces ? first : second;
    auto& follower = first_produces ? second : first;

    // Only the producer staged anything, and only it can complete.
    RNET_CHECK(producer.participant->Pending() != nullptr);
    RNET_CHECK(follower.participant->Pending() == nullptr);
    RNET_CHECK_ERR(follower.ApplyPendingAsWorkerWould(after));
    RNET_CHECK_OK(producer.ApplyPendingAsWorkerWould(after));
    RNET_CHECK_EQ(producer.applier_calls, 1);
    RNET_CHECK_EQ(follower.applier_calls, 0);
    RNET_CHECK_EQ(producer.participant->chain().Height(), uint64_t{1});
    RNET_CHECK_EQ(follower.participant->chain().Height(), uint64_t{0});
}

// The producer's own checkpoint goes through the checks it would apply to a peer's.
// A node that accepts from itself what it would refuse from a stranger is a node
// whose bugs reach the network before anyone notices.
RNET_TEST(Round, TheProducerValidatesItsOwnCheckpoint) {
    DrivenParticipant driven(1);
    const uint64_t opened = 100'000;
    driven.Contribute(1, opened);
    driven.Contribute(2, opened);
    const uint64_t closed = opened + P().policy.round_deadline_ms + 1;
    RNET_CHECK_OK(driven.participant->Tick(closed));
    RNET_CHECK_OK(driven.ApplyPendingAsWorkerWould(closed));

    auto tip = driven.participant->chain().Tip();
    RNET_CHECK_OK(tip);
    // It is in the chain, which it could only reach through OnCheckpoint.
    RNET_CHECK(driven.participant->chain().Has(tip.value().header.Id()));
    // And the round moved on with it.
    RNET_CHECK_EQ(driven.participant->outer_step(), uint64_t{1});
    RNET_CHECK_EQ(driven.participant->submission_count(), uint32_t{0});
}

// A round whose payloads have not all arrived must not close: producing from a
// subset while naming the whole set gives an evidence hash nobody can reproduce.
RNET_TEST(Round, DoesNotCloseWhileAPayloadIsStillMissing) {
    DrivenParticipant driven(1);
    const uint64_t opened = 100'000;
    driven.Contribute(1, opened);

    // A second contribution whose bytes never arrive.
    auto stranded = Contribution(2, GenesisCheckpoint().Id());
    stranded.payload_hash = crypto::Sha3_256(std::string_view("bytes that never came"));
    RNET_CHECK(driven.participant
                   ->OnObject(net::InvType::Contribution, stranded.Id(), stranded.ToContainer(),
                              opened)
                   .accepted());

    RNET_CHECK_OK(driven.participant->Tick(opened + P().policy.round_deadline_ms + 1));
    RNET_CHECK_EQ(driven.participant->chain().Height(), uint64_t{0});
    RNET_CHECK(driven.participant->Pending() == nullptr);
    RNET_CHECK_EQ(driven.applier_calls, 0);
    RNET_CHECK_EQ(driven.participant->MissingPayloads().size(), size_t{1});
}

// A round that expires without a quorum must be reopened, not left in place.
//
// An abandoned round refuses every later submission and never closes, so a node
// holding one goes on ticking, logging nothing, and never accepts another
// contribution for the rest of its life. The network sees a peer that handshakes
// normally and contributes nothing, forever.
RNET_TEST(Round, AnExpiredRoundIsReopenedRatherThanLeftDead) {
    DrivenParticipant driven(1);
    auto& participant = *driven.participant;

    // regtest needs one contributor, so a quorum-less expiry needs an empty round:
    // opened by a tick, expired by the next.
    const uint64_t opened = 100'000;
    RNET_CHECK_OK(participant.Tick(opened));
    RNET_CHECK_OK(participant.Tick(opened + P().policy.round_deadline_ms + 1));

    // The proof that it reopened is that a contribution is still accepted long
    // after the first round's deadline had passed.
    const uint64_t late = opened + P().policy.round_deadline_ms * 3;
    const auto header = Contribution(2, GenesisCheckpoint().Id());
    const auto verdict = participant.OnObject(net::InvType::Contribution, header.Id(),
                                              header.ToContainer(), late);
    if (!verdict.accepted()) {
        RNET_FAIL("a contribution after an expired round was refused: {}", verdict.reason);
    }
    RNET_CHECK_EQ(participant.submission_count(), uint32_t{1});
}

// A CLOSED round is a place to die in, and it took longer to find than Abandoned
// because it has two exits and both can silently lead nowhere: a payload that
// never lands, or an elected producer that crashed. Until one resolves,
// Round::Submit refuses every contribution because the round is not Open — so the
// node keeps ticking, logs nothing, and never accepts work again.
//
// The first fix for this class covered Abandoned only. These cover the rest.

RNET_TEST(Round, AClosedRoundWithAStrandedPayloadEventuallyReopens) {
    DrivenParticipant driven(1);
    const uint64_t opened = 100'000;
    driven.Contribute(1, opened);

    auto stranded = Contribution(2, GenesisCheckpoint().Id());
    stranded.payload_hash = crypto::Sha3_256(std::string_view("bytes that never came"));
    RNET_CHECK(driven.participant
                   ->OnObject(net::InvType::Contribution, stranded.Id(), stranded.ToContainer(),
                              opened)
                   .accepted());

    const uint64_t deadline = P().policy.round_deadline_ms;

    // Just past closing the round is right to wait; the payload may still land.
    RNET_CHECK_OK(driven.participant->Tick(opened + deadline + 1));
    RNET_CHECK_EQ(driven.participant->MissingPayloads().size(), size_t{1});

    // A full further deadline later it has waited long enough.
    RNET_CHECK_OK(driven.participant->Tick(opened + deadline * 2 + 2));

    const uint64_t later = opened + deadline * 2 + 3;
    const auto fresh = Contribution(3, GenesisCheckpoint().Id());
    const auto verdict = driven.participant->OnObject(net::InvType::Contribution, fresh.Id(),
                                                      fresh.ToContainer(), later);
    if (!verdict.accepted()) {
        RNET_FAIL("a node whose round closed on a stranded payload refused new work: {}",
                  verdict.reason);
    }
}

RNET_TEST(Round, AFollowerWhoseProducerNeverPublishesEventuallyReopens) {
    DrivenParticipant first(1);
    DrivenParticipant second(2);
    const uint64_t opened = 100'000;
    for (auto* driven : {&first, &second}) {
        driven->Contribute(1, opened);
        driven->Contribute(2, opened);
    }
    auto& follower = first.participant->IsProducerForThisStep() ? second : first;
    RNET_CHECK(!follower.participant->IsProducerForThisStep());

    const uint64_t deadline = P().policy.round_deadline_ms;

    // Waiting for the producer's checkpoint is correct here — and that is what
    // makes the bug invisible: waiting for one that is coming and waiting for one
    // that never will look identical from inside.
    RNET_CHECK_OK(follower.participant->Tick(opened + deadline + 1));
    RNET_CHECK_EQ(follower.participant->chain().Height(), uint64_t{0});

    RNET_CHECK_OK(follower.participant->Tick(opened + deadline * 2 + 2));

    const uint64_t later = opened + deadline * 2 + 3;
    const auto fresh = Contribution(3, GenesisCheckpoint().Id());
    const auto verdict = follower.participant->OnObject(net::InvType::Contribution, fresh.Id(),
                                                        fresh.ToContainer(), later);
    if (!verdict.accepted()) {
        RNET_FAIL("a follower whose producer never published refused new work: {}", verdict.reason);
    }
}

// Reopening must not cost a checkpoint that was merely late: it names the same
// parent, so it is still valid when it finally arrives.
RNET_TEST(Round, ALateCheckpointIsStillAcceptedAfterReopening) {
    DrivenParticipant driven(9);
    const uint64_t opened = 100'000;
    driven.Contribute(1, opened);
    driven.Contribute(2, opened);

    const uint64_t deadline = P().policy.round_deadline_ms;
    RNET_CHECK_OK(driven.participant->Tick(opened + deadline * 2 + 2));

    canon::CheckpointHeader late;
    late.parent = GenesisCheckpoint().Id();
    late.round_id = P().round.round_id;
    late.outer_step = 1;
    late.weights_hash = crypto::Sha3_256(std::string_view("weights published late"));
    late.n_params = P().round.model.ParameterCount();
    late.contributor_count = 2;
    late.evidence_hash = crypto::Sha3_256(std::string_view("what it aggregated"));
    // The state this node itself reached. A fabricated value here is refused, and
    // that refusal is the subject of its own test below — here the point is that a
    // LATE but honest checkpoint still lands.
    late.optimizer_state_hash = driven.participant->optimizer_state();
    late.weight_dtype = 1;

    const auto verdict = driven.participant->OnObject(net::InvType::Checkpoint, late.Id(),
                                                      late.ToContainer(), opened + deadline * 3);
    if (!verdict.accepted()) {
        RNET_FAIL("a late checkpoint was refused after the round reopened: {}", verdict.reason);
    }
    RNET_CHECK_EQ(driven.participant->outer_step(), uint64_t{1});
}

// The finding this whole change came from, made permanent.
//
// The optimizer used to step only inside ProduceCheckpoint, so a node that was not
// the elected producer never advanced its momentum. Election rotates. A follower
// therefore arrived at its own turn with momentum nobody else had and computed a
// different update from identical contributions — measured at the time: 56 against
// 33 from the same aggregate, a seventy per cent gap on the first step, compounding
// after that because Nesterov feeds momentum through the lookahead point rather
// than adding it.
//
// Nothing detected it. The checkpoint committed to which contributions were used
// and to the resulting weights hash, but nothing said what state produced them, so
// the update was accepted on the producer's word and every test stayed green while
// the only product of the protocol diverged on every node.
RNET_TEST(Round, EveryNodeAdvancesTheOptimizerNotOnlyTheProducer) {
    DrivenParticipant first(1);
    DrivenParticipant second(2);
    const uint64_t opened = 100'000;
    for (auto* driven : {&first, &second}) {
        driven->Contribute(1, opened);
        driven->Contribute(2, opened);
    }
    auto& producer = first.participant->IsProducerForThisStep() ? first : second;
    auto& follower = first.participant->IsProducerForThisStep() ? second : first;

    const uint64_t closed = opened + P().policy.round_deadline_ms + 1;
    RNET_CHECK_OK(producer.participant->Tick(closed));
    RNET_CHECK_OK(follower.participant->Tick(closed));

    // Both stepped, from the same contributions, so both reached the same state.
    // Before the fix the follower's optimizer had never run at all.
    RNET_CHECK(!(producer.participant->optimizer_state() == crypto::Hash256{}));
    if (!(producer.participant->optimizer_state() == follower.participant->optimizer_state())) {
        RNET_FAIL("producer reached {} and follower reached {} from identical contributions",
                  util::ToHex(producer.participant->optimizer_state()).substr(0, 16),
                  util::ToHex(follower.participant->optimizer_state()).substr(0, 16));
    }
    // And neither considers itself unable to produce next time.
    RNET_CHECK(!producer.participant->optimizer_desynced());
    RNET_CHECK(!follower.participant->optimizer_desynced());
}

// A checkpoint whose optimizer state disagrees is refused, and scored: the two
// nodes computed different updates from the same inputs, and every step after it
// would widen the gap.
RNET_TEST(Round, ACheckpointWithForeignOptimizerStateIsRefused) {
    DrivenParticipant driven(1);
    const uint64_t opened = 100'000;
    driven.Contribute(1, opened);
    driven.Contribute(2, opened);
    RNET_CHECK_OK(driven.participant->Tick(opened + P().policy.round_deadline_ms + 1));
    RNET_CHECK(!(driven.participant->optimizer_state() == crypto::Hash256{}));

    canon::CheckpointHeader forged;
    forged.parent = GenesisCheckpoint().Id();
    forged.round_id = P().round.round_id;
    forged.outer_step = 1;
    forged.weights_hash = crypto::Sha3_256(std::string_view("plausible weights"));
    forged.n_params = P().round.model.ParameterCount();
    forged.contributor_count = 2;
    forged.evidence_hash = crypto::Sha3_256(std::string_view("evidence"));
    // Momentum this node did not reach. Everything else about the checkpoint is
    // well-formed, which is exactly why nothing else would catch it.
    forged.optimizer_state_hash = crypto::Sha3_256(std::string_view("someone else's momentum"));
    forged.weight_dtype = 1;

    const auto verdict = driven.participant->OnObject(net::InvType::Checkpoint, forged.Id(),
                                                      forged.ToContainer(), opened);
    RNET_CHECK_EQ(static_cast<int>(verdict.admission), static_cast<int>(Admission::Rejected));
    RNET_CHECK(verdict.misbehaviour >= 50);
    if (verdict.reason.find("optimizer state") == std::string::npos) {
        RNET_FAIL("the refusal should name what disagreed, got '{}'", verdict.reason);
    }
    RNET_CHECK_EQ(driven.participant->chain().Height(), uint64_t{0});
}

// A checkpoint is a claim about a step this node could not reproduce — it joined
// late, or its payloads never came. It may follow the chain, but it must know it
// cannot produce, and say so.
RNET_TEST(Round, ANodeThatCannotReproduceTheStateFollowsButRefusesToProduce) {
    DrivenParticipant driven(1);
    const uint64_t opened = 100'000;

    // Never contributed, never aggregated, never stepped.
    canon::CheckpointHeader arrived;
    arrived.parent = GenesisCheckpoint().Id();
    arrived.round_id = P().round.round_id;
    arrived.outer_step = 1;
    arrived.weights_hash = crypto::Sha3_256(std::string_view("weights from elsewhere"));
    arrived.n_params = P().round.model.ParameterCount();
    arrived.contributor_count = 2;
    arrived.evidence_hash = crypto::Sha3_256(std::string_view("evidence from elsewhere"));
    arrived.optimizer_state_hash = crypto::Sha3_256(std::string_view("momentum from elsewhere"));
    arrived.weight_dtype = 1;

    const auto verdict = driven.participant->OnObject(net::InvType::Checkpoint, arrived.Id(),
                                                      arrived.ToContainer(), opened);
    // Followed: the chain advanced.
    RNET_CHECK(verdict.accepted());
    RNET_CHECK_EQ(driven.participant->outer_step(), uint64_t{1});
    // But this node knows what it cannot do.
    RNET_CHECK(driven.participant->optimizer_desynced());
    RNET_CHECK(!driven.participant->IsProducerForThisStep());
}

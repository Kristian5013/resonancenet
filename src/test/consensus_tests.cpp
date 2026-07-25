#include <set>
#include <cstring>
#include <cmath>
#include "consensus/genesis.h"
#include "consensus/params.h"
#include "test/framework.h"
#include "util/hex.h"

using namespace rnet;
using namespace rnet::consensus;

RNET_TEST(Params, AllNetworksValidate) {
    for (Network net : {Network::Main, Network::Test, Network::Regtest}) {
        RNET_CHECK_OK(Params(net).Validate());
        RNET_CHECK_OK(Params(net).round.model.Validate());
    }
}

RNET_TEST(Params, NetworksHaveDistinctMagicAndPorts) {
    const auto& main = Params(Network::Main);
    const auto& test = Params(Network::Test);
    const auto& reg = Params(Network::Regtest);

    RNET_CHECK(main.message_magic != test.message_magic);
    RNET_CHECK(main.message_magic != reg.message_magic);
    RNET_CHECK(test.message_magic != reg.message_magic);
    RNET_CHECK(main.default_port != test.default_port);
    RNET_CHECK(main.default_port != reg.default_port);
}

// Every pair must differ. An earlier version of this test only compared against
// regtest and missed that main and test — which train the same architecture —
// produced byte-identical genesis artifacts, so one network's artifact would have
// been accepted by the other. The descriptor now carries the network magic.
RNET_TEST(Params, NetworksHaveDistinctGenesis) {
    const std::string main = GenesisHash(Network::Main);
    const std::string test = GenesisHash(Network::Test);
    const std::string reg = GenesisHash(Network::Regtest);
    RNET_CHECK(main != test);
    RNET_CHECK(main != reg);
    RNET_CHECK(test != reg);
    RNET_CHECK_EQ(main.size(), size_t{64});
}

RNET_TEST(Params, RoundIsBoundToItsNetworkMagic) {
    for (Network net : {Network::Main, Network::Test, Network::Regtest}) {
        const auto& p = Params(net);
        const uint32_t expected = (static_cast<uint32_t>(p.message_magic[0]) << 24) |
                                  (static_cast<uint32_t>(p.message_magic[1]) << 16) |
                                  (static_cast<uint32_t>(p.message_magic[2]) << 8) |
                                  static_cast<uint32_t>(p.message_magic[3]);
        RNET_CHECK_EQ(p.round.network_magic, expected);
    }
}

// A genesis artifact from one network must not load under another network's
// anchor, even when the two train an identical model.
RNET_TEST(Genesis, ArtifactDoesNotCrossNetworks) {
    const auto main_raw = GenesisContainer(Network::Main);
    RNET_CHECK_ERR(LoadGenesis(main_raw, GenesisHash(Network::Test)));
    RNET_CHECK_ERR(LoadGenesis(main_raw, GenesisHash(Network::Regtest)));
}

RNET_TEST(Params, MainnetIsRn1b) {
    const auto& m = Params(Network::Main).round.model;
    RNET_CHECK_EQ(m.d_model, 2048u);
    RNET_CHECK_EQ(m.n_layers, 16u);
    RNET_CHECK_EQ(m.vocab_size, 128000u);   // our own tokenizer, chosen by experiment
    RNET_CHECK_EQ(m.seq_len, 8192u);        // fits 24 GB in deterministic mode
    RNET_CHECK_EQ(m.n_heads / m.n_kv_heads, 4u);
}

RNET_TEST(Params, RegtestTokenizerIsByteLevel) {
    const auto& round = Params(Network::Regtest).round;
    RNET_CHECK_EQ(round.model.vocab_size, 256u);
    RNET_CHECK_EQ(round.tokenizer_hash, crypto::Hash256{});   // all-zero = no artifact
}

RNET_TEST(Params, MainnetPinsOurSovereignTokenizer) {
    const auto& round = Params(Network::Main).round;
    RNET_CHECK(round.tokenizer_hash != crypto::Hash256{});
    RNET_CHECK_STREQ(util::ToHex(round.tokenizer_hash),
                     "11878ae15ef43a42a92e5d02231b780731b39c0067868d66c29f12042919febc");
}

// The pinned anchors must equal what the code actually produces. If this fails,
// either a consensus parameter changed (a hard fork, which must be deliberate) or
// the anchors were not refreshed — both are release blockers.
RNET_TEST(Genesis, PinnedAnchorsMatchEmittedArtifacts) {
    for (Network net : {Network::Main, Network::Test, Network::Regtest}) {
        const auto& p = Params(net);
        if (p.genesis_hash_hex.empty()) RNET_FAIL("{}: genesis anchor is not pinned", p.name);
        RNET_CHECK_STREQ(GenesisHash(net), p.genesis_hash_hex);
    }
}

// A node must be able to bootstrap the way it does in production: read the
// shipped artifact and validate it against the compiled-in anchor.
RNET_TEST(Genesis, ShippedArtifactsLoadUnderPinnedAnchors) {
    for (Network net : {Network::Main, Network::Test, Network::Regtest}) {
        const auto raw = GenesisContainer(net);
        auto round = LoadGenesis(raw, Params(net).genesis_hash_hex);
        RNET_CHECK_OK(round);
        RNET_CHECK_EQ(round.value().network_magic, Params(net).round.network_magic);
    }
}

RNET_TEST(Genesis, ContainerDecodesToTheSameRound) {
    for (Network net : {Network::Main, Network::Test, Network::Regtest}) {
        const auto raw = GenesisContainer(net);
        auto round = canon::RoundDescriptor::FromContainer(raw);
        RNET_CHECK_OK(round);
        RNET_CHECK_EQ(round.value().Id(), Params(net).round.Id());
    }
}

RNET_TEST(Genesis, GenesisHashIsStable) {
    // Emitting twice must produce identical bytes — the artifact is deterministic,
    // so a release build and a developer build agree.
    RNET_CHECK_EQ(GenesisContainer(Network::Main), GenesisContainer(Network::Main));
    RNET_CHECK_STREQ(GenesisHash(Network::Main), GenesisHash(Network::Main));
}

RNET_TEST(Genesis, WrongHashIsRefused) {
    const auto raw = GenesisContainer(Network::Main);
    const std::string wrong(64, '0');
    RNET_CHECK_ERR(LoadGenesis(raw, wrong));
}

RNET_TEST(Genesis, CorrectHashIsAccepted) {
    const auto raw = GenesisContainer(Network::Main);
    RNET_CHECK_OK(LoadGenesis(raw, GenesisHash(Network::Main)));
}

// Loading with no anchor must fail closed: a node with an unpinned network has no
// basis to trust any artifact, and "accept whatever is on disk" is exactly the
// hole this design exists to prevent.
RNET_TEST(Genesis, EmptyAnchorIsRefused) {
    const auto raw = GenesisContainer(Network::Main);
    RNET_CHECK_ERR(LoadGenesis(raw, ""));
}

RNET_TEST(Genesis, TamperedArtifactIsRefused) {
    auto raw = GenesisContainer(Network::Main);
    const std::string anchor = GenesisHash(Network::Main);
    raw[20] ^= 0x01;
    RNET_CHECK_ERR(LoadGenesis(raw, anchor));
}

// Policy is a second, independently updatable artifact — and it carries the
// invariant that ties verification to storage.
RNET_TEST(Policy, EveryNetworkHasAValidPolicy) {
    for (Network net : {Network::Main, Network::Test, Network::Regtest}) {
        RNET_CHECK_OK(Params(net).policy.Validate());
        RNET_CHECK(Params(net).policy.challenge_deadline_steps <
                   Params(net).policy.retained_checkpoints);
    }
}

RNET_TEST(Policy, PoliciesAreDistinctPerNetwork) {
    const std::string main = PolicyHash(Network::Main);
    const std::string test = PolicyHash(Network::Test);
    const std::string reg = PolicyHash(Network::Regtest);
    RNET_CHECK(main != test);
    RNET_CHECK(main != reg);
    RNET_CHECK(test != reg);
}

// The point of splitting the artifacts: tuning an operational knob must not
// disturb the model's identity.
RNET_TEST(Policy, TuningPolicyDoesNotChangeTheRoundHash) {
    const std::string round_before = GenesisHash(Network::Main);
    auto policy = Params(Network::Main).policy;
    policy.challenge_percent = 20;
    policy.policy_version = 2;
    RNET_CHECK_OK(policy.Validate());
    RNET_CHECK(util::ToHex(policy.Id()) != PolicyHash(Network::Main));   // policy identity moved
    RNET_CHECK_STREQ(GenesisHash(Network::Main), round_before);          // round identity did not
}

RNET_TEST(Policy, UnanswerableChallengeWindowIsRejected) {
    auto policy = Params(Network::Main).policy;
    policy.challenge_deadline_steps = policy.retained_checkpoints;
    RNET_CHECK_ERR(policy.Validate());
    RNET_CHECK_ERR(canon::PolicyDescriptor::FromContainer(policy.ToContainer()));
}

RNET_TEST(Policy, PinnedAnchorsMatchEmittedArtifacts) {
    for (Network net : {Network::Main, Network::Test, Network::Regtest}) {
        const auto& p = Params(net);
        if (p.policy_hash_hex.empty()) RNET_FAIL("{}: policy anchor is not pinned", p.name);
        RNET_CHECK_STREQ(PolicyHash(net), p.policy_hash_hex);
        RNET_CHECK_OK(LoadPolicy(PolicyContainer(net), p.policy_hash_hex));
    }
}

RNET_TEST(Policy, ArtifactDoesNotCrossNetworks) {
    RNET_CHECK_ERR(LoadPolicy(PolicyContainer(Network::Main), PolicyHash(Network::Test)));
}

RNET_TEST(Network, NamesParseRoundTrip) {
    for (Network net : {Network::Main, Network::Test, Network::Regtest}) {
        auto parsed = ParseNetwork(NetworkName(net));
        RNET_CHECK_OK(parsed);
        RNET_CHECK(parsed.value() == net);
    }
    RNET_CHECK_ERR(ParseNetwork("mainnet"));
}

// ---------------------------------------------------------------------------
// Where training starts.
//
// Every node must begin from the same weights, byte for byte. Nodes that start
// from different states compute deltas that cannot be aggregated and cannot be
// verified, and the failure looks like every worker cheating at once — which is
// the hardest kind of bug to read. So the starting state is derived rather than
// distributed, and these tests are about that derivation being pinned down hard
// enough for two independent implementations to agree.
// ---------------------------------------------------------------------------
#include "consensus/init.h"

// The layout and the parameter count are written independently. If they disagree,
// one of them is wrong about the model, and shipping either would mean a network
// whose idea of its own size does not match its weights.
RNET_TEST(Init, TheTensorLayoutAndTheParameterCountAgree) {
    for (auto net : {Network::Main, Network::Test, Network::Regtest}) {
        const auto& model = Params(net).round.model;
        uint64_t total = 0;
        for (const auto& tensor : TensorLayout(model)) total += tensor.elements();
        if (total != model.ParameterCount()) {
            RNET_FAIL("{}: layout holds {} parameters, the spec says {}", Params(net).name, total,
                      model.ParameterCount());
        }
    }
}

// Distinct names, or two tensors would share a stream position and the "random"
// initialisation would contain exact duplicates.
RNET_TEST(Init, EveryTensorHasADistinctName) {
    const auto layout = TensorLayout(Params(Network::Regtest).round.model);
    std::set<std::string> names;
    for (const auto& tensor : layout) {
        if (!names.insert(tensor.name).second) RNET_FAIL("duplicate tensor name '{}'", tensor.name);
    }
    RNET_CHECK_EQ(names.size(), layout.size());
}

// Two networks running the same architecture must not start from the same
// weights: the network magic is part of the descriptor, so it is part of the seed.
RNET_TEST(Init, NetworksWithTheSameModelStillStartDifferently) {
    const auto& main_round = Params(Network::Main).round;
    const auto& test_round = Params(Network::Test).round;
    RNET_CHECK_EQ(main_round.model.ParameterCount(), test_round.model.ParameterCount());
    RNET_CHECK(!(InitSeed(main_round) == InitSeed(test_round)));
    if (Params(Network::Main).genesis_weights_hash_hex ==
        Params(Network::Test).genesis_weights_hash_hex) {
        RNET_FAIL("main and test start from identical weights");
    }
}

// Any slice must be reachable without generating what precedes it — that is what
// makes a 1B model initialisable in bounded memory, and what lets a verifier
// re-derive one tensor without re-deriving the model.
RNET_TEST(Init, AnySliceCanBeGeneratedOnItsOwn) {
    const auto& round = Params(Network::Regtest).round;
    const auto layout = TensorLayout(round.model);
    const auto seed = InitSeed(round);

    // A tensor big enough to cross several stream blocks.
    uint32_t index = 0;
    for (uint32_t i = 0; i < layout.size(); ++i) {
        if (!layout[i].is_norm && layout[i].elements() >= 256) { index = i; break; }
    }
    const auto& tensor = layout[index];

    std::vector<float> whole(256);
    RNET_CHECK_OK(InitTensorSlice(seed, index, tensor, 0, whole));

    // Offsets deliberately chosen to be off a block boundary: the stream is eight
    // values per block, so 3 and 37 land mid-block.
    for (uint64_t offset : {uint64_t{0}, uint64_t{3}, uint64_t{37}, uint64_t{200}}) {
        std::vector<float> slice(40);
        RNET_CHECK_OK(InitTensorSlice(seed, index, tensor, offset, slice));
        for (size_t i = 0; i < slice.size(); ++i) {
            if (slice[i] != whole[offset + i]) {
                RNET_FAIL("slice at offset {} disagrees at element {}", offset, i);
            }
        }
    }
}

// Norm weights start at one. Random noise in the one place the model has no
// capacity to correct it would make the first forward pass depend on chance.
RNET_TEST(Init, NormWeightsStartAtOne) {
    const auto& round = Params(Network::Regtest).round;
    const auto layout = TensorLayout(round.model);
    const auto seed = InitSeed(round);

    bool found = false;
    for (uint32_t i = 0; i < layout.size(); ++i) {
        if (!layout[i].is_norm) continue;
        found = true;
        std::vector<float> values(std::min<uint64_t>(16, layout[i].elements()));
        RNET_CHECK_OK(InitTensorSlice(seed, i, layout[i], 0, values));
        for (float v : values) RNET_CHECK_EQ(v, 1.0f);
    }
    RNET_CHECK(found);
}

// Values stay inside 1/sqrt(fan_in). A bound that leaked would put the model
// outside the regime the scaling was chosen for, at the one step where nothing
// downstream can notice.
RNET_TEST(Init, ValuesStayWithinTheirBound) {
    const auto& round = Params(Network::Regtest).round;
    const auto layout = TensorLayout(round.model);
    const auto seed = InitSeed(round);

    for (uint32_t i = 0; i < layout.size(); ++i) {
        if (layout[i].is_norm) continue;
        const float bound = 1.0f / std::sqrt(static_cast<float>(layout[i].cols));
        std::vector<float> values(std::min<uint64_t>(4096, layout[i].elements()));
        RNET_CHECK_OK(InitTensorSlice(seed, i, layout[i], 0, values));
        for (float v : values) {
            if (v < -bound || v > bound) {
                RNET_FAIL("tensor '{}' produced {} outside ±{}", layout[i].name, v, bound);
            }
        }
    }
}

// bf16 conversion rounds half to even — the same rule the hardware follows, so a
// node converting in software and one converting on a GPU agree.
RNET_TEST(Init, Bf16RoundsHalfToEven) {
    // Exactly representable values pass through untouched.
    RNET_CHECK_EQ(FloatToBf16(1.0f), uint16_t{0x3F80});
    RNET_CHECK_EQ(FloatToBf16(-1.0f), uint16_t{0xBF80});
    RNET_CHECK_EQ(FloatToBf16(0.0f), uint16_t{0x0000});

    // A tie rounds to the even neighbour, not away from zero.
    float tie_to_even;
    uint32_t bits = 0x3F808000u;   // halfway between 0x3F80 and 0x3F81
    std::memcpy(&tie_to_even, &bits, sizeof(bits));
    RNET_CHECK_EQ(FloatToBf16(tie_to_even), uint16_t{0x3F80});

    float tie_up;
    bits = 0x3F818000u;            // halfway between 0x3F81 and 0x3F82
    std::memcpy(&tie_up, &bits, sizeof(bits));
    RNET_CHECK_EQ(FloatToBf16(tie_up), uint16_t{0x3F82});

    // A NaN must stay a NaN rather than rounding into infinity.
    float nan_value;
    bits = 0x7F800001u;
    std::memcpy(&nan_value, &bits, sizeof(bits));
    const uint16_t converted = FloatToBf16(nan_value);
    RNET_CHECK_EQ(uint16_t(converted & 0x7F80), uint16_t{0x7F80});
    RNET_CHECK(uint16_t(converted & 0x007F) != 0);
}

// The pinned anchor is the whole point: a node checks that what it derived is what
// the network agreed on. Regtest is small enough to recompute in a test; the
// larger networks are checked by `rnet-tool genesis-weights`.
RNET_TEST(Init, TheDerivedWeightsMatchThePinnedAnchor) {
    const auto& params = Params(Network::Regtest);
    RNET_CHECK(!params.genesis_weights_hash_hex.empty());

    auto derived = GenesisWeightsHash(params.round);
    RNET_CHECK_OK(derived);
    RNET_CHECK_STREQ(util::ToHex(derived.value()), params.genesis_weights_hash_hex);
}

// And the checkpoint every node starts from follows from that anchor, with no
// second file to disagree with it.
RNET_TEST(Init, TheGenesisCheckpointFollowsFromTheAnchor) {
    for (auto net : {Network::Main, Network::Test, Network::Regtest}) {
        auto checkpoint = GenesisCheckpointFor(net);
        RNET_CHECK_OK(checkpoint);
        RNET_CHECK(checkpoint.value().IsGenesis());
        RNET_CHECK_EQ(checkpoint.value().outer_step, uint64_t{0});
        RNET_CHECK_EQ(checkpoint.value().n_params, Params(net).round.model.ParameterCount());
        RNET_CHECK_STREQ(util::ToHex(checkpoint.value().weights_hash),
                         Params(net).genesis_weights_hash_hex);
    }
}

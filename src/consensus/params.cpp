#include "consensus/params.h"

#include <format>

#include "util/hex.h"

namespace rnet::consensus {

namespace {

using canon::ActivationId;
using canon::DeterminismClass;
using canon::ModelSpec;
using canon::NormId;
using canon::OptimizerId;
using canon::RoundDescriptor;

// SHA3-256 of share/genesis/tokenizer_round0.json — our own 128k byte-level BPE.
// The vocabulary size was chosen by experiment (matched-parameter bpb sweep:
// spending budget on vocabulary beat spending it on depth, monotonically up to
// 128k), not copied from another model.
constexpr std::string_view kTokenizerRound0 =
    "11878ae15ef43a42a92e5d02231b780731b39c0067868d66c29f12042919febc";

// Packs the 4-byte wire magic into the u32 the round descriptor is bound to.
uint32_t MagicWord(const std::array<uint8_t, 4>& magic) {
    return (static_cast<uint32_t>(magic[0]) << 24) | (static_cast<uint32_t>(magic[1]) << 16) |
           (static_cast<uint32_t>(magic[2]) << 8) | static_cast<uint32_t>(magic[3]);
}

// Operational policy shared by every network, differing only in its knobs.
// shadow_mode stays true everywhere: settlement waits on both a demonstrated
// convergence and a measured determinism class.
canon::PolicyDescriptor MakePolicy(uint32_t network_magic, uint32_t inner_steps,
                                   uint32_t micro_batch, uint32_t min_contributors,
                                   uint32_t checkpoint_interval, uint32_t chunk_tokens, uint32_t round_deadline_ms,
                                   uint32_t challenge_percent, uint32_t challenge_deadline,
                                   uint32_t retained, uint32_t slash_quorum) {
    canon::PolicyDescriptor p;
    p.network_magic = network_magic;
    p.policy_version = 1;
    p.inner_steps = inner_steps;
    p.micro_batch = micro_batch;
    p.min_contributors = min_contributors;
    p.checkpoint_interval = checkpoint_interval;
    p.dataset_chunk_tokens = chunk_tokens;
    p.round_deadline_ms = round_deadline_ms;
    p.outer_momentum_q16 = 58982;   // 0.90
    p.outer_lr_q16 = 45875;         // 0.70
    p.nesterov = 1;
    p.challenge_percent = challenge_percent;
    p.challenge_deadline_steps = challenge_deadline;
    p.retained_checkpoints = retained;
    p.slash_quorum = slash_quorum;
    p.shadow_mode = 1;
    return p;
}

crypto::Hash256 HashFromHex(std::string_view hex) {
    auto h = util::FromHexFixed<32>(hex);
    // Parameters are compile-time constants authored in this file; a malformed
    // literal is a build-time mistake, not a runtime condition.
    return h.ok() ? h.value() : crypto::Hash256{};
}

// RN-1B — the launch model.
//
// Sizing is measured, not guessed: in deterministic mode (which verification
// requires) this fits a 24 GB card at seq_len 8192 with room to spare, while the
// next size up does not. See doc/design/architecture.md.
ModelSpec Rn1bModel() {
    ModelSpec m;
    m.d_model = 2048;
    m.n_layers = 16;
    m.n_heads = 16;
    m.n_kv_heads = 4;          // GQA 4:1
    m.d_ff = 5632;
    m.vocab_size = 128000;
    m.seq_len = 8192;
    m.rope_theta = 500000;
    m.tie_embeddings = 1;
    m.qk_norm = 1;             // stabilises high-LR continuous training
    m.norm = NormId::RmsNorm;
    m.activation = ActivationId::SwiGlu;
    return m;
}

// A deliberately tiny model so functional tests run in seconds.
ModelSpec RegtestModel() {
    ModelSpec m;
    m.d_model = 256;
    m.n_layers = 4;
    m.n_heads = 4;
    m.n_kv_heads = 2;
    m.d_ff = 688;
    m.vocab_size = 256;        // byte-level
    m.seq_len = 256;
    m.rope_theta = 500000;
    m.tie_embeddings = 1;
    m.qk_norm = 1;
    m.norm = NormId::RmsNorm;
    m.activation = ActivationId::SwiGlu;
    return m;
}

NetworkParams MakeMain() {
    NetworkParams p;
    p.network = Network::Main;
    p.name = "main";
    p.message_magic = {0x52, 0x4E, 0x4D, 0x31};   // "RNM1"
    p.default_port = 9444;
    p.round.network_magic = MagicWord(p.message_magic);
    p.round.round_id = 0;
    p.round.model = Rn1bModel();
    p.round.determinism_class = DeterminismClass::Pending;
    p.round.optimizer = OptimizerId::Adafactor;
    p.round.tokenizer_hash = HashFromHex(kTokenizerRound0);
    p.round.dataset_root = crypto::Hash256{};     // pinned when the corpus is published
    p.policy = MakePolicy(MagicWord(p.message_magic),
                          /*inner_steps=*/250, /*micro_batch=*/1, /*min_contributors=*/2,
                          /*checkpoint_interval=*/4, /*chunk_tokens=*/1u << 20,
                          // Ten minutes: long enough for a consumer GPU to finish
                          // 250 inner steps and upload ~1 GB, short enough that one
                          // absent worker does not hold the network.
                          /*round_deadline_ms=*/600'000,
                          /*challenge_percent=*/10, /*challenge_deadline=*/3,
                          /*retained=*/5, /*slash_quorum=*/3);
    // Populated when the seed infrastructure is stood up. Empty is honest: a node
    // with no seeds and no peers says so rather than pretending it is connected.
    p.dns_seeds = {};
    p.fixed_seeds = {};
    // The state every node starts from. Not distributed as a file — derived from
    // this very descriptor (see consensus/init.h) and pinned here, so a node can
    // reproduce the starting weights offline and check that it did.
    p.genesis_weights_hash_hex =
        "d98d2b9797be6e7a304679d44f2b488d9adca4eadc69568607e038964be8fe49";
    p.genesis_hash_hex = "305780b8c751e8c9a540344de2eda40df7c2b84fb2cd7ed385ae9b6aaddcb78a";
    p.policy_hash_hex = "d9a253a89a6381cd8cfdfcb219b253b36d55e47d808c91452e2b03e9d30d88d7";
    return p;
}

NetworkParams MakeTest() {
    NetworkParams p = MakeMain();
    p.network = Network::Test;
    p.name = "test";
    p.message_magic = {0x52, 0x4E, 0x54, 0x31};   // "RNT1"
    p.default_port = 19444;
    p.round.network_magic = MagicWord(p.message_magic);   // distinct genesis from main
    p.policy = MakePolicy(MagicWord(p.message_magic), 50, 1, 2, 2, 1u << 20,
                          /*round_deadline_ms=*/120'000,
                          /*challenge_percent=*/25,   // cheap chain: verify aggressively
                          /*challenge_deadline=*/5, /*retained=*/8, /*slash_quorum=*/2);
    // Differs from main despite the identical architecture: the network magic is
    // part of the descriptor, so the seed derived from it differs too.
    p.genesis_weights_hash_hex =
        "d51566a9ef437ce5cabb14cbda76825abd8ae35c3ef0366aec4e73433822e607";
    p.genesis_hash_hex = "199fa0bd946726a0a416da7b5103891fc6302b337ba84f558740e0bd8338c61d";
    p.policy_hash_hex = "ed4cfcfa75c2e2a29464dbc00bcc4882aac706d00be60e6ddd2278523b3a0992";
    return p;
}

NetworkParams MakeRegtest() {
    NetworkParams p;
    p.network = Network::Regtest;
    p.name = "regtest";
    p.message_magic = {0x52, 0x4E, 0x52, 0x31};   // "RNR1"
    p.default_port = 19555;
    p.round.network_magic = MagicWord(p.message_magic);
    p.round.round_id = 0;
    p.round.model = RegtestModel();
    p.round.determinism_class = DeterminismClass::Pending;
    p.round.optimizer = OptimizerId::Adafactor;
    p.round.tokenizer_hash = crypto::Hash256{};   // all-zero: byte-level tokenizer
    p.round.dataset_root = crypto::Hash256{};
    p.policy = MakePolicy(MagicWord(p.message_magic), 4, 1, 1, 1, 4096,
                          /*round_deadline_ms=*/1'000,   // tests must not wait
                          /*challenge_percent=*/100,   // verify everything in tests
                          /*challenge_deadline=*/2, /*retained=*/4, /*slash_quorum=*/1);
    p.genesis_weights_hash_hex =
        "006d448c68865f6fdef93915d848cbc03e23d6b6f8256915c414de2b8211ce51";
    p.genesis_hash_hex = "af61f8fdebe3b92867fdf54010c040363adadb4d32f4b76740b502c789f71928";
    p.policy_hash_hex = "a2d482f1f4d85fe786c259a4e38074691e830e2902e3f1ccd878ec35c0fa067f";
    return p;
}

const NetworkParams kMain = MakeMain();
const NetworkParams kTest = MakeTest();
const NetworkParams kRegtest = MakeRegtest();

}  // namespace

Result<Network> ParseNetwork(std::string_view name) {
    if (name == "main") return Network::Main;
    if (name == "test") return Network::Test;
    if (name == "regtest") return Network::Regtest;
    return Err("unknown network '" + std::string(name) + "' (expected main|test|regtest)");
}

std::string NetworkName(Network net) {
    switch (net) {
        case Network::Main: return "main";
        case Network::Test: return "test";
        case Network::Regtest: return "regtest";
    }
    return "unknown";
}

Status NetworkParams::Validate() const {
    if (auto st = round.model.Validate(); !st) return st;
    if (auto st = policy.Validate(); !st) return st;
    if (policy.network_magic != round.network_magic) {
        return Err("params: policy and round belong to different networks");
    }
    return Status::Ok();
}

const NetworkParams& Params(Network net) {
    switch (net) {
        case Network::Main: return kMain;
        case Network::Test: return kTest;
        case Network::Regtest: return kRegtest;
    }
    return kMain;
}

}  // namespace rnet::consensus

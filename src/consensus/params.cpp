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

// SHA3-256 of share/genesis/tokenizer_round0.json — a 32k byte-level BPE, trained
// by worker/tools/train_tokenizer.py on 3 GB of FineWeb-Edu.
//
// 32k rather than the 128k this replaces, because the model is 400M rather than
// 1B: a 128000-entry embedding at d_model 1024 would be a third of the parameter
// budget spent on a lookup table. Measured on the same text, the two compress
// almost identically — 4.77 characters per token against 5.05 — so the hundred
// million parameters move into layers for a 5.5%% penalty in window content.
//
// A byte vocabulary was measured against both and rejected: it removes the corpus
// pipeline entirely, which is attractive, but needs five times the positions for
// the same text, and that lands on context length.
constexpr std::string_view kTokenizerRound0 =
    "1f8d0c4bc23d000c4f602654e615a53fc61f0fb3f56afdce492b640ad54b9d93";

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
// RN-400M: the round-0 model.
//
// Shaped by measurement on a 24 GB consumer card, which is the hardware this
// network is for. Three candidates at 400M parameters were timed at seq_len
// 16384 with deterministic flash attention, forward and backward, 24 cores:
//
//   d1024 L16 H8  ff6656   1.44 s/step   11403 tok/s   3.42 GiB
//   d1024 L24 H8  ff4096   1.86 s/step    8819 tok/s   3.20 GiB   <- this
//   d1280 L20 H10 ff3584   1.86 s/step    8829 tok/s   3.32 GiB
//
// The first is 28% faster and was not chosen: an FFN 6.5x the model width is off
// the beaten path, while 24 layers of 1024 is the shape GPT-2 medium, OPT-350M
// and Pythia-410M all use, so its hyperparameters are known rather than guessed.
// For a first round that is worth more than 28%.
ModelSpec Rn400mModel() {
    ModelSpec m;
    m.d_model = 1024;
    m.n_layers = 24;
    m.n_heads = 8;             // head_dim 128, which is what flash kernels want
    m.n_kv_heads = 2;          // GQA 4:1
    m.d_ff = 4096;
    // 32k, not 128k.
    //
    // A 128000-entry embedding at d_model 1024 is 131 million parameters — a
    // third of a 400M model spent on a lookup table. The sweep that chose 128k
    // was run at 1B, where it is 27%; at this size the balance is different and
    // the 100 million parameters go into layers instead.
    m.vocab_size = 32000;
    // 16384, not 8192.
    //
    // Measured: throughput falls faster than the window grows — 19234 tok/s at
    // 8192 against 2501 at 131072 — so a long training context is paid for in
    // data seen, and data is what a 400M model is short of. 16384 doubles the
    // window for 31% of the throughput, and 8x NTK scaling takes it to 131072 at
    // inference for almost nothing. Training at 131072 would have cost 8x the
    // data for a length that can be added afterwards.
    m.seq_len = 16384;
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
    p.round.model = Rn400mModel();
    p.round.determinism_class = DeterminismClass::Pending;
    p.round.optimizer = OptimizerId::Adafactor;
    p.round.tokenizer_hash = HashFromHex(kTokenizerRound0);
    p.round.dataset_root = crypto::Hash256{};     // pinned when the corpus is published
    p.policy = MakePolicy(MagicWord(p.message_magic),
                          // Two hundred, not two hundred and fifty. Measured on a
                          // 24 GB card at seq_len 16384: 1.86 s per inner step, so
                          // 250 of them is 7.7 minutes, and a 398 MB pseudo-gradient
                          // still has to be uploaded inside the same deadline. 200
                          // is 6.2 minutes and leaves room for a worker slower than
                          // the one this was measured on.
                          /*inner_steps=*/200, /*micro_batch=*/1, /*min_contributors=*/2,
                          /*checkpoint_interval=*/4, /*chunk_tokens=*/1u << 20,
                          // Twenty minutes, and the number is a measurement rather
                          // than a round figure.
                          //
                          // An inner step of this model at seq_len 16384 takes 4.09
                          // seconds on an RTX 5090 — measured on the real thing, not
                          // on a bare transformer block, which predicted 1.86 and
                          // omitted the output projection over 32000, the loss and
                          // the optimizer. So 200 steps is 818 seconds there, and a
                          // card a third slower needs 1060.
                          //
                          // This deadline and inner_steps together decide what
                          // hardware may participate. Ten minutes would have meant a
                          // network only for cards as fast as the one it was
                          // measured on, which is not the network this is for.
                          /*round_deadline_ms=*/1'200'000,
                          /*challenge_percent=*/10, /*challenge_deadline=*/3,
                          /*retained=*/5, /*slash_quorum=*/3);
    // Where a node with no peers looks first. Not part of any hashed artifact —
    // which hostnames a network trusts is operational, not consensus — so adding
    // one does not move an anchor.
    //
    // A hostname that does not resolve is not an error: BootstrapFromSeeds logs it
    // and moves on, because a node that refuses to start when one hostname is down
    // is a node that can be stopped by taking down one hostname.
    p.dns_seeds = {"seed.resonancenet.org"};
    // Literal addresses, consulted only when every seed is unreachable. Filled in
    // once the seed node has a stable address.
    p.fixed_seeds = {};
    // The state every node starts from. Not distributed as a file — derived from
    // this very descriptor (see consensus/init.h) and pinned here, so a node can
    // reproduce the starting weights offline and check that it did.
    p.genesis_weights_hash_hex =
        "db92552dc1a8115ebec224c0c8b8fae459d30b47a4606dfdd932366d659c76d7";
    p.genesis_hash_hex = "93e932f33d0d86eadb5c16fef2d20784c45bb819f5c167440009979915e69d25";
    p.policy_hash_hex = "8a3426b6eee6c16c4c9d13aa12dae5bfc66ada1353aa0625243faebcc26a4e74";
    return p;
}

NetworkParams MakeTest() {
    NetworkParams p = MakeMain();
    p.network = Network::Test;
    p.name = "test";
    p.message_magic = {0x52, 0x4E, 0x54, 0x31};   // "RNT1"
    p.default_port = 19444;
    p.round.network_magic = MagicWord(p.message_magic);   // distinct genesis from main
    // Cleared deliberately. This function starts from MakeMain(), so without this
    // line the test network would inherit main's seed and every test node would
    // knock on the door of the production one. A test network needs its own
    // hostname, and until that exists it has none.
    p.dns_seeds = {};
    p.fixed_seeds = {};
    p.policy = MakePolicy(MagicWord(p.message_magic), 50, 1, 2, 2, 1u << 20,
                          /*round_deadline_ms=*/120'000,
                          /*challenge_percent=*/25,   // cheap chain: verify aggressively
                          /*challenge_deadline=*/5, /*retained=*/8, /*slash_quorum=*/2);
    // Differs from main despite the identical architecture: the network magic is
    // part of the descriptor, so the seed derived from it differs too.
    p.genesis_weights_hash_hex =
        "cc3edecb0dcd5dae9d5c6736c449c7fddc78654acbc84e67c3323299ca1b5df0";
    p.genesis_hash_hex = "9c8db75ddaaebd51e9e609eba8fb667d84a6cd210b6c89a14103fcf019525f92";
    p.policy_hash_hex = "42b2726ab6163d8d1dc29b1ee532c36c13be4f3553e21482da7ff4bbed1e5546";
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
    // Regtest is a single machine talking to itself. Seeds would be meaningless
    // and a lookup against a public hostname from a local test is worse than
    // meaningless.
    p.dns_seeds = {};
    p.fixed_seeds = {};
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
        "55341a2c5c05ce2bab43e24e09bf153ccb9be7c2f5158fbd89980c6ef2cbd7d4";
    p.genesis_hash_hex = "29c86631c8a991646dac442788a6fdc88b6056434a0b9fbd5eaa8f2ff5c5ca3d";
    p.policy_hash_hex = "3ef13240e05e7346ad9c173b050a16711239d9a58c21a0b2cb777293994d34ac";
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

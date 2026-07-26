// rnet-tool — offline utilities for genesis artifacts and corpora.
//
//   rnet-tool genesis-emit  --network main --out share/genesis
//   rnet-tool genesis-show    --network main
//   rnet-tool genesis-weights --network main   (re-derives the initial weights)
//   rnet-tool genesis-check --file share/genesis/main.rnet --hash <sha3>
//   rnet-tool dataset-build --file corpus.bin --out data/corpus --tokenizer-hash <sha3>
//   rnet-tool dataset-check --file corpus.bin --manifest data/corpus.rnds
//   rnet-tool schedule-show --manifest data/corpus.rnds --worker 1 --step 0
#include <cstdio>
#include <filesystem>
#include <string>

#include "consensus/genesis.h"
#include "consensus/init.h"
#include "consensus/params.h"
#include "dataset/manifest.h"
#include "dataset/scheduler.h"
#include "util/args.h"
#include "util/hex.h"
#include "util/json.h"
#include "util/logging.h"

using namespace rnet;

namespace {

int Fail(const std::string& message) {
    std::fprintf(stderr, "error: %s\n", message.c_str());
    return 1;
}

Result<consensus::Network> NetworkFrom(const util::ArgParser& args) {
    return consensus::ParseNetwork(args.GetString("network", "main"));
}

int CmdGenesisEmit(const util::ArgParser& args) {
    auto net = NetworkFrom(args);
    if (!net) return Fail(net.error());

    const std::filesystem::path dir = args.GetString("out", "share/genesis");
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);

    const auto round_path = dir / (consensus::NetworkName(net.value()) + ".rnet");
    if (auto st = consensus::WriteGenesisFile(round_path, net.value()); !st) return Fail(st.error());
    std::printf("%-28s %s\n", round_path.c_str(), consensus::GenesisHash(net.value()).c_str());

    const auto policy_path = dir / (consensus::NetworkName(net.value()) + ".rnpol");
    if (auto st = consensus::WritePolicyFile(policy_path, net.value()); !st) return Fail(st.error());
    std::printf("%-28s %s\n", policy_path.c_str(), consensus::PolicyHash(net.value()).c_str());

    std::printf("\nPin both hashes as the trust anchors for '%s'.\n",
                consensus::NetworkName(net.value()).c_str());
    return 0;
}

// Re-derives the initial weights and checks them against the pinned anchor.
//
// Worth having as a command rather than only as a test: an operator joining the
// network can verify for themselves that the starting state their node will use
// is the one the network agreed on, without trusting the binary that told them so.
int CmdGenesisWeights(const util::ArgParser& args) {
    auto net = NetworkFrom(args);
    if (!net) return Fail(net.error());
    const auto& p = consensus::Params(net.value());

    std::printf("deriving %llu parameters for '%s' — this is real work, not a lookup\n",
                static_cast<unsigned long long>(p.round.model.ParameterCount()), p.name.c_str());
    auto derived = consensus::GenesisWeightsHash(p.round);
    if (!derived) return Fail(derived.error());

    const auto hex = util::ToHex(derived.value());
    std::printf("derived  %s\n", hex.c_str());
    if (p.genesis_weights_hash_hex.empty()) {
        std::printf("anchor   (unpinned)\n\nPin this value as genesis_weights_hash_hex for '%s'.\n",
                    p.name.c_str());
        return 0;
    }
    std::printf("anchor   %s\n", p.genesis_weights_hash_hex.c_str());
    if (hex != p.genesis_weights_hash_hex) {
        return Fail("derived weights do not match the pinned anchor — this build would start "
                    "training from a different state than the network agreed on");
    }
    std::printf("\nmatch: this node starts from the state the network agreed on.\n");
    return 0;
}

int CmdGenesisShow(const util::ArgParser& args) {
    auto net = NetworkFrom(args);
    if (!net) return Fail(net.error());
    const auto& p = consensus::Params(net.value());
    const auto& m = p.round.model;

    util::Json j = util::Json::Object();
    j.Set("network", p.name);
    j.Set("genesis_hash", consensus::GenesisHash(net.value()));
    j.Set("pinned_anchor", p.genesis_hash_hex.empty() ? "(unpinned)" : p.genesis_hash_hex);
    j.Set("default_port", static_cast<int64_t>(p.default_port));
    j.Set("round_id", p.round.round_id);
    j.Set("parameters", p.round.model.ParameterCount());

    util::Json model = util::Json::Object();
    model.Set("d_model", static_cast<int64_t>(m.d_model));
    model.Set("n_layers", static_cast<int64_t>(m.n_layers));
    model.Set("n_heads", static_cast<int64_t>(m.n_heads));
    model.Set("n_kv_heads", static_cast<int64_t>(m.n_kv_heads));
    model.Set("d_ff", static_cast<int64_t>(m.d_ff));
    model.Set("vocab_size", static_cast<int64_t>(m.vocab_size));
    model.Set("seq_len", static_cast<int64_t>(m.seq_len));
    model.Set("rope_theta", static_cast<int64_t>(m.rope_theta));
    model.Set("tie_embeddings", m.tie_embeddings != 0);
    model.Set("qk_norm", m.qk_norm != 0);
    j.Set("model", model);

    j.Set("genesis_weights_hash",
          p.genesis_weights_hash_hex.empty() ? "(unpinned)" : p.genesis_weights_hash_hex);
    j.Set("tokenizer_hash", util::ToHex(p.round.tokenizer_hash));
    j.Set("dataset_root", util::ToHex(p.round.dataset_root));
    j.Set("policy_hash", consensus::PolicyHash(net.value()));
    j.Set("pinned_policy_anchor", p.policy_hash_hex.empty() ? "(unpinned)" : p.policy_hash_hex);
    j.Set("inner_steps", static_cast<int64_t>(p.policy.inner_steps));
    j.Set("micro_batch", static_cast<int64_t>(p.policy.micro_batch));
    j.Set("chunk_tokens", static_cast<int64_t>(p.policy.dataset_chunk_tokens));
    j.Set("challenge_percent", static_cast<int64_t>(p.policy.challenge_percent));
    j.Set("challenge_deadline_steps", static_cast<int64_t>(p.policy.challenge_deadline_steps));
    j.Set("retained_checkpoints", static_cast<int64_t>(p.policy.retained_checkpoints));
    j.Set("slash_quorum", static_cast<int64_t>(p.policy.slash_quorum));
    j.Set("shadow_mode", p.policy.shadow_mode != 0);

    std::printf("%s\n", j.Dump(2).c_str());
    return 0;
}

int CmdGenesisCheck(const util::ArgParser& args) {
    const std::string file = args.GetString("file");
    const std::string hash = args.GetString("hash");
    if (file.empty() || hash.empty()) return Fail("genesis-check requires --file and --hash");

    auto round = consensus::LoadGenesisFile(file, hash);
    if (!round) return Fail(round.error());

    std::printf("OK  round_id=%llu d_model=%u n_layers=%u vocab=%u seq_len=%u params=%llu\n",
                static_cast<unsigned long long>(round.value().round_id), round.value().model.d_model,
                round.value().model.n_layers, round.value().model.vocab_size,
                round.value().model.seq_len,
                static_cast<unsigned long long>(round.value().model.ParameterCount()));
    return 0;
}

int CmdDatasetBuild(const util::ArgParser& args) {
    const std::string file = args.GetString("file");
    if (file.empty()) return Fail("dataset-build requires --file");
    const std::string out = args.GetString("out", "dataset");

    auto net = NetworkFrom(args);
    if (!net) return Fail(net.error());
    const auto& params = consensus::Params(net.value());

    uint32_t chunk = params.policy.dataset_chunk_tokens;
    if (args.Has("chunk-bytes")) {
        auto v = args.GetUInt("chunk-bytes");
        if (!v) return Fail(v.error());
        chunk = static_cast<uint32_t>(v.value());
    }

    crypto::Hash256 tokenizer = params.round.tokenizer_hash;
    if (args.Has("tokenizer-hash")) {
        auto h = util::FromHexFixed<32>(args.GetString("tokenizer-hash"));
        if (!h) return Fail(h.error());
        tokenizer = h.value();
    }



    RNET_LOG_INFO("indexing {} (target_chunk_bytes={})", file, chunk);
    auto manifest = dataset::BuildAndWriteManifest(file, out, chunk, tokenizer);
    if (!manifest) return Fail(manifest.error());

    std::printf("%s\n", dataset::ManifestToJson(manifest.value()).Dump(2).c_str());
    std::printf("\nwrote %s.rnds and %s.json\n", out.c_str(), out.c_str());
    return 0;
}

int CmdDatasetCheck(const util::ArgParser& args) {
    const std::string file = args.GetString("file");
    const std::string manifest_path = args.GetString("manifest");
    if (file.empty() || manifest_path.empty()) {
        return Fail("dataset-check requires --file and --manifest");
    }

    auto manifest = dataset::LoadManifestFile(manifest_path);
    if (!manifest) return Fail(manifest.error());
    if (auto st = dataset::VerifyCorpus(file, manifest.value()); !st) return Fail(st.error());

    std::printf("OK  %llu bytes, %u chunks, root %s\n",
                static_cast<unsigned long long>(manifest.value().n_bytes),
                manifest.value().n_chunks, util::ToHex(manifest.value().dataset_root).c_str());
    return 0;
}

int CmdScheduleShow(const util::ArgParser& args) {
    const std::string manifest_path = args.GetString("manifest");
    if (manifest_path.empty()) return Fail("schedule-show requires --manifest");

    auto manifest = dataset::LoadManifestFile(manifest_path);
    if (!manifest) return Fail(manifest.error());

    auto net = NetworkFrom(args);
    if (!net) return Fail(net.error());
    const auto& params = consensus::Params(net.value());

    const uint64_t worker = args.Has("worker") ? args.GetUInt("worker").value_or(0) : 0;
    const uint64_t step = args.Has("step") ? args.GetUInt("step").value_or(0) : 0;

    auto chunks = dataset::BatchChunks(manifest.value().dataset_root, params.round.round_id,
                                       worker, step, params.policy.micro_batch,
                                       manifest.value().n_chunks);
    if (!chunks) return Fail(chunks.error());

    std::printf("worker=%llu step=%llu seq_len=%u micro_batch=%u  (%u chunks in corpus)\n",
                static_cast<unsigned long long>(worker), static_cast<unsigned long long>(step),
                params.round.model.seq_len, params.policy.micro_batch,
                manifest.value().n_chunks);
    // The chunk, not the token offset: where inside it the window starts depends
    // on how many tokens that chunk holds, and nothing here can tokenize.
    for (size_t i = 0; i < chunks.value().size(); ++i) {
        std::printf("  window[%zu] chunk=%llu\n", i,
                    static_cast<unsigned long long>(chunks.value()[i]));
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    util::ArgParser args("rnet-tool", "ResonanceNet offline tool: genesis artifacts and corpora.");
    args.AddPositional("command",
                       "genesis-emit | genesis-show | genesis-weights | genesis-check | dataset-build | "
                       "dataset-check | schedule-show");
    args.AddOption("network", "main | test | regtest", true, "main");
    args.AddOption("out", "output path or prefix");
    args.AddOption("file", "input file");
    args.AddOption("hash", "expected hash (hex)");
    args.AddOption("manifest", "dataset manifest (.rnds)");
    args.AddOption("tokenizer-hash", "tokenizer artifact hash (hex)");
    args.AddOption("chunk-bytes", "target bytes per chunk; a chunk ends at the first document separator past it");
    args.AddOption("worker", "worker id");
    args.AddOption("step", "training step");
    args.AddOption("log", "error|warn|info|debug|trace", true, "info");

    if (auto st = args.Parse(argc, argv); !st) return Fail(st.error());

    util::LogLevel level = util::LogLevel::Info;
    if (!util::ParseLogLevel(args.GetString("log", "info"), level)) return Fail("bad --log level");
    util::SetLogLevel(level);

    if (args.positionals().empty()) {
        std::fprintf(stderr, "%s\n", args.Usage().c_str());
        return 1;
    }

    const std::string& cmd = args.positionals()[0];
    if (cmd == "genesis-emit") return CmdGenesisEmit(args);
    if (cmd == "genesis-show") return CmdGenesisShow(args);
    if (cmd == "genesis-weights") return CmdGenesisWeights(args);
    if (cmd == "genesis-check") return CmdGenesisCheck(args);
    if (cmd == "dataset-build") return CmdDatasetBuild(args);
    if (cmd == "dataset-check") return CmdDatasetCheck(args);
    if (cmd == "schedule-show") return CmdScheduleShow(args);

    std::fprintf(stderr, "unknown command '%s'\n\n%s\n", cmd.c_str(), args.Usage().c_str());
    return 1;
}

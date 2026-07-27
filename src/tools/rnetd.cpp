// rnetd — the node daemon.
//
// What it does before it will talk to anyone:
//
//   1. Loads the consensus artifacts from disk and CHECKS THEM AGAINST THE HASHES
//      COMPILED INTO THIS BINARY. A node whose genesis file says something other
//      than what its code was built for is not a node with a configuration
//      problem — it is a node that would train a different model and call the
//      result consensus. It refuses to start.
//
//   2. Restores its address database and its bucket key, so a restart is not a
//      fresh eclipse opportunity.
//
//   3. Bootstraps from seeds only when it has nothing else. Seeds are a last
//      resort, not a routine dependency: whatever a node trusts to find its first
//      peer is the best place to attack it.
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>

#include "consensus/genesis.h"
#include "consensus/params.h"
#include "crypto/random.h"
#include "dataset/manifest.h"
#include "net/bootstrap.h"
#include "net/node.h"
#include "protocol/service.h"
#include "protocol/workerservice.h"
#include "util/args.h"
#include "util/fs.h"
#include "util/hex.h"
#include "util/logging.h"
#include "util/time.h"

using namespace rnet;

namespace {

volatile std::sig_atomic_t g_shutdown = 0;

void OnSignal(int) { g_shutdown = 1; }

std::filesystem::path DefaultDataDir(const std::string& network) {
    const char* home = std::getenv("HOME");
    const std::filesystem::path base =
        home != nullptr ? std::filesystem::path(home) / ".rnet" : std::filesystem::path(".rnet");
    return network == "main" ? base : base / network;
}

// The check that makes the daemon trustworthy. The artifacts on disk are data;
// the hashes in `params` are what this binary was built to agree with. If they
// disagree, everything downstream — batch derivation, replay, verdicts — would be
// computed against rules nobody else is following, and the node would look
// healthy the whole time.
Status VerifyConsensusArtifacts(const std::filesystem::path& datadir,
                                const consensus::NetworkParams& params) {
    const auto genesis_path = datadir / (params.name + ".rnet");
    const auto policy_path = datadir / (params.name + ".rnpol");

    auto round = consensus::LoadGenesisFileForNetwork(genesis_path, params.network);
    if (!round) {
        return Err(std::format(
            "{}: {}\nEmit the artifacts with `rnet-tool genesis-emit --network {} --out {}` and "
            "check the hash against the release before trusting them.",
            genesis_path.string(), round.error(), params.name, datadir.string()));
    }

    auto policy = consensus::LoadPolicyFile(policy_path, consensus::PolicyHash(params.network));
    if (!policy) return Err(std::format("{}: {}", policy_path.string(), policy.error()));

    RNET_LOG_INFO("consensus: round {} on {}, model {}L/{}d/{} vocab, {} inner steps per outer",
                  round.value().round_id, params.name, round.value().model.n_layers,
                  round.value().model.d_model, round.value().model.vocab_size,
                  policy.value().inner_steps);
    return Status::Ok();
}

// The bucket key decides where addresses land, so it must be unpredictable to an
// attacker AND stable across restarts. Regenerating it every start would reshuffle
// the table on every restart and undo the protection it exists to provide.
Result<crypto::Hash256> LoadOrCreateAddrKey(const std::filesystem::path& path) {
    if (auto existing = util::ReadFile(path); existing && existing.value().size() == 32) {
        crypto::Hash256 key{};
        std::copy(existing.value().begin(), existing.value().end(), key.begin());
        return key;
    }
    const crypto::Hash256 key = crypto::RandomHashOrDie();
    if (auto st = util::WriteFileAtomic(path, key); !st) {
        return Err("cannot persist the address key: " + st.error());
    }
    return key;
}

}  // namespace

int main(int argc, char** argv) {
    util::ArgParser args("rnetd", "ResonanceNet node daemon");
    args.AddOption("network", "which consensus rules to run: main, test, regtest", true, "main");
    args.AddOption("datadir", "state directory (default: ~/.rnet)");
    args.AddOption("bind", "listen address", true, "0.0.0.0");
    args.AddOption("port", "listen port (default: the network's own)");
    args.AddRepeated("connect", "connect only to this peer; may be given more than once");
    args.AddOption("max-inbound", "inbound connection slots", true, "64");
    args.AddOption("max-outbound", "outbound connection slots", true, "8");
    args.AddOption("no-listen", "do not accept inbound connections", false);
    args.AddOption("no-seeds", "do not query DNS seeds", false);
    args.AddOption("worker-id", "this node's worker identity; required to participate");
    args.AddOption("verify", "answer verification challenges assigned to this node", false);
    args.AddOption("corpus", "tokenized corpus file to serve (needs --manifest)");
    args.AddOption("manifest", "dataset manifest (.rnds) describing that corpus");
    args.AddOption("log", "error | warn | info | debug | trace", true, "info");
    args.AddOption("help", "show this text", false);

    // An unknown option is an error, not something to ignore: a daemon started
    // with a misspelled flag must not run with a setting its operator believes is
    // in effect.
    if (auto st = args.Parse(argc, argv); !st) {
        std::fprintf(stderr, "error: %s\n", st.error().c_str());
        return 1;
    }
    if (args.Has("help")) {
        std::printf("%s\n", args.Usage().c_str());
        return 0;
    }

    util::LogLevel level = util::LogLevel::Info;
    if (!util::ParseLogLevel(args.GetString("log", "info"), level)) {
        std::fprintf(stderr, "error: unknown log level '%s'\n", args.GetString("log").c_str());
        return 1;
    }
    util::SetLogLevel(level);

    auto network = consensus::ParseNetwork(args.GetString("network", "main"));
    if (!network) {
        std::fprintf(stderr, "error: %s\n", network.error().c_str());
        return 1;
    }
    const auto& params = consensus::Params(network.value());

    const std::string datadir_option = args.GetString("datadir");
    const std::filesystem::path datadir =
        datadir_option.empty() ? DefaultDataDir(params.name) : std::filesystem::path(datadir_option);
    std::error_code ec;
    std::filesystem::create_directories(datadir, ec);
    if (ec) {
        std::fprintf(stderr, "error: cannot create datadir %s: %s\n", datadir.c_str(),
                     ec.message().c_str());
        return 1;
    }

    if (auto st = VerifyConsensusArtifacts(datadir, params); !st) {
        std::fprintf(stderr, "error: %s\n", st.error().c_str());
        return 1;
    }

    net::NodeConfig config;
    config.network = network.value();

    auto max_inbound = args.GetUInt("max-inbound");
    auto max_outbound = args.GetUInt("max-outbound");
    if (!max_inbound || !max_outbound) {
        std::fprintf(stderr, "error: connection limits must be numbers\n");
        return 1;
    }
    config.max_inbound = static_cast<uint32_t>(max_inbound.value());
    config.max_outbound = static_cast<uint32_t>(max_outbound.value());

    if (!args.Has("no-listen")) {
        uint16_t port = params.default_port;
        if (args.Has("port")) {
            auto given = args.GetUInt("port");
            if (!given || given.value() == 0 || given.value() > 65535) {
                std::fprintf(stderr, "error: --port must be 1..65535\n");
                return 1;
            }
            port = static_cast<uint16_t>(given.value());
        }
        auto bind = net::NetAddress::FromString(
            std::format("{}:{}", args.GetString("bind", "0.0.0.0"), port), port);
        if (!bind) {
            std::fprintf(stderr, "error: bad bind address: %s\n", bind.error().c_str());
            return 1;
        }
        config.bind_address = bind.value();
    }
    if (auto st = config.Validate(); !st) {
        std::fprintf(stderr, "error: %s\n", st.error().c_str());
        return 1;
    }

    auto addrman_key = LoadOrCreateAddrKey(datadir / "addrkey");
    if (!addrman_key) {
        std::fprintf(stderr, "error: %s\n", addrman_key.error().c_str());
        return 1;
    }

    net::Node node(config, addrman_key.value(), crypto::RandomU64OrDie());
    if (auto st = node.Start(); !st) {
        std::fprintf(stderr, "error: cannot start: %s\n", st.error().c_str());
        return 1;
    }
    RNET_LOG_INFO("rnetd on {} — genesis {}", params.name, params.genesis_hash_hex);

    // Consensus participation is opt-in. A node without a worker identity is a
    // relay: it moves and verifies objects for the network without claiming to
    // have produced any, which is a useful thing to be and must not require
    // pretending to be a worker.
    std::unique_ptr<protocol::Participant> participant;
    std::unique_ptr<protocol::Service> service;
    ipc::Server ipc_server;
    // On disk, under the datadir: a checkpoint is two gigabytes for the launch
    // model and several are retained, which is not something to hold in memory
    // beside a training process.
    net::WeightsStore weights_store(datadir / "weights");
    node.SetWeightsStore(&weights_store);

    // A seed node serves the corpus; an ordinary node does not have to. Opening it
    // rebuilds the tree and refuses a file that is not the corpus it claims to be,
    // so a misconfigured seed fails at startup rather than by serving tokens that
    // fail every proof.
    std::unique_ptr<net::CorpusSource> corpus;
    if (args.Has("corpus") != args.Has("manifest")) {
        std::fprintf(stderr, "error: --corpus and --manifest go together\n");
        return 1;
    }
    if (args.Has("corpus")) {
        auto manifest = dataset::LoadManifestFile(args.GetString("manifest"));
        if (!manifest) {
            std::fprintf(stderr, "error: %s\n", manifest.error().c_str());
            return 1;
        }
        // The corpus must be the one this round pinned, or workers would derive
        // correct offsets into the wrong tokens.
        if (manifest.value().dataset_root != params.round.dataset_root) {
            std::fprintf(stderr,
                         "error: this corpus has root %s, but '%s' pins %s — serving it would "
                         "feed workers tokens the network did not agree on\n",
                         util::ToHex(manifest.value().dataset_root).c_str(), params.name.c_str(),
                         util::ToHex(params.round.dataset_root).c_str());
            return 1;
        }
        std::printf("indexing %s — this reads the whole corpus\n",
                    args.GetString("corpus").c_str());
        auto opened = net::CorpusSource::Open(args.GetString("corpus"), manifest.value());
        if (!opened) {
            std::fprintf(stderr, "error: %s\n", opened.error().c_str());
            return 1;
        }
        corpus = std::make_unique<net::CorpusSource>(std::move(opened.value()));
        node.SetCorpus(corpus.get());
        RNET_LOG_INFO("serving the corpus: {} bytes of text in {} chunks, root {}",
                      corpus->n_bytes(), corpus->n_chunks(),
                      util::ToHex(corpus->root()).substr(0, 16));
    }
    std::unique_ptr<protocol::WorkerService> worker_service;
    if (args.Has("worker-id")) {
        auto worker_id = args.GetUInt("worker-id");
        if (!worker_id || worker_id.value() == 0) {
            std::fprintf(stderr, "error: --worker-id must be a non-zero number\n");
            return 1;
        }
        protocol::ParticipantConfig participant_config;
        participant_config.network = network.value();
        participant_config.worker_id = worker_id.value();
        participant_config.verifier = args.Has("verify");
        if (auto st = participant_config.Validate(); !st) {
            std::fprintf(stderr, "error: %s\n", st.error().c_str());
            return 1;
        }

        // The genesis checkpoint is derived from the artifact this node already
        // verified, not read from a second file: two sources for one fact is two
        // chances to disagree.
        auto genesis_checkpoint = consensus::GenesisCheckpointFor(network.value());
        if (!genesis_checkpoint) {
            std::fprintf(stderr, "error: %s\n", genesis_checkpoint.error().c_str());
            return 1;
        }
        participant = std::make_unique<protocol::Participant>(participant_config,
                                                              genesis_checkpoint.value());
        service = std::make_unique<protocol::Service>(node, *participant);
        service->Attach();

        // The local channel. Opened only for a participating node: a relay has no
        // worker identity to lend, so a socket would grant a privilege that does
        // not exist.
        if (auto st = ipc_server.Listen(datadir / "rnet.sock"); !st) {
            std::fprintf(stderr, "error: %s\n", st.error().c_str());
            return 1;
        }
        worker_service =
            std::make_unique<protocol::WorkerService>(ipc_server, node, *participant);
        RNET_LOG_INFO("participating as worker {} ({})", worker_id.value(),
                      participant_config.verifier ? "verifier" : "contributor only");
    } else {
        RNET_LOG_INFO("relay only — no worker identity given, so nothing is claimed or submitted");
    }

    const auto peers_path = (datadir / "peers.dat").string();
    if (auto st = node.addresses().Load(peers_path); !st) {
        // A missing or corrupt peers file costs the node its memory of the
        // network, not its ability to join one.
        RNET_LOG_INFO("no usable peers file ({}); starting cold", st.error());
    }

    const int64_t start_ms = util::NowMillis();
    const auto& explicit_peers = args.GetRepeated("connect");

    for (const auto& text : explicit_peers) {
        auto address = net::NetAddress::FromString(text, params.default_port);
        if (!address) {
            std::fprintf(stderr, "error: bad --connect '%s': %s\n", text.c_str(),
                         address.error().c_str());
            return 1;
        }
        if (auto st = node.ConnectTo(address.value(), start_ms); !st) {
            RNET_LOG_ERROR("cannot connect to {}: {}", text, st.error());
        }
    }

    // Seeds are consulted only when there is nothing else, and never when the
    // operator named explicit peers — being told who to talk to is a stronger
    // instruction than a hostname lookup.
    if (explicit_peers.empty() && !args.Has("no-seeds") && node.addresses().Size() == 0) {
        auto result = net::BootstrapFromSeeds(params, node.addresses(), start_ms / 1000);
        RNET_LOG_INFO("bootstrap: {}/{} seeds answered, {} addresses", result.seeds_answered,
                      result.seeds_queried, result.addresses_added);
        if (result.addresses_added == 0) {
            const auto fixed =
                net::BootstrapFromFixedSeeds(params, node.addresses(), start_ms / 1000);
            if (fixed.addresses_added > 0) {
                RNET_LOG_INFO("bootstrap: {} addresses from fixed seeds", fixed.addresses_added);
            }
        }
        if (node.addresses().Size() == 0) {
            // Stated plainly rather than hidden behind a hopeful retry loop: a node
            // that cannot find anyone is not participating, whatever its process
            // state looks like.
            RNET_LOG_ERROR(
                "no peers and no reachable seeds on {} — the node is running but isolated. Give it "
                "--connect <host:port>, or wait for seed infrastructure to come up.",
                params.name);
        }
    }

    std::signal(SIGINT, OnSignal);
    std::signal(SIGTERM, OnSignal);
    std::signal(SIGPIPE, SIG_IGN);   // a peer closing mid-write must not kill the node

    int64_t last_save = start_ms;
    int64_t last_status = start_ms;
    int64_t last_dial = 0;

    while (g_shutdown == 0) {
        const int64_t now = util::NowMillis();
        if (auto st = node.Tick(now, 100); !st) {
            RNET_LOG_ERROR("tick: {}", st.error());
            break;
        }
        if (service) {
            if (auto st = service->Tick(now); !st) RNET_LOG_ERROR("consensus tick: {}", st.error());
        }
        if (worker_service) {
            if (auto st = worker_service->Poll(now); !st) RNET_LOG_ERROR("ipc: {}", st.error());
        }

        // Reaching out is the node's own initiative. A node that only accepts
        // inbound connections lets other people decide who it knows, which is
        // exactly the position an eclipse attack wants it in. Rate-limited so a
        // large address database is not dialled through in one burst.
        if (explicit_peers.empty() && node.OutboundCount() < config.max_outbound &&
            now - last_dial > 2000) {
            last_dial = now;
            if (auto candidate = node.addresses().Select(crypto::RandomU64OrDie())) {
                (void)node.ConnectTo(candidate->address, now);
            }
        }

        if (now - last_save > 300'000) {
            last_save = now;
            if (auto st = node.addresses().Save(peers_path); !st) {
                RNET_LOG_ERROR("cannot save peers: {}", st.error());
            }
        }
        if (now - last_status > 60'000) {
            last_status = now;
            RNET_LOG_INFO(
                "peers {} ({} ready, {} in, {} out) — addresses {} ({} tried) — objects {}",
                node.PeerCount(), node.ReadyCount(), node.InboundCount(), node.OutboundCount(),
                node.addresses().Size(), node.addresses().TriedCount(), node.objects().Size());
            if (participant) {
                RNET_LOG_INFO("consensus: step {}, {} contributions this round, {} objects "
                              "accepted / {} refused",
                              participant->outer_step(), participant->submission_count(),
                              service->objects_accepted(), service->objects_rejected());
                RNET_LOG_INFO(
                    "local workers: {} connected, {} contributions accepted, {} refused, "
                    "{} corpus fetches deferred",
                    ipc_server.client_count(), worker_service->contributions_accepted(),
                    worker_service->requests_refused(), worker_service->requests_deferred());
                auto held = weights_store.List();
                auto bytes = weights_store.TotalBytes();
                RNET_LOG_INFO("weights: {} checkpoint(s) held, {} MB",
                              held.ok() ? held.value().size() : 0,
                              (bytes.ok() ? bytes.value() : 0) / (1024 * 1024));
            }
        }
    }

    RNET_LOG_INFO("shutting down");
    ipc_server.Stop();
    if (auto st = node.addresses().Save(peers_path); !st) {
        RNET_LOG_ERROR("cannot save peers: {}", st.error());
    }
    node.Stop();
    return 0;
}

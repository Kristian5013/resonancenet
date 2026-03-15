// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

// Project headers.
#include "consensus/params.h"
#include "core/config.h"
#include "core/error.h"
#include "core/hex.h"
#include "core/logging.h"
#include "crypto/ed25519.h"
#include "crypto/hash.h"
#include "crypto/keccak.h"
#include "gpu/backend.h"
#include "gpu/context.h"
#include "miner/difficulty.h"
#include "primitives/address.h"
#include "training/checkpoint_io.h"
#include "training/data_loader.h"
#include "training/evaluator.h"
#include "training/model_config.h"
#include "training/training_engine.h"

// Standard library.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

// Platform-specific socket headers.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using socket_t = SOCKET;
static constexpr socket_t INVALID_SOCK = INVALID_SOCKET;
static void close_socket(socket_t s) { closesocket(s); }
#else
#include <csignal>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using socket_t = int;
static constexpr socket_t INVALID_SOCK = -1;
static void close_socket(socket_t s) { close(s); }
#endif

// ===========================================================================
//  Minimal RPC client (same pattern as rnet-cli)
// ===========================================================================

// ---------------------------------------------------------------------------
// connect_to_host
// ---------------------------------------------------------------------------
static socket_t connect_to_host(const std::string& host, uint16_t port)
{
    struct addrinfo hints{}, *result = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    std::string port_str = std::to_string(port);
    if (getaddrinfo(host.c_str(), port_str.c_str(), &hints, &result) != 0)
        return INVALID_SOCK;
    socket_t sock = INVALID_SOCK;
    for (auto* rp = result; rp; rp = rp->ai_next) {
        sock = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sock == INVALID_SOCK) continue;
        if (connect(sock, rp->ai_addr, static_cast<int>(rp->ai_addrlen)) == 0) break;
        close_socket(sock);
        sock = INVALID_SOCK;
    }
    freeaddrinfo(result);
    return sock;
}

// ---------------------------------------------------------------------------
// send_all
// ---------------------------------------------------------------------------
static bool send_all(socket_t sock, const std::string& data)
{
    size_t total = 0;
    while (total < data.size()) {
        int sent = send(sock, data.data() + total,
                        static_cast<int>(data.size() - total), 0);
        if (sent <= 0) return false;
        total += static_cast<size_t>(sent);
    }
    return true;
}

// ---------------------------------------------------------------------------
// recv_all
// ---------------------------------------------------------------------------
static std::string recv_all(socket_t sock)
{
    std::string result;
    std::array<char, 8192> buf{};
    for (;;) {
        int n = recv(sock, buf.data(), static_cast<int>(buf.size()), 0);
        if (n <= 0) break;
        result.append(buf.data(), static_cast<size_t>(n));
    }
    return result;
}

// ---------------------------------------------------------------------------
// base64_encode
// ---------------------------------------------------------------------------
static std::string base64_encode(const std::string& input)
{
    static constexpr char t[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string o;
    o.reserve(((input.size() + 2) / 3) * 4);
    for (size_t i = 0; i < input.size(); i += 3) {
        uint32_t n = static_cast<uint8_t>(input[i]) << 16;
        if (i + 1 < input.size()) n |= static_cast<uint8_t>(input[i + 1]) << 8;
        if (i + 2 < input.size()) n |= static_cast<uint8_t>(input[i + 2]);
        o.push_back(t[(n >> 18) & 0x3F]);
        o.push_back(t[(n >> 12) & 0x3F]);
        o.push_back((i + 1 < input.size()) ? t[(n >> 6) & 0x3F] : '=');
        o.push_back((i + 2 < input.size()) ? t[n & 0x3F] : '=');
    }
    return o;
}

struct RpcResponse {
    int http_status = 0;
    std::string body;
};

// ---------------------------------------------------------------------------
// rpc_call
// ---------------------------------------------------------------------------
// Sends a JSON-RPC 1.0 request and returns the HTTP status + body.
// ---------------------------------------------------------------------------
static RpcResponse rpc_call(const std::string& host, uint16_t port,
                             const std::string& auth,
                             const std::string& method,
                             const std::string& params_json = "[]")
{
    RpcResponse resp;
    socket_t sock = connect_to_host(host, port);
    if (sock == INVALID_SOCK) {
        resp.http_status = -1;
        resp.body = "Connection refused";
        return resp;
    }

    std::string json_body = "{\"jsonrpc\":\"1.0\",\"id\":1,\"method\":\"" + method
                            + "\",\"params\":" + params_json + "}";

    std::ostringstream req;
    req << "POST / HTTP/1.0\r\n"
        << "Host: " << host << "\r\n"
        << "Content-Type: application/json\r\n"
        << "Content-Length: " << json_body.size() << "\r\n";
    if (!auth.empty()) {
        req << "Authorization: Basic " << auth << "\r\n";
    }
    req << "\r\n" << json_body;

    send_all(sock, req.str());
#ifdef _WIN32
    shutdown(sock, SD_SEND);
#else
    shutdown(sock, SHUT_WR);
#endif
    std::string raw = recv_all(sock);
    close_socket(sock);

    auto header_end = raw.find("\r\n\r\n");
    if (header_end == std::string::npos) {
        resp.http_status = -2;
        resp.body = raw;
        return resp;
    }
    auto first_line_end = raw.find("\r\n");
    auto space = raw.find(' ');
    if (space != std::string::npos && space < first_line_end)
        resp.http_status = std::atoi(raw.c_str() + space + 1);
    resp.body = raw.substr(header_end + 4);
    return resp;
}

// ---------------------------------------------------------------------------
// extract_json_field
// ---------------------------------------------------------------------------
// Minimal JSON field extractor -- handles strings, objects, arrays, null,
// numbers, and booleans.  Borrowed from rnet-cli; avoids a JSON library dep.
// ---------------------------------------------------------------------------
static std::string extract_json_field(const std::string& json,
                                       const std::string& field)
{
    std::string key = "\"" + field + "\"";
    auto pos = json.find(key);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + key.size());
    if (pos == std::string::npos) return "";
    pos++;
    while (pos < json.size() && json[pos] == ' ') pos++;
    if (pos >= json.size()) return "";

    if (json[pos] == '"') {
        // 1. String value.
        pos++;
        std::string result;
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                pos++;
                if (json[pos] == 'n') result += '\n';
                else if (json[pos] == 't') result += '\t';
                else result += json[pos];
            } else {
                result += json[pos];
            }
            pos++;
        }
        return result;
    } else if (json[pos] == '{' || json[pos] == '[') {
        // 2. Object or array -- find matching brace.
        int depth = 1;
        char open = json[pos], close_c = (open == '{') ? '}' : ']';
        size_t start = pos;
        pos++;
        while (pos < json.size() && depth > 0) {
            if (json[pos] == open) depth++;
            else if (json[pos] == close_c) depth--;
            pos++;
        }
        return json.substr(start, pos - start);
    } else if (json.substr(pos, 4) == "null") {
        // 3. Null literal.
        return "null";
    } else {
        // 4. Number or boolean.
        size_t start = pos;
        while (pos < json.size() && json[pos] != ',' && json[pos] != '}')
            pos++;
        std::string val = json.substr(start, pos - start);
        while (!val.empty() &&
               (val.back() == ' ' || val.back() == '\r' || val.back() == '\n'))
            val.pop_back();
        return val;
    }
}

// ===========================================================================
//  Mining configuration
// ===========================================================================

struct MinerCliConfig {
    std::string host = "127.0.0.1";
    uint16_t port = 9554;
    std::string rpcuser;
    std::string rpcpassword;
    std::string miner_address;
    std::string train_data = "data/train.bin";
    std::string val_data = "data/val.bin";
    std::string checkpoint_dir = "checkpoints";
    int steps_per_attempt = 0;  // 0 = auto (use suggest_step_count)
    int num_workers = 1;
    bool benchmark = false;
};

static std::atomic<bool> g_running{true};

#ifndef _WIN32
static void sigint_handler(int) { g_running.store(false); }
#endif

// ---------------------------------------------------------------------------
// print_usage
// ---------------------------------------------------------------------------
static void print_usage()
{
    fprintf(stderr,
        "Usage: rnet-miner [options]\n"
        "\n"
        "Options:\n"
        "  -rpcconnect=<host>         RPC host (default: 127.0.0.1)\n"
        "  -rpcport=<port>            RPC port (default: 9554)\n"
        "  -rpcuser=<user>            RPC username\n"
        "  -rpcpassword=<pw>          RPC password\n"
        "  -address=<rn1...>          Miner's reward address (bech32)\n"
        "  -traindata=<path>          Path to training dataset\n"
        "  -valdata=<path>            Path to validation dataset\n"
        "  -checkpointdir=<path>      Directory for checkpoints (default: ./checkpoints)\n"
        "  -steps=<n>                 Training steps per attempt (0 = auto, default: auto)\n"
        "  -workers=<n>               Number of parallel workers (default: 1)\n"
        "  -benchmark                 Run benchmark mode (no RPC, just measure speed)\n"
        "  -help                      Show this help\n"
    );
}

// ---------------------------------------------------------------------------
// parse_args
// ---------------------------------------------------------------------------
static MinerCliConfig parse_args(int argc, char* argv[])
{
    MinerCliConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        // Strip leading double-dash to single-dash (accept both --flag and -flag).
        if (arg.size() > 2 && arg[0] == '-' && arg[1] == '-') {
            arg = arg.substr(1);
        }
        if (arg.find("-rpcconnect=") == 0) cfg.host = arg.substr(12);
        else if (arg.find("-rpcport=") == 0) cfg.port = static_cast<uint16_t>(std::atoi(arg.c_str() + 9));
        else if (arg.find("-rpcuser=") == 0) cfg.rpcuser = arg.substr(9);
        else if (arg.find("-rpcpassword=") == 0) cfg.rpcpassword = arg.substr(13);
        else if (arg.find("-address=") == 0) cfg.miner_address = arg.substr(9);
        else if (arg.find("-traindata=") == 0) cfg.train_data = arg.substr(11);
        else if (arg.find("-valdata=") == 0) cfg.val_data = arg.substr(9);
        else if (arg.find("-checkpointdir=") == 0) cfg.checkpoint_dir = arg.substr(15);
        else if (arg.find("-steps=") == 0) cfg.steps_per_attempt = std::atoi(arg.c_str() + 7);
        else if (arg.find("-workers=") == 0) cfg.num_workers = std::atoi(arg.c_str() + 9);
        else if (arg == "-benchmark") cfg.benchmark = true;
        else if (arg == "-help") { print_usage(); std::exit(0); }
    }
    return cfg;
}

// ===========================================================================
//  Mining loop
// ===========================================================================

// ---------------------------------------------------------------------------
// run_mining_loop
// ---------------------------------------------------------------------------
// Proof-of-Training mining loop.  Repeatedly fetches a block template from
// rnetd, loads the parent checkpoint, trains the MinGRU model on the GPU,
// evaluates validation loss, and submits a new block when the loss improves
// by at least the difficulty_delta threshold.
// ---------------------------------------------------------------------------
static int run_mining_loop(const MinerCliConfig& cfg)
{
    std::string auth;
    if (!cfg.rpcuser.empty()) {
        auth = base64_encode(cfg.rpcuser + ":" + cfg.rpcpassword);
    } else {
        // Auto-detect cookie file from default data directory.
        std::string cookie_path;
#ifdef _WIN32
        const char* appdata = std::getenv("APPDATA");
        if (appdata) {
            cookie_path = std::string(appdata) + "\\ResonanceNet\\.cookie";
        }
#else
        const char* home = std::getenv("HOME");
        if (home) {
            cookie_path = std::string(home) + "/.resonancenet/.cookie";
        }
#endif
        if (!cookie_path.empty()) {
            FILE* f = fopen(cookie_path.c_str(), "rb");
            if (f) {
                std::array<char, 512> buf{};
                size_t n = fread(buf.data(), 1, buf.size() - 1, f);
                fclose(f);
                while (n > 0 && (buf[n-1] == '\n' || buf[n-1] == '\r')) --n;
                if (n > 0) {
                    auth = base64_encode(std::string(buf.data(), n));
                }
            }
        }
    }

    // -- Pre-flight checks --------------------------------------------------

    if (cfg.miner_address.empty()) {
        fprintf(stderr, "Error: -address=<rn1...> is required\n");
        fprintf(stderr, "Use your wallet address to receive mining rewards.\n");
        fprintf(stderr, "Get one with: rnet-wallet-tool create && rnet-cli getnewaddress\n");
        return 1;
    }

    // 1. Validate the miner address.
    if (!rnet::primitives::is_valid_address(cfg.miner_address)) {
        fprintf(stderr, "Error: invalid address '%s'\n", cfg.miner_address.c_str());
        fprintf(stderr, "Expected bech32 address starting with 'rn1'\n");
        return 1;
    }

    // 2. Verify dataset files exist.
    if (!std::filesystem::exists(cfg.train_data)) {
        fprintf(stderr, "Error: training dataset not found: %s\n", cfg.train_data.c_str());
        fprintf(stderr, "Use -traindata=<path> to specify the training data file.\n");
        fprintf(stderr, "\nDataset preparation:\n");
        fprintf(stderr, "  1. Download a text corpus (e.g. OpenWebText, Wikipedia)\n");
        fprintf(stderr, "  2. Tokenize with GPT-2 tokenizer (vocab_size=50257)\n");
        fprintf(stderr, "  3. Save as binary uint16 tokens: data/train.bin, data/val.bin\n");
        return 1;
    }
    if (!std::filesystem::exists(cfg.val_data)) {
        fprintf(stderr, "Error: validation dataset not found: %s\n", cfg.val_data.c_str());
        fprintf(stderr, "Use -valdata=<path> to specify the validation data file.\n");
        return 1;
    }

    // 3. Create checkpoint directory if needed.
    if (!std::filesystem::exists(cfg.checkpoint_dir)) {
        std::filesystem::create_directories(cfg.checkpoint_dir);
    }

    // 4. Detect and create GPU backend.
    auto devices = rnet::gpu::GpuContext::enumerate_devices();
    if (devices.empty()) {
        fprintf(stderr, "Error: no GPU device found (need CUDA, Vulkan, or Metal)\n");
        fprintf(stderr, "Falling back to CPU backend (very slow).\n");
    }
    auto backend = rnet::gpu::GpuBackend::create_best();
    if (!backend) {
        fprintf(stderr, "Error: failed to create GPU backend\n");
        return 1;
    }
    printf("GPU: %s (%.0f MB free / %.0f MB total)\n",
           backend->device_name().c_str(),
           static_cast<double>(backend->free_memory()) / (1024.0 * 1024.0),
           static_cast<double>(backend->total_memory()) / (1024.0 * 1024.0));

    // 5. Load training and validation datasets.
    rnet::training::DataLoader train_loader;
    auto train_rc = train_loader.load_dataset(cfg.train_data);
    if (train_rc.is_err()) {
        fprintf(stderr, "Error loading training data: %s\n",
                train_rc.error().c_str());
        return 1;
    }
    printf("Training data: %zu tokens\n", train_loader.total_tokens());

    rnet::training::DataLoader val_loader;
    auto val_rc = val_loader.load_dataset(cfg.val_data);
    if (val_rc.is_err()) {
        fprintf(stderr, "Error loading validation data: %s\n",
                val_rc.error().c_str());
        return 1;
    }
    printf("Validation data: %zu tokens\n", val_loader.total_tokens());

    // 5b. Compute dataset hash (keccak256d of train + val files).
    std::string dataset_hash_hex;
    {
        rnet::crypto::HashWriter hasher{};
        auto append_file = [&](const std::string& path) {
            std::ifstream f(path, std::ios::binary);
            char buf[65536];
            while (f.read(buf, sizeof(buf)) || f.gcount() > 0) {
                hasher.write(reinterpret_cast<const uint8_t*>(buf),
                             static_cast<size_t>(f.gcount()));
            }
        };
        append_file(cfg.train_data);
        append_file(cfg.val_data);
        dataset_hash_hex = hasher.get_hash256().to_hex();
        printf("Dataset hash: %s\n", dataset_hash_hex.c_str());
    }

    // 6. Load consensus parameters (mainnet defaults).
    auto consensus = rnet::consensus::ConsensusParams::mainnet();

    // -- Banner -------------------------------------------------------------

    printf("\nResonanceNet Miner v0.1\n");
    printf("Connecting to %s:%u\n", cfg.host.c_str(), cfg.port);
    printf("Reward address: %s\n", cfg.miner_address.c_str());
    printf("Train data: %s\n", cfg.train_data.c_str());
    printf("Val data: %s\n", cfg.val_data.c_str());
    printf("Checkpoint dir: %s\n", cfg.checkpoint_dir.c_str());
    printf("Workers: %d\n", cfg.num_workers);
    printf("\n");

    uint64_t total_attempts = 0;
    uint64_t blocks_found = 0;
    auto start_time = std::chrono::steady_clock::now();

    // -- Main mining loop ---------------------------------------------------

    while (g_running.load()) {

        // 1. RPC: getblocktemplate -- obtain parent block info.
        auto tmpl_resp = rpc_call(cfg.host, cfg.port, auth,
                                   "getblocktemplate",
                                   "[{\"address\":\"" + cfg.miner_address + "\"}]");
        if (tmpl_resp.http_status < 0) {
            fprintf(stderr, "Error: cannot connect to rnetd: %s\n",
                    tmpl_resp.body.c_str());
            fprintf(stderr, "Retrying in 5 seconds...\n");
            std::this_thread::sleep_for(std::chrono::seconds(5));
            continue;
        }
        if (tmpl_resp.http_status != 200) {
            fprintf(stderr, "RPC error (HTTP %d): %s\n",
                    tmpl_resp.http_status, tmpl_resp.body.c_str());
            std::this_thread::sleep_for(std::chrono::seconds(5));
            continue;
        }

        // 2. Parse the block template fields from the RPC result.
        std::string result_json = extract_json_field(tmpl_resp.body, "result");
        if (result_json.empty() || result_json == "null") {
            std::string err_obj = extract_json_field(tmpl_resp.body, "error");
            fprintf(stderr, "RPC getblocktemplate failed: %s\n", err_obj.c_str());
            std::this_thread::sleep_for(std::chrono::seconds(5));
            continue;
        }

        std::string height_str    = extract_json_field(result_json, "height");
        std::string val_loss_str  = extract_json_field(result_json, "val_loss");
        std::string diff_str      = extract_json_field(result_json, "difficulty_delta");
        std::string ckpt_hash_str = extract_json_field(result_json, "checkpoint_hash");
        std::string d_model_str   = extract_json_field(result_json, "d_model");
        std::string n_layers_str  = extract_json_field(result_json, "n_layers");
        std::string n_slots_str   = extract_json_field(result_json, "n_slots");
        std::string d_ff_str      = extract_json_field(result_json, "d_ff");
        std::string vocab_str     = extract_json_field(result_json, "vocab_size");

        uint64_t height         = std::strtoull(height_str.c_str(), nullptr, 10);
        float    parent_val_loss = static_cast<float>(std::atof(val_loss_str.c_str()));
        float    difficulty_delta = static_cast<float>(std::atof(diff_str.c_str()));
        uint32_t d_model        = static_cast<uint32_t>(std::atoi(d_model_str.c_str()));
        uint32_t n_layers       = static_cast<uint32_t>(std::atoi(n_layers_str.c_str()));

        // Use genesis defaults if template returns zeros.
        if (parent_val_loss <= 0.0f) parent_val_loss = 10.0f;
        if (difficulty_delta <= 0.0f) difficulty_delta = consensus.genesis_difficulty_delta;
        if (d_model == 0) d_model = consensus.genesis_d_model;
        if (n_layers == 0) n_layers = consensus.genesis_n_layers;
        uint32_t n_slots        = n_slots_str.empty() ? 64u : static_cast<uint32_t>(std::atoi(n_slots_str.c_str()));
        uint32_t d_ff           = d_ff_str.empty() ? d_model * 2 : static_cast<uint32_t>(std::atoi(d_ff_str.c_str()));
        uint32_t vocab_size     = vocab_str.empty() ? 50257u : static_cast<uint32_t>(std::atoi(vocab_str.c_str()));

        printf("[attempt %llu] height=%llu  val_loss=%.6f  diff_delta=%.8f  "
               "d_model=%u  n_layers=%u\n",
               static_cast<unsigned long long>(total_attempts + 1),
               static_cast<unsigned long long>(height),
               static_cast<double>(parent_val_loss), static_cast<double>(difficulty_delta),
               d_model, n_layers);

        // 3. Build the model configuration from the template.
        rnet::training::ModelConfig model_cfg;
        model_cfg.d_model   = d_model;
        model_cfg.n_layers  = n_layers;
        model_cfg.n_slots   = n_slots;
        model_cfg.d_ff      = d_ff;
        model_cfg.vocab_size = vocab_size;

        // 4. Create TrainingEngine.
        rnet::training::TrainingEngine engine(*backend);
        auto init_rc = engine.init(model_cfg);
        if (init_rc.is_err()) {
            fprintf(stderr, "Error initializing model: %s\n",
                    init_rc.error().c_str());
            std::this_thread::sleep_for(std::chrono::seconds(5));
            continue;
        }

        // 5. Load parent checkpoint, or init random weights for genesis.
        bool is_genesis_mining = (ckpt_hash_str.empty() ||
            ckpt_hash_str == std::string(64, '0'));

        if (is_genesis_mining) {
            // Genesis: no parent checkpoint — engine.init() already
            // initialized weights with Xavier random. Ready to train.
            printf("[attempt %llu] Genesis mining: training from random init\n",
                   static_cast<unsigned long long>(total_attempts + 1));
        } else {
            std::filesystem::path ckpt_path =
                std::filesystem::path(cfg.checkpoint_dir) / (ckpt_hash_str + ".rnet");

            // 5a. If checkpoint is missing, request it from the node via
            //     RPC which broadcasts a getchkpt to P2P peers.  Poll
            //     up to 60 seconds for the file to arrive.
            if (!std::filesystem::exists(ckpt_path)) {
                printf("[attempt %llu] Checkpoint %s not found locally, "
                       "requesting from peers...\n",
                       static_cast<unsigned long long>(total_attempts + 1),
                       ckpt_hash_str.substr(0, 16).c_str());

                auto req_resp = rpc_call(cfg.host, cfg.port, auth,
                    "requestcheckpoint",
                    "[\"" + ckpt_hash_str + "\"]");

                if (req_resp.http_status == 200) {
                    std::string found_str =
                        extract_json_field(req_resp.body, "found");
                    if (found_str == "true") {
                        // Node already had it — check the path it returned.
                        std::string node_path =
                            extract_json_field(req_resp.body, "path");
                        if (!node_path.empty() &&
                            std::filesystem::exists(node_path)) {
                            // Copy from the node's checkpoint dir to ours
                            // if they differ.
                            if (node_path != ckpt_path.string()) {
                                std::error_code ec;
                                std::filesystem::copy_file(
                                    node_path, ckpt_path,
                                    std::filesystem::copy_options::skip_existing,
                                    ec);
                            }
                        }
                    }
                }

                // 5b. Poll for the checkpoint file to appear (peer transfer).
                int wait_rounds = 0;
                constexpr int max_wait_rounds = 12;  // 12 * 5s = 60s
                while (!std::filesystem::exists(ckpt_path) &&
                       wait_rounds < max_wait_rounds && g_running.load()) {
                    if (wait_rounds == 0) {
                        printf("[attempt %llu] Waiting for checkpoint from "
                               "peers (up to 60s)...\n",
                               static_cast<unsigned long long>(
                                   total_attempts + 1));
                    }
                    std::this_thread::sleep_for(std::chrono::seconds(5));
                    wait_rounds++;

                    // 5c. Re-request periodically in case first request
                    //     was missed.
                    if (wait_rounds % 3 == 0) {
                        rpc_call(cfg.host, cfg.port, auth,
                                 "requestcheckpoint",
                                 "[\"" + ckpt_hash_str + "\"]");
                    }
                }

                if (!std::filesystem::exists(ckpt_path)) {
                    fprintf(stderr, "Warning: checkpoint %s not received "
                            "after %d seconds, retrying...\n",
                            ckpt_hash_str.substr(0, 16).c_str(),
                            wait_rounds * 5);
                    continue;
                }

                printf("[attempt %llu] Checkpoint %s received!\n",
                       static_cast<unsigned long long>(total_attempts + 1),
                       ckpt_hash_str.substr(0, 16).c_str());
            }

            auto load_rc = engine.load_checkpoint(ckpt_path);
            if (load_rc.is_err()) {
                fprintf(stderr, "Error loading checkpoint: %s\n",
                        load_rc.error().c_str());
                std::this_thread::sleep_for(std::chrono::seconds(5));
                continue;
            }
        }

        // 6. Determine the number of training steps.
        int n_steps = rnet::miner::suggest_step_count(
            parent_val_loss, d_model, consensus);
        if (cfg.steps_per_attempt > 0) {
            n_steps = cfg.steps_per_attempt;
        }

        printf("[attempt %llu] Training %d steps (params: %llu)...\n",
               static_cast<unsigned long long>(total_attempts + 1), n_steps,
               static_cast<unsigned long long>(model_cfg.param_count()));

        auto attempt_start = std::chrono::steady_clock::now();

        // 7. Train: run forward+backward for n_steps.
        train_loader.reset();
        auto train_result = engine.train_steps(n_steps, train_loader);
        if (train_result.is_err()) {
            fprintf(stderr, "Training error: %s\n",
                    train_result.error().c_str());
            total_attempts++;
            continue;
        }
        float train_loss = train_result.value();

        // 8. Evaluate on the validation set.
        val_loader.reset();
        auto val_result = engine.evaluate(val_loader, consensus.eval_batches);
        if (val_result.is_err()) {
            fprintf(stderr, "Evaluation error: %s\n",
                    val_result.error().c_str());
            total_attempts++;
            continue;
        }
        float new_val_loss = val_result.value();

        auto attempt_end = std::chrono::steady_clock::now();
        double attempt_sec =
            std::chrono::duration<double>(attempt_end - attempt_start).count();

        float improvement = parent_val_loss - new_val_loss;
        total_attempts++;

        printf("[attempt %llu] train_loss=%.6f  val_loss=%.6f  "
               "improvement=%.8f  required=%.8f  (%.1fs)\n",
               static_cast<unsigned long long>(total_attempts),
               static_cast<double>(train_loss), static_cast<double>(new_val_loss),
               static_cast<double>(improvement), static_cast<double>(difficulty_delta),
               attempt_sec);

        // 9. Check whether the improvement meets the difficulty threshold.
        if (improvement >= difficulty_delta) {
            // 9a. Save the new checkpoint.
            std::filesystem::path tmp_ckpt =
                std::filesystem::path(cfg.checkpoint_dir) / "pending.rnet";
            auto save_rc = engine.save_checkpoint(tmp_ckpt);
            if (save_rc.is_err()) {
                fprintf(stderr, "Error saving checkpoint: %s\n",
                        save_rc.error().c_str());
                continue;
            }

            // 9b. Hash the checkpoint file with keccak256.
            auto hash_result = rnet::crypto::keccak256_file(tmp_ckpt);
            if (hash_result.is_err()) {
                fprintf(stderr, "Error hashing checkpoint: %s\n",
                        hash_result.error().c_str());
                continue;
            }
            std::string new_hash = hash_result.value().to_hex();

            // 9c. Rename to final path: {new_hash}.rnet.
            std::filesystem::path final_ckpt =
                std::filesystem::path(cfg.checkpoint_dir) /
                (new_hash + ".rnet");
            std::filesystem::rename(tmp_ckpt, final_ckpt);

            // 9d. RPC: submittrainingblock with training proof.
            std::ostringstream submit_params;
            submit_params << "[{"
                          << "\"checkpoint_hash\":\"" << new_hash << "\","
                          << "\"val_loss\":" << new_val_loss << ","
                          << "\"train_steps\":" << n_steps << ","
                          << "\"address\":\"" << cfg.miner_address << "\","
                          << "\"dataset_hash\":\"" << dataset_hash_hex << "\""
                          << "}]";

            auto submit_resp = rpc_call(cfg.host, cfg.port, auth,
                                         "submittrainingblock", submit_params.str());

            if (submit_resp.http_status == 200) {
                std::string accepted =
                    extract_json_field(submit_resp.body, "accepted");
                if (accepted == "true") {
                    blocks_found++;
                    printf("\n*** BLOCK FOUND! ***\n");
                    printf("  Height:          %llu\n",
                           static_cast<unsigned long long>(height));
                    printf("  Val loss:        %.6f -> %.6f (delta %.8f)\n",
                           static_cast<double>(parent_val_loss),
                           static_cast<double>(new_val_loss),
                           static_cast<double>(improvement));
                    printf("  Checkpoint:      %s\n", new_hash.c_str());
                    printf("  Training steps:  %d\n", n_steps);
                    printf("  Blocks found:    %llu\n\n",
                           static_cast<unsigned long long>(blocks_found));
                } else {
                    std::string reason =
                        extract_json_field(submit_resp.body, "reject_reason");
                    fprintf(stderr, "Submit rejected: %s\n", reason.c_str());
                }
            } else {
                fprintf(stderr, "Submit RPC error (HTTP %d): %s\n",
                        submit_resp.http_status, submit_resp.body.c_str());
            }
        }

        // 10. Auto-cleanup: keep only the last 2 checkpoints (current
        //     tip and its parent).  Delete all older .rnet files from
        //     the checkpoint directory to reclaim disk space.
        {
            std::vector<std::filesystem::path> ckpt_files;
            std::error_code ec;
            for (const auto& entry :
                 std::filesystem::directory_iterator(cfg.checkpoint_dir, ec)) {
                if (entry.is_regular_file() &&
                    entry.path().extension() == ".rnet" &&
                    entry.path().filename() != "pending.rnet") {
                    ckpt_files.push_back(entry.path());
                }
            }

            // Sort by modification time (newest first).
            std::sort(ckpt_files.begin(), ckpt_files.end(),
                [](const std::filesystem::path& a,
                   const std::filesystem::path& b) {
                    std::error_code ec2;
                    return std::filesystem::last_write_time(a, ec2) >
                           std::filesystem::last_write_time(b, ec2);
                });

            // Delete all but the newest 2.
            if (ckpt_files.size() > 2) {
                for (size_t i = 2; i < ckpt_files.size(); ++i) {
                    std::error_code rm_ec;
                    std::filesystem::remove(ckpt_files[i], rm_ec);
                    if (!rm_ec) {
                        printf("Cleaned up old checkpoint: %s\n",
                               ckpt_files[i].filename().string().c_str());
                    }
                }
            }
        }

        // 11. Print periodic stats.
        if (total_attempts % 5 == 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed_sec =
                std::chrono::duration<double>(now - start_time).count();
            double rate = static_cast<double>(total_attempts) / elapsed_sec;
            printf("Stats: %llu attempts, %llu blocks found, %.2f attempts/hr\n",
                   static_cast<unsigned long long>(total_attempts),
                   static_cast<unsigned long long>(blocks_found),
                   rate * 3600.0);
        }
    }

    // -- Shutdown summary ---------------------------------------------------

    printf("\nMiner stopped.\n");
    auto end_time = std::chrono::steady_clock::now();
    double total_sec = std::chrono::duration<double>(end_time - start_time).count();
    printf("Total: %llu attempts in %.1f seconds\n",
           static_cast<unsigned long long>(total_attempts), total_sec);
    printf("Blocks found: %llu\n", static_cast<unsigned long long>(blocks_found));

    return 0;
}

// ===========================================================================
//  Benchmark mode
// ===========================================================================

// ---------------------------------------------------------------------------
// run_benchmark
// ---------------------------------------------------------------------------
// Benchmarks Keccak-256 hashing, Ed25519 signing, and Ed25519 verification
// as a proxy for PoT verification overhead.
// ---------------------------------------------------------------------------
static int run_benchmark(const MinerCliConfig& cfg)
{
    printf("ResonanceNet Miner Benchmark\n");
    printf("Steps per attempt: %d\n\n", cfg.steps_per_attempt);

    // 1. Keccak-256 throughput benchmark.
    printf("Keccak-256 throughput benchmark...\n");
    std::vector<uint8_t> data(64, 0xAB);
    auto start = std::chrono::high_resolution_clock::now();
    rnet::uint256 hash;
    constexpr int ITERATIONS = 500000;
    for (int i = 0; i < ITERATIONS; ++i) {
        hash = rnet::crypto::keccak256(data);
        std::memcpy(data.data(), hash.data(), 32);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("  %d hashes in %.2f ms = %.0f hashes/sec\n",
           ITERATIONS, ms, ITERATIONS / (ms / 1000.0));

    // 2. Ed25519 signing benchmark.
    printf("\nEd25519 signing benchmark...\n");
    auto kp = rnet::crypto::ed25519_generate();
    if (kp.is_err()) {
        fprintf(stderr, "Error generating keypair\n");
        return 1;
    }
    std::vector<uint8_t> msg(256, 0xCD);
    start = std::chrono::high_resolution_clock::now();
    constexpr int SIG_ITERS = 10000;
    for (int i = 0; i < SIG_ITERS; ++i) {
        auto sig = rnet::crypto::ed25519_sign(kp.value().secret, msg);
        (void)sig;
    }
    end = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("  %d signatures in %.2f ms = %.0f sigs/sec\n",
           SIG_ITERS, ms, SIG_ITERS / (ms / 1000.0));

    // 3. Ed25519 verification benchmark.
    printf("\nEd25519 verification benchmark...\n");
    auto sig_result = rnet::crypto::ed25519_sign(kp.value().secret, msg);
    if (sig_result.is_err()) {
        fprintf(stderr, "Error signing\n");
        return 1;
    }
    auto sig = sig_result.value();
    start = std::chrono::high_resolution_clock::now();
    constexpr int VER_ITERS = 10000;
    for (int i = 0; i < VER_ITERS; ++i) {
        bool ok = rnet::crypto::ed25519_verify(kp.value().public_key, msg, sig);
        (void)ok;
    }
    end = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("  %d verifications in %.2f ms = %.0f verif/sec\n",
           VER_ITERS, ms, VER_ITERS / (ms / 1000.0));

    return 0;
}

// ===========================================================================
//  Main
// ===========================================================================

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
// Entry point for rnet-miner.  Parses arguments, sets up signal handlers,
// and dispatches to either benchmark mode or the mining loop.
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
#ifdef _WIN32
    WSADATA wsa_data;
    WSAStartup(MAKEWORD(2, 2), &wsa_data);
#else
    std::signal(SIGINT, sigint_handler);
    std::signal(SIGTERM, sigint_handler);
#endif

    MinerCliConfig cfg = parse_args(argc, argv);

    int rc;
    if (cfg.benchmark) {
        rc = run_benchmark(cfg);
    } else {
        rc = run_mining_loop(cfg);
    }

#ifdef _WIN32
    WSACleanup();
#endif
    return rc;
}

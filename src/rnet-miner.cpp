// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

// Project headers.
#include "core/config.h"
#include "core/error.h"
#include "core/hex.h"
#include "core/logging.h"
#include "crypto/ed25519.h"
#include "crypto/keccak.h"

// Standard library.
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
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

// ===========================================================================
//  Mining configuration
// ===========================================================================

struct MinerCliConfig {
    std::string host = "127.0.0.1";
    uint16_t port = 9554;
    std::string rpcuser;
    std::string rpcpassword;
    std::string miner_pubkey;
    std::string train_data = "data/train.bin";
    std::string val_data = "data/val.bin";
    std::string checkpoint_dir = "checkpoints";
    int steps_per_attempt = 1000;
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
        "  -minerkey=<pubkey_hex>     Miner's Ed25519 public key (64 hex chars)\n"
        "  -traindata=<path>          Path to training dataset\n"
        "  -valdata=<path>            Path to validation dataset\n"
        "  -checkpointdir=<path>      Directory for checkpoints (default: ./checkpoints)\n"
        "  -steps=<n>                 Training steps per mining attempt (default: 1000)\n"
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
        if (arg.find("-rpcconnect=") == 0) cfg.host = arg.substr(12);
        else if (arg.find("-rpcport=") == 0) cfg.port = static_cast<uint16_t>(std::atoi(arg.c_str() + 9));
        else if (arg.find("-rpcuser=") == 0) cfg.rpcuser = arg.substr(9);
        else if (arg.find("-rpcpassword=") == 0) cfg.rpcpassword = arg.substr(13);
        else if (arg.find("-minerkey=") == 0) cfg.miner_pubkey = arg.substr(10);
        else if (arg.find("-traindata=") == 0) cfg.train_data = arg.substr(11);
        else if (arg.find("-valdata=") == 0) cfg.val_data = arg.substr(9);
        else if (arg.find("-checkpointdir=") == 0) cfg.checkpoint_dir = arg.substr(15);
        else if (arg.find("-steps=") == 0) cfg.steps_per_attempt = std::atoi(arg.c_str() + 7);
        else if (arg.find("-workers=") == 0) cfg.num_workers = std::atoi(arg.c_str() + 9);
        else if (arg == "-benchmark") cfg.benchmark = true;
        else if (arg == "-help" || arg == "--help") { print_usage(); std::exit(0); }
    }
    return cfg;
}

// ===========================================================================
//  Mining loop
// ===========================================================================

// ---------------------------------------------------------------------------
// run_mining_loop
// ---------------------------------------------------------------------------
// Main mining loop: repeatedly fetches a block template from rnetd, runs
// training steps (currently a placeholder), and submits the result.
// ---------------------------------------------------------------------------
static int run_mining_loop(const MinerCliConfig& cfg)
{
    std::string auth;
    if (!cfg.rpcuser.empty()) {
        auth = base64_encode(cfg.rpcuser + ":" + cfg.rpcpassword);
    }

    if (cfg.miner_pubkey.empty()) {
        fprintf(stderr, "Error: -minerkey=<pubkey_hex> is required\n");
        return 1;
    }

    // Verify dataset files exist.
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

    // Create checkpoint directory if needed.
    if (!std::filesystem::exists(cfg.checkpoint_dir)) {
        std::filesystem::create_directories(cfg.checkpoint_dir);
    }

    printf("ResonanceNet Miner v0.1\n");
    printf("Connecting to %s:%u\n", cfg.host.c_str(), cfg.port);
    printf("Miner pubkey: %s\n", cfg.miner_pubkey.c_str());
    printf("Train data: %s\n", cfg.train_data.c_str());
    printf("Val data: %s\n", cfg.val_data.c_str());
    printf("Checkpoint dir: %s\n", cfg.checkpoint_dir.c_str());
    printf("Training steps per attempt: %d\n", cfg.steps_per_attempt);
    printf("Workers: %d\n", cfg.num_workers);
    printf("\n");

    uint64_t total_attempts = 0;
    uint64_t blocks_found = 0;
    auto start_time = std::chrono::steady_clock::now();

    while (g_running.load()) {
        // 1. Get block template from rnetd.
        auto tmpl_resp = rpc_call(cfg.host, cfg.port, auth,
                                   "getblocktemplate",
                                   "[{\"minerkey\":\"" + cfg.miner_pubkey + "\"}]");
        if (tmpl_resp.http_status < 0) {
            fprintf(stderr, "Error: cannot connect to rnetd: %s\n", tmpl_resp.body.c_str());
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

        // 2. Train the model (placeholder -- actual training uses GPU backend).
        //    In production, this would:
        //    a) Load the latest checkpoint
        //    b) Run forward+backward pass for steps_per_attempt steps
        //    c) Evaluate on validation set
        //    d) Save new checkpoint
        printf("[%llu] Training %d steps...\n",
               static_cast<unsigned long long>(total_attempts), cfg.steps_per_attempt);

        // Simulate training time (in production, this is real GPU work).
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        total_attempts++;

        // 3. Submit block.
        //    In production, we would serialize the completed block and submit
        //    via the "submitblock" RPC method.
        auto submit_resp = rpc_call(cfg.host, cfg.port, auth,
                                     "submitblock", "[\"placeholder\"]");

        if (submit_resp.http_status == 200) {
            printf("[%llu] Block submitted.\n",
                   static_cast<unsigned long long>(total_attempts));
        }

        // 4. Print stats periodically.
        if (total_attempts % 10 == 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed_sec = std::chrono::duration<double>(now - start_time).count();
            double rate = static_cast<double>(total_attempts) / elapsed_sec;
            printf("Stats: %llu attempts, %llu blocks found, %.2f attempts/sec\n",
                   static_cast<unsigned long long>(total_attempts),
                   static_cast<unsigned long long>(blocks_found),
                   rate);
        }
    }

    printf("\nMiner stopped.\n");
    auto end_time = std::chrono::steady_clock::now();
    double total_sec = std::chrono::duration<double>(end_time - start_time).count();
    printf("Total: %llu attempts in %.1f seconds (%.2f attempts/sec)\n",
           static_cast<unsigned long long>(total_attempts), total_sec,
           static_cast<double>(total_attempts) / total_sec);
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

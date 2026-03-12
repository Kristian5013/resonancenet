// rnet-util — General utility tool
// Provides hash, encode/decode, and benchmarking operations.

#include "core/base58.h"
#include "core/bech32.h"
#include "core/error.h"
#include "core/hex.h"
#include "core/types.h"
#include "crypto/hash.h"
#include "crypto/keccak.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace rnet;

// ─── Usage ──────────────────────────────────────────────────────────

static void print_usage() {
    fprintf(stderr,
        "Usage: rnet-util <command> [args...]\n"
        "\n"
        "Hashing:\n"
        "  keccak256 <hex>            Keccak-256 hash of hex data\n"
        "  keccak256d <hex>           Double Keccak-256 hash of hex data\n"
        "  hash160 <hex>              Hash160 (first 20 bytes of keccak256d)\n"
        "  hash256 <hex>              Hash256 (alias for keccak256d)\n"
        "\n"
        "Encoding:\n"
        "  base58enc <hex>            Base58-encode hex data\n"
        "  base58dec <string>         Base58-decode to hex\n"
        "  base58check_enc <hex>      Base58Check-encode hex data\n"
        "  base58check_dec <string>   Base58Check-decode to hex\n"
        "  bech32enc <hrp> <ver> <hex>  Encode segwit address\n"
        "  bech32dec <hrp> <address>    Decode segwit address\n"
        "\n"
        "Conversion:\n"
        "  hexrev <hex>               Reverse byte order of hex string\n"
        "\n"
        "Benchmark:\n"
        "  grind [iterations]         Benchmark Keccak-256 hashing speed\n"
    );
}

// ─── Hash commands ──────────────────────────────────────────────────

static int cmd_keccak256(const std::string& hex_input) {
    auto data = core::from_hex(hex_input);
    auto hash = crypto::keccak256(data);
    printf("%s\n", hash.to_hex().c_str());
    return 0;
}

static int cmd_keccak256d(const std::string& hex_input) {
    auto data = core::from_hex(hex_input);
    auto hash = crypto::keccak256d(data);
    printf("%s\n", hash.to_hex().c_str());
    return 0;
}

static int cmd_hash160(const std::string& hex_input) {
    auto data = core::from_hex(hex_input);
    auto hash = crypto::hash160(data);
    printf("%s\n", hash.to_hex().c_str());
    return 0;
}

static int cmd_hash256(const std::string& hex_input) {
    auto data = core::from_hex(hex_input);
    auto hash = crypto::hash256(data);
    printf("%s\n", hash.to_hex().c_str());
    return 0;
}

// ─── Encoding commands ──────────────────────────────────────────────

static int cmd_base58_encode(const std::string& hex_input) {
    auto data = core::from_hex(hex_input);
    auto result = core::base58_encode(data);
    printf("%s\n", result.c_str());
    return 0;
}

static int cmd_base58_decode(const std::string& str) {
    auto result = core::base58_decode(str);
    if (!result.has_value()) {
        fprintf(stderr, "Error: invalid Base58 string\n");
        return 1;
    }
    printf("%s\n", core::to_hex(result.value()).c_str());
    return 0;
}

static int cmd_base58check_encode(const std::string& hex_input) {
    auto data = core::from_hex(hex_input);
    auto result = core::base58check_encode_simple(data);
    printf("%s\n", result.c_str());
    return 0;
}

static int cmd_base58check_decode(const std::string& str) {
    auto result = core::base58check_decode_simple(str);
    if (!result.has_value()) {
        fprintf(stderr, "Error: invalid Base58Check string (bad checksum?)\n");
        return 1;
    }
    printf("%s\n", core::to_hex(result.value()).c_str());
    return 0;
}

static int cmd_bech32_encode(const std::string& hrp, int version,
                              const std::string& hex_program) {
    auto program = core::from_hex(hex_program);
    auto result = core::encode_segwit_addr(hrp, version, program);
    if (result.empty()) {
        fprintf(stderr, "Error: bech32 encoding failed\n");
        return 1;
    }
    printf("%s\n", result.c_str());
    return 0;
}

static int cmd_bech32_decode(const std::string& hrp, const std::string& addr) {
    auto result = core::decode_segwit_addr(hrp, addr);
    if (!result.valid) {
        fprintf(stderr, "Error: invalid bech32 address\n");
        return 1;
    }
    printf("version: %d\n", result.witness_version);
    printf("program: %s\n", core::to_hex(result.witness_program).c_str());
    return 0;
}

static int cmd_hexrev(const std::string& hex_input) {
    auto result = core::reverse_hex(hex_input);
    printf("%s\n", result.c_str());
    return 0;
}

// ─── Benchmark ──────────────────────────────────────────────────────

static int cmd_grind(int iterations) {
    if (iterations <= 0) iterations = 1000000;

    printf("Benchmarking Keccak-256: %d iterations...\n", iterations);

    // Prepare test data: 64 bytes
    std::vector<uint8_t> data(64, 0xAB);

    auto start = std::chrono::high_resolution_clock::now();
    uint256 hash;
    for (int i = 0; i < iterations; ++i) {
        hash = crypto::keccak256(data);
        // Feed hash back as input to prevent optimization
        std::memcpy(data.data(), hash.data(), 32);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double per_sec = static_cast<double>(iterations) / (elapsed_ms / 1000.0);
    double mbps = (static_cast<double>(iterations) * 64.0) / (elapsed_ms / 1000.0) / 1e6;

    printf("Time: %.2f ms\n", elapsed_ms);
    printf("Rate: %.0f hashes/sec\n", per_sec);
    printf("Throughput: %.2f MB/s\n", mbps);
    printf("Last hash: %s\n", hash.to_hex().c_str());

    return 0;
}

// ─── Main ───────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string cmd = argv[1];

    if (cmd == "-help" || cmd == "--help" || cmd == "-h") {
        print_usage();
        return 0;
    }

    // Hash commands
    if (cmd == "keccak256" && argc >= 3)  return cmd_keccak256(argv[2]);
    if (cmd == "keccak256d" && argc >= 3) return cmd_keccak256d(argv[2]);
    if (cmd == "hash160" && argc >= 3)    return cmd_hash160(argv[2]);
    if (cmd == "hash256" && argc >= 3)    return cmd_hash256(argv[2]);

    // Encoding commands
    if (cmd == "base58enc" && argc >= 3)       return cmd_base58_encode(argv[2]);
    if (cmd == "base58dec" && argc >= 3)       return cmd_base58_decode(argv[2]);
    if (cmd == "base58check_enc" && argc >= 3) return cmd_base58check_encode(argv[2]);
    if (cmd == "base58check_dec" && argc >= 3) return cmd_base58check_decode(argv[2]);

    if (cmd == "bech32enc" && argc >= 5) {
        return cmd_bech32_encode(argv[2], std::atoi(argv[3]), argv[4]);
    }
    if (cmd == "bech32dec" && argc >= 4) {
        return cmd_bech32_decode(argv[2], argv[3]);
    }

    if (cmd == "hexrev" && argc >= 3) return cmd_hexrev(argv[2]);

    // Benchmark
    if (cmd == "grind") {
        int iters = (argc >= 3) ? std::atoi(argv[2]) : 1000000;
        return cmd_grind(iters);
    }

    fprintf(stderr, "Unknown command or missing arguments: %s\n", cmd.c_str());
    print_usage();
    return 1;
}

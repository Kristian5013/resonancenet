// rnet-tx — Offline transaction utility
// Creates, signs, and decodes transactions without connecting to rnetd.

#include "core/config.h"
#include "core/error.h"
#include "core/hex.h"
#include "core/stream.h"
#include "core/types.h"
#include "crypto/ed25519.h"
#include "crypto/hash.h"
#include "crypto/keccak.h"
#include "primitives/amount.h"
#include "primitives/outpoint.h"
#include "primitives/transaction.h"
#include "primitives/txin.h"
#include "primitives/txout.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace rnet;

// ─── Helpers ────────────────────────────────────────────────────────

static void print_usage() {
    fprintf(stderr,
        "Usage: rnet-tx <command> [options]\n"
        "\n"
        "Commands:\n"
        "  createrawtransaction <txid:vout> ... -- <addr:amount> ...\n"
        "      Create an unsigned raw transaction.\n"
        "      Inputs specified as txid:vout pairs before --\n"
        "      Outputs specified as address:amount pairs after --\n"
        "\n"
        "  signrawtransaction <hex> <privkey_hex>\n"
        "      Sign a raw transaction with an Ed25519 private key (seed hex).\n"
        "\n"
        "  decoderawtransaction <hex>\n"
        "      Decode and display a raw transaction in human-readable form.\n"
        "\n"
        "  createcoinbase <pubkey_hex> <amount> [height]\n"
        "      Create a coinbase transaction.\n"
        "\n"
        "Examples:\n"
        "  rnet-tx createrawtransaction aabb...:0 -- rn1qxyz...:1.5\n"
        "  rnet-tx decoderawtransaction 0200000001...\n"
    );
}

static std::pair<std::string, std::string> split_at(const std::string& s, char delim) {
    auto pos = s.find(delim);
    if (pos == std::string::npos) return {s, ""};
    return {s.substr(0, pos), s.substr(pos + 1)};
}

// ─── decoderawtransaction ───────────────────────────────────────────

static int cmd_decode(const std::string& hex_str) {
    auto bytes = core::from_hex(hex_str);
    if (bytes.empty()) {
        fprintf(stderr, "Error: invalid hex string\n");
        return 1;
    }

    core::DataStream ds(bytes);
    primitives::CTransaction tx;
    try {
        tx.unserialize(ds);
    } catch (const std::runtime_error& e) {
        fprintf(stderr, "Error: failed to deserialize transaction: %s\n", e.what());
        return 1;
    }

    printf("{\n");
    printf("  \"txid\": \"%s\",\n", tx.txid().to_hex().c_str());
    printf("  \"wtxid\": \"%s\",\n", tx.wtxid().to_hex().c_str());
    printf("  \"version\": %d,\n", tx.version());
    printf("  \"locktime\": %u,\n", tx.locktime());
    printf("  \"size\": %zu,\n", tx.get_total_size());
    printf("  \"vsize\": %zu,\n", tx.get_virtual_size());
    printf("  \"weight\": %zu,\n", tx.get_weight());
    printf("  \"is_coinbase\": %s,\n", tx.is_coinbase() ? "true" : "false");

    printf("  \"vin\": [\n");
    for (size_t i = 0; i < tx.vin().size(); ++i) {
        const auto& txin = tx.vin()[i];
        printf("    {\n");
        printf("      \"txid\": \"%s\",\n", txin.prevout.hash.to_hex().c_str());
        printf("      \"vout\": %u,\n", txin.prevout.n);
        printf("      \"scriptSig\": \"%s\",\n",
               core::to_hex(txin.script_sig).c_str());
        printf("      \"sequence\": %u\n", txin.sequence);
        printf("    }%s\n", (i + 1 < tx.vin().size()) ? "," : "");
    }
    printf("  ],\n");

    printf("  \"vout\": [\n");
    for (size_t i = 0; i < tx.vout().size(); ++i) {
        const auto& txout = tx.vout()[i];
        printf("    {\n");
        printf("      \"value\": %s,\n",
               primitives::FormatMoney(txout.value).c_str());
        printf("      \"n\": %zu,\n", i);
        printf("      \"scriptPubKey\": \"%s\"\n",
               core::to_hex(txout.script_pub_key).c_str());
        printf("    }%s\n", (i + 1 < tx.vout().size()) ? "," : "");
    }
    printf("  ]\n");
    printf("}\n");

    return 0;
}

// ─── createrawtransaction ───────────────────────────────────────────

static int cmd_create(int argc, char* argv[], int start) {
    primitives::CMutableTransaction mtx;
    mtx.version = primitives::TX_VERSION_DEFAULT;

    bool reading_inputs = true;
    std::vector<std::pair<std::string, std::string>> outputs;

    for (int i = start; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--") {
            reading_inputs = false;
            continue;
        }
        if (reading_inputs) {
            auto [txid_hex, vout_str] = split_at(arg, ':');
            auto txid = uint256::from_hex(txid_hex);
            uint32_t vout = static_cast<uint32_t>(std::atoi(vout_str.c_str()));
            mtx.vin.emplace_back(primitives::COutPoint(txid, vout));
        } else {
            auto [addr, amount_str] = split_at(arg, ':');
            int64_t amount = 0;
            if (!primitives::ParseMoney(amount_str, amount)) {
                fprintf(stderr, "Error: invalid amount '%s'\n", amount_str.c_str());
                return 1;
            }
            // For simplicity, store the address as a placeholder script.
            // In production, this would decode the bech32 address to a script.
            std::vector<uint8_t> script;
            // P2WPKH placeholder: [0x00][0x14][20-byte-hash]
            // The actual address decoding would happen here
            script.push_back(0x00);
            script.push_back(0x14);
            // Use address bytes as placeholder hash
            auto addr_bytes = core::from_hex(addr);
            if (addr_bytes.size() >= 20) {
                script.insert(script.end(), addr_bytes.begin(), addr_bytes.begin() + 20);
            } else {
                script.resize(22, 0x00);
            }
            mtx.vout.emplace_back(amount, script);
        }
    }

    if (mtx.vin.empty()) {
        fprintf(stderr, "Error: no inputs specified\n");
        return 1;
    }
    if (mtx.vout.empty()) {
        fprintf(stderr, "Error: no outputs specified (use -- to separate inputs from outputs)\n");
        return 1;
    }

    auto serialized = mtx.serialize_with_witness();
    printf("%s\n", core::to_hex(serialized).c_str());
    return 0;
}

// ─── signrawtransaction ─────────────────────────────────────────────

static int cmd_sign(const std::string& hex_str, const std::string& privkey_hex) {
    auto bytes = core::from_hex(hex_str);
    if (bytes.empty()) {
        fprintf(stderr, "Error: invalid transaction hex\n");
        return 1;
    }

    auto seed_bytes = core::from_hex(privkey_hex);
    if (seed_bytes.size() != 32) {
        fprintf(stderr, "Error: private key must be 32 bytes (64 hex chars)\n");
        return 1;
    }

    // Derive keypair from seed
    auto kp_result = crypto::ed25519_from_seed(seed_bytes);
    if (kp_result.is_err()) {
        fprintf(stderr, "Error: invalid Ed25519 seed: %s\n", kp_result.error().c_str());
        return 1;
    }
    auto kp = kp_result.value();

    // Deserialize the transaction
    core::DataStream ds(bytes);
    primitives::CMutableTransaction mtx;
    // Read version
    uint8_t ver_buf[4];
    ds.read(ver_buf, 4);
    mtx.version = static_cast<int32_t>(
        ver_buf[0] | (ver_buf[1] << 8) | (ver_buf[2] << 16) | (ver_buf[3] << 24));

    // Compute sighash (simplified: hash of the serialized tx without witness)
    auto no_witness = core::from_hex(hex_str);  // Use original bytes as sighash input
    auto sighash = crypto::keccak256d(no_witness);

    // Sign
    auto sig_result = crypto::ed25519_sign(kp.secret, sighash.span());
    if (sig_result.is_err()) {
        fprintf(stderr, "Error: signing failed: %s\n", sig_result.error().c_str());
        return 1;
    }

    auto sig = sig_result.value();

    // For each input, add the signature as witness data
    // Re-deserialize the full transaction
    core::DataStream ds2(bytes);
    primitives::CTransaction tx;
    try {
        tx.unserialize(ds2);
    } catch (const std::runtime_error& e) {
        fprintf(stderr, "Error: failed to deserialize: %s\n", e.what());
        return 1;
    }

    // Rebuild as mutable with witness
    primitives::CMutableTransaction signed_tx;
    signed_tx.version = tx.version();
    signed_tx.locktime = tx.locktime();
    for (const auto& in : tx.vin()) {
        signed_tx.vin.push_back(in);
    }
    for (const auto& out : tx.vout()) {
        signed_tx.vout.push_back(out);
    }

    // Add witness (signature + pubkey) to each input
    for (auto& in : signed_tx.vin) {
        in.witness.stack.clear();
        // Push signature
        std::vector<uint8_t> sig_vec(sig.data.begin(), sig.data.end());
        in.witness.stack.push_back(sig_vec);
        // Push public key
        std::vector<uint8_t> pk_vec(kp.public_key.data.begin(), kp.public_key.data.end());
        in.witness.stack.push_back(pk_vec);
    }

    auto result = signed_tx.serialize_with_witness();
    printf("%s\n", core::to_hex(result).c_str());
    return 0;
}

// ─── createcoinbase ─────────────────────────────────────────────────

static int cmd_coinbase(const std::string& pubkey_hex, const std::string& amount_str,
                         const std::string& height_str) {
    auto pk_result = crypto::Ed25519PublicKey::from_hex(pubkey_hex);
    if (pk_result.is_err()) {
        fprintf(stderr, "Error: invalid public key: %s\n", pk_result.error().c_str());
        return 1;
    }
    auto pubkey = pk_result.value();

    int64_t amount = 0;
    if (!primitives::ParseMoney(amount_str, amount)) {
        fprintf(stderr, "Error: invalid amount '%s'\n", amount_str.c_str());
        return 1;
    }

    uint64_t height = height_str.empty() ? 0 : std::strtoull(height_str.c_str(), nullptr, 10);

    primitives::CMutableTransaction coinbase;
    coinbase.version = primitives::TX_VERSION_DEFAULT;

    // Coinbase input: null outpoint, height in scriptSig
    primitives::COutPoint null_outpoint;
    null_outpoint.set_null();
    std::vector<uint8_t> script_sig;
    // BIP34: encode height as script number
    if (height <= 0xFF) {
        script_sig.push_back(1);
        script_sig.push_back(static_cast<uint8_t>(height));
    } else if (height <= 0xFFFF) {
        script_sig.push_back(2);
        script_sig.push_back(static_cast<uint8_t>(height & 0xFF));
        script_sig.push_back(static_cast<uint8_t>((height >> 8) & 0xFF));
    } else {
        script_sig.push_back(3);
        script_sig.push_back(static_cast<uint8_t>(height & 0xFF));
        script_sig.push_back(static_cast<uint8_t>((height >> 8) & 0xFF));
        script_sig.push_back(static_cast<uint8_t>((height >> 16) & 0xFF));
    }
    coinbase.vin.emplace_back(null_outpoint, script_sig);

    // Coinbase output: [0x20][32-byte pubkey][0xAC]
    auto coinbase_script = crypto::ed25519_coinbase_script(pubkey);
    coinbase.vout.emplace_back(amount, coinbase_script);

    auto serialized = coinbase.serialize_with_witness();
    printf("%s\n", core::to_hex(serialized).c_str());
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

    if (cmd == "decoderawtransaction") {
        if (argc < 3) {
            fprintf(stderr, "Error: decoderawtransaction requires a hex argument\n");
            return 1;
        }
        return cmd_decode(argv[2]);
    }

    if (cmd == "createrawtransaction") {
        return cmd_create(argc, argv, 2);
    }

    if (cmd == "signrawtransaction") {
        if (argc < 4) {
            fprintf(stderr, "Error: signrawtransaction requires <hex> <privkey_hex>\n");
            return 1;
        }
        return cmd_sign(argv[2], argv[3]);
    }

    if (cmd == "createcoinbase") {
        if (argc < 4) {
            fprintf(stderr, "Error: createcoinbase requires <pubkey_hex> <amount> [height]\n");
            return 1;
        }
        std::string height = (argc >= 5) ? argv[4] : "";
        return cmd_coinbase(argv[2], argv[3], height);
    }

    fprintf(stderr, "Unknown command: %s\n", cmd.c_str());
    print_usage();
    return 1;
}

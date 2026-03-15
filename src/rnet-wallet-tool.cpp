// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

// Project headers.
#include "core/config.h"
#include "core/error.h"
#include "core/hex.h"
#include "core/logging.h"
#include "crypto/bip39.h"
#include "crypto/ed25519.h"
#include "crypto/hash.h"
#include "primitives/address.h"
#include "script/recovery_script.h"
#include "wallet/addresses.h"
#include "wallet/wallet.h"

// Standard library.
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

using namespace rnet;

// ===========================================================================
//  Helpers
// ===========================================================================

// ---------------------------------------------------------------------------
// print_usage
// ---------------------------------------------------------------------------
static void print_usage()
{
    fprintf(stderr,
        "Usage: rnet-wallet-tool [options] <command>\n"
        "\n"
        "Commands:\n"
        "  create [path]              Create a new wallet (default: data dir)\n"
        "  info <path>                Display wallet information\n"
        "  dump <path>                Dump wallet keys (requires unlock)\n"
        "  encrypt <path>             Encrypt an unencrypted wallet\n"
        "  mnemonic                   Generate a new BIP39 mnemonic (no wallet)\n"
        "\n"
        "Options:\n"
        "  -passphrase=<pass>         Passphrase for encryption/decryption\n"
        "  -network=<net>             Network: mainnet, testnet, regtest (default: mainnet)\n"
        "  -recovery=<type>           Recovery type: heartbeat, social, emission\n"
        "  -heartbeat_interval=<n>    Heartbeat interval in blocks (default: 100000)\n"
        "  -help                      Show this help message\n"
    );
}

struct WalletToolConfig {
    std::string command;
    std::string path;
    std::string passphrase;
    std::string network = "mainnet";
    std::string recovery = "heartbeat";
    uint64_t heartbeat_interval = 100000;
};

// ---------------------------------------------------------------------------
// parse_args
// ---------------------------------------------------------------------------
static WalletToolConfig parse_args(int argc, char* argv[])
{
    WalletToolConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("-passphrase=") == 0) {
            cfg.passphrase = arg.substr(12);
        } else if (arg.find("-network=") == 0) {
            cfg.network = arg.substr(9);
        } else if (arg.find("-recovery=") == 0) {
            cfg.recovery = arg.substr(10);
        } else if (arg.find("-heartbeat_interval=") == 0) {
            cfg.heartbeat_interval = std::strtoull(arg.c_str() + 20, nullptr, 10);
        } else if (arg == "-help" || arg == "--help" || arg == "-h") {
            print_usage();
            std::exit(0);
        } else if (arg[0] != '-') {
            if (cfg.command.empty()) {
                cfg.command = arg;
            } else if (cfg.path.empty()) {
                cfg.path = arg;
            }
        }
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// parse_network
// ---------------------------------------------------------------------------
static primitives::NetworkType parse_network(const std::string& net)
{
    if (net == "testnet") return primitives::NetworkType::TESTNET;
    if (net == "regtest") return primitives::NetworkType::REGTEST;
    return primitives::NetworkType::MAINNET;
}

// ===========================================================================
//  Commands
// ===========================================================================

// ---------------------------------------------------------------------------
// cmd_create
// ---------------------------------------------------------------------------
// Creates a new wallet file, optionally encrypts it, displays the BIP39
// mnemonic for backup, and generates the first receive address.
// ---------------------------------------------------------------------------
static int cmd_create(WalletToolConfig cfg)
{
    // 1. Default wallet path if none given.
    if (cfg.path.empty()) {
#ifdef _WIN32
        const char* appdata = std::getenv("APPDATA");
        if (appdata) {
            cfg.path = std::string(appdata) + "\\ResonanceNet\\wallet.dat";
        } else {
            cfg.path = "wallet.dat";
        }
#else
        const char* home = std::getenv("HOME");
        if (home) {
            cfg.path = std::string(home) + "/.resonancenet/wallet.dat";
        } else {
            cfg.path = "wallet.dat";
        }
#endif
        printf("Using default wallet path: %s\n", cfg.path.c_str());
    }

    // 2. Create parent directory if it does not exist.
    auto parent = std::filesystem::path(cfg.path).parent_path();
    if (!parent.empty() && !std::filesystem::exists(parent)) {
        std::filesystem::create_directories(parent);
    }

    // 3. Reject if a file (not directory) already exists at this path.
    if (std::filesystem::exists(cfg.path) && !std::filesystem::is_directory(cfg.path)) {
        fprintf(stderr, "Error: wallet already exists at '%s'\n", cfg.path.c_str());
        return 1;
    }

    // 4. If user gave a directory, append wallet.dat.
    if (std::filesystem::exists(cfg.path) && std::filesystem::is_directory(cfg.path)) {
        cfg.path = (std::filesystem::path(cfg.path) / "wallet.dat").string();
        if (std::filesystem::exists(cfg.path)) {
            fprintf(stderr, "Error: wallet already exists at '%s'\n", cfg.path.c_str());
            return 1;
        }
    }

    auto network = parse_network(cfg.network);

    // 1. Build recovery policy.
    script::RecoveryType recovery_type = script::RecoveryType::HEARTBEAT;
    script::RecoveryPolicy policy;

    if (cfg.recovery == "heartbeat") {
        recovery_type = script::RecoveryType::HEARTBEAT;
        script::HeartbeatPolicy hp;
        hp.interval = cfg.heartbeat_interval;
        hp.recovery_pubkey_hash.resize(20, 0);
        policy = hp;
    } else if (cfg.recovery == "emission") {
        recovery_type = script::RecoveryType::EMISSION;
        script::EmissionPolicy ep;
        ep.inactivity_period = 200000;
        policy = ep;
    } else {
        fprintf(stderr, "Error: unsupported recovery type '%s'\n", cfg.recovery.c_str());
        return 1;
    }

    // 2. Create the wallet.
    auto result = wallet::CWallet::create(
        cfg.path, "default", network, recovery_type, policy);

    if (result.is_err()) {
        fprintf(stderr, "Error: failed to create wallet: %s\n", result.error().c_str());
        return 1;
    }

    auto& w = result.value();

    // 3. Encrypt if passphrase given.
    if (!cfg.passphrase.empty()) {
        auto enc_result = w->encrypt_wallet(cfg.passphrase);
        if (enc_result.is_err()) {
            fprintf(stderr, "Warning: encryption failed: %s\n", enc_result.error().c_str());
        } else {
            printf("Wallet encrypted.\n");
        }
    }

    // 4. Display mnemonic for backup.
    auto mnemonic_result = w->get_mnemonic();
    if (mnemonic_result.is_ok()) {
        printf("\nWallet created successfully at: %s\n", cfg.path.c_str());
        printf("Network: %s\n", cfg.network.c_str());
        printf("Recovery type: %s\n", cfg.recovery.c_str());
        printf("\n*** BACKUP YOUR MNEMONIC ***\n");
        printf("%s\n", mnemonic_result.value().c_str());
        printf("***************************\n\n");
    } else {
        printf("Wallet created at: %s\n", cfg.path.c_str());
    }

    // 5. Generate first address.
    auto addr_result = w->get_new_address("default");
    if (addr_result.is_ok()) {
        printf("First receive address: %s\n", addr_result.value().c_str());
    }

    auto save_result = w->save();
    if (save_result.is_err()) {
        fprintf(stderr, "Warning: failed to save wallet: %s\n", save_result.error().c_str());
    }

    return 0;
}

// ---------------------------------------------------------------------------
// cmd_info
// ---------------------------------------------------------------------------
// Loads and displays wallet metadata: name, encryption status, balance,
// and UTXO count.
// ---------------------------------------------------------------------------
static int cmd_info(const WalletToolConfig& cfg)
{
    if (cfg.path.empty()) {
        fprintf(stderr, "Error: wallet path required\n");
        return 1;
    }

    auto result = wallet::CWallet::load(cfg.path);
    if (result.is_err()) {
        fprintf(stderr, "Error: failed to load wallet: %s\n", result.error().c_str());
        return 1;
    }

    auto& w = result.value();
    printf("Wallet: %s\n", w->name().c_str());
    printf("Path: %s\n", cfg.path.c_str());
    printf("Encrypted: %s\n", w->is_encrypted() ? "yes" : "no");
    printf("Locked: %s\n", w->is_locked() ? "yes" : "no");

    auto balance = w->get_balance(0);
    printf("Confirmed balance: %s RNT\n",
           primitives::FormatMoney(balance.confirmed).c_str());
    printf("Unconfirmed balance: %s RNT\n",
           primitives::FormatMoney(balance.unconfirmed).c_str());

    auto coins = w->get_unspent_coins();
    printf("UTXOs: %zu\n", coins.size());

    // 1. Show receive addresses.
    auto addrs = w->address_manager().get_receive_addresses();
    if (!addrs.empty()) {
        printf("\nReceive addresses:\n");
        for (const auto& a : addrs) {
            printf("  %s", a.address.c_str());
            if (!a.label.empty()) printf("  (%s)", a.label.c_str());
            if (a.is_used) printf("  [used]");
            printf("\n");
        }
    }

    return 0;
}

// ---------------------------------------------------------------------------
// cmd_dump
// ---------------------------------------------------------------------------
// Dumps wallet private key material (mnemonic).  Requires unlock if the
// wallet is encrypted.
// ---------------------------------------------------------------------------
static int cmd_dump(const WalletToolConfig& cfg)
{
    if (cfg.path.empty()) {
        fprintf(stderr, "Error: wallet path required\n");
        return 1;
    }

    auto result = wallet::CWallet::load(cfg.path);
    if (result.is_err()) {
        fprintf(stderr, "Error: failed to load wallet: %s\n", result.error().c_str());
        return 1;
    }

    auto& w = result.value();

    // 1. Unlock if encrypted.
    if (w->is_encrypted() && w->is_locked()) {
        if (cfg.passphrase.empty()) {
            fprintf(stderr, "Error: wallet is encrypted, provide -passphrase=...\n");
            return 1;
        }
        auto unlock_result = w->unlock(cfg.passphrase);
        if (unlock_result.is_err()) {
            fprintf(stderr, "Error: failed to unlock: %s\n", unlock_result.error().c_str());
            return 1;
        }
    }

    // 2. Dump mnemonic.
    auto mnemonic_result = w->get_mnemonic();
    if (mnemonic_result.is_ok()) {
        printf("# Wallet dump: %s\n", w->name().c_str());
        printf("# WARNING: This contains private key material!\n\n");
        printf("mnemonic: %s\n\n", mnemonic_result.value().c_str());
    } else {
        fprintf(stderr, "Warning: could not retrieve mnemonic: %s\n",
                mnemonic_result.error().c_str());
    }

    return 0;
}

// ---------------------------------------------------------------------------
// cmd_encrypt
// ---------------------------------------------------------------------------
// Encrypts an unencrypted wallet with the provided passphrase.
// ---------------------------------------------------------------------------
static int cmd_encrypt(const WalletToolConfig& cfg)
{
    if (cfg.path.empty()) {
        fprintf(stderr, "Error: wallet path required\n");
        return 1;
    }
    if (cfg.passphrase.empty()) {
        fprintf(stderr, "Error: -passphrase=... required for encryption\n");
        return 1;
    }

    auto result = wallet::CWallet::load(cfg.path);
    if (result.is_err()) {
        fprintf(stderr, "Error: failed to load wallet: %s\n", result.error().c_str());
        return 1;
    }

    auto& w = result.value();
    if (w->is_encrypted()) {
        fprintf(stderr, "Error: wallet is already encrypted\n");
        return 1;
    }

    auto enc_result = w->encrypt_wallet(cfg.passphrase);
    if (enc_result.is_err()) {
        fprintf(stderr, "Error: encryption failed: %s\n", enc_result.error().c_str());
        return 1;
    }

    auto save_result = w->save();
    if (save_result.is_err()) {
        fprintf(stderr, "Warning: failed to save: %s\n", save_result.error().c_str());
    }

    printf("Wallet encrypted successfully.\n");
    return 0;
}

// ---------------------------------------------------------------------------
// cmd_mnemonic
// ---------------------------------------------------------------------------
// Generates and prints a 24-word BIP39 mnemonic (no wallet file created).
// ---------------------------------------------------------------------------
static int cmd_mnemonic()
{
    auto result = crypto::generate_mnemonic(24);
    if (result.is_err()) {
        fprintf(stderr, "Error: failed to generate mnemonic: %s\n", result.error().c_str());
        return 1;
    }
    printf("%s\n", result.value().c_str());
    return 0;
}

// ===========================================================================
//  Main
// ===========================================================================

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
// Entry point for rnet-wallet-tool.  Parses arguments and dispatches to the
// appropriate wallet subcommand.
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc < 2) {
        print_usage();
        return 1;
    }

    WalletToolConfig cfg = parse_args(argc, argv);

    if (cfg.command.empty()) {
        print_usage();
        return 1;
    }

    if (cfg.command == "create")    return cmd_create(cfg);
    if (cfg.command == "info")      return cmd_info(cfg);
    if (cfg.command == "dump")      return cmd_dump(cfg);
    if (cfg.command == "encrypt")   return cmd_encrypt(cfg);
    if (cfg.command == "mnemonic")  return cmd_mnemonic();

    fprintf(stderr, "Unknown command: %s\n", cfg.command.c_str());
    print_usage();
    return 1;
}

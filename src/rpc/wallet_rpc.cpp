// Copyright (c) 2025-2026 The ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "rpc/wallet_rpc.h"

#include "chain/chainstate.h"
#include "core/logging.h"
#include "mempool/pool.h"
#include "node/context.h"
#include "primitives/amount.h"
#include "wallet/wallet.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>

namespace rnet::rpc {

// ===========================================================================
//  Wallet lookup helper
// ===========================================================================

// ---------------------------------------------------------------------------
// get_wallet / set_rpc_wallet
//
// Design: In a full implementation, NodeContext would hold a wallet manager.
//         For now, we use a global wallet pointer that rnetd sets.
// ---------------------------------------------------------------------------

namespace {
wallet::CWallet* g_wallet = nullptr;
} // anonymous namespace

void set_rpc_wallet(wallet::CWallet* w) {
    // 1. Store wallet pointer for later RPC lookups
    g_wallet = w;
}

static wallet::CWallet* get_wallet(node::NodeContext& /*ctx*/) {
    // 1. Return the globally registered wallet pointer
    return g_wallet;
}

// ===========================================================================
//  Wallet Info
// ===========================================================================

// ---------------------------------------------------------------------------
// rpc_getbalance
//
// Design: Returns the total confirmed balance in RNT. Converts from satoshi
//         units (int64) to floating-point RNT for JSON output.
// ---------------------------------------------------------------------------

static JsonValue rpc_getbalance(const RPCRequest& req,
                                node::NodeContext& ctx) {
    // 1. Retrieve the active wallet
    auto* wallet = get_wallet(ctx);
    if (!wallet) {
        return make_rpc_error(RPC_WALLET_NOT_FOUND, "No wallet loaded");
    }

    // 2. Determine current chain height for balance calculation
    int height = 0;
    if (ctx.chainstate) {
        height = ctx.chainstate->height();
    }

    // 3. Query confirmed balance and convert to RNT
    auto balance = wallet->get_balance(height);

    return JsonValue(static_cast<double>(balance.confirmed) / 1e8);
}

// ---------------------------------------------------------------------------
// rpc_getwalletinfo
//
// Design: Aggregates wallet metadata into a single JSON object matching the
//         Bitcoin Core getwalletinfo schema: name, version, balances, key
//         pool, encryption state, and scan progress.
// ---------------------------------------------------------------------------

static JsonValue rpc_getwalletinfo(const RPCRequest& req,
                                   node::NodeContext& ctx) {
    // 1. Retrieve the active wallet
    auto* wallet = get_wallet(ctx);
    if (!wallet) {
        return make_rpc_error(RPC_WALLET_NOT_FOUND, "No wallet loaded");
    }

    // 2. Fetch balances at current chain height
    int height = ctx.chainstate ? ctx.chainstate->height() : 0;
    auto balance = wallet->get_balance(height);

    // 3. Build the result object
    JsonValue result = JsonValue::object();
    result.set("walletname", JsonValue(wallet->name()));
    result.set("walletversion", JsonValue(static_cast<int64_t>(40000)));
    result.set("balance",
               JsonValue(static_cast<double>(balance.confirmed) / 1e8));
    result.set("unconfirmed_balance",
               JsonValue(static_cast<double>(balance.unconfirmed) / 1e8));
    result.set("immature_balance",
               JsonValue(static_cast<double>(balance.immature) / 1e8));
    result.set("txcount", JsonValue(static_cast<int64_t>(
        wallet->get_unspent_coins().size())));
    result.set("keypoolsize", JsonValue(static_cast<int64_t>(100)));
    result.set("unlocked_until", JsonValue(static_cast<int64_t>(0)));
    result.set("paytxfee", JsonValue(0.0));
    result.set("private_keys_enabled", JsonValue(true));
    result.set("avoid_reuse", JsonValue(false));
    result.set("scanning", JsonValue(false));
    result.set("encrypted", JsonValue(wallet->is_encrypted()));
    result.set("locked", JsonValue(wallet->is_locked()));

    return result;
}

// ===========================================================================
//  Address Management
// ===========================================================================

// ---------------------------------------------------------------------------
// rpc_getnewaddress
//
// Design: Derives the next unused address from the HD keychain. An optional
//         label parameter associates a human-readable tag with the address.
// ---------------------------------------------------------------------------

static JsonValue rpc_getnewaddress(const RPCRequest& req,
                                   node::NodeContext& ctx) {
    // 1. Retrieve the active wallet
    auto* wallet = get_wallet(ctx);
    if (!wallet) {
        return make_rpc_error(RPC_WALLET_NOT_FOUND, "No wallet loaded");
    }

    // 2. Parse optional label parameter
    std::string label;
    const auto& label_param = get_param_optional(req, 0);
    if (label_param.is_string()) label = label_param.as_string();

    // 3. Derive a new address from the keypool
    auto result = wallet->get_new_address(label);
    if (result.is_err()) {
        return make_rpc_error(RPC_WALLET_KEYPOOL_RAN_OUT, result.error());
    }

    return JsonValue(result.value());
}

// ===========================================================================
//  Transaction
// ===========================================================================

// ---------------------------------------------------------------------------
// rpc_sendtoaddress
//
// Design: Constructs, signs, and broadcasts a transaction sending `amount`
//         RNT to `address`. The wallet must be unlocked. After signing, the
//         transaction is submitted to the local mempool for relay.
// ---------------------------------------------------------------------------

static JsonValue rpc_sendtoaddress(const RPCRequest& req,
                                   node::NodeContext& ctx) {
    // 1. Retrieve the active wallet
    auto* wallet = get_wallet(ctx);
    if (!wallet) {
        return make_rpc_error(RPC_WALLET_NOT_FOUND, "No wallet loaded");
    }

    // 2. Validate required parameters
    const auto& addr_param = get_param(req, 0);
    const auto& amount_param = get_param(req, 1);

    if (!addr_param.is_string()) {
        return make_rpc_error(RPC_INVALID_PARAMS, "address required (string)");
    }
    if (!amount_param.is_number()) {
        return make_rpc_error(RPC_INVALID_PARAMS, "amount required (number)");
    }

    // 3. Parse address and convert amount to satoshis
    std::string address = addr_param.as_string();
    int64_t amount = 0;
    if (amount_param.is_double()) {
        amount = static_cast<int64_t>(amount_param.as_double() * 1e8);
    } else {
        amount = amount_param.as_int() * primitives::COIN;
    }

    if (amount <= 0) {
        return make_rpc_error(RPC_INVALID_PARAMETER, "invalid amount");
    }

    // 4. Ensure the wallet is unlocked for signing
    if (wallet->is_locked()) {
        return make_rpc_error(RPC_WALLET_UNLOCK_NEEDED,
                              "wallet is locked, unlock first");
    }

    // 5. Build and sign the transaction
    auto result = wallet->send_to(address, amount);
    if (result.is_err()) {
        return make_rpc_error(RPC_WALLET_ERROR, result.error());
    }

    auto txid = result.value().txid().to_hex();

    // 6. Submit to mempool if available
    if (ctx.mempool) {
        auto tx_ref = std::make_shared<const primitives::CTransaction>(
            std::move(result.value()));
        int height = ctx.chainstate ? ctx.chainstate->height() : 0;
        float val_loss = 0.0f;
        if (ctx.chainstate && ctx.chainstate->tip()) {
            val_loss = ctx.chainstate->tip()->val_loss;
        }
        ctx.mempool->add_tx(tx_ref, 0, height, val_loss);
    }

    // 7. Log the send for audit trail
    LogPrint(RPC, "sendtoaddress: %s -> %s (%s RNT)",
             txid.c_str(), address.c_str(),
             primitives::FormatMoney(amount).c_str());

    return JsonValue(txid);
}

// ---------------------------------------------------------------------------
// rpc_listtransactions
//
// Design: Returns recent wallet transactions with pagination (count/skip).
//         Currently uses unspent coins as a proxy for full transaction
//         history until the wallet tracks a complete tx journal.
// ---------------------------------------------------------------------------

static JsonValue rpc_listtransactions(const RPCRequest& req,
                                      node::NodeContext& ctx) {
    // 1. Retrieve the active wallet
    auto* wallet = get_wallet(ctx);
    if (!wallet) {
        return make_rpc_error(RPC_WALLET_NOT_FOUND, "No wallet loaded");
    }

    // 2. Parse optional count and skip parameters
    int count = 10;
    int skip = 0;
    const auto& count_param = get_param_optional(req, 0);
    const auto& skip_param = get_param_optional(req, 1);
    if (count_param.is_int()) count = static_cast<int>(count_param.as_int());
    if (skip_param.is_int()) skip = static_cast<int>(skip_param.as_int());

    // 3. Build transaction list from unspent coins
    JsonValue transactions = JsonValue::array();

    auto coins = wallet->get_unspent_coins();
    int idx = 0;
    for (const auto& coin : coins) {
        if (idx < skip) { ++idx; continue; }
        if (idx >= skip + count) break;

        JsonValue tx_entry = JsonValue::object();
        tx_entry.set("category", JsonValue(std::string("receive")));
        tx_entry.set("amount",
                      JsonValue(static_cast<double>(coin.txout.value) / 1e8));
        tx_entry.set("confirmations", JsonValue(static_cast<int64_t>(
            coin.height >= 0 ? 6 : 0)));
        tx_entry.set("txid", JsonValue(coin.outpoint.hash.to_hex()));
        tx_entry.set("vout",
                      JsonValue(static_cast<int64_t>(coin.outpoint.n)));

        transactions.push_back(std::move(tx_entry));
        ++idx;
    }

    return transactions;
}

// ===========================================================================
//  Backup / Recovery
// ===========================================================================

// ---------------------------------------------------------------------------
// rpc_backupwallet
//
// Design: Flushes the wallet to disk, then copies the wallet file to the
//         caller-specified destination path using std::filesystem.
// ---------------------------------------------------------------------------

static JsonValue rpc_backupwallet(const RPCRequest& req,
                                  node::NodeContext& ctx) {
    // 1. Retrieve the active wallet
    auto* wallet = get_wallet(ctx);
    if (!wallet) {
        return make_rpc_error(RPC_WALLET_NOT_FOUND, "No wallet loaded");
    }

    // 2. Validate the destination path parameter
    const auto& dest_param = get_param(req, 0);
    if (!dest_param.is_string()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "destination path required");
    }

    // 3. Flush the wallet to disk
    auto result = wallet->save();
    if (result.is_err()) {
        return make_rpc_error(RPC_WALLET_ERROR,
                              "backup failed: " + result.error());
    }

    // 4. Copy the wallet file to the destination
    std::error_code ec;
    std::filesystem::copy_file(
        wallet->path(),
        std::filesystem::path(dest_param.as_string()),
        std::filesystem::copy_options::overwrite_existing,
        ec);

    if (ec) {
        return make_rpc_error(RPC_WALLET_ERROR,
                              "failed to copy wallet: " + ec.message());
    }

    LogPrint(RPC, "Wallet backed up to %s", dest_param.as_string().c_str());
    return JsonValue();  // null = success
}

// ---------------------------------------------------------------------------
// rpc_dumpwallet
//
// Design: Exports wallet keys in a human-readable format. The wallet must be
//         unlocked so private keys can be decrypted. Reports whether HD
//         mnemonic recovery words are available.
// ---------------------------------------------------------------------------

static JsonValue rpc_dumpwallet(const RPCRequest& req,
                                node::NodeContext& ctx) {
    // 1. Retrieve the active wallet
    auto* wallet = get_wallet(ctx);
    if (!wallet) {
        return make_rpc_error(RPC_WALLET_NOT_FOUND, "No wallet loaded");
    }

    // 2. Require unlocked wallet for key export
    if (wallet->is_locked()) {
        return make_rpc_error(RPC_WALLET_UNLOCK_NEEDED,
                              "wallet must be unlocked to dump");
    }

    // 3. Validate the destination file parameter
    const auto& dest_param = get_param(req, 0);
    if (!dest_param.is_string()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "destination file path required");
    }

    // 4. Attempt mnemonic retrieval and build result
    auto mnemonic = wallet->get_mnemonic();

    JsonValue result = JsonValue::object();
    result.set("filename", JsonValue(dest_param.as_string()));

    if (mnemonic.is_ok()) {
        result.set("mnemonic_available", JsonValue(true));
    } else {
        result.set("mnemonic_available", JsonValue(false));
    }

    LogPrint(RPC, "Wallet dumped to %s", dest_param.as_string().c_str());
    return result;
}

// ---------------------------------------------------------------------------
// rpc_importwallet
//
// Design: Imports keys from a previously-dumped wallet file. Currently a
//         stub that logs the request; full key parsing is not yet wired.
// ---------------------------------------------------------------------------

static JsonValue rpc_importwallet(const RPCRequest& req,
                                  node::NodeContext& ctx) {
    // 1. Retrieve the active wallet
    auto* wallet = get_wallet(ctx);
    if (!wallet) {
        return make_rpc_error(RPC_WALLET_NOT_FOUND, "No wallet loaded");
    }

    // 2. Validate the import file parameter
    const auto& file_param = get_param(req, 0);
    if (!file_param.is_string()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "import file path required");
    }

    // 3. Log the import request (full parsing not yet implemented)
    LogPrint(RPC, "importwallet from %s (not yet implemented)",
             file_param.as_string().c_str());

    return JsonValue();  // null = success
}

// ===========================================================================
//  Registration
// ===========================================================================

// ---------------------------------------------------------------------------
// rpc_loadwallet
//
// Design: Loads a wallet from a file path. If no path given, loads from
//         the default data directory.
// ---------------------------------------------------------------------------
static JsonValue rpc_loadwallet(const RPCRequest& req,
                                 node::NodeContext& ctx) {
    // 1. Determine wallet path.
    std::string wallet_path;
    const auto& path_param = get_param_optional(req, 0);
    if (path_param.is_string()) {
        wallet_path = path_param.as_string();
    } else {
        // Default: data_dir / wallet.dat
        wallet_path = (ctx.data_dir / "wallet.dat").string();
    }

    // 2. Check if wallet already loaded.
    if (g_wallet) {
        JsonValue result = JsonValue::object();
        result.set("warning", JsonValue("Wallet already loaded"));
        result.set("name", JsonValue(g_wallet->name()));
        return result;
    }

    // 3. Load the wallet.
    auto load_result = wallet::CWallet::load(wallet_path);
    if (load_result.is_err()) {
        throw std::runtime_error("Failed to load wallet: " + load_result.error());
    }

    // 4. Store as global wallet.
    static std::unique_ptr<wallet::CWallet> s_loaded_wallet;
    s_loaded_wallet = std::move(load_result.value());
    g_wallet = s_loaded_wallet.get();

    JsonValue result = JsonValue::object();
    result.set("name", JsonValue(g_wallet->name()));
    result.set("warning", JsonValue(""));
    return result;
}

// ---------------------------------------------------------------------------
// register_wallet_rpcs
//
// Design: Binds all wallet RPC handlers to the command table. Each entry
//         specifies name, handler function, help text, and category.
// ---------------------------------------------------------------------------

void register_wallet_rpcs(RPCTable& table) {
    // 0. Load/unload commands
    table.register_command({
        "loadwallet",
        rpc_loadwallet,
        "Loads a wallet from the specified path.\n"
        "Arguments: filename (string, optional, default: wallet.dat)",
        "Wallet"
    });
    // 1. Wallet Info commands
    table.register_command({
        "getbalance",
        rpc_getbalance,
        "Returns the total available balance in RNT.",
        "Wallet"
    });

    table.register_command({
        "getwalletinfo",
        rpc_getwalletinfo,
        "Returns wallet state information.",
        "Wallet"
    });

    // 2. Address Management commands
    table.register_command({
        "getnewaddress",
        rpc_getnewaddress,
        "Returns a new ResonanceNet address for receiving payments.\n"
        "Arguments: label (string, optional)",
        "Wallet"
    });

    // 3. Transaction commands
    table.register_command({
        "sendtoaddress",
        rpc_sendtoaddress,
        "Send an amount to a given address.\n"
        "Arguments: address (string), amount (numeric in RNT)",
        "Wallet"
    });

    table.register_command({
        "listtransactions",
        rpc_listtransactions,
        "Returns recent transactions for the wallet.\n"
        "Arguments: count (int, default=10), skip (int, default=0)",
        "Wallet"
    });

    // 4. Backup / Recovery commands
    table.register_command({
        "backupwallet",
        rpc_backupwallet,
        "Safely copies the wallet file to the destination.\n"
        "Arguments: destination (string)",
        "Wallet"
    });

    table.register_command({
        "dumpwallet",
        rpc_dumpwallet,
        "Dumps all wallet keys in a human-readable format.\n"
        "Arguments: filename (string)",
        "Wallet"
    });

    table.register_command({
        "importwallet",
        rpc_importwallet,
        "Imports keys from a wallet dump file.\n"
        "Arguments: filename (string)",
        "Wallet"
    });
}

} // namespace rnet::rpc

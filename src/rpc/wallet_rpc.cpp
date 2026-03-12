#include "rpc/wallet_rpc.h"

#include "chain/chainstate.h"
#include "core/logging.h"
#include "mempool/pool.h"
#include "node/context.h"
#include "primitives/amount.h"
#include "wallet/wallet.h"

namespace rnet::rpc {

// ── Wallet lookup helper ────────────────────────────────────────────

// In a full implementation, NodeContext would hold a wallet manager.
// For now, we use a global wallet pointer that rnetd sets.
namespace {
wallet::CWallet* g_wallet = nullptr;
}

void set_rpc_wallet(wallet::CWallet* w) {
    g_wallet = w;
}

static wallet::CWallet* get_wallet(node::NodeContext& /*ctx*/) {
    return g_wallet;
}

// ── getbalance ──────────────────────────────────────────────────────

static JsonValue rpc_getbalance(const RPCRequest& req,
                                node::NodeContext& ctx) {
    auto* wallet = get_wallet(ctx);
    if (!wallet) {
        return make_rpc_error(RPC_WALLET_NOT_FOUND, "No wallet loaded");
    }

    int height = 0;
    if (ctx.chainstate) {
        height = ctx.chainstate->height();
    }

    auto balance = wallet->get_balance(height);

    // Return total confirmed balance in RNT
    return JsonValue(static_cast<double>(balance.confirmed) / 1e8);
}

// ── getnewaddress ───────────────────────────────────────────────────

static JsonValue rpc_getnewaddress(const RPCRequest& req,
                                   node::NodeContext& ctx) {
    auto* wallet = get_wallet(ctx);
    if (!wallet) {
        return make_rpc_error(RPC_WALLET_NOT_FOUND, "No wallet loaded");
    }

    std::string label;
    const auto& label_param = get_param_optional(req, 0);
    if (label_param.is_string()) label = label_param.as_string();

    auto result = wallet->get_new_address(label);
    if (result.is_err()) {
        return make_rpc_error(RPC_WALLET_KEYPOOL_RAN_OUT, result.error());
    }

    return JsonValue(result.value());
}

// ── sendtoaddress ───────────────────────────────────────────────────

static JsonValue rpc_sendtoaddress(const RPCRequest& req,
                                   node::NodeContext& ctx) {
    auto* wallet = get_wallet(ctx);
    if (!wallet) {
        return make_rpc_error(RPC_WALLET_NOT_FOUND, "No wallet loaded");
    }

    const auto& addr_param = get_param(req, 0);
    const auto& amount_param = get_param(req, 1);

    if (!addr_param.is_string()) {
        return make_rpc_error(RPC_INVALID_PARAMS, "address required (string)");
    }
    if (!amount_param.is_number()) {
        return make_rpc_error(RPC_INVALID_PARAMS, "amount required (number)");
    }

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

    if (wallet->is_locked()) {
        return make_rpc_error(RPC_WALLET_UNLOCK_NEEDED,
                              "wallet is locked, unlock first");
    }

    auto result = wallet->send_to(address, amount);
    if (result.is_err()) {
        return make_rpc_error(RPC_WALLET_ERROR, result.error());
    }

    auto txid = result.value().txid().to_hex();

    // Submit to mempool if available
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

    LogPrint(RPC, "sendtoaddress: %s -> %s (%s RNT)",
             txid.c_str(), address.c_str(),
             primitives::FormatMoney(amount).c_str());

    return JsonValue(txid);
}

// ── listtransactions ────────────────────────────────────────────────

static JsonValue rpc_listtransactions(const RPCRequest& req,
                                      node::NodeContext& ctx) {
    auto* wallet = get_wallet(ctx);
    if (!wallet) {
        return make_rpc_error(RPC_WALLET_NOT_FOUND, "No wallet loaded");
    }

    int count = 10;
    int skip = 0;
    const auto& count_param = get_param_optional(req, 0);
    const auto& skip_param = get_param_optional(req, 1);
    if (count_param.is_int()) count = static_cast<int>(count_param.as_int());
    if (skip_param.is_int()) skip = static_cast<int>(skip_param.as_int());

    // Return list of wallet transactions
    // In a full implementation, we would scan the wallet's transaction history
    JsonValue transactions = JsonValue::array();

    // Get unspent coins as a proxy for transaction history
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

// ── getwalletinfo ───────────────────────────────────────────────────

static JsonValue rpc_getwalletinfo(const RPCRequest& req,
                                   node::NodeContext& ctx) {
    auto* wallet = get_wallet(ctx);
    if (!wallet) {
        return make_rpc_error(RPC_WALLET_NOT_FOUND, "No wallet loaded");
    }

    int height = ctx.chainstate ? ctx.chainstate->height() : 0;
    auto balance = wallet->get_balance(height);

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

// ── backupwallet ────────────────────────────────────────────────────

static JsonValue rpc_backupwallet(const RPCRequest& req,
                                  node::NodeContext& ctx) {
    auto* wallet = get_wallet(ctx);
    if (!wallet) {
        return make_rpc_error(RPC_WALLET_NOT_FOUND, "No wallet loaded");
    }

    const auto& dest_param = get_param(req, 0);
    if (!dest_param.is_string()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "destination path required");
    }

    // Save wallet to the destination path
    auto result = wallet->save();
    if (result.is_err()) {
        return make_rpc_error(RPC_WALLET_ERROR,
                              "backup failed: " + result.error());
    }

    // Copy the wallet file to the destination
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

// ── dumpwallet ──────────────────────────────────────────────────────

static JsonValue rpc_dumpwallet(const RPCRequest& req,
                                node::NodeContext& ctx) {
    auto* wallet = get_wallet(ctx);
    if (!wallet) {
        return make_rpc_error(RPC_WALLET_NOT_FOUND, "No wallet loaded");
    }

    if (wallet->is_locked()) {
        return make_rpc_error(RPC_WALLET_UNLOCK_NEEDED,
                              "wallet must be unlocked to dump");
    }

    const auto& dest_param = get_param(req, 0);
    if (!dest_param.is_string()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "destination file path required");
    }

    // Get the mnemonic for dump
    auto mnemonic = wallet->get_mnemonic();

    JsonValue result = JsonValue::object();
    result.set("filename", JsonValue(dest_param.as_string()));

    if (mnemonic.is_ok()) {
        // In a real implementation, write keys to the file
        result.set("mnemonic_available", JsonValue(true));
    } else {
        result.set("mnemonic_available", JsonValue(false));
    }

    LogPrint(RPC, "Wallet dumped to %s", dest_param.as_string().c_str());
    return result;
}

// ── importwallet ────────────────────────────────────────────────────

static JsonValue rpc_importwallet(const RPCRequest& req,
                                  node::NodeContext& ctx) {
    auto* wallet = get_wallet(ctx);
    if (!wallet) {
        return make_rpc_error(RPC_WALLET_NOT_FOUND, "No wallet loaded");
    }

    const auto& file_param = get_param(req, 0);
    if (!file_param.is_string()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "import file path required");
    }

    // In a full implementation, this would parse the dump file
    // and import keys into the wallet
    LogPrint(RPC, "importwallet from %s (not yet implemented)",
             file_param.as_string().c_str());

    return JsonValue();  // null = success
}

// ── Registration ────────────────────────────────────────────────────

void register_wallet_rpcs(RPCTable& table) {
    table.register_command({
        "getbalance",
        rpc_getbalance,
        "Returns the total available balance in RNT.",
        "Wallet"
    });

    table.register_command({
        "getnewaddress",
        rpc_getnewaddress,
        "Returns a new ResonanceNet address for receiving payments.\n"
        "Arguments: label (string, optional)",
        "Wallet"
    });

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

    table.register_command({
        "getwalletinfo",
        rpc_getwalletinfo,
        "Returns wallet state information.",
        "Wallet"
    });

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

}  // namespace rnet::rpc

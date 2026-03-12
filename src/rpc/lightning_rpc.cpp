#include "rpc/lightning_rpc.h"

#include "chain/chainstate.h"
#include "core/logging.h"
#include "node/context.h"

namespace rnet::rpc {

// ── Lightning manager access ────────────────────────────────────────
// In a full implementation, NodeContext would hold a LightningManager.
// For now, these RPCs return placeholder data.

// ── openchannel ─────────────────────────────────────────────────────

static JsonValue rpc_openchannel(const RPCRequest& req,
                                 node::NodeContext& ctx) {
    const auto& node_id_param = get_param(req, 0);
    const auto& capacity_param = get_param(req, 1);

    if (!node_id_param.is_string()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "node_id required (hex pubkey)");
    }
    if (!capacity_param.is_number()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "capacity required (amount in RNT)");
    }

    std::string node_id = node_id_param.as_string();
    int64_t capacity = 0;
    if (capacity_param.is_double()) {
        capacity = static_cast<int64_t>(capacity_param.as_double() * 1e8);
    } else {
        capacity = capacity_param.as_int();
    }

    int64_t push_amount = 0;
    const auto& push_param = get_param_optional(req, 2);
    if (push_param.is_number()) {
        if (push_param.is_double()) {
            push_amount = static_cast<int64_t>(push_param.as_double() * 1e8);
        } else {
            push_amount = push_param.as_int();
        }
    }

    LogPrint(LIGHTNING, "openchannel: node=%s capacity=%lld push=%lld",
             node_id.c_str(), static_cast<long long>(capacity),
             static_cast<long long>(push_amount));

    // Placeholder: would interact with LightningManager to open a channel
    JsonValue result = JsonValue::object();
    result.set("channel_id", JsonValue(std::string(64, '0')));
    result.set("node_id", JsonValue(node_id));
    result.set("capacity", JsonValue(capacity));
    result.set("push_amount", JsonValue(push_amount));
    result.set("status", JsonValue(std::string("pending_open")));
    result.set("funding_txid", JsonValue(std::string(64, '0')));

    return result;
}

// ── closechannel ────────────────────────────────────────────────────

static JsonValue rpc_closechannel(const RPCRequest& req,
                                  node::NodeContext& ctx) {
    const auto& channel_id_param = get_param(req, 0);
    if (!channel_id_param.is_string()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "channel_id required (hex string)");
    }

    bool force = false;
    const auto& force_param = get_param_optional(req, 1);
    if (force_param.is_bool()) force = force_param.as_bool();

    std::string channel_id = channel_id_param.as_string();

    LogPrint(LIGHTNING, "closechannel: %s force=%d",
             channel_id.c_str(), force ? 1 : 0);

    JsonValue result = JsonValue::object();
    result.set("channel_id", JsonValue(channel_id));
    result.set("status", JsonValue(std::string(force ? "force_closing" : "closing")));
    result.set("closing_txid", JsonValue(std::string(64, '0')));

    return result;
}

// ── pay ─────────────────────────────────────────────────────────────

static JsonValue rpc_pay(const RPCRequest& req,
                         node::NodeContext& ctx) {
    const auto& invoice_param = get_param(req, 0);
    if (!invoice_param.is_string()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "invoice string required (BOLT11 format)");
    }

    std::string invoice = invoice_param.as_string();

    // Optional amount override (for zero-amount invoices)
    int64_t amount_override = 0;
    const auto& amount_param = get_param_optional(req, 1);
    if (amount_param.is_number()) {
        if (amount_param.is_double()) {
            amount_override = static_cast<int64_t>(amount_param.as_double() * 1e8);
        } else {
            amount_override = amount_param.as_int();
        }
    }

    LogPrint(LIGHTNING, "pay: invoice=%s amount_override=%lld",
             invoice.c_str(), static_cast<long long>(amount_override));

    // Placeholder response
    JsonValue result = JsonValue::object();
    result.set("payment_hash", JsonValue(std::string(64, '0')));
    result.set("payment_preimage", JsonValue(std::string(64, '0')));
    result.set("status", JsonValue(std::string("pending")));
    result.set("amount", JsonValue(amount_override));
    result.set("fee", JsonValue(static_cast<int64_t>(0)));
    result.set("hops", JsonValue(static_cast<int64_t>(0)));

    return result;
}

// ── addinvoice ──────────────────────────────────────────────────────

static JsonValue rpc_addinvoice(const RPCRequest& req,
                                node::NodeContext& ctx) {
    const auto& amount_param = get_param(req, 0);
    if (!amount_param.is_number()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "amount required (in resonances)");
    }

    int64_t amount = 0;
    if (amount_param.is_double()) {
        amount = static_cast<int64_t>(amount_param.as_double() * 1e8);
    } else {
        amount = amount_param.as_int();
    }

    std::string description;
    const auto& desc_param = get_param_optional(req, 1);
    if (desc_param.is_string()) description = desc_param.as_string();

    int64_t expiry = 3600;  // 1 hour default
    const auto& expiry_param = get_param_optional(req, 2);
    if (expiry_param.is_int()) expiry = expiry_param.as_int();

    LogPrint(LIGHTNING, "addinvoice: amount=%lld desc=%s",
             static_cast<long long>(amount), description.c_str());

    // Placeholder response
    JsonValue result = JsonValue::object();
    result.set("payment_hash", JsonValue(std::string(64, '0')));
    result.set("payment_request",
               JsonValue(std::string("rnt1placeholder...")));
    result.set("amount", JsonValue(amount));
    result.set("description", JsonValue(description));
    result.set("expiry", JsonValue(expiry));
    result.set("add_index", JsonValue(static_cast<int64_t>(1)));

    return result;
}

// ── listchannels ────────────────────────────────────────────────────

static JsonValue rpc_listchannels(const RPCRequest& req,
                                  node::NodeContext& ctx) {
    // Return all channels
    JsonValue channels = JsonValue::array();

    // Placeholder: in full implementation, iterate LightningManager channels
    // For now, return empty array

    return channels;
}

// ── getlightninginfo ────────────────────────────────────────────────

static JsonValue rpc_getlightninginfo(const RPCRequest& req,
                                      node::NodeContext& ctx) {
    JsonValue result = JsonValue::object();

    result.set("version", JsonValue(std::string("0.1.0")));
    result.set("identity_pubkey", JsonValue(std::string(64, '0')));
    result.set("alias", JsonValue(std::string("rnet-lightning")));
    result.set("num_pending_channels", JsonValue(static_cast<int64_t>(0)));
    result.set("num_active_channels", JsonValue(static_cast<int64_t>(0)));
    result.set("num_inactive_channels", JsonValue(static_cast<int64_t>(0)));
    result.set("num_peers", JsonValue(static_cast<int64_t>(0)));
    result.set("block_height", JsonValue(static_cast<int64_t>(
        ctx.chainstate ? ctx.chainstate->height() : 0)));
    result.set("synced_to_chain", JsonValue(true));
    result.set("synced_to_graph", JsonValue(false));
    result.set("best_header_timestamp", JsonValue(static_cast<int64_t>(0)));

    // Network info
    result.set("network", JsonValue(ctx.network));
    result.set("port", JsonValue(static_cast<int64_t>(9556)));

    return result;
}

// ── Registration ────────────────────────────────────────────────────

void register_lightning_rpcs(RPCTable& table) {
    table.register_command({
        "openchannel",
        rpc_openchannel,
        "Open a new Lightning channel with a peer.\n"
        "Arguments: node_id (hex pubkey), capacity (RNT), push_amount (optional)",
        "Lightning"
    });

    table.register_command({
        "closechannel",
        rpc_closechannel,
        "Close a Lightning channel.\n"
        "Arguments: channel_id (hex), force (bool, optional)",
        "Lightning"
    });

    table.register_command({
        "pay",
        rpc_pay,
        "Pay a Lightning invoice.\n"
        "Arguments: invoice (BOLT11 string), amount (optional, for zero-amount invoices)",
        "Lightning"
    });

    table.register_command({
        "addinvoice",
        rpc_addinvoice,
        "Create a new Lightning invoice.\n"
        "Arguments: amount (resonances), description (optional), expiry (optional, seconds)",
        "Lightning"
    });

    table.register_command({
        "listchannels",
        rpc_listchannels,
        "List all Lightning channels.",
        "Lightning"
    });

    table.register_command({
        "getlightninginfo",
        rpc_getlightninginfo,
        "Returns Lightning Network node information.",
        "Lightning"
    });
}

}  // namespace rnet::rpc

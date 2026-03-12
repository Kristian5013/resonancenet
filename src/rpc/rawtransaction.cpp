#include "rpc/rawtransaction.h"

#include "chain/chainstate.h"
#include "core/logging.h"
#include "core/stream.h"
#include "mempool/pool.h"
#include "node/context.h"
#include "primitives/amount.h"
#include "primitives/transaction.h"

namespace rnet::rpc {

// ── getrawtransaction ───────────────────────────────────────────────

static JsonValue rpc_getrawtransaction(const RPCRequest& req,
                                       node::NodeContext& ctx) {
    const auto& txid_param = get_param(req, 0);
    if (!txid_param.is_string()) {
        return make_rpc_error(RPC_INVALID_PARAMS, "txid required (hex string)");
    }

    bool verbose = false;
    const auto& verb_param = get_param_optional(req, 1);
    if (verb_param.is_bool()) verbose = verb_param.as_bool();
    if (verb_param.is_int()) verbose = verb_param.as_int() != 0;

    auto txid = rnet::uint256::from_hex(txid_param.as_string());

    // Check mempool first
    primitives::CTransactionRef tx;
    if (ctx.mempool) {
        tx = ctx.mempool->get(txid);
    }

    // TODO: look up confirmed transactions in the block storage

    if (!tx) {
        return make_rpc_error(RPC_INVALID_ADDRESS_OR_KEY,
                              "No such mempool or blockchain transaction");
    }

    if (!verbose) {
        // Return raw serialized hex
        core::DataStream ss;
        tx->serialize(ss);
        auto sp = ss.span();
        return JsonValue(bytes_to_hex(sp.data(), sp.size()));
    }

    // Verbose: decode the transaction
    JsonValue result = JsonValue::object();
    result.set("txid", JsonValue(tx->txid().to_hex()));
    result.set("wtxid", JsonValue(tx->wtxid().to_hex()));
    result.set("version", JsonValue(static_cast<int64_t>(tx->version())));
    result.set("size", JsonValue(static_cast<int64_t>(tx->get_total_size())));
    result.set("vsize", JsonValue(static_cast<int64_t>(tx->get_virtual_size())));
    result.set("weight", JsonValue(static_cast<int64_t>(tx->get_weight())));
    result.set("locktime", JsonValue(static_cast<int64_t>(tx->locktime())));

    // Inputs
    JsonValue vin = JsonValue::array();
    for (size_t i = 0; i < tx->vin().size(); ++i) {
        const auto& input = tx->vin()[i];
        JsonValue in_obj = JsonValue::object();

        if (tx->is_coinbase()) {
            in_obj.set("coinbase", JsonValue(true));
        } else {
            in_obj.set("txid", JsonValue(input.prevout.hash.to_hex()));
            in_obj.set("vout",
                       JsonValue(static_cast<int64_t>(input.prevout.n)));
        }
        in_obj.set("sequence",
                    JsonValue(static_cast<int64_t>(input.sequence)));
        vin.push_back(std::move(in_obj));
    }
    result.set("vin", std::move(vin));

    // Outputs
    JsonValue vout = JsonValue::array();
    for (size_t i = 0; i < tx->vout().size(); ++i) {
        const auto& output = tx->vout()[i];
        JsonValue out_obj = JsonValue::object();
        out_obj.set("value",
                     JsonValue(static_cast<double>(output.value) / 1e8));
        out_obj.set("n", JsonValue(static_cast<int64_t>(i)));

        JsonValue script = JsonValue::object();
        script.set("hex", JsonValue(bytes_to_hex(
            output.script_pub_key.data(), output.script_pub_key.size())));
        out_obj.set("scriptPubKey", std::move(script));

        vout.push_back(std::move(out_obj));
    }
    result.set("vout", std::move(vout));

    return result;
}

// ── createrawtransaction ────────────────────────────────────────────

static JsonValue rpc_createrawtransaction(const RPCRequest& req,
                                          node::NodeContext& ctx) {
    const auto& inputs_param = get_param(req, 0);
    const auto& outputs_param = get_param(req, 1);

    if (!inputs_param.is_array()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "inputs array required as first argument");
    }
    if (!outputs_param.is_object() && !outputs_param.is_array()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "outputs required as second argument");
    }

    primitives::CMutableTransaction mtx;

    // Optional locktime
    const auto& locktime_param = get_param_optional(req, 2);
    if (locktime_param.is_int()) {
        mtx.locktime = static_cast<uint32_t>(locktime_param.as_int());
    }

    // Parse inputs
    for (size_t i = 0; i < inputs_param.size(); ++i) {
        const auto& inp = inputs_param[i];
        if (!inp.is_object()) continue;

        const auto& txid_val = inp["txid"];
        const auto& vout_val = inp["vout"];

        if (!txid_val.is_string() || !vout_val.is_int()) {
            return make_rpc_error(RPC_INVALID_PARAMETER,
                                  "each input must have txid and vout");
        }

        primitives::CTxIn txin;
        txin.prevout.hash = rnet::uint256::from_hex(txid_val.as_string());
        txin.prevout.n = static_cast<uint32_t>(vout_val.as_int());

        const auto& seq_val = inp["sequence"];
        if (seq_val.is_int()) {
            txin.sequence = static_cast<uint32_t>(seq_val.as_int());
        }

        mtx.vin.push_back(std::move(txin));
    }

    // Parse outputs — object format: {"address": amount, ...}
    if (outputs_param.is_object()) {
        for (const auto& [addr, amt] : outputs_param.as_object()) {
            primitives::CTxOut txout;
            if (amt.is_double()) {
                txout.value = static_cast<int64_t>(amt.as_double() * 1e8);
            } else if (amt.is_int()) {
                txout.value = amt.as_int() * primitives::COIN;
            } else {
                continue;
            }
            // In a real implementation, we would decode the address to
            // a scriptPubKey here. For now, store the address as a
            // placeholder script.
            // TODO: decode bech32/base58 address to script
            mtx.vout.push_back(std::move(txout));
        }
    }

    // Serialize and return hex
    auto data = mtx.serialize_with_witness();
    return JsonValue(bytes_to_hex(data.data(), data.size()));
}

// ── decoderawtransaction ────────────────────────────────────────────

static JsonValue rpc_decoderawtransaction(const RPCRequest& req,
                                          node::NodeContext& ctx) {
    const auto& hex_param = get_param(req, 0);
    if (!hex_param.is_string()) {
        return make_rpc_error(RPC_INVALID_PARAMS, "hex string required");
    }

    auto bytes = hex_to_bytes(hex_param.as_string());
    if (bytes.empty()) {
        return make_rpc_error(RPC_DESERIALIZATION_ERROR,
                              "invalid hex encoding");
    }

    core::DataStream ss(std::span<const uint8_t>(bytes.data(), bytes.size()));
    primitives::CTransaction tx;
    try {
        tx.unserialize(ss);
    } catch (...) {
        return make_rpc_error(RPC_DESERIALIZATION_ERROR,
                              "TX decode failed");
    }

    JsonValue result = JsonValue::object();
    result.set("txid", JsonValue(tx.txid().to_hex()));
    result.set("wtxid", JsonValue(tx.wtxid().to_hex()));
    result.set("version", JsonValue(static_cast<int64_t>(tx.version())));
    result.set("size", JsonValue(static_cast<int64_t>(tx.get_total_size())));
    result.set("vsize", JsonValue(static_cast<int64_t>(tx.get_virtual_size())));
    result.set("weight", JsonValue(static_cast<int64_t>(tx.get_weight())));
    result.set("locktime", JsonValue(static_cast<int64_t>(tx.locktime())));

    // Inputs
    JsonValue vin = JsonValue::array();
    for (size_t i = 0; i < tx.vin().size(); ++i) {
        const auto& input = tx.vin()[i];
        JsonValue in_obj = JsonValue::object();
        if (tx.is_coinbase()) {
            in_obj.set("coinbase", JsonValue(true));
        } else {
            in_obj.set("txid", JsonValue(input.prevout.hash.to_hex()));
            in_obj.set("vout",
                       JsonValue(static_cast<int64_t>(input.prevout.n)));
        }
        in_obj.set("sequence",
                    JsonValue(static_cast<int64_t>(input.sequence)));
        vin.push_back(std::move(in_obj));
    }
    result.set("vin", std::move(vin));

    // Outputs
    JsonValue vout = JsonValue::array();
    for (size_t i = 0; i < tx.vout().size(); ++i) {
        const auto& output = tx.vout()[i];
        JsonValue out_obj = JsonValue::object();
        out_obj.set("value",
                     JsonValue(static_cast<double>(output.value) / 1e8));
        out_obj.set("n", JsonValue(static_cast<int64_t>(i)));
        JsonValue script = JsonValue::object();
        script.set("hex", JsonValue(bytes_to_hex(
            output.script_pub_key.data(), output.script_pub_key.size())));
        out_obj.set("scriptPubKey", std::move(script));
        vout.push_back(std::move(out_obj));
    }
    result.set("vout", std::move(vout));

    return result;
}

// ── sendrawtransaction ──────────────────────────────────────────────

static JsonValue rpc_sendrawtransaction(const RPCRequest& req,
                                        node::NodeContext& ctx) {
    const auto& hex_param = get_param(req, 0);
    if (!hex_param.is_string()) {
        return make_rpc_error(RPC_INVALID_PARAMS, "hex string required");
    }

    auto bytes = hex_to_bytes(hex_param.as_string());
    if (bytes.empty()) {
        return make_rpc_error(RPC_DESERIALIZATION_ERROR,
                              "invalid hex encoding");
    }

    core::DataStream ss(std::span<const uint8_t>(bytes.data(), bytes.size()));
    auto tx = std::make_shared<primitives::CTransaction>();
    try {
        const_cast<primitives::CTransaction&>(*tx).unserialize(ss);
    } catch (...) {
        return make_rpc_error(RPC_DESERIALIZATION_ERROR,
                              "TX decode failed");
    }

    if (!ctx.mempool) {
        return make_rpc_error(RPC_INTERNAL_ERROR, "mempool not available");
    }

    // Add to mempool (fee=0 placeholder, would need proper fee calculation)
    int height = 0;
    float val_loss = 0.0f;
    if (ctx.chainstate && ctx.chainstate->tip()) {
        height = ctx.chainstate->tip()->height;
        val_loss = ctx.chainstate->tip()->val_loss;
    }

    auto result = ctx.mempool->add_tx(tx, 0, height, val_loss);
    if (result.is_err()) {
        return make_rpc_error(RPC_VERIFY_REJECTED, result.error());
    }

    LogPrint(RPC, "TX submitted via RPC: %s", tx->txid().to_hex().c_str());

    // TODO: relay to connected peers via connman

    return JsonValue(tx->txid().to_hex());
}

// ── Registration ────────────────────────────────────────────────────

void register_rawtransaction_rpcs(RPCTable& table) {
    table.register_command({
        "getrawtransaction",
        rpc_getrawtransaction,
        "Returns raw transaction data.\n"
        "Arguments: txid (hex string), verbose (bool, default=false)",
        "Rawtransactions"
    });

    table.register_command({
        "createrawtransaction",
        rpc_createrawtransaction,
        "Create a raw transaction (unsigned).\n"
        "Arguments: inputs (array), outputs (object), locktime (int, optional)",
        "Rawtransactions"
    });

    table.register_command({
        "decoderawtransaction",
        rpc_decoderawtransaction,
        "Decode a hex-encoded raw transaction.\n"
        "Arguments: hexstring (string)",
        "Rawtransactions"
    });

    table.register_command({
        "sendrawtransaction",
        rpc_sendrawtransaction,
        "Submit a raw transaction to the mempool and network.\n"
        "Arguments: hexstring (string)",
        "Rawtransactions"
    });
}

}  // namespace rnet::rpc

#include "rpc/blockchain.h"

#include "chain/block_index.h"
#include "chain/chainstate.h"
#include "core/logging.h"
#include "core/time.h"
#include "node/context.h"
#include "primitives/amount.h"
#include "primitives/block.h"
#include "primitives/block_header.h"

namespace rnet::rpc {

// ── getblockchaininfo ───────────────────────────────────────────────

static JsonValue rpc_getblockchaininfo(const RPCRequest& req,
                                       node::NodeContext& ctx) {
    JsonValue result = JsonValue::object();
    result.set("chain", JsonValue(ctx.network));

    if (!ctx.chainstate) {
        result.set("blocks", JsonValue(static_cast<int64_t>(0)));
        result.set("headers", JsonValue(static_cast<int64_t>(0)));
        result.set("bestblockhash", JsonValue(std::string(64, '0')));
        result.set("difficulty", JsonValue(0.0));
        result.set("val_loss", JsonValue(0.0));
        result.set("d_model", JsonValue(static_cast<int64_t>(384)));
        result.set("n_layers", JsonValue(static_cast<int64_t>(6)));
        result.set("initialblockdownload", JsonValue(true));
        return result;
    }

    auto* tip = ctx.chainstate->tip();
    int height = tip ? tip->height : 0;

    result.set("blocks", JsonValue(static_cast<int64_t>(height)));
    result.set("headers", JsonValue(
        static_cast<int64_t>(ctx.chainstate->block_index_size())));
    result.set("bestblockhash",
               JsonValue(tip ? tip->block_hash.to_hex() : std::string(64, '0')));

    if (tip) {
        result.set("val_loss", JsonValue(static_cast<double>(tip->val_loss)));
        result.set("d_model", JsonValue(static_cast<int64_t>(tip->d_model)));
        result.set("n_layers", JsonValue(static_cast<int64_t>(tip->n_layers)));
        result.set("mediantime",
                    JsonValue(static_cast<int64_t>(tip->timestamp)));
    } else {
        result.set("val_loss", JsonValue(0.0));
        result.set("d_model", JsonValue(static_cast<int64_t>(384)));
        result.set("n_layers", JsonValue(static_cast<int64_t>(6)));
        result.set("mediantime", JsonValue(static_cast<int64_t>(0)));
    }

    result.set("initialblockdownload", JsonValue(height < 1));
    result.set("chainwork", JsonValue(std::string("0")));  // PoT uses val_loss not work
    result.set("size_on_disk", JsonValue(static_cast<int64_t>(0)));
    result.set("pruned", JsonValue(false));

    return result;
}

// ── getblockcount ───────────────────────────────────────────────────

static JsonValue rpc_getblockcount(const RPCRequest& req,
                                   node::NodeContext& ctx) {
    if (!ctx.chainstate) return JsonValue(static_cast<int64_t>(0));
    return JsonValue(static_cast<int64_t>(ctx.chainstate->height()));
}

// ── getblockhash ────────────────────────────────────────────────────

static JsonValue rpc_getblockhash(const RPCRequest& req,
                                  node::NodeContext& ctx) {
    const auto& height_param = get_param(req, 0);
    if (!height_param.is_int()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "height parameter required (integer)");
    }

    int target = static_cast<int>(height_param.as_int());

    if (!ctx.chainstate) {
        return make_rpc_error(RPC_INTERNAL_ERROR, "chainstate not available");
    }

    auto* index = ctx.chainstate->get_block_by_height(target);
    if (!index) {
        return make_rpc_error(RPC_INVALID_PARAMETER,
                              "Block height out of range");
    }

    return JsonValue(index->block_hash.to_hex());
}

// ── getblock ────────────────────────────────────────────────────────

static JsonValue block_index_to_json(const chain::CBlockIndex* index,
                                     int verbosity) {
    if (!index) return JsonValue();

    JsonValue result = JsonValue::object();
    result.set("hash", JsonValue(index->block_hash.to_hex()));
    result.set("height", JsonValue(static_cast<int64_t>(index->height)));
    result.set("version", JsonValue(static_cast<int64_t>(index->header.version)));

    result.set("merkleroot", JsonValue(index->header.merkle_root.to_hex()));
    result.set("time", JsonValue(static_cast<int64_t>(index->timestamp)));

    // PoT-specific fields
    result.set("val_loss", JsonValue(static_cast<double>(index->val_loss)));
    result.set("prev_val_loss",
               JsonValue(static_cast<double>(index->header.prev_val_loss)));
    result.set("train_steps",
               JsonValue(static_cast<int64_t>(index->header.train_steps)));
    result.set("checkpoint_hash",
               JsonValue(index->header.checkpoint_hash.to_hex()));
    result.set("dataset_hash",
               JsonValue(index->header.dataset_hash.to_hex()));

    // Model config
    result.set("d_model", JsonValue(static_cast<int64_t>(index->d_model)));
    result.set("n_layers", JsonValue(static_cast<int64_t>(index->n_layers)));
    result.set("n_slots",
               JsonValue(static_cast<int64_t>(index->header.n_slots)));
    result.set("d_ff", JsonValue(static_cast<int64_t>(index->header.d_ff)));
    result.set("vocab_size",
               JsonValue(static_cast<int64_t>(index->header.vocab_size)));
    result.set("max_seq_len",
               JsonValue(static_cast<int64_t>(index->header.max_seq_len)));

    // Growth fields
    result.set("stagnation_count",
               JsonValue(static_cast<int64_t>(index->header.stagnation_count)));
    result.set("growth_delta",
               JsonValue(static_cast<int64_t>(index->header.growth_delta)));

    // Navigation
    result.set("previousblockhash",
               JsonValue(index->header.prev_hash.to_hex()));

    if (index->prev) {
        // nothing extra needed
    }

    result.set("nTx", JsonValue(static_cast<int64_t>(index->chain_tx)));
    result.set("confirmations", JsonValue(static_cast<int64_t>(1)));
    result.set("status", JsonValue(static_cast<int64_t>(index->status)));

    // Miner pubkey
    result.set("miner_pubkey",
               JsonValue(std::string(
                   reinterpret_cast<const char*>(index->header.miner_pubkey.data()),
                   0)));  // Don't include raw bytes in JSON
    std::string miner_hex;
    miner_hex.reserve(64);
    static constexpr char hx[] = "0123456789abcdef";
    for (int i = 0; i < 32; ++i) {
        miner_hex.push_back(hx[(index->header.miner_pubkey[i] >> 4) & 0xF]);
        miner_hex.push_back(hx[index->header.miner_pubkey[i] & 0xF]);
    }
    result.set("miner_pubkey", JsonValue(std::move(miner_hex)));

    return result;
}

static JsonValue rpc_getblock(const RPCRequest& req,
                              node::NodeContext& ctx) {
    const auto& hash_param = get_param(req, 0);
    if (!hash_param.is_string()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "blockhash parameter required (hex string)");
    }

    int verbosity = 1;
    const auto& verb_param = get_param_optional(req, 1);
    if (verb_param.is_int()) verbosity = static_cast<int>(verb_param.as_int());
    if (verb_param.is_bool()) verbosity = verb_param.as_bool() ? 1 : 0;

    if (!ctx.chainstate) {
        return make_rpc_error(RPC_INTERNAL_ERROR, "chainstate not available");
    }

    auto hash = rnet::uint256::from_hex(hash_param.as_string());
    auto* index = ctx.chainstate->lookup_block_index(hash);
    if (!index) {
        return make_rpc_error(RPC_INVALID_ADDRESS_OR_KEY, "Block not found");
    }

    if (verbosity == 0) {
        // Return raw serialized block as hex
        // For now, return the hash since we don't have block data cached
        return JsonValue(index->block_hash.to_hex());
    }

    return block_index_to_json(index, verbosity);
}

// ── getblockheader ──────────────────────────────────────────────────

static JsonValue rpc_getblockheader(const RPCRequest& req,
                                    node::NodeContext& ctx) {
    const auto& hash_param = get_param(req, 0);
    if (!hash_param.is_string()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "blockhash parameter required");
    }

    bool verbose = true;
    const auto& verb_param = get_param_optional(req, 1);
    if (verb_param.is_bool()) verbose = verb_param.as_bool();

    if (!ctx.chainstate) {
        return make_rpc_error(RPC_INTERNAL_ERROR, "chainstate not available");
    }

    auto hash = rnet::uint256::from_hex(hash_param.as_string());
    auto* index = ctx.chainstate->lookup_block_index(hash);
    if (!index) {
        return make_rpc_error(RPC_INVALID_ADDRESS_OR_KEY,
                              "Block header not found");
    }

    if (!verbose) {
        // Return serialized header as hex
        auto data = index->header.serialize_unsigned();
        return JsonValue(bytes_to_hex(data.data(), data.size()));
    }

    return block_index_to_json(index, 1);
}

// ── gettxout ────────────────────────────────────────────────────────

static JsonValue rpc_gettxout(const RPCRequest& req,
                              node::NodeContext& ctx) {
    const auto& txid_param = get_param(req, 0);
    const auto& vout_param = get_param(req, 1);

    if (!txid_param.is_string() || !vout_param.is_int()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "txid (string) and vout (int) required");
    }

    if (!ctx.chainstate) {
        return make_rpc_error(RPC_INTERNAL_ERROR, "chainstate not available");
    }

    // Look up the UTXO in the coins view
    // For now return a placeholder indicating the UTXO lookup would go here
    JsonValue result = JsonValue::object();
    result.set("bestblock", JsonValue(
        ctx.chainstate->tip() ?
        ctx.chainstate->tip()->block_hash.to_hex() :
        std::string(64, '0')));
    result.set("confirmations", JsonValue(static_cast<int64_t>(0)));
    result.set("value", JsonValue(0.0));
    result.set("coinbase", JsonValue(false));

    JsonValue script_pubkey = JsonValue::object();
    script_pubkey.set("asm", JsonValue(std::string("")));
    script_pubkey.set("hex", JsonValue(std::string("")));
    script_pubkey.set("type", JsonValue(std::string("unknown")));
    result.set("scriptPubKey", std::move(script_pubkey));

    return result;
}

// ── gettxoutsetinfo ─────────────────────────────────────────────────

static JsonValue rpc_gettxoutsetinfo(const RPCRequest& req,
                                     node::NodeContext& ctx) {
    JsonValue result = JsonValue::object();

    if (!ctx.chainstate) {
        result.set("height", JsonValue(static_cast<int64_t>(0)));
        result.set("bestblock", JsonValue(std::string(64, '0')));
        result.set("txouts", JsonValue(static_cast<int64_t>(0)));
        result.set("total_amount", JsonValue(0.0));
        return result;
    }

    auto* tip = ctx.chainstate->tip();
    result.set("height", JsonValue(
        static_cast<int64_t>(tip ? tip->height : 0)));
    result.set("bestblock", JsonValue(
        tip ? tip->block_hash.to_hex() : std::string(64, '0')));
    result.set("txouts", JsonValue(static_cast<int64_t>(0)));
    result.set("total_amount", JsonValue(0.0));
    result.set("hash_serialized", JsonValue(std::string(64, '0')));

    return result;
}

// ── Registration ────────────────────────────────────────────────────

void register_blockchain_rpcs(RPCTable& table) {
    table.register_command({
        "getblockchaininfo",
        rpc_getblockchaininfo,
        "Returns an object containing various state info regarding blockchain processing.\n"
        "Includes PoT-specific fields: val_loss, d_model, n_layers.",
        "Blockchain"
    });

    table.register_command({
        "getblockcount",
        rpc_getblockcount,
        "Returns the height of the most-work fully-validated chain.",
        "Blockchain"
    });

    table.register_command({
        "getblockhash",
        rpc_getblockhash,
        "Returns hash of block at the given height.\n"
        "Arguments: height (int)",
        "Blockchain"
    });

    table.register_command({
        "getblock",
        rpc_getblock,
        "Returns block data for the given block hash.\n"
        "Arguments: blockhash (hex string), verbosity (int, default=1)",
        "Blockchain"
    });

    table.register_command({
        "getblockheader",
        rpc_getblockheader,
        "Returns block header data for the given block hash.\n"
        "Arguments: blockhash (hex string), verbose (bool, default=true)",
        "Blockchain"
    });

    table.register_command({
        "gettxout",
        rpc_gettxout,
        "Returns details about an unspent transaction output.\n"
        "Arguments: txid (hex string), vout (int)",
        "Blockchain"
    });

    table.register_command({
        "gettxoutsetinfo",
        rpc_gettxoutsetinfo,
        "Returns statistics about the unspent transaction output set.",
        "Blockchain"
    });
}

}  // namespace rnet::rpc

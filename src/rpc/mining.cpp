#include "rpc/mining.h"

#include "chain/chainstate.h"
#include "consensus/block_reward.h"
#include "consensus/growth_policy.h"
#include "consensus/merkle.h"
#include "core/logging.h"
#include "core/stream.h"
#include "core/time.h"
#include "mempool/pool.h"
#include "node/context.h"
#include "primitives/block.h"
#include "primitives/outpoint.h"
#include "primitives/transaction.h"
#include "primitives/txin.h"
#include "primitives/txout.h"

namespace rnet::rpc {

// ── getmininginfo ───────────────────────────────────────────────────

static JsonValue rpc_getmininginfo(const RPCRequest& req,
                                   node::NodeContext& ctx) {
    JsonValue result = JsonValue::object();

    int height = 0;
    float val_loss = 0.0f;
    uint32_t d_model = 384;
    uint32_t n_layers = 6;

    if (ctx.chainstate) {
        auto* tip = ctx.chainstate->tip();
        if (tip) {
            height = tip->height;
            val_loss = tip->val_loss;
            d_model = tip->d_model;
            n_layers = tip->n_layers;
        }
    }

    result.set("blocks", JsonValue(static_cast<int64_t>(height)));
    result.set("currentblockweight", JsonValue(static_cast<int64_t>(0)));
    result.set("currentblocktx", JsonValue(static_cast<int64_t>(0)));

    // PoT-specific mining info
    result.set("val_loss", JsonValue(static_cast<double>(val_loss)));
    result.set("d_model", JsonValue(static_cast<int64_t>(d_model)));
    result.set("n_layers", JsonValue(static_cast<int64_t>(n_layers)));
    result.set("model_params",
               JsonValue(static_cast<int64_t>(0)));  // Would compute from config

    result.set("networkhashps", JsonValue(0.0));  // PoT equivalent: training rate
    result.set("pooledtx", JsonValue(static_cast<int64_t>(
        ctx.mempool ? ctx.mempool->size() : 0)));
    result.set("chain", JsonValue(ctx.network));

    return result;
}

// ── getblocktemplate ────────────────────────────────────────────────

static JsonValue rpc_getblocktemplate(const RPCRequest& req,
                                      node::NodeContext& ctx) {
    if (!ctx.chainstate) {
        return make_rpc_error(RPC_INTERNAL_ERROR, "chainstate not available");
    }

    auto* tip = ctx.chainstate->tip();
    if (!tip) {
        return make_rpc_error(RPC_INTERNAL_ERROR, "no chain tip");
    }

    JsonValue result = JsonValue::object();

    // Template fields
    result.set("version", JsonValue(static_cast<int64_t>(tip->header.version)));
    result.set("previousblockhash", JsonValue(tip->block_hash.to_hex()));
    result.set("height", JsonValue(static_cast<int64_t>(tip->height + 1)));
    result.set("curtime", JsonValue(static_cast<int64_t>(core::get_time())));
    result.set("mintime", JsonValue(static_cast<int64_t>(tip->timestamp + 1)));

    // PoT fields for the template
    result.set("current_val_loss",
               JsonValue(static_cast<double>(tip->val_loss)));
    result.set("checkpoint_hash", JsonValue(tip->header.checkpoint_hash.to_hex()));
    result.set("dataset_hash", JsonValue(tip->header.dataset_hash.to_hex()));
    result.set("d_model", JsonValue(static_cast<int64_t>(tip->d_model)));
    result.set("n_layers", JsonValue(static_cast<int64_t>(tip->n_layers)));
    result.set("n_slots",
               JsonValue(static_cast<int64_t>(tip->header.n_slots)));
    result.set("d_ff", JsonValue(static_cast<int64_t>(tip->header.d_ff)));
    result.set("stagnation_count",
               JsonValue(static_cast<int64_t>(tip->header.stagnation_count)));

    // Transactions available from mempool
    JsonValue transactions = JsonValue::array();
    if (ctx.mempool) {
        auto sorted = ctx.mempool->get_sorted_txs();
        for (const auto& tx : sorted) {
            JsonValue tx_entry = JsonValue::object();
            tx_entry.set("txid", JsonValue(tx->txid().to_hex()));
            tx_entry.set("weight",
                         JsonValue(static_cast<int64_t>(tx->get_weight())));
            tx_entry.set("fee", JsonValue(static_cast<int64_t>(0)));
            transactions.push_back(std::move(tx_entry));
        }
    }
    result.set("transactions", std::move(transactions));

    // Coinbase info
    result.set("coinbasevalue", JsonValue(static_cast<int64_t>(0)));

    return result;
}

// ── submitblock ─────────────────────────────────────────────────────

static JsonValue rpc_submitblock(const RPCRequest& req,
                                 node::NodeContext& ctx) {
    const auto& hex_param = get_param(req, 0);
    if (!hex_param.is_string()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "hex block data required");
    }

    if (!ctx.chainstate) {
        return make_rpc_error(RPC_INTERNAL_ERROR, "chainstate not available");
    }

    const std::string& hex_data = hex_param.as_string();
    if (!is_valid_hex(hex_data)) {
        return make_rpc_error(RPC_DESERIALIZATION_ERROR,
                              "invalid hex encoding");
    }

    auto bytes = hex_to_bytes(hex_data);
    if (bytes.empty()) {
        return make_rpc_error(RPC_DESERIALIZATION_ERROR,
                              "failed to decode hex data");
    }

    // Deserialize the block
    core::DataStream ss(std::span<const uint8_t>(bytes.data(), bytes.size()));
    primitives::CBlock block;
    try {
        block.unserialize(ss);
    } catch (...) {
        return make_rpc_error(RPC_DESERIALIZATION_ERROR,
                              "block decode failed");
    }

    // Submit to chainstate
    auto result = ctx.chainstate->accept_block(block);
    if (result.is_err()) {
        return make_rpc_error(RPC_VERIFY_ERROR, result.error());
    }

    LogPrint(RPC, "Block submitted via RPC: height=%d",
             result.value()->height);

    // Success returns null (Bitcoin convention)
    return JsonValue();
}

// ── generate (regtest only) ─────────────────────────────────────────

static JsonValue rpc_generate(const RPCRequest& req,
                               node::NodeContext& ctx) {
    if (ctx.network != "regtest") {
        return make_rpc_error(RPC_MISC_ERROR,
                              "generate is only available in regtest mode");
    }
    if (!ctx.chainstate) {
        return make_rpc_error(RPC_INTERNAL_ERROR, "chainstate not available");
    }

    // Parse nblocks (default 1)
    int nblocks = 1;
    const auto& p0 = get_param(req, 0);
    if (p0.is_number()) {
        nblocks = static_cast<int>(p0.as_int());
    }
    if (nblocks < 1 || nblocks > 1000) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "nblocks must be between 1 and 1000");
    }

    const auto& params = ctx.chainstate->params();
    JsonValue hashes = JsonValue::array();

    for (int i = 0; i < nblocks; ++i) {
        auto* tip = ctx.chainstate->tip();
        if (!tip) {
            return make_rpc_error(RPC_INTERNAL_ERROR, "no chain tip");
        }

        primitives::CBlock block;

        // Header fields
        block.version = 1;
        block.height = tip->height + 1;
        block.prev_hash = tip->block_hash;
        block.timestamp = static_cast<uint64_t>(core::get_time());
        if (block.timestamp <= tip->timestamp) {
            block.timestamp = tip->timestamp + 1;
        }

        // PoT fields: simulate slight improvement
        block.prev_val_loss = tip->val_loss;
        block.val_loss = tip->val_loss * 0.99f;  // 1% improvement
        block.train_steps = static_cast<uint32_t>(params.min_steps_per_block);

        // Growth fields — must match growth policy exactly
        bool loss_improved = block.val_loss < tip->val_loss;
        consensus::GrowthState gstate{};
        gstate.d_model = tip->d_model;
        gstate.n_layers = tip->n_layers;
        gstate.stagnation = tip->header.stagnation_count;
        gstate.last_loss = tip->val_loss;

        auto growth = consensus::GrowthPolicy::compute_growth(gstate, loss_improved);

        block.d_model = growth.new_d_model;
        block.n_layers = growth.new_n_layers;
        block.d_ff = growth.new_d_ff;
        block.n_slots = tip->header.n_slots;
        block.vocab_size = tip->header.vocab_size;
        block.max_seq_len = tip->header.max_seq_len;
        block.n_conv_branches = tip->header.n_conv_branches;
        block.kernel_sizes = tip->header.kernel_sizes;
        block.stagnation_count = growth.new_stagnation;
        block.growth_delta = growth.delta_d_model;

        // Coinbase transaction
        consensus::EmissionState emission{};
        auto reward = consensus::compute_block_reward(
            block.height, loss_improved ? 0.01f : 0.0f, emission, params);

        primitives::CMutableTransaction mtx;
        mtx.version = 1;

        // Coinbase input
        primitives::COutPoint null_outpoint;
        null_outpoint.set_null();
        std::string height_msg = "regtest block " + std::to_string(block.height);
        std::vector<uint8_t> script_sig(height_msg.begin(), height_msg.end());
        primitives::CTxIn coinbase_in(null_outpoint, std::move(script_sig), 0xFFFFFFFF);
        mtx.vin.push_back(std::move(coinbase_in));

        // Coinbase output (to unspendable key, like genesis)
        std::vector<uint8_t> coinbase_script;
        coinbase_script.push_back(0x20);
        coinbase_script.resize(33, 0x00);
        coinbase_script.push_back(0xAC);

        primitives::CTxOut coinbase_out(reward.total(), std::move(coinbase_script));
        mtx.vout.push_back(std::move(coinbase_out));
        mtx.locktime = 0;

        block.vtx.push_back(primitives::MakeTransactionRef(std::move(mtx)));

        // Merkle root
        block.merkle_root = consensus::block_merkle_root(block);

        // Submit
        auto result = ctx.chainstate->accept_block(block);
        if (result.is_err()) {
            return make_rpc_error(RPC_VERIFY_ERROR,
                "generate block " + std::to_string(i) + " failed: " + result.error());
        }

        auto bhash = block.hash();
        hashes.push_back(JsonValue(bhash.to_hex()));

        LogPrintf("Generated regtest block height=%d hash=%s val_loss=%.4f d_model=%u",
                  block.height, bhash.to_hex().c_str(),
                  block.val_loss, block.d_model);
    }

    return hashes;
}

// ── Registration ────────────────────────────────────────────────────

void register_mining_rpcs(RPCTable& table) {
    table.register_command({
        "getmininginfo",
        rpc_getmininginfo,
        "Returns mining-related information including PoT metrics.\n"
        "Fields: blocks, val_loss, d_model, n_layers, pooledtx.",
        "Mining"
    });

    table.register_command({
        "getblocktemplate",
        rpc_getblocktemplate,
        "Returns data needed to construct a block to work on.\n"
        "Includes PoT-specific fields for training configuration.",
        "Mining"
    });

    table.register_command({
        "submitblock",
        rpc_submitblock,
        "Submit a new block to the network.\n"
        "Arguments: hexdata (string) - hex-encoded serialized block",
        "Mining"
    });

    table.register_command({
        "generate",
        rpc_generate,
        "Generate nblocks blocks immediately (regtest only).\n"
        "Arguments: nblocks (int, default 1)\n"
        "Returns: array of block hashes",
        "Mining"
    });
}

}  // namespace rnet::rpc

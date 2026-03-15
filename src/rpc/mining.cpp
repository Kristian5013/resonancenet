// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "rpc/mining.h"

#include "rpc/wallet_rpc.h"
#include "wallet/wallet.h"
#include "chain/chainstate.h"
#include "consensus/block_reward.h"
#include "consensus/growth_policy.h"
#include "consensus/merkle.h"
#include "consensus/proof_of_training.h"
#include "crypto/ed25519.h"
#include "core/logging.h"
#include "core/stream.h"
#include "core/time.h"
#include "mempool/pool.h"
#include "net/conn_manager.h"
#include "net/protocol.h"
#include "node/context.h"
#include "primitives/address.h"
#include "primitives/block.h"
#include "primitives/outpoint.h"
#include "primitives/transaction.h"
#include "primitives/txin.h"
#include "primitives/txout.h"

#include <string>
#include <vector>

namespace rnet::rpc {

// ===========================================================================
//  getmininginfo
// ===========================================================================

// ---------------------------------------------------------------------------
// Returns mining-related information including PoT (Proof-of-Training)
// metrics.  Unlike Bitcoin's hashrate-centric view, ResonanceNet mining
// info centres on val_loss, d_model and growth-policy state.
// ---------------------------------------------------------------------------
static JsonValue rpc_getmininginfo(const RPCRequest& req,
                                   node::NodeContext& ctx) {
    JsonValue result = JsonValue::object();

    // 1. Gather chain-tip metrics (height, val_loss, architecture dims).
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

    // 2. Standard Bitcoin-style fields.
    result.set("blocks", JsonValue(static_cast<int64_t>(height)));
    result.set("currentblockweight", JsonValue(static_cast<int64_t>(0)));
    result.set("currentblocktx", JsonValue(static_cast<int64_t>(0)));

    // 3. PoT-specific mining info.
    result.set("val_loss", JsonValue(static_cast<double>(val_loss)));
    result.set("d_model", JsonValue(static_cast<int64_t>(d_model)));
    result.set("n_layers", JsonValue(static_cast<int64_t>(n_layers)));
    result.set("model_params",
               JsonValue(static_cast<int64_t>(0)));  // Would compute from config

    // 4. Network activity.
    result.set("networkhashps", JsonValue(0.0));  // PoT equivalent: training rate
    result.set("pooledtx", JsonValue(static_cast<int64_t>(
        ctx.mempool ? ctx.mempool->size() : 0)));
    result.set("chain", JsonValue(ctx.network));

    return result;
}

// ===========================================================================
//  getblocktemplate
// ===========================================================================

// ---------------------------------------------------------------------------
// Returns everything a miner needs to construct the next block.  Extends the
// Bitcoin getblocktemplate schema with PoT fields: current_val_loss,
// checkpoint_hash, dataset_hash, architecture dimensions and stagnation
// count so the miner can configure its training run.
// ---------------------------------------------------------------------------
static JsonValue rpc_getblocktemplate(const RPCRequest& req,
                                      node::NodeContext& ctx) {
    // 1. Validate chainstate availability.
    if (!ctx.chainstate) {
        return make_rpc_error(RPC_INTERNAL_ERROR, "chainstate not available");
    }

    auto* tip = ctx.chainstate->tip();
    if (!tip) {
        return make_rpc_error(RPC_INTERNAL_ERROR, "no chain tip");
    }

    JsonValue result = JsonValue::object();

    // 2. Standard template fields.
    result.set("version", JsonValue(static_cast<int64_t>(tip->header.version)));
    result.set("previousblockhash", JsonValue(tip->block_hash.to_hex()));
    result.set("height", JsonValue(static_cast<int64_t>(tip->height + 1)));
    result.set("curtime", JsonValue(static_cast<int64_t>(core::get_time())));
    result.set("mintime", JsonValue(static_cast<int64_t>(tip->timestamp + 1)));

    // 3. PoT fields for the template.
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

    // 4. Transactions available from mempool.
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

    // 5. Coinbase info.
    result.set("coinbasevalue", JsonValue(static_cast<int64_t>(0)));

    return result;
}

// ===========================================================================
//  submitblock
// ===========================================================================

// ---------------------------------------------------------------------------
// Accepts a hex-encoded serialised block and submits it to the chainstate
// for validation and connection.  Returns JSON null on success (Bitcoin
// convention) or an RPC error on failure.
// ---------------------------------------------------------------------------
static JsonValue rpc_submitblock(const RPCRequest& req,
                                 node::NodeContext& ctx) {
    // 1. Validate hex parameter.
    const auto& hex_param = get_param(req, 0);
    if (!hex_param.is_string()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "hex block data required");
    }

    // 2. Validate chainstate.
    if (!ctx.chainstate) {
        return make_rpc_error(RPC_INTERNAL_ERROR, "chainstate not available");
    }

    // 3. Decode hex data.
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

    // 4. Deserialise the block.
    core::DataStream ss(std::span<const uint8_t>(bytes.data(), bytes.size()));
    primitives::CBlock block;
    try {
        block.unserialize(ss);
    } catch (...) {
        return make_rpc_error(RPC_DESERIALIZATION_ERROR,
                              "block decode failed");
    }

    // 5. Submit to chainstate.
    auto result = ctx.chainstate->accept_block(block);
    if (result.is_err()) {
        return make_rpc_error(RPC_VERIFY_ERROR, result.error());
    }

    LogPrint(RPC, "Block submitted via RPC: height=%d",
             result.value()->height);

    // 6. Scan the block for wallet-relevant outputs.
    auto* wallet = get_rpc_wallet();
    if (wallet) {
        wallet->scan_block(block);
    }

    // 7. Broadcast the accepted block to all connected peers.
    if (ctx.connman) {
        core::DataStream blk_ss;
        block.serialize(blk_ss);
        ctx.connman->broadcast(net::msg::BLOCK, blk_ss.span());
        LogPrintf("Broadcast submitblock height=%d (%zu bytes) to peers",
                 result.value()->height, blk_ss.size());
    }

    // 8. Success returns null (Bitcoin convention).
    return JsonValue();
}

// ===========================================================================
//  generate (regtest only)
// ===========================================================================

// ---------------------------------------------------------------------------
// Instantly generates nblocks blocks on regtest by simulating a 1% val_loss
// improvement per block and applying the growth policy.  Each block contains
// a minimal coinbase paying to an unspendable key.  This is the primary
// mechanism for integration testing without real GPU training.
//
// Arguments:
//   nblocks (int) - number of blocks to generate, 1-1000 (default 1)
//
// Returns: JSON array of generated block hashes.
// ---------------------------------------------------------------------------
static JsonValue rpc_generate(const RPCRequest& req,
                               node::NodeContext& ctx) {
    // 1. Restrict to regtest.
    if (ctx.network != "regtest") {
        return make_rpc_error(RPC_MISC_ERROR,
                              "generate is only available in regtest mode");
    }
    if (!ctx.chainstate) {
        return make_rpc_error(RPC_INTERNAL_ERROR, "chainstate not available");
    }

    // 2. Parse nblocks (default 1).
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
        // 3. Fetch current tip.
        auto* tip = ctx.chainstate->tip();
        if (!tip) {
            return make_rpc_error(RPC_INTERNAL_ERROR, "no chain tip");
        }

        primitives::CBlock block;

        // 4. Header fields.
        block.version = 1;
        block.height = tip->height + 1;
        block.prev_hash = tip->block_hash;
        block.timestamp = static_cast<uint64_t>(core::get_time());
        if (block.timestamp <= tip->timestamp) {
            block.timestamp = tip->timestamp + 1;
        }

        // 5. PoT fields: simulate slight improvement (1% per block).
        block.prev_val_loss = tip->val_loss;
        block.val_loss = tip->val_loss * 0.99f;
        block.train_steps = static_cast<uint32_t>(params.min_steps_per_block);

        // 6. Growth fields -- must match growth policy exactly.
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

        // 7. Coinbase transaction.
        consensus::EmissionState emission{};
        auto reward = consensus::compute_block_reward(
            block.height, emission, params);

        primitives::CMutableTransaction mtx;
        mtx.version = 1;

        // 8. Coinbase input (null outpoint, height message as script_sig).
        primitives::COutPoint null_outpoint;
        null_outpoint.set_null();
        std::string height_msg = "regtest block " + std::to_string(block.height);
        std::vector<uint8_t> script_sig(height_msg.begin(), height_msg.end());
        primitives::CTxIn coinbase_in(null_outpoint, std::move(script_sig), 0xFFFFFFFF);
        mtx.vin.push_back(std::move(coinbase_in));

        // 9. Coinbase output (unspendable key: [0x20][32-byte zero pubkey][0xAC]).
        std::vector<uint8_t> coinbase_script;
        coinbase_script.push_back(0x20);
        coinbase_script.resize(33, 0x00);
        coinbase_script.push_back(0xAC);

        primitives::CTxOut coinbase_out(reward.total(), std::move(coinbase_script));
        mtx.vout.push_back(std::move(coinbase_out));
        mtx.locktime = 0;

        block.vtx.push_back(primitives::MakeTransactionRef(std::move(mtx)));

        // 10. Compute merkle root and submit.
        block.merkle_root = consensus::block_merkle_root(block);

        auto result = ctx.chainstate->accept_block(block);
        if (result.is_err()) {
            return make_rpc_error(RPC_VERIFY_ERROR,
                "generate block " + std::to_string(i) + " failed: " + result.error());
        }

        // 11. Scan the block for wallet-relevant outputs.
        auto* gen_wallet = get_rpc_wallet();
        if (gen_wallet) {
            gen_wallet->scan_block(block);
        }

        // 12. Broadcast the generated block to all connected peers.
        if (ctx.connman) {
            core::DataStream blk_ss;
            block.serialize(blk_ss);
            ctx.connman->broadcast(net::msg::BLOCK, blk_ss.span());
        }

        auto bhash = block.hash();
        hashes.push_back(JsonValue(bhash.to_hex()));

        LogPrintf("Generated regtest block height=%d hash=%s val_loss=%.4f d_model=%u",
                  block.height, bhash.to_hex().c_str(),
                  block.val_loss, block.d_model);
    }

    return hashes;
}

// ===========================================================================
//  Registration
// ===========================================================================

// ---------------------------------------------------------------------------
// Registers all mining-related RPCs into the global RPC dispatch table.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// rpc_submittrainingblock
// ---------------------------------------------------------------------------
// Accepts a training proof (val_loss, checkpoint_hash, train_steps, address)
// and constructs + validates the block internally.  This is the PoT-native
// submit path used by rnet-miner instead of hex-encoded submitblock.
// ---------------------------------------------------------------------------
static JsonValue rpc_submittrainingblock(const RPCRequest& req,
                                          node::NodeContext& ctx) {
    // 1. Parse the training proof object.
    const auto& proof = get_param(req, 0);
    if (proof.is_null()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "training proof object required");
    }

    const auto& val_loss_v = proof["val_loss"];
    const auto& ckpt_hash_v = proof["checkpoint_hash"];
    const auto& steps_v = proof["train_steps"];
    const auto& addr_v = proof["address"];

    if (!val_loss_v.is_number() || !ckpt_hash_v.is_string() ||
        !steps_v.is_number() || !addr_v.is_string()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "required: val_loss (number), checkpoint_hash (string), "
                              "train_steps (number), address (string)");
    }

    float val_loss = static_cast<float>(val_loss_v.as_double());
    std::string checkpoint_hash = ckpt_hash_v.as_string();
    int train_steps = static_cast<int>(steps_v.as_int());
    std::string address = addr_v.as_string();

    // 1b. Dataset hash — pins the training data for verifiability.
    std::string dataset_hash_str;
    const auto& dh_v = proof["dataset_hash"];
    if (dh_v.is_string()) {
        dataset_hash_str = dh_v.as_string();
    }

    // 2. Validate chainstate.
    if (!ctx.chainstate) {
        return make_rpc_error(RPC_INTERNAL_ERROR, "chainstate not available");
    }

    // 3. Get the current tip.
    auto* tip = ctx.chainstate->tip();
    if (!tip) {
        return make_rpc_error(RPC_INTERNAL_ERROR, "no chain tip");
    }

    const auto& params = ctx.chainstate->params();

    // 4. Decode the miner reward address into a scriptPubKey.
    auto decoded = primitives::decode_address(address);
    if (decoded.is_err()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "invalid address: " + decoded.error());
    }
    auto script_pub_key = primitives::script_from_address(decoded.value());

    // 5. Construct the block header fields.
    primitives::CBlock block;
    block.version   = 1;
    block.height    = tip->height + 1;
    block.prev_hash = tip->block_hash;
    block.timestamp = static_cast<uint64_t>(core::get_time());
    if (block.timestamp <= tip->timestamp) {
        block.timestamp = tip->timestamp + 1;
    }

    // 6. Block timing regulated by difficulty_delta retarget, not by
    //    a minimum interval.  Any timestamp advancing past the parent
    //    is valid — the retarget algorithm handles pacing.

    // 7. PoT fields from the submitted training proof.
    block.val_loss       = val_loss;
    block.prev_val_loss  = tip->val_loss;
    block.train_steps    = static_cast<uint32_t>(train_steps);
    block.checkpoint_hash = rnet::uint256::from_hex(checkpoint_hash);
    // Use miner-provided dataset hash, or inherit from parent.
    if (!dataset_hash_str.empty()) {
        block.dataset_hash = rnet::uint256::from_hex(dataset_hash_str);
    } else {
        block.dataset_hash = tip->header.dataset_hash;
    }

    // 8. Compute difficulty_delta for this block height.
    uint64_t period_start_ts = tip->timestamp;
    if (tip->prev) {
        period_start_ts = tip->prev->timestamp;
    }
    block.difficulty_delta = consensus::compute_next_difficulty(
        block.height, tip->header.difficulty_delta,
        period_start_ts, tip->timestamp, params);

    // 9. Check PoT improvement meets difficulty threshold.
    float improvement = block.prev_val_loss - block.val_loss;
    if (improvement < block.difficulty_delta) {
        return make_rpc_error(RPC_VERIFY_ERROR,
            "insufficient improvement: " + std::to_string(improvement) +
            " < required " + std::to_string(block.difficulty_delta));
    }

    // 10. Growth policy — compute new model dimensions.
    bool loss_improved = block.val_loss < tip->val_loss;
    consensus::GrowthState gstate{};
    gstate.d_model   = tip->d_model;
    gstate.n_layers  = tip->n_layers;
    gstate.stagnation = tip->header.stagnation_count;
    gstate.last_loss = tip->val_loss;

    auto growth = consensus::GrowthPolicy::compute_growth(gstate, loss_improved);

    block.d_model          = growth.new_d_model;
    block.n_layers         = growth.new_n_layers;
    block.d_ff             = growth.new_d_ff;
    block.n_slots          = tip->header.n_slots;
    block.vocab_size       = tip->header.vocab_size;
    block.max_seq_len      = tip->header.max_seq_len;
    block.n_conv_branches  = tip->header.n_conv_branches;
    block.kernel_sizes     = tip->header.kernel_sizes;
    block.stagnation_count = growth.new_stagnation;
    block.growth_delta     = growth.delta_d_model;

    // 11. Coinbase transaction — block reward to the miner's address.
    consensus::EmissionState emission{};
    auto reward = consensus::compute_block_reward(
        block.height, emission, params);

    primitives::CMutableTransaction mtx;
    mtx.version = 1;

    primitives::COutPoint null_outpoint;
    null_outpoint.set_null();
    std::string height_msg = "block " + std::to_string(block.height);
    std::vector<uint8_t> script_sig(height_msg.begin(), height_msg.end());
    primitives::CTxIn coinbase_in(null_outpoint, std::move(script_sig),
                                  0xFFFFFFFF);
    mtx.vin.push_back(std::move(coinbase_in));

    primitives::CTxOut coinbase_out(reward.total(), std::move(script_pub_key));
    mtx.vout.push_back(std::move(coinbase_out));
    mtx.locktime = 0;

    block.vtx.push_back(primitives::MakeTransactionRef(std::move(mtx)));

    // 12. Compute merkle root.
    block.merkle_root = consensus::block_merkle_root(block);

    // 12b. Ed25519 sign the block — generate ephemeral keypair per block.
    auto keypair_result = crypto::ed25519_generate();
    if (!keypair_result) {
        return make_rpc_error(RPC_INTERNAL_ERROR,
                              "failed to generate Ed25519 keypair");
    }
    auto& keypair = keypair_result.value();

    // Copy public key into block header.
    std::copy(keypair.public_key.data.begin(), keypair.public_key.data.end(),
              block.miner_pubkey.begin());

    // Sign the unsigned header bytes.
    const auto unsigned_bytes = block.serialize_unsigned();
    auto sig_result = crypto::ed25519_sign(
        keypair.secret,
        std::span<const uint8_t>(unsigned_bytes.data(), unsigned_bytes.size()));
    if (!sig_result) {
        return make_rpc_error(RPC_INTERNAL_ERROR,
                              "Ed25519 signing failed");
    }
    std::copy(sig_result.value().data.begin(), sig_result.value().data.end(),
              block.signature.begin());

    // 13. Submit block to chainstate for full validation and connection.
    auto result = ctx.chainstate->accept_block(block);
    if (result.is_err()) {
        LogPrintf("SubmitTrainingBlock REJECTED: %s", result.error().c_str());
        JsonValue resp = JsonValue::object();
        resp.set("accepted", JsonValue(false));
        resp.set("reject_reason", JsonValue(result.error()));
        return resp;
    }

    // 14. Scan the block for wallet-relevant outputs.
    auto* stb_wallet = get_rpc_wallet();
    if (stb_wallet) {
        stb_wallet->scan_block(block);
    }

    // 15. Log the accepted block.
    auto bhash = block.hash();
    LogPrintf("SubmitTrainingBlock ACCEPTED: height=%d hash=%s "
              "val_loss=%.6f d_model=%u",
              block.height, bhash.to_hex().c_str(),
              static_cast<double>(block.val_loss), block.d_model);

    // 16. Broadcast the full block to all connected peers.
    if (ctx.connman) {
        core::DataStream blk_ss;
        block.serialize(blk_ss);
        ctx.connman->broadcast(net::msg::BLOCK, blk_ss.span());
        LogPrintf("Broadcast training block height=%d (%zu bytes) to peers",
                 block.height, blk_ss.size());
    }

    // 17. Return result.
    JsonValue resp = JsonValue::object();
    resp.set("accepted", JsonValue(true));
    resp.set("hash", JsonValue(bhash.to_hex()));
    resp.set("height", JsonValue(static_cast<int64_t>(block.height)));
    resp.set("val_loss", JsonValue(static_cast<double>(block.val_loss)));
    resp.set("checkpoint_hash", JsonValue(checkpoint_hash));
    resp.set("train_steps", JsonValue(static_cast<int64_t>(train_steps)));
    resp.set("d_model", JsonValue(static_cast<int64_t>(block.d_model)));
    resp.set("n_layers", JsonValue(static_cast<int64_t>(block.n_layers)));
    return resp;
}

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
        "submittrainingblock",
        rpc_submittrainingblock,
        "Submit a training proof for a new block.\n"
        "Arguments: {val_loss, checkpoint_hash, train_steps, address}",
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

} // namespace rnet::rpc

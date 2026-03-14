// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "rpc/blockchain.h"

#include "chain/block_index.h"
#include "chain/chainstate.h"
#include "core/logging.h"
#include "core/time.h"
#include "node/context.h"
#include "primitives/amount.h"
#include "primitives/block.h"
#include "primitives/block_header.h"

#include <cstdint>
#include <string>

namespace rnet::rpc {

// ===========================================================================
//  Constants
// ===========================================================================

static constexpr int64_t DEFAULT_D_MODEL  = 384;   // (units)
static constexpr int64_t DEFAULT_N_LAYERS = 6;     // (layers)
static constexpr int     MINER_PUBKEY_LEN = 32;    // (bytes)

// ===========================================================================
//  Blockchain Info
// ===========================================================================

// ---------------------------------------------------------------------------
// rpc_getblockchaininfo
// ---------------------------------------------------------------------------
// Design: Returns chain name, height, best block hash, val_loss, d_model,
//         n_layers. When chainstate is unavailable, returns safe defaults so
//         clients always receive a well-formed response.
// ---------------------------------------------------------------------------
static JsonValue rpc_getblockchaininfo(const RPCRequest& req,
                                       node::NodeContext& ctx) {
    // 1. Build the result object
    JsonValue result = JsonValue::object();
    result.set("chain", JsonValue(ctx.network));

    // 2. Handle missing chainstate with safe defaults
    if (!ctx.chainstate) {
        result.set("blocks", JsonValue(static_cast<int64_t>(0)));
        result.set("headers", JsonValue(static_cast<int64_t>(0)));
        result.set("bestblockhash", JsonValue(std::string(64, '0')));
        result.set("difficulty", JsonValue(0.0));
        result.set("val_loss", JsonValue(0.0));
        result.set("d_model", JsonValue(DEFAULT_D_MODEL));
        result.set("n_layers", JsonValue(DEFAULT_N_LAYERS));
        result.set("initialblockdownload", JsonValue(true));
        return result;
    }

    // 3. Read chain tip
    auto* tip = ctx.chainstate->tip();
    int height = tip ? tip->height : 0;

    // 4. Populate height and header count
    result.set("blocks", JsonValue(static_cast<int64_t>(height)));
    result.set("headers", JsonValue(
        static_cast<int64_t>(ctx.chainstate->block_index_size())));
    result.set("bestblockhash",
               JsonValue(tip ? tip->block_hash.to_hex() : std::string(64, '0')));

    // 5. PoT-specific fields from tip
    if (tip) {
        result.set("val_loss", JsonValue(static_cast<double>(tip->val_loss)));
        result.set("d_model", JsonValue(static_cast<int64_t>(tip->d_model)));
        result.set("n_layers", JsonValue(static_cast<int64_t>(tip->n_layers)));
        result.set("mediantime",
                    JsonValue(static_cast<int64_t>(tip->timestamp)));
    } else {
        result.set("val_loss", JsonValue(0.0));
        result.set("d_model", JsonValue(DEFAULT_D_MODEL));
        result.set("n_layers", JsonValue(DEFAULT_N_LAYERS));
        result.set("mediantime", JsonValue(static_cast<int64_t>(0)));
    }

    // 6. Remaining status fields
    result.set("initialblockdownload", JsonValue(height < 1));
    result.set("chainwork", JsonValue(std::string("0")));  // PoT uses val_loss not work
    result.set("size_on_disk", JsonValue(static_cast<int64_t>(0)));
    result.set("pruned", JsonValue(false));

    return result;
}

// ===========================================================================
//  Block Data
// ===========================================================================

// ---------------------------------------------------------------------------
// rpc_getblockcount
// ---------------------------------------------------------------------------
// Design: Simple height query. Returns the height of the most-work
//         fully-validated chain, or 0 if chainstate is unavailable.
// ---------------------------------------------------------------------------
static JsonValue rpc_getblockcount(const RPCRequest& req,
                                   node::NodeContext& ctx) {
    // 1. Return 0 when chainstate is missing
    if (!ctx.chainstate) return JsonValue(static_cast<int64_t>(0));

    // 2. Return current chain height
    return JsonValue(static_cast<int64_t>(ctx.chainstate->height()));
}

// ---------------------------------------------------------------------------
// rpc_getblockhash
// ---------------------------------------------------------------------------
// Design: Height to hash lookup. Accepts an integer height parameter and
//         returns the block hash at that height, or an error if out of range.
// ---------------------------------------------------------------------------
static JsonValue rpc_getblockhash(const RPCRequest& req,
                                  node::NodeContext& ctx) {
    // 1. Validate height parameter
    const auto& height_param = get_param(req, 0);
    if (!height_param.is_int()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "height parameter required (integer)");
    }

    int target = static_cast<int>(height_param.as_int());

    // 2. Verify chainstate availability
    if (!ctx.chainstate) {
        return make_rpc_error(RPC_INTERNAL_ERROR, "chainstate not available");
    }

    // 3. Look up block index by height
    auto* index = ctx.chainstate->get_block_by_height(target);
    if (!index) {
        return make_rpc_error(RPC_INVALID_PARAMETER,
                              "Block height out of range");
    }

    return JsonValue(index->block_hash.to_hex());
}

// ---------------------------------------------------------------------------
// block_index_to_json
// ---------------------------------------------------------------------------
// Design: Converts a CBlockIndex into a JSON object with full block data
//         including tx details and PoT growth fields. Used by both getblock
//         and getblockheader for consistent serialization.
// ---------------------------------------------------------------------------
static JsonValue block_index_to_json(const chain::CBlockIndex* index,
                                     int verbosity) {
    // 1. Guard against null index
    if (!index) return JsonValue();

    // 2. Core block identity fields
    JsonValue result = JsonValue::object();
    result.set("hash", JsonValue(index->block_hash.to_hex()));
    result.set("height", JsonValue(static_cast<int64_t>(index->height)));
    result.set("version", JsonValue(static_cast<int64_t>(index->header.version)));

    result.set("merkleroot", JsonValue(index->header.merkle_root.to_hex()));
    result.set("time", JsonValue(static_cast<int64_t>(index->timestamp)));

    // 3. PoT-specific fields
    result.set("val_loss", JsonValue(static_cast<double>(index->val_loss)));
    result.set("prev_val_loss",
               JsonValue(static_cast<double>(index->header.prev_val_loss)));
    result.set("train_steps",
               JsonValue(static_cast<int64_t>(index->header.train_steps)));
    result.set("checkpoint_hash",
               JsonValue(index->header.checkpoint_hash.to_hex()));
    result.set("dataset_hash",
               JsonValue(index->header.dataset_hash.to_hex()));

    // 4. Model configuration
    result.set("d_model", JsonValue(static_cast<int64_t>(index->d_model)));
    result.set("n_layers", JsonValue(static_cast<int64_t>(index->n_layers)));
    result.set("n_slots",
               JsonValue(static_cast<int64_t>(index->header.n_slots)));
    result.set("d_ff", JsonValue(static_cast<int64_t>(index->header.d_ff)));
    result.set("vocab_size",
               JsonValue(static_cast<int64_t>(index->header.vocab_size)));
    result.set("max_seq_len",
               JsonValue(static_cast<int64_t>(index->header.max_seq_len)));

    // 5. Growth fields
    result.set("stagnation_count",
               JsonValue(static_cast<int64_t>(index->header.stagnation_count)));
    result.set("growth_delta",
               JsonValue(static_cast<int64_t>(index->header.growth_delta)));

    // 6. Navigation links
    result.set("previousblockhash",
               JsonValue(index->header.prev_hash.to_hex()));

    if (index->prev) {
        // nothing extra needed
    }

    // 7. Transaction and status metadata
    result.set("nTx", JsonValue(static_cast<int64_t>(index->chain_tx)));
    result.set("confirmations", JsonValue(static_cast<int64_t>(1)));
    result.set("status", JsonValue(static_cast<int64_t>(index->status)));

    // 8. Miner pubkey as hex string
    result.set("miner_pubkey",
               JsonValue(std::string(
                   reinterpret_cast<const char*>(index->header.miner_pubkey.data()),
                   0)));  // Don't include raw bytes in JSON
    std::string miner_hex;
    miner_hex.reserve(64);
    static constexpr char hx[] = "0123456789abcdef";
    for (int i = 0; i < MINER_PUBKEY_LEN; ++i) {
        miner_hex.push_back(hx[(index->header.miner_pubkey[i] >> 4) & 0xF]);
        miner_hex.push_back(hx[index->header.miner_pubkey[i] & 0xF]);
    }
    result.set("miner_pubkey", JsonValue(std::move(miner_hex)));

    return result;
}

// ---------------------------------------------------------------------------
// rpc_getblock
// ---------------------------------------------------------------------------
// Design: Full block data with tx details, verbosity levels. Verbosity 0
//         returns raw hex, verbosity >= 1 returns a JSON object with all
//         block fields including PoT growth fields.
// ---------------------------------------------------------------------------
static JsonValue rpc_getblock(const RPCRequest& req,
                              node::NodeContext& ctx) {
    // 1. Validate block hash parameter
    const auto& hash_param = get_param(req, 0);
    if (!hash_param.is_string()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "blockhash parameter required (hex string)");
    }

    // 2. Parse optional verbosity (default 1)
    int verbosity = 1;
    const auto& verb_param = get_param_optional(req, 1);
    if (verb_param.is_int()) verbosity = static_cast<int>(verb_param.as_int());
    if (verb_param.is_bool()) verbosity = verb_param.as_bool() ? 1 : 0;

    // 3. Verify chainstate availability
    if (!ctx.chainstate) {
        return make_rpc_error(RPC_INTERNAL_ERROR, "chainstate not available");
    }

    // 4. Look up the block index by hash
    auto hash = rnet::uint256::from_hex(hash_param.as_string());
    auto* index = ctx.chainstate->lookup_block_index(hash);
    if (!index) {
        return make_rpc_error(RPC_INVALID_ADDRESS_OR_KEY, "Block not found");
    }

    // 5. Return raw hex at verbosity 0, JSON object otherwise
    if (verbosity == 0) {
        // Return raw serialized block as hex
        // For now, return the hash since we don't have block data cached
        return JsonValue(index->block_hash.to_hex());
    }

    return block_index_to_json(index, verbosity);
}

// ---------------------------------------------------------------------------
// rpc_getblockheader
// ---------------------------------------------------------------------------
// Design: Header fields including PoT growth fields. When verbose is false,
//         returns the serialized header as hex. When true, returns the same
//         JSON structure as getblock.
// ---------------------------------------------------------------------------
static JsonValue rpc_getblockheader(const RPCRequest& req,
                                    node::NodeContext& ctx) {
    // 1. Validate block hash parameter
    const auto& hash_param = get_param(req, 0);
    if (!hash_param.is_string()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "blockhash parameter required");
    }

    // 2. Parse optional verbose flag (default true)
    bool verbose = true;
    const auto& verb_param = get_param_optional(req, 1);
    if (verb_param.is_bool()) verbose = verb_param.as_bool();

    // 3. Verify chainstate availability
    if (!ctx.chainstate) {
        return make_rpc_error(RPC_INTERNAL_ERROR, "chainstate not available");
    }

    // 4. Look up the block index by hash
    auto hash = rnet::uint256::from_hex(hash_param.as_string());
    auto* index = ctx.chainstate->lookup_block_index(hash);
    if (!index) {
        return make_rpc_error(RPC_INVALID_ADDRESS_OR_KEY,
                              "Block header not found");
    }

    // 5. Return serialized hex or JSON based on verbose flag
    if (!verbose) {
        // Return serialized header as hex
        auto data = index->header.serialize_unsigned();
        return JsonValue(bytes_to_hex(data.data(), data.size()));
    }

    return block_index_to_json(index, 1);
}

// ===========================================================================
//  UTXO Queries
// ===========================================================================

// ---------------------------------------------------------------------------
// rpc_gettxout
// ---------------------------------------------------------------------------
// Design: UTXO lookup by txid:vout. Returns the unspent output details
//         including bestblock, confirmations, value, and scriptPubKey.
//         Currently returns a placeholder pending full coins-view integration.
// ---------------------------------------------------------------------------
static JsonValue rpc_gettxout(const RPCRequest& req,
                              node::NodeContext& ctx) {
    // 1. Validate txid and vout parameters
    const auto& txid_param = get_param(req, 0);
    const auto& vout_param = get_param(req, 1);

    if (!txid_param.is_string() || !vout_param.is_int()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "txid (string) and vout (int) required");
    }

    // 2. Verify chainstate availability
    if (!ctx.chainstate) {
        return make_rpc_error(RPC_INTERNAL_ERROR, "chainstate not available");
    }

    // 3. Build placeholder UTXO result
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

    // 4. Attach scriptPubKey sub-object
    JsonValue script_pubkey = JsonValue::object();
    script_pubkey.set("asm", JsonValue(std::string("")));
    script_pubkey.set("hex", JsonValue(std::string("")));
    script_pubkey.set("type", JsonValue(std::string("unknown")));
    result.set("scriptPubKey", std::move(script_pubkey));

    return result;
}

// ---------------------------------------------------------------------------
// rpc_gettxoutsetinfo
// ---------------------------------------------------------------------------
// Design: Returns statistics about the unspent transaction output set
//         including height, bestblock hash, utxo count, and total amount.
// ---------------------------------------------------------------------------
static JsonValue rpc_gettxoutsetinfo(const RPCRequest& req,
                                     node::NodeContext& ctx) {
    JsonValue result = JsonValue::object();

    // 1. Handle missing chainstate with zero-value defaults
    if (!ctx.chainstate) {
        result.set("height", JsonValue(static_cast<int64_t>(0)));
        result.set("bestblock", JsonValue(std::string(64, '0')));
        result.set("txouts", JsonValue(static_cast<int64_t>(0)));
        result.set("total_amount", JsonValue(0.0));
        return result;
    }

    // 2. Populate from chain tip
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

// ===========================================================================
//  Chain Tips
// ===========================================================================

// ---------------------------------------------------------------------------
// rpc_getchaintips (not yet implemented — placeholder for future use)
// ---------------------------------------------------------------------------
// Design: Active chain tip info. Will return an array of chain tip objects
//         with height, hash, branch length, and status.
// ---------------------------------------------------------------------------

// ===========================================================================
//  Registration
// ===========================================================================

// ---------------------------------------------------------------------------
// register_blockchain_rpcs
// ---------------------------------------------------------------------------
// Design: Command table registration. Registers all blockchain RPC commands
//         with their names, handlers, help text, and category.
// ---------------------------------------------------------------------------
void register_blockchain_rpcs(RPCTable& table) {
    // 1. Chain state overview
    table.register_command({
        "getblockchaininfo",
        rpc_getblockchaininfo,
        "Returns an object containing various state info regarding blockchain processing.\n"
        "Includes PoT-specific fields: val_loss, d_model, n_layers.",
        "Blockchain"
    });

    // 2. Height query
    table.register_command({
        "getblockcount",
        rpc_getblockcount,
        "Returns the height of the most-work fully-validated chain.",
        "Blockchain"
    });

    // 3. Height to hash
    table.register_command({
        "getblockhash",
        rpc_getblockhash,
        "Returns hash of block at the given height.\n"
        "Arguments: height (int)",
        "Blockchain"
    });

    // 4. Full block data
    table.register_command({
        "getblock",
        rpc_getblock,
        "Returns block data for the given block hash.\n"
        "Arguments: blockhash (hex string), verbosity (int, default=1)",
        "Blockchain"
    });

    // 5. Block header
    table.register_command({
        "getblockheader",
        rpc_getblockheader,
        "Returns block header data for the given block hash.\n"
        "Arguments: blockhash (hex string), verbose (bool, default=true)",
        "Blockchain"
    });

    // 6. UTXO lookup
    table.register_command({
        "gettxout",
        rpc_gettxout,
        "Returns details about an unspent transaction output.\n"
        "Arguments: txid (hex string), vout (int)",
        "Blockchain"
    });

    // 7. UTXO set statistics
    table.register_command({
        "gettxoutsetinfo",
        rpc_gettxoutsetinfo,
        "Returns statistics about the unspent transaction output set.",
        "Blockchain"
    });
}

} // namespace rnet::rpc

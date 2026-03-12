#include "rpc/training_rpc.h"

#include "chain/block_index.h"
#include "chain/chainstate.h"
#include "core/logging.h"
#include "node/context.h"
#include "primitives/block_header.h"

namespace rnet::rpc {

// ── gettraininginfo ─────────────────────────────────────────────────

static JsonValue rpc_gettraininginfo(const RPCRequest& req,
                                     node::NodeContext& ctx) {
    JsonValue result = JsonValue::object();

    if (!ctx.chainstate || !ctx.chainstate->tip()) {
        result.set("height", JsonValue(static_cast<int64_t>(0)));
        result.set("val_loss", JsonValue(0.0));
        result.set("prev_val_loss", JsonValue(0.0));
        result.set("train_steps", JsonValue(static_cast<int64_t>(0)));
        result.set("checkpoint_hash", JsonValue(std::string(64, '0')));
        result.set("dataset_hash", JsonValue(std::string(64, '0')));
        result.set("stagnation_count", JsonValue(static_cast<int64_t>(0)));
        result.set("model_params", JsonValue(static_cast<int64_t>(0)));
        return result;
    }

    auto* tip = ctx.chainstate->tip();
    const auto& hdr = tip->header;

    result.set("height", JsonValue(static_cast<int64_t>(tip->height)));
    result.set("val_loss", JsonValue(static_cast<double>(hdr.val_loss)));
    result.set("prev_val_loss",
               JsonValue(static_cast<double>(hdr.prev_val_loss)));
    result.set("train_steps",
               JsonValue(static_cast<int64_t>(hdr.train_steps)));
    result.set("checkpoint_hash",
               JsonValue(hdr.checkpoint_hash.to_hex()));
    result.set("dataset_hash",
               JsonValue(hdr.dataset_hash.to_hex()));
    result.set("stagnation_count",
               JsonValue(static_cast<int64_t>(hdr.stagnation_count)));

    // Compute model parameter count from header config
    uint64_t param_count = hdr.model_param_count();
    result.set("model_params",
               JsonValue(static_cast<int64_t>(param_count)));

    // Loss improvement ratio (for gauging training progress)
    if (hdr.prev_val_loss > 0.0f && hdr.val_loss > 0.0f) {
        double improvement = static_cast<double>(hdr.prev_val_loss - hdr.val_loss)
                           / static_cast<double>(hdr.prev_val_loss);
        result.set("loss_improvement", JsonValue(improvement));
    } else {
        result.set("loss_improvement", JsonValue(0.0));
    }

    // Training history: last N blocks' val_loss
    JsonValue loss_history = JsonValue::array();
    const chain::CBlockIndex* cur = tip;
    for (int i = 0; i < 20 && cur; ++i) {
        JsonValue entry = JsonValue::object();
        entry.set("height", JsonValue(static_cast<int64_t>(cur->height)));
        entry.set("val_loss",
                   JsonValue(static_cast<double>(cur->val_loss)));
        entry.set("train_steps",
                   JsonValue(static_cast<int64_t>(cur->header.train_steps)));
        loss_history.push_back(std::move(entry));
        cur = cur->prev;
    }
    result.set("loss_history", std::move(loss_history));

    return result;
}

// ── getmodelstate ───────────────────────────────────────────────────

static JsonValue rpc_getmodelstate(const RPCRequest& req,
                                   node::NodeContext& ctx) {
    JsonValue result = JsonValue::object();

    if (!ctx.chainstate || !ctx.chainstate->tip()) {
        result.set("d_model", JsonValue(static_cast<int64_t>(384)));
        result.set("n_layers", JsonValue(static_cast<int64_t>(6)));
        result.set("n_slots", JsonValue(static_cast<int64_t>(64)));
        result.set("d_ff", JsonValue(static_cast<int64_t>(768)));
        result.set("vocab_size", JsonValue(static_cast<int64_t>(50257)));
        result.set("max_seq_len", JsonValue(static_cast<int64_t>(2048)));
        result.set("n_conv_branches", JsonValue(static_cast<int64_t>(5)));
        result.set("model_params", JsonValue(static_cast<int64_t>(0)));
        return result;
    }

    const auto& hdr = ctx.chainstate->tip()->header;

    result.set("d_model", JsonValue(static_cast<int64_t>(hdr.d_model)));
    result.set("n_layers", JsonValue(static_cast<int64_t>(hdr.n_layers)));
    result.set("n_slots", JsonValue(static_cast<int64_t>(hdr.n_slots)));
    result.set("d_ff", JsonValue(static_cast<int64_t>(hdr.d_ff)));
    result.set("vocab_size",
               JsonValue(static_cast<int64_t>(hdr.vocab_size)));
    result.set("max_seq_len",
               JsonValue(static_cast<int64_t>(hdr.max_seq_len)));
    result.set("n_conv_branches",
               JsonValue(static_cast<int64_t>(hdr.n_conv_branches)));
    result.set("model_params",
               JsonValue(static_cast<int64_t>(hdr.model_param_count())));

    // Kernel sizes
    JsonValue kernels = JsonValue::array();
    for (int i = 0; i < hdr.n_conv_branches && i < 8; ++i) {
        if (hdr.kernel_sizes[i] == 0) break;
        kernels.push_back(JsonValue(static_cast<int64_t>(hdr.kernel_sizes[i])));
    }
    result.set("kernel_sizes", std::move(kernels));

    return result;
}

// ── getgrowthinfo ───────────────────────────────────────────────────

static JsonValue rpc_getgrowthinfo(const RPCRequest& req,
                                   node::NodeContext& ctx) {
    JsonValue result = JsonValue::object();

    if (!ctx.chainstate || !ctx.chainstate->tip()) {
        result.set("growth_events", JsonValue(static_cast<int64_t>(0)));
        result.set("current_d_model", JsonValue(static_cast<int64_t>(384)));
        result.set("current_n_layers", JsonValue(static_cast<int64_t>(6)));
        result.set("stagnation_count", JsonValue(static_cast<int64_t>(0)));
        result.set("next_growth_threshold", JsonValue(static_cast<int64_t>(0)));
        return result;
    }

    const auto& hdr = ctx.chainstate->tip()->header;

    result.set("current_d_model",
               JsonValue(static_cast<int64_t>(hdr.d_model)));
    result.set("current_n_layers",
               JsonValue(static_cast<int64_t>(hdr.n_layers)));
    result.set("stagnation_count",
               JsonValue(static_cast<int64_t>(hdr.stagnation_count)));
    result.set("growth_delta",
               JsonValue(static_cast<int64_t>(hdr.growth_delta)));

    // Count growth events in chain history
    int growth_events = 0;
    const chain::CBlockIndex* cur = ctx.chainstate->tip();
    uint32_t prev_d_model = 384;
    uint32_t prev_n_layers = 6;

    while (cur) {
        if (cur->header.growth_delta > 0) {
            ++growth_events;
        }
        if (cur->d_model != prev_d_model || cur->n_layers != prev_n_layers) {
            // Track dimension changes
        }
        prev_d_model = cur->d_model;
        prev_n_layers = cur->n_layers;
        cur = cur->prev;
    }

    result.set("growth_events",
               JsonValue(static_cast<int64_t>(growth_events)));

    // Growth history: list recent growth events
    JsonValue history = JsonValue::array();
    cur = ctx.chainstate->tip();
    int found = 0;
    while (cur && found < 10) {
        if (cur->header.growth_delta > 0) {
            JsonValue entry = JsonValue::object();
            entry.set("height",
                       JsonValue(static_cast<int64_t>(cur->height)));
            entry.set("growth_delta",
                       JsonValue(static_cast<int64_t>(
                           cur->header.growth_delta)));
            entry.set("d_model",
                       JsonValue(static_cast<int64_t>(cur->d_model)));
            entry.set("n_layers",
                       JsonValue(static_cast<int64_t>(cur->n_layers)));
            entry.set("val_loss",
                       JsonValue(static_cast<double>(cur->val_loss)));
            history.push_back(std::move(entry));
            ++found;
        }
        cur = cur->prev;
    }
    result.set("growth_history", std::move(history));

    return result;
}

// ── Registration ────────────────────────────────────────────────────

void register_training_rpcs(RPCTable& table) {
    table.register_command({
        "gettraininginfo",
        rpc_gettraininginfo,
        "Returns Proof-of-Training information at the chain tip.\n"
        "Includes val_loss, train_steps, model_params, and loss history.",
        "Training"
    });

    table.register_command({
        "getmodelstate",
        rpc_getmodelstate,
        "Returns the current neural network model configuration.\n"
        "Fields: d_model, n_layers, n_slots, d_ff, vocab_size, kernel_sizes.",
        "Training"
    });

    table.register_command({
        "getgrowthinfo",
        rpc_getgrowthinfo,
        "Returns information about model growth events.\n"
        "Shows stagnation count, growth history, and dimension changes.",
        "Training"
    });
}

}  // namespace rnet::rpc

// Copyright (c) 2024-2026 The ResonanceNet Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "miner/training_solver.h"

#include "core/logging.h"
#include "core/random.h"
#include "core/time.h"
#include "crypto/keccak.h"
#include "training/checkpoint_io.h"
#include "training/model_config.h"

#include <sstream>

namespace rnet::miner {

// ---------------------------------------------------------------------------
// TrainingSolver (constructor)
// ---------------------------------------------------------------------------
TrainingSolver::TrainingSolver(gpu::GpuBackend& backend,
                               const consensus::ConsensusParams& params)
    : backend_(backend)
    , params_(params)
    , engine_(backend) {}

// ---------------------------------------------------------------------------
// ~TrainingSolver
// ---------------------------------------------------------------------------
TrainingSolver::~TrainingSolver() = default;

// ---------------------------------------------------------------------------
// set_output_dir
// ---------------------------------------------------------------------------
void TrainingSolver::set_output_dir(const std::filesystem::path& dir) {
    output_dir_ = dir;
}

// ---------------------------------------------------------------------------
// is_interrupted
// ---------------------------------------------------------------------------
bool TrainingSolver::is_interrupted() const {
    return interrupted_.load(std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// interrupt
// ---------------------------------------------------------------------------
void TrainingSolver::interrupt() {
    interrupted_.store(true, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// reset_interrupt
// ---------------------------------------------------------------------------
void TrainingSolver::reset_interrupt() {
    interrupted_.store(false, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// solve
// ---------------------------------------------------------------------------
// Core Proof-of-Training mining procedure.
//
// Workflow:
//   1. Validate step count against consensus bounds
//   2. Load parent checkpoint into training engine
//   3. Check for model growth event (d_model += 2 on improvement)
//   4. Train for N steps on consensus dataset
//   5. Evaluate on validation set
//   6. Accept if val_loss < parent_val_loss (block found)
//   7. Save checkpoint and compute Keccak-256d hash
//
// Success criterion:
//   val_loss_new < val_loss_parent
//
// Improvement metric:
//   improvement = (parent_loss - val_loss) / parent_loss
// ---------------------------------------------------------------------------
Result<SolverResult> TrainingSolver::solve(
    const primitives::CBlockHeader& parent_header,
    const std::filesystem::path& checkpoint_path,
    training::DataLoader& train_data,
    training::DataLoader& val_data,
    int n_steps) {

    core::Timer timer;

    // 1. Validate step count against consensus bounds.
    if (n_steps < params_.min_steps_per_block) {
        return Result<SolverResult>::err("step count below minimum: " +
            std::to_string(n_steps) + " < " +
            std::to_string(params_.min_steps_per_block));
    }
    if (n_steps > params_.max_steps_per_block) {
        return Result<SolverResult>::err("step count above maximum: " +
            std::to_string(n_steps) + " > " +
            std::to_string(params_.max_steps_per_block));
    }

    // 2. Load parent checkpoint into the training engine.
    auto parent_config = training::ModelConfig::from_block_header(parent_header);

    LogPrint(MINING, "TrainingSolver: loading checkpoint for height %llu, "
             "d_model=%u, n_layers=%u, parent_loss=%.6f",
             static_cast<unsigned long long>(parent_header.height),
             parent_config.d_model, parent_config.n_layers,
             parent_header.val_loss);

    auto init_result = engine_.init(parent_config);
    if (init_result.is_err()) {
        return Result<SolverResult>::err("failed to init engine: " + init_result.error());
    }

    auto load_result = engine_.load_checkpoint(checkpoint_path);
    if (load_result.is_err()) {
        return Result<SolverResult>::err("failed to load checkpoint: " + load_result.error());
    }

    if (is_interrupted()) {
        return Result<SolverResult>::err("interrupted before training");
    }

    // 3. Check for model growth event (d_model += 2 on improvement).
    consensus::GrowthState gstate{
        parent_header.d_model,
        parent_header.n_layers,
        parent_header.stagnation_count,
        parent_header.val_loss
    };

    // Optimistically assume improvement — growth is computed pre-training.
    auto growth = consensus::GrowthPolicy::compute_growth(gstate, true);

    training::ModelConfig new_config = parent_config;
    if (growth.delta_d_model > 0 || growth.layer_added) {
        new_config.d_model = growth.new_d_model;
        new_config.n_layers = growth.new_n_layers;
        new_config.d_ff = growth.new_d_ff;

        LogPrint(MINING, "TrainingSolver: growth event! d_model %u->%u, "
                 "n_layers %u->%u",
                 parent_config.d_model, new_config.d_model,
                 parent_config.n_layers, new_config.n_layers);

        auto expand_result = engine_.expand_model(new_config);
        if (expand_result.is_err()) {
            return Result<SolverResult>::err("failed to expand model: " +
                expand_result.error());
        }
    }

    if (is_interrupted()) {
        return Result<SolverResult>::err("interrupted before training");
    }

    // 4. Train for N steps on the consensus dataset.
    LogPrint(MINING, "TrainingSolver: training for %d steps...", n_steps);

    train_data.reset();
    auto train_result = engine_.train_steps(n_steps, train_data);
    if (train_result.is_err()) {
        return Result<SolverResult>::err("training failed: " + train_result.error());
    }

    float train_loss = train_result.value();
    LogPrint(MINING, "TrainingSolver: training loss = %.6f", train_loss);

    if (is_interrupted()) {
        return Result<SolverResult>::err("interrupted after training");
    }

    // 5. Evaluate on the validation set.
    LogPrint(MINING, "TrainingSolver: evaluating on validation set...");

    val_data.reset();
    auto eval_result = engine_.evaluate(val_data, params_.eval_batches);
    if (eval_result.is_err()) {
        return Result<SolverResult>::err("evaluation failed: " + eval_result.error());
    }

    float val_loss = eval_result.value();
    float parent_loss = parent_header.val_loss;

    LogPrint(MINING, "TrainingSolver: val_loss=%.6f, parent_loss=%.6f, "
             "improved=%s, time=%.1fs",
             val_loss, parent_loss,
             (val_loss < parent_loss) ? "YES" : "NO",
             timer.elapsed_sec());

    // 6. Accept only if improvement meets difficulty threshold.
    //
    //    required: parent_loss - val_loss >= difficulty_delta
    //
    const float delta = parent_loss - val_loss;
    if (delta < parent_header.difficulty_delta) {
        return Result<SolverResult>::err("insufficient improvement: delta=" +
            std::to_string(delta) + " < required=" +
            std::to_string(parent_header.difficulty_delta));
    }

    // 7. Save checkpoint and compute Keccak-256d hash.
    auto ckpt_path = make_checkpoint_path(parent_header.height + 1);
    auto save_result = engine_.save_checkpoint(ckpt_path);
    if (save_result.is_err()) {
        return Result<SolverResult>::err("failed to save checkpoint: " +
            save_result.error());
    }

    auto hash_result = hash_checkpoint(ckpt_path);
    if (hash_result.is_err()) {
        return Result<SolverResult>::err("failed to hash checkpoint: " +
            hash_result.error());
    }

    // Build result.
    //
    //   improvement = (parent_loss - val_loss) / parent_loss
    //
    SolverResult result;
    result.val_loss = val_loss;
    result.improvement = (parent_loss - val_loss) / parent_loss;
    result.train_steps = static_cast<uint32_t>(n_steps);
    result.checkpoint_hash = hash_result.value();
    result.dataset_hash = train_data.dataset_hash();
    result.checkpoint_path = ckpt_path;
    result.model_config = new_config;

    LogPrint(MINING, "TrainingSolver: BLOCK FOUND! height=%llu, "
             "val_loss=%.6f, improvement=%.4f%%, time=%.1fs",
             static_cast<unsigned long long>(parent_header.height + 1),
             val_loss, result.improvement * 100.0f,
             timer.elapsed_sec());

    return Result<SolverResult>::ok(std::move(result));
}

// ---------------------------------------------------------------------------
// hash_checkpoint
// ---------------------------------------------------------------------------
Result<rnet::uint256> TrainingSolver::hash_checkpoint(
    const std::filesystem::path& path) {
    auto result = crypto::keccak256d_file(path);
    if (result.is_err()) {
        return Result<rnet::uint256>::err("keccak256d_file failed: " + result.error());
    }
    return Result<rnet::uint256>::ok(result.value());
}

// ---------------------------------------------------------------------------
// make_checkpoint_path
// ---------------------------------------------------------------------------
std::filesystem::path TrainingSolver::make_checkpoint_path(uint64_t height) {
    std::ostringstream oss;
    oss << "checkpoint_" << height << "_"
        << core::get_rand_u32() << ".rnet";
    return output_dir_ / oss.str();
}

} // namespace rnet::miner

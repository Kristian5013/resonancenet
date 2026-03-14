#include "miner/training_solver.h"

#include <sstream>

#include "core/logging.h"
#include "core/random.h"
#include "core/time.h"
#include "crypto/keccak.h"
#include "training/checkpoint_io.h"
#include "training/model_config.h"

namespace rnet::miner {

TrainingSolver::TrainingSolver(gpu::GpuBackend& backend,
                               const consensus::ConsensusParams& params)
    : backend_(backend)
    , params_(params)
    , engine_(backend) {}

TrainingSolver::~TrainingSolver() = default;

void TrainingSolver::set_output_dir(const std::filesystem::path& dir) {
    output_dir_ = dir;
}

bool TrainingSolver::is_interrupted() const {
    return interrupted_.load(std::memory_order_relaxed);
}

void TrainingSolver::interrupt() {
    interrupted_.store(true, std::memory_order_relaxed);
}

void TrainingSolver::reset_interrupt() {
    interrupted_.store(false, std::memory_order_relaxed);
}

Result<SolverResult> TrainingSolver::solve(
    const primitives::CBlockHeader& parent_header,
    const std::filesystem::path& checkpoint_path,
    training::DataLoader& train_data,
    training::DataLoader& val_data,
    int n_steps) {

    core::Timer timer;

    // Validate step count
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

    // Step 1: Derive model config from parent header
    auto parent_config = training::ModelConfig::from_block_header(parent_header);

    LogPrint(MINING, "TrainingSolver: loading checkpoint for height %llu, "
             "d_model=%u, n_layers=%u, parent_loss=%.6f",
             static_cast<unsigned long long>(parent_header.height),
             parent_config.d_model, parent_config.n_layers,
             parent_header.val_loss);

    // Step 2: Load parent checkpoint into the training engine
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

    // Step 3: Check for growth event
    consensus::GrowthState gstate{
        parent_header.d_model,
        parent_header.n_layers,
        parent_header.stagnation_count,
        parent_header.val_loss
    };

    // We optimistically assume improvement (growth is computed pre-training).
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

    // Step 4: Train for N steps
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

    // Step 5: Evaluate on validation set
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

    // Step 6: Check if we improved
    if (val_loss >= parent_loss) {
        return Result<SolverResult>::err("no improvement: val_loss=" +
            std::to_string(val_loss) + " >= parent_loss=" +
            std::to_string(parent_loss));
    }

    // Step 7: Save checkpoint
    auto ckpt_path = make_checkpoint_path(parent_header.height + 1);
    auto save_result = engine_.save_checkpoint(ckpt_path);
    if (save_result.is_err()) {
        return Result<SolverResult>::err("failed to save checkpoint: " +
            save_result.error());
    }

    // Step 8: Hash the checkpoint
    auto hash_result = hash_checkpoint(ckpt_path);
    if (hash_result.is_err()) {
        return Result<SolverResult>::err("failed to hash checkpoint: " +
            hash_result.error());
    }

    // Build result
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

Result<rnet::uint256> TrainingSolver::hash_checkpoint(
    const std::filesystem::path& path) {
    auto result = crypto::keccak256d_file(path);
    if (result.is_err()) {
        return Result<rnet::uint256>::err("keccak256d_file failed: " + result.error());
    }
    return Result<rnet::uint256>::ok(result.value());
}

std::filesystem::path TrainingSolver::make_checkpoint_path(uint64_t height) {
    std::ostringstream oss;
    oss << "checkpoint_" << height << "_"
        << core::get_rand_u32() << ".rnet";
    return output_dir_ / oss.str();
}

}  // namespace rnet::miner

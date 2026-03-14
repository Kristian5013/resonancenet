#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>

#include "consensus/growth_policy.h"
#include "consensus/params.h"
#include "core/error.h"
#include "gpu/backend.h"
#include "primitives/block_header.h"
#include "training/data_loader.h"
#include "training/model_config.h"
#include "training/training_engine.h"

namespace rnet::miner {

/// Result of a successful PoT solve attempt.
struct SolverResult {
    float val_loss = 0.0f;               ///< Achieved validation loss
    float improvement = 0.0f;            ///< Fractional improvement over parent
    uint32_t train_steps = 0;            ///< Actual steps trained
    rnet::uint256 checkpoint_hash{};     ///< Hash of the output checkpoint
    rnet::uint256 dataset_hash{};        ///< Keccak-256d hash of the training dataset
    std::filesystem::path checkpoint_path; ///< Path to saved checkpoint
    training::ModelConfig model_config;  ///< Model config used (may have grown)
};

/// Training solver: the core PoT mining engine.
///
/// Workflow per attempt:
///   1. Load parent checkpoint
///   2. Check if growth event triggered -> expand model
///   3. Train for N steps
///   4. Evaluate on validation set
///   5. If val_loss < parent_val_loss -> success (block found)
class TrainingSolver {
public:
    TrainingSolver(gpu::GpuBackend& backend,
                   const consensus::ConsensusParams& params);
    ~TrainingSolver();

    // Non-copyable
    TrainingSolver(const TrainingSolver&) = delete;
    TrainingSolver& operator=(const TrainingSolver&) = delete;

    /// Attempt to mine a block by training the model.
    ///
    /// @param parent_header    Parent block header (for model config, val_loss).
    /// @param checkpoint_path  Path to the parent's model checkpoint.
    /// @param train_data       Training data loader.
    /// @param val_data         Validation data loader.
    /// @param n_steps          Number of training steps to perform.
    /// @return SolverResult on success, error if training failed or no improvement.
    Result<SolverResult> solve(const primitives::CBlockHeader& parent_header,
                               const std::filesystem::path& checkpoint_path,
                               training::DataLoader& train_data,
                               training::DataLoader& val_data,
                               int n_steps);

    /// Set the output directory for new checkpoints.
    void set_output_dir(const std::filesystem::path& dir);

    /// Check if the solver has been interrupted.
    bool is_interrupted() const;

    /// Request interruption (thread-safe).
    void interrupt();

    /// Reset interruption flag.
    void reset_interrupt();

private:
    gpu::GpuBackend& backend_;
    const consensus::ConsensusParams& params_;
    training::TrainingEngine engine_;
    std::filesystem::path output_dir_;
    std::atomic<bool> interrupted_{false};

    /// Compute the checkpoint hash for a saved checkpoint file.
    Result<rnet::uint256> hash_checkpoint(const std::filesystem::path& path);

    /// Generate a unique checkpoint filename.
    std::filesystem::path make_checkpoint_path(uint64_t height);
};

}  // namespace rnet::miner

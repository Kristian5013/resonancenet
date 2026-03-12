#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <thread>

#include "consensus/params.h"
#include "core/error.h"
#include "core/sync.h"
#include "crypto/ed25519.h"
#include "gpu/backend.h"
#include "miner/block_template.h"
#include "miner/training_solver.h"
#include "primitives/block.h"
#include "training/data_loader.h"

namespace rnet::miner {

/// Mining worker statistics.
struct WorkerStats {
    uint64_t attempts = 0;           ///< Total solve attempts
    uint64_t solutions_found = 0;    ///< Successful blocks mined
    uint64_t total_steps = 0;        ///< Total training steps across all attempts
    double total_time_sec = 0.0;     ///< Total wall-clock time mining
    float best_loss = 1e9f;          ///< Best validation loss achieved
};

/// Callback invoked when a worker finds a valid block.
using BlockFoundCallback = std::function<void(const primitives::CBlock& block)>;

/// Mining worker: runs a single training solver in a dedicated thread.
class MiningWorker {
public:
    MiningWorker(int id,
                 gpu::GpuBackend& backend,
                 const consensus::ConsensusParams& params);
    ~MiningWorker();

    // Non-copyable, non-movable
    MiningWorker(const MiningWorker&) = delete;
    MiningWorker& operator=(const MiningWorker&) = delete;

    /// Start the worker thread.
    /// The worker loops: build template -> solve -> callback if found.
    void start(BlockFoundCallback callback);

    /// Stop the worker thread gracefully.
    void stop();

    /// Update the block template for the next mining attempt.
    /// Thread-safe: the worker picks up the new template on its next iteration.
    void update_template(std::shared_ptr<BlockTemplate> tmpl);

    /// Update training and validation data paths.
    void set_data_paths(const std::filesystem::path& train_data_path,
                        const std::filesystem::path& val_data_path);

    /// Set the checkpoint directory.
    void set_checkpoint_dir(const std::filesystem::path& dir);

    /// Set the parent checkpoint path (tip of chain).
    void set_parent_checkpoint(const std::filesystem::path& path);

    /// Set the number of training steps per attempt.
    void set_steps_per_attempt(int steps);

    /// Get worker ID.
    int id() const { return id_; }

    /// Get worker statistics.
    WorkerStats stats() const;

    /// Check if the worker is currently running.
    bool is_running() const { return running_.load(); }

private:
    void run();

    int id_;
    gpu::GpuBackend& backend_;
    const consensus::ConsensusParams& params_;
    TrainingSolver solver_;

    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};

    mutable core::Mutex mutex_;
    std::shared_ptr<BlockTemplate> current_template_;
    BlockFoundCallback callback_;

    std::filesystem::path train_data_path_;
    std::filesystem::path val_data_path_;
    std::filesystem::path checkpoint_dir_;
    std::filesystem::path parent_checkpoint_;
    int steps_per_attempt_ = 0;

    WorkerStats stats_;
    training::DataLoader train_loader_;
    training::DataLoader val_loader_;
    bool data_loaded_ = false;
};

}  // namespace rnet::miner

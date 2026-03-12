#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include "consensus/block_reward.h"
#include "consensus/params.h"
#include "core/error.h"
#include "core/sync.h"
#include "crypto/ed25519.h"
#include "gpu/backend.h"
#include "miner/block_template.h"
#include "miner/worker.h"
#include "primitives/block.h"
#include "primitives/transaction.h"

namespace rnet::miner {

/// Miner configuration.
struct MinerConfig {
    int num_workers = 1;                             ///< Number of parallel mining workers
    int steps_per_attempt = 1000;                    ///< Training steps per mining attempt
    std::filesystem::path train_data_path;           ///< Path to training dataset
    std::filesystem::path val_data_path;             ///< Path to validation dataset
    std::filesystem::path checkpoint_dir;            ///< Directory for model checkpoints
    crypto::Ed25519PublicKey miner_pubkey{};         ///< Miner's reward public key
};

/// Aggregate mining statistics.
struct MinerStats {
    uint64_t total_attempts = 0;
    uint64_t blocks_found = 0;
    double total_time_sec = 0.0;
    float best_loss = 1e9f;
    double hashrate_equivalent = 0.0;  ///< Attempts per second (PoT analog of hashrate)
};

/// Main miner class: coordinates workers, templates, and chain tip updates.
class Miner {
public:
    Miner(gpu::GpuBackend& backend,
          const consensus::ConsensusParams& params);
    ~Miner();

    // Non-copyable
    Miner(const Miner&) = delete;
    Miner& operator=(const Miner&) = delete;

    /// Initialize the miner with the given configuration.
    Result<void> init(const MinerConfig& config);

    /// Start mining on the current chain tip.
    Result<void> start();

    /// Stop mining gracefully. Blocks until all workers have stopped.
    void stop();

    /// Update the chain tip. Rebuilds the block template and notifies workers.
    ///
    /// @param tip_header     New chain tip header.
    /// @param tip_checkpoint Path to the tip's model checkpoint.
    /// @param mempool_txs    Current mempool transactions (sorted by fee rate).
    /// @param tx_fees        Fees for each mempool transaction.
    /// @param emission       Current emission state.
    void update_tip(const primitives::CBlockHeader& tip_header,
                    const std::filesystem::path& tip_checkpoint,
                    const std::vector<primitives::CTransactionRef>& mempool_txs,
                    const std::vector<int64_t>& tx_fees,
                    const consensus::EmissionState& emission);

    /// Set the callback for when a block is found.
    void set_block_found_callback(BlockFoundCallback callback);

    /// Check if the miner is currently active.
    bool is_mining() const { return mining_.load(); }

    /// Get aggregate statistics.
    MinerStats stats() const;

    /// Get per-worker statistics.
    std::vector<WorkerStats> worker_stats() const;

private:
    void on_block_found(const primitives::CBlock& block);
    void rebuild_template();

    gpu::GpuBackend& backend_;
    const consensus::ConsensusParams& params_;
    MinerConfig config_;

    mutable core::Mutex mutex_;
    std::vector<std::unique_ptr<MiningWorker>> workers_;
    BlockFoundCallback block_found_callback_;
    std::atomic<bool> mining_{false};
    std::atomic<bool> initialized_{false};

    // Current chain tip state (protected by mutex_)
    primitives::CBlockHeader tip_header_;
    std::filesystem::path tip_checkpoint_;
    std::vector<primitives::CTransactionRef> mempool_txs_;
    std::vector<int64_t> tx_fees_;
    consensus::EmissionState emission_;
};

}  // namespace rnet::miner

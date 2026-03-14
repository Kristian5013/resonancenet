// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "miner/miner.h"

#include "core/logging.h"
#include "core/thread.h"
#include "core/time.h"
#include "miner/block_template.h"
#include "miner/difficulty.h"

#include <algorithm>
#include <filesystem>

namespace rnet::miner {

// ---------------------------------------------------------------------------
// Miner lifecycle
//
// The Miner is the top-level coordinator for Proof-of-Training mining.
// It owns N MiningWorker threads, each running an independent training
// loop on the GPU.  When the chain tip advances the Miner rebuilds the
// block template and distributes it to every worker so they begin
// training from the new parent checkpoint.
//
// Start/stop is idempotent — calling start() twice is harmless, and the
// destructor always calls stop() to guarantee clean thread shutdown.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Miner::Miner
// ---------------------------------------------------------------------------
Miner::Miner(gpu::GpuBackend& backend,
             const consensus::ConsensusParams& params)
    : backend_(backend)
    , params_(params) {}

// ---------------------------------------------------------------------------
// Miner::~Miner
// ---------------------------------------------------------------------------
Miner::~Miner() {
    stop();
}

// ---------------------------------------------------------------------------
// Miner::init
// ---------------------------------------------------------------------------
Result<void> Miner::init(const MinerConfig& config) {
    // 1. Reject double-init.
    if (initialized_.load()) {
        return Result<void>::err("miner already initialized");
    }

    // 2. Validate required fields.
    if (config.miner_pubkey.is_zero()) {
        return Result<void>::err("miner public key not set");
    }
    if (config.train_data_path.empty()) {
        return Result<void>::err("training data path not set");
    }
    if (config.val_data_path.empty()) {
        return Result<void>::err("validation data path not set");
    }
    if (config.checkpoint_dir.empty()) {
        return Result<void>::err("checkpoint directory not set");
    }

    // 3. Ensure the checkpoint directory exists on disk.
    std::error_code ec;
    std::filesystem::create_directories(config.checkpoint_dir, ec);
    if (ec) {
        return Result<void>::err("failed to create checkpoint dir: " + ec.message());
    }

    // 4. Stash config and create the worker pool.
    {
        LOCK(mutex_);
        config_ = config;
    }

    int num_workers = std::max(1, config.num_workers);

    LOCK(mutex_);
    workers_.clear();
    workers_.reserve(static_cast<size_t>(num_workers));

    for (int i = 0; i < num_workers; ++i) {
        auto worker = std::make_unique<MiningWorker>(i, backend_, params_);
        worker->set_data_paths(config.train_data_path, config.val_data_path);
        worker->set_checkpoint_dir(config.checkpoint_dir);
        worker->set_steps_per_attempt(config.steps_per_attempt);
        workers_.push_back(std::move(worker));
    }

    // 5. Mark initialised so start() becomes callable.
    initialized_.store(true);

    LogPrint(MINING, "Miner: initialized with %d workers, steps=%d",
             num_workers, config.steps_per_attempt);

    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// Miner::start
// ---------------------------------------------------------------------------
Result<void> Miner::start() {
    // 1. Pre-conditions.
    if (!initialized_.load()) {
        return Result<void>::err("miner not initialized");
    }
    if (mining_.load()) {
        return Result<void>::err("miner already running");
    }

    // 2. Launch every worker thread with our block-found relay.
    LOCK(mutex_);

    for (auto& worker : workers_) {
        worker->start([this](const primitives::CBlock& block) {
            on_block_found(block);
        });
    }

    // 3. Flip the running flag.
    mining_.store(true);
    LogPrint(MINING, "Miner: started %zu workers", workers_.size());

    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// Miner::stop
// ---------------------------------------------------------------------------
void Miner::stop() {
    if (!mining_.load()) {
        return;
    }

    LogPrint(MINING, "Miner: stopping...");

    // 1. Signal every worker to halt and join its thread.
    {
        LOCK(mutex_);
        for (auto& worker : workers_) {
            worker->stop();
        }
    }

    // 2. Clear the running flag.
    mining_.store(false);
    LogPrint(MINING, "Miner: stopped");
}

// ---------------------------------------------------------------------------
// Miner::update_tip
// ---------------------------------------------------------------------------
void Miner::update_tip(const primitives::CBlockHeader& tip_header,
                        const std::filesystem::path& tip_checkpoint,
                        const std::vector<primitives::CTransactionRef>& mempool_txs,
                        const std::vector<int64_t>& tx_fees,
                        const consensus::EmissionState& emission) {
    LOCK(mutex_);

    // 1. Cache the new chain-tip state.
    tip_header_ = tip_header;
    tip_checkpoint_ = tip_checkpoint;
    mempool_txs_ = mempool_txs;
    tx_fees_ = tx_fees;
    emission_ = emission;

    // 2. Build a fresh block template and push it to workers.
    rebuild_template();
}

// ---------------------------------------------------------------------------
// Miner::set_block_found_callback
// ---------------------------------------------------------------------------
void Miner::set_block_found_callback(BlockFoundCallback callback) {
    LOCK(mutex_);
    block_found_callback_ = std::move(callback);
}

// ---------------------------------------------------------------------------
// Miner::stats
// ---------------------------------------------------------------------------
MinerStats Miner::stats() const {
    MinerStats aggregate;

    LOCK(mutex_);
    for (const auto& worker : workers_) {
        auto ws = worker->stats();
        aggregate.total_attempts += ws.attempts;
        aggregate.blocks_found += ws.solutions_found;
        aggregate.total_time_sec += ws.total_time_sec;
        if (ws.best_loss < aggregate.best_loss) {
            aggregate.best_loss = ws.best_loss;
        }
    }

    if (aggregate.total_time_sec > 0.0) {
        aggregate.hashrate_equivalent =
            static_cast<double>(aggregate.total_attempts) / aggregate.total_time_sec;
    }

    return aggregate;
}

// ---------------------------------------------------------------------------
// Miner::worker_stats
// ---------------------------------------------------------------------------
std::vector<WorkerStats> Miner::worker_stats() const {
    std::vector<WorkerStats> result;
    LOCK(mutex_);
    result.reserve(workers_.size());
    for (const auto& worker : workers_) {
        result.push_back(worker->stats());
    }
    return result;
}

// ---------------------------------------------------------------------------
// Miner::on_block_found
// ---------------------------------------------------------------------------
void Miner::on_block_found(const primitives::CBlock& block) {
    LogPrint(MINING, "Miner: BLOCK FOUND! height=%llu, val_loss=%.6f, txs=%zu",
             static_cast<unsigned long long>(block.height),
             block.val_loss,
             block.vtx.size());

    // 1. Snapshot the callback under the lock.
    BlockFoundCallback cb;
    {
        LOCK(mutex_);
        cb = block_found_callback_;
    }

    // 2. Invoke outside the lock to avoid re-entrant deadlock.
    if (cb) {
        cb(block);
    }
}

// ---------------------------------------------------------------------------
// Miner::rebuild_template
// ---------------------------------------------------------------------------
void Miner::rebuild_template() {
    // Must be called under mutex_.

    // 1. Nothing to build if there is no parent checkpoint yet.
    if (tip_checkpoint_.empty()) {
        return;
    }

    // 2. Assemble a new block template from the current tip + mempool.
    auto tmpl = std::make_shared<BlockTemplate>(
        create_block_template(
            tip_header_,
            mempool_txs_,
            tx_fees_,
            config_.miner_pubkey,
            emission_,
            params_));

    // 3. Distribute the template and parent checkpoint to every worker.
    for (auto& worker : workers_) {
        worker->update_template(tmpl);
        worker->set_parent_checkpoint(tip_checkpoint_);
    }

    LogPrint(MINING, "Miner: template updated for height %llu, %zu txs",
             static_cast<unsigned long long>(tip_header_.height + 1),
             tmpl->tx_count());
}

} // namespace rnet::miner

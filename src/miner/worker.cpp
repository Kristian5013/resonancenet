#include "miner/worker.h"

#include "core/logging.h"
#include "core/thread.h"
#include "core/time.h"
#include "miner/difficulty.h"

namespace rnet::miner {

MiningWorker::MiningWorker(int id,
                           gpu::GpuBackend& backend,
                           const consensus::ConsensusParams& params)
    : id_(id)
    , backend_(backend)
    , params_(params)
    , solver_(backend, params) {}

MiningWorker::~MiningWorker() {
    stop();
}

void MiningWorker::start(BlockFoundCallback callback) {
    if (running_.load()) {
        return;
    }

    {
        LOCK(mutex_);
        callback_ = std::move(callback);
    }

    stop_requested_.store(false);
    running_.store(true);

    thread_ = std::thread([this]() {
        core::set_thread_name("miner-worker-" + std::to_string(id_));
        run();
    });
}

void MiningWorker::stop() {
    stop_requested_.store(true);
    solver_.interrupt();

    if (thread_.joinable()) {
        thread_.join();
    }

    running_.store(false);
}

void MiningWorker::update_template(std::shared_ptr<BlockTemplate> tmpl) {
    LOCK(mutex_);
    current_template_ = std::move(tmpl);
    // Interrupt current solve so worker picks up new template
    solver_.interrupt();
}

void MiningWorker::set_data_paths(const std::filesystem::path& train_data_path,
                                  const std::filesystem::path& val_data_path) {
    LOCK(mutex_);
    train_data_path_ = train_data_path;
    val_data_path_ = val_data_path;
    data_loaded_ = false;
}

void MiningWorker::set_checkpoint_dir(const std::filesystem::path& dir) {
    LOCK(mutex_);
    checkpoint_dir_ = dir;
    solver_.set_output_dir(dir);
}

void MiningWorker::set_parent_checkpoint(const std::filesystem::path& path) {
    LOCK(mutex_);
    parent_checkpoint_ = path;
}

void MiningWorker::set_steps_per_attempt(int steps) {
    LOCK(mutex_);
    steps_per_attempt_ = steps;
}

WorkerStats MiningWorker::stats() const {
    LOCK(mutex_);
    return stats_;
}

void MiningWorker::run() {
    LogPrint(MINING, "MiningWorker[%d]: started", id_);

    while (!stop_requested_.load()) {
        // Snapshot current state under lock
        std::shared_ptr<BlockTemplate> tmpl;
        std::filesystem::path ckpt_path;
        int steps = 0;

        {
            LOCK(mutex_);
            tmpl = current_template_;
            ckpt_path = parent_checkpoint_;
            steps = steps_per_attempt_;
        }

        // Wait if we don't have a template yet
        if (!tmpl || ckpt_path.empty() || steps <= 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // Load data if not yet loaded
        if (!data_loaded_) {
            std::filesystem::path train_path, val_path;
            {
                LOCK(mutex_);
                train_path = train_data_path_;
                val_path = val_data_path_;
            }

            if (!train_path.empty() && !val_path.empty()) {
                auto r1 = train_loader_.load_dataset(train_path);
                auto r2 = val_loader_.load_dataset(val_path);
                if (r1.is_ok() && r2.is_ok()) {
                    data_loaded_ = true;
                    LogPrint(MINING, "MiningWorker[%d]: loaded training data "
                             "(%zu tokens) and validation data (%zu tokens)",
                             id_,
                             train_loader_.total_tokens(),
                             val_loader_.total_tokens());
                } else {
                    LogError("MiningWorker[%d]: failed to load data", id_);
                    std::this_thread::sleep_for(std::chrono::seconds(5));
                    continue;
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
        }

        // Get suggested step count if not explicitly set
        if (steps <= 0) {
            steps = suggest_step_count(
                tmpl->block.prev_val_loss,
                tmpl->block.d_model,
                params_);
        }

        // Reset solver interrupt flag
        solver_.reset_interrupt();

        core::Timer timer;

        LogPrint(MINING, "MiningWorker[%d]: attempting solve, height=%llu, "
                 "steps=%d",
                 id_,
                 static_cast<unsigned long long>(tmpl->block.height),
                 steps);

        // Attempt to solve
        auto result = solver_.solve(
            // Use the template's parent header info
            // The template block has height = parent_height + 1,
            // so we reconstruct a parent-like header from the template.
            [&]() -> primitives::CBlockHeader {
                primitives::CBlockHeader parent;
                parent.height = tmpl->block.height - 1;
                parent.val_loss = tmpl->block.prev_val_loss;
                parent.d_model = tmpl->block.d_model - tmpl->block.growth_delta;
                if (tmpl->growth.layer_added) {
                    parent.n_layers = tmpl->block.n_layers - 1;
                } else {
                    parent.n_layers = tmpl->block.n_layers;
                }
                parent.n_slots = tmpl->block.n_slots;
                parent.d_ff = parent.d_model * 2;
                parent.vocab_size = tmpl->block.vocab_size;
                parent.max_seq_len = tmpl->block.max_seq_len;
                parent.n_conv_branches = tmpl->block.n_conv_branches;
                parent.kernel_sizes = tmpl->block.kernel_sizes;
                parent.stagnation_count = tmpl->block.stagnation_count;
                return parent;
            }(),
            ckpt_path,
            train_loader_,
            val_loader_,
            steps);

        double elapsed = timer.elapsed_sec();

        {
            LOCK(mutex_);
            stats_.attempts++;
            stats_.total_steps += static_cast<uint64_t>(steps);
            stats_.total_time_sec += elapsed;
        }

        if (result.is_ok()) {
            auto& sol = result.value();

            // Fill in the template block with solver results
            primitives::CBlock found_block = tmpl->block;
            found_block.checkpoint_hash = sol.checkpoint_hash;
            found_block.val_loss = sol.val_loss;
            found_block.train_steps = sol.train_steps;

            // Recompute merkle root (it shouldn't change but ensure consistency)
            found_block.merkle_root = found_block.compute_merkle_root();

            {
                LOCK(mutex_);
                stats_.solutions_found++;
                if (sol.val_loss < stats_.best_loss) {
                    stats_.best_loss = sol.val_loss;
                }
            }

            // Invoke callback
            BlockFoundCallback cb;
            {
                LOCK(mutex_);
                cb = callback_;
            }
            if (cb) {
                cb(found_block);
            }
        } else {
            LogDebug(MINING, "MiningWorker[%d]: no solution: %s (%.1fs)",
                     id_, result.error().c_str(), elapsed);
        }
    }

    LogPrint(MINING, "MiningWorker[%d]: stopped", id_);
}

}  // namespace rnet::miner

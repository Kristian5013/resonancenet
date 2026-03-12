#include "miner/block_assembler.h"

#include <algorithm>

#include "core/logging.h"
#include "core/time.h"
#include "miner/coinbase.h"

namespace rnet::miner {

BlockAssembler::BlockAssembler(const consensus::ConsensusParams& params)
    : params_(params) {}

void BlockAssembler::add_transactions(const std::vector<TxEntry>& entries) {
    candidates_.insert(candidates_.end(), entries.begin(), entries.end());
}

void BlockAssembler::select_transactions() {
    selected_.clear();
    total_fees_ = 0;
    total_weight_ = 0;

    // Sort candidates by fee rate, descending
    std::sort(candidates_.begin(), candidates_.end(),
              [](const TxEntry& a, const TxEntry& b) {
                  return a.fee_rate > b.fee_rate;
              });

    // Weight limit: max_block_size is in bytes, weight = 4 * base_size for witness discount.
    // Use max_block_size as a conservative weight limit.
    const size_t max_weight = static_cast<size_t>(params_.max_block_size);

    // Reserve space for the coinbase transaction (~200 weight units)
    constexpr size_t coinbase_weight_reserve = 400;

    for (const auto& entry : candidates_) {
        if (!entry.tx) {
            continue;
        }

        // Check weight limit
        if (total_weight_ + entry.weight + coinbase_weight_reserve > max_weight) {
            continue;  // Skip this tx, try smaller ones
        }

        selected_.push_back(entry.tx);
        total_fees_ += entry.fee;
        total_weight_ += entry.weight;
    }

    LogPrint(MINING, "BlockAssembler: selected %zu txs, fees=%lld, weight=%zu",
             selected_.size(), static_cast<long long>(total_fees_), total_weight_);
}

primitives::CBlock BlockAssembler::assemble(
    const primitives::CBlockHeader& parent,
    const crypto::Ed25519PublicKey& miner_pubkey,
    const consensus::BlockReward& reward,
    const consensus::GrowthResult& growth) {

    // Select transactions from candidates
    select_transactions();

    primitives::CBlock block;

    // --- Header fields ---
    block.version = parent.version;
    block.height = parent.height + 1;
    block.prev_hash = parent.hash();
    block.timestamp = static_cast<uint64_t>(core::get_time());

    // Ensure timestamp is strictly greater than parent
    if (block.timestamp <= parent.timestamp) {
        block.timestamp = parent.timestamp + 1;
    }

    // PoT fields: val_loss, checkpoint_hash filled in later by solver
    block.prev_val_loss = parent.val_loss;
    block.train_steps = 0;  // Filled by solver

    // Growth fields
    block.d_model = growth.new_d_model;
    block.n_layers = growth.new_n_layers;
    block.d_ff = growth.new_d_ff;
    block.n_slots = parent.n_slots;  // Slots don't change with growth
    block.vocab_size = parent.vocab_size;
    block.max_seq_len = parent.max_seq_len;
    block.n_conv_branches = parent.n_conv_branches;
    block.kernel_sizes = parent.kernel_sizes;
    block.growth_delta = growth.delta_d_model;
    block.stagnation_count = growth.new_stagnation;

    // Miner pubkey in header
    std::memcpy(block.miner_pubkey.data(), miner_pubkey.data.data(), 32);

    // --- Transactions ---

    // Create coinbase: reward + total_fees
    int64_t total_reward = reward.total() + total_fees_;
    auto coinbase_tx = create_coinbase_tx(block.height, total_reward, miner_pubkey);
    block.vtx.push_back(std::move(coinbase_tx));

    // Add selected transactions
    for (auto& tx : selected_) {
        block.vtx.push_back(std::move(tx));
    }

    // Compute merkle root
    block.merkle_root = block.compute_merkle_root();

    return block;
}

}  // namespace rnet::miner

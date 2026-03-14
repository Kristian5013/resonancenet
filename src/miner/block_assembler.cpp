// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "miner/block_assembler.h"

#include "core/logging.h"
#include "core/time.h"
#include "miner/coinbase.h"

#include <algorithm>
#include <cstring>

namespace rnet::miner {

// ---------------------------------------------------------------------------
// Design note — Transaction selection by fee rate, coinbase construction
//
// BlockAssembler implements a greedy knapsack: candidate transactions are
// sorted by fee-rate (resonances per 1 000 weight units, descending) and
// packed into the block until the weight ceiling is reached.  A fixed
// reserve is held back for the coinbase transaction so that the block
// never exceeds the consensus weight limit.
//
// After selection the assembler builds the full CBlock:
//   1. Populate header fields from the parent (height, prev_hash, growth).
//   2. Create the coinbase transaction (base reward + collected fees).
//   3. Append the selected transactions.
//   4. Compute the Merkle root over all vtx entries.
//
// PoT-specific fields (val_loss, train_steps, checkpoint_hash) are left
// at their default values here — the mining solver fills them in once
// training completes.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// BlockAssembler — constructor
// ---------------------------------------------------------------------------

BlockAssembler::BlockAssembler(const consensus::ConsensusParams& params)
    : params_(params) {}

// ---------------------------------------------------------------------------
// add_transactions
// ---------------------------------------------------------------------------

void BlockAssembler::add_transactions(const std::vector<TxEntry>& entries) {
    candidates_.insert(candidates_.end(), entries.begin(), entries.end());
}

// ---------------------------------------------------------------------------
// select_transactions
// ---------------------------------------------------------------------------

void BlockAssembler::select_transactions() {
    selected_.clear();
    total_fees_ = 0;
    total_weight_ = 0;

    // 1. Sort candidates by fee rate (descending) — highest-paying first.
    std::sort(candidates_.begin(), candidates_.end(),
              [](const TxEntry& a, const TxEntry& b) {
                  return a.fee_rate > b.fee_rate;
              });

    // 2. Determine the weight ceiling from consensus params.
    //    max_block_size is in bytes; weight = 4 * base_size with witness
    //    discount, so using max_block_size directly is conservative.
    const size_t max_weight = static_cast<size_t>(params_.max_block_size);

    // 3. Reserve space for the coinbase transaction (~400 weight units).
    constexpr size_t kCoinbaseWeightReserve = 400;

    // 4. Greedily pack transactions that fit under the limit.
    for (const auto& entry : candidates_) {
        if (!entry.tx) {
            continue;
        }

        if (total_weight_ + entry.weight + kCoinbaseWeightReserve > max_weight) {
            continue;  // skip this tx, try smaller ones
        }

        selected_.push_back(entry.tx);
        total_fees_ += entry.fee;
        total_weight_ += entry.weight;
    }

    LogPrint(MINING, "BlockAssembler: selected %zu txs, fees=%lld, weight=%zu",
             selected_.size(), static_cast<long long>(total_fees_), total_weight_);
}

// ---------------------------------------------------------------------------
// assemble
// ---------------------------------------------------------------------------

primitives::CBlock BlockAssembler::assemble(
    const primitives::CBlockHeader& parent,
    const crypto::Ed25519PublicKey& miner_pubkey,
    const consensus::BlockReward& reward,
    const consensus::GrowthResult& growth) {

    // 1. Run the greedy fee-rate selection over all candidates.
    select_transactions();

    primitives::CBlock block;

    // 2. Populate header fields from the parent block.
    block.version   = parent.version;
    block.height    = parent.height + 1;
    block.prev_hash = parent.hash();
    block.timestamp = static_cast<uint64_t>(core::get_time());

    // 3. Enforce strictly-increasing timestamps.
    if (block.timestamp <= parent.timestamp) {
        block.timestamp = parent.timestamp + 1;
    }

    // 4. PoT fields — val_loss and checkpoint_hash filled by the solver.
    block.prev_val_loss = parent.val_loss;
    block.train_steps   = 0;

    // 5. Growth fields from the consensus growth policy.
    block.d_model          = growth.new_d_model;
    block.n_layers         = growth.new_n_layers;
    block.d_ff             = growth.new_d_ff;
    block.n_slots          = parent.n_slots;
    block.vocab_size       = parent.vocab_size;
    block.max_seq_len      = parent.max_seq_len;
    block.n_conv_branches  = parent.n_conv_branches;
    block.kernel_sizes     = parent.kernel_sizes;
    block.growth_delta     = growth.delta_d_model;
    block.stagnation_count = growth.new_stagnation;

    // 6. Copy the miner's 32-byte Ed25519 public key into the header.
    std::memcpy(block.miner_pubkey.data(), miner_pubkey.data.data(), 32);

    // 7. Create coinbase: base reward + total collected fees.
    int64_t total_reward = reward.total() + total_fees_;
    auto coinbase_tx = create_coinbase_tx(block.height, total_reward, miner_pubkey);
    block.vtx.push_back(std::move(coinbase_tx));

    // 8. Append the selected (fee-paying) transactions.
    for (auto& tx : selected_) {
        block.vtx.push_back(std::move(tx));
    }

    // 9. Compute the Merkle root over all transactions.
    block.merkle_root = block.compute_merkle_root();

    return block;
}

} // namespace rnet::miner

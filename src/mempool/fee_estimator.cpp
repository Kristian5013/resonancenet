// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "mempool/fee_estimator.h"

#include <algorithm>
#include <numeric>

namespace rnet::mempool {

// ---------------------------------------------------------------------------
// FeeEstimator :: constructor
// ---------------------------------------------------------------------------

FeeEstimator::FeeEstimator() {
    buckets_.fill({0, 0});
}

// ---------------------------------------------------------------------------
// FeeEstimator :: track_tx
// ---------------------------------------------------------------------------
//
// Design note — fee-rate history
//   Every transaction entering the mempool is recorded with its fee
//   rate and entry height.  A rolling window of recent_fee_rates_
//   (capped at MAX_HISTORY) serves as fallback data when the
//   confirmation-bucket statistics are too sparse to be useful.
// ---------------------------------------------------------------------------

void FeeEstimator::track_tx(const rnet::uint256& txid, int64_t fee_rate,
                            int entry_height) {
    LOCK(mutex_);

    // 1. Store the transaction for later confirmation matching.
    tracked_[txid] = TrackedTx{fee_rate, entry_height};

    // 2. Append to the rolling fee-rate window.
    recent_fee_rates_.push_back(fee_rate);
    if (recent_fee_rates_.size() > MAX_HISTORY) {
        recent_fee_rates_.pop_front();
    }
}

// ---------------------------------------------------------------------------
// FeeEstimator :: confirm_tx
// ---------------------------------------------------------------------------
//
// Design note — confirmation buckets
//   When a tracked transaction is confirmed we compute how many blocks
//   elapsed since it entered the mempool.  That delta selects one of
//   MAX_TARGET buckets, where we accumulate the total fee rate and the
//   number of samples.  The ratio gives the average fee rate needed to
//   confirm within N blocks.
// ---------------------------------------------------------------------------

void FeeEstimator::confirm_tx(const rnet::uint256& txid,
                              int confirm_height) {
    LOCK(mutex_);

    // 1. Look up the tracked entry; bail if unknown.
    auto it = tracked_.find(txid);
    if (it == tracked_.end()) return;

    // 2. Compute how many blocks the tx waited.
    int blocks_to_confirm = confirm_height - it->second.entry_height;
    if (blocks_to_confirm < 1) blocks_to_confirm = 1;

    // 3. Record in the corresponding bucket (if within range).
    if (blocks_to_confirm <= MAX_TARGET) {
        auto& bucket = buckets_[static_cast<size_t>(blocks_to_confirm - 1)];
        bucket.total_fee_rate += it->second.fee_rate;
        bucket.count++;
    }

    // 4. Stop tracking this transaction.
    tracked_.erase(it);
}

// ---------------------------------------------------------------------------
// FeeEstimator :: process_block
// ---------------------------------------------------------------------------

void FeeEstimator::process_block(
    int height, const std::vector<rnet::uint256>& confirmed_txids)
{
    for (const auto& txid : confirmed_txids) {
        confirm_tx(txid, height);
    }
}

// ---------------------------------------------------------------------------
// FeeEstimator :: estimate_fee
// ---------------------------------------------------------------------------
//
// Design note — estimation strategy
//   The estimator aggregates bucket data from block 1 up to the
//   requested target.  If there are enough samples the weighted
//   average gives a reliable fee rate.  When data is scarce (e.g.
//   right after startup) the median of the recent fee-rate window
//   is used as a heuristic fallback.  If even that is empty the
//   minimum relay fee is returned so the caller always receives a
//   usable value.
// ---------------------------------------------------------------------------

primitives::CFeeRate FeeEstimator::estimate_fee(int target_blocks) const {
    LOCK(mutex_);

    // 1. Clamp the target to the valid range.
    if (target_blocks < 1) target_blocks = 1;
    if (target_blocks > MAX_TARGET) target_blocks = MAX_TARGET;

    // 2. Accumulate bucket statistics up to the target depth.
    int64_t total_rate = 0;
    int64_t total_count = 0;

    for (int i = 0; i < target_blocks && i < MAX_TARGET; ++i) {
        total_rate += buckets_[static_cast<size_t>(i)].total_fee_rate;
        total_count += buckets_[static_cast<size_t>(i)].count;
    }

    // 3. Return the weighted average if we have data.
    if (total_count > 0) {
        int64_t avg_rate = total_rate / total_count;
        return primitives::CFeeRate(avg_rate);
    }

    // 4. Fallback: median of the recent fee-rate window.
    if (!recent_fee_rates_.empty()) {
        std::vector<int64_t> sorted(recent_fee_rates_.begin(),
                                    recent_fee_rates_.end());
        int64_t med = median(sorted);
        return primitives::CFeeRate(med);
    }

    // 5. No data at all — return the network minimum.
    return primitives::MIN_RELAY_TX_FEE;
}

// ---------------------------------------------------------------------------
// FeeEstimator :: estimate
// ---------------------------------------------------------------------------
//
// Design note — estimation modes
//   ECONOMICAL returns the raw estimate.  CONSERVATIVE adds a 10 %
//   safety margin to reduce the chance of the tx languishing in the
//   pool.  HIGH_PRIORITY doubles the estimate for users who need
//   next-block confirmation and are willing to overpay.
// ---------------------------------------------------------------------------

primitives::FeeEstimation FeeEstimator::estimate(
    int target_blocks, primitives::FeeEstimateTarget mode) const
{
    primitives::FeeEstimation result;
    result.target_blocks = target_blocks;
    result.mode = mode;

    // 1. Obtain the base estimate.
    auto rate = estimate_fee(target_blocks);

    // 2. Apply the mode-specific adjustment.
    switch (mode) {
        case primitives::FeeEstimateTarget::ECONOMICAL:
            break;
        case primitives::FeeEstimateTarget::CONSERVATIVE:
            rate = primitives::CFeeRate(
                rate.get_fee_per_kvb() + rate.get_fee_per_kvb() / 10);
            break;
        case primitives::FeeEstimateTarget::HIGH_PRIORITY:
            rate = primitives::CFeeRate(rate.get_fee_per_kvb() * 2);
            break;
    }

    result.rate = rate;
    return result;
}

// ---------------------------------------------------------------------------
// FeeEstimator :: min_relay_fee
// ---------------------------------------------------------------------------

primitives::CFeeRate FeeEstimator::min_relay_fee() const {
    return primitives::MIN_RELAY_TX_FEE;
}

// ---------------------------------------------------------------------------
// FeeEstimator :: clear
// ---------------------------------------------------------------------------

void FeeEstimator::clear() {
    LOCK(mutex_);

    // 1. Reset all internal state.
    tracked_.clear();
    recent_fee_rates_.clear();
    buckets_.fill({0, 0});
}

// ---------------------------------------------------------------------------
// FeeEstimator :: median  (static helper)
// ---------------------------------------------------------------------------

int64_t FeeEstimator::median(std::vector<int64_t>& values) {
    if (values.empty()) return 0;

    // 1. Sort the input.
    std::sort(values.begin(), values.end());

    // 2. Return the middle element (or average of the two middle).
    size_t mid = values.size() / 2;
    if (values.size() % 2 == 0) {
        return (values[mid - 1] + values[mid]) / 2;
    }
    return values[mid];
}

} // namespace rnet::mempool

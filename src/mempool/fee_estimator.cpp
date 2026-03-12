#include "mempool/fee_estimator.h"

#include <algorithm>
#include <numeric>

namespace rnet::mempool {

FeeEstimator::FeeEstimator() {
    buckets_.fill({0, 0});
}

void FeeEstimator::track_tx(const rnet::uint256& txid, int64_t fee_rate,
                            int entry_height) {
    LOCK(mutex_);
    tracked_[txid] = TrackedTx{fee_rate, entry_height};

    recent_fee_rates_.push_back(fee_rate);
    if (recent_fee_rates_.size() > MAX_HISTORY) {
        recent_fee_rates_.pop_front();
    }
}

void FeeEstimator::confirm_tx(const rnet::uint256& txid,
                              int confirm_height) {
    LOCK(mutex_);

    auto it = tracked_.find(txid);
    if (it == tracked_.end()) return;

    int blocks_to_confirm = confirm_height - it->second.entry_height;
    if (blocks_to_confirm < 1) blocks_to_confirm = 1;

    // Record in the appropriate bucket
    if (blocks_to_confirm <= MAX_TARGET) {
        auto& bucket = buckets_[static_cast<size_t>(blocks_to_confirm - 1)];
        bucket.total_fee_rate += it->second.fee_rate;
        bucket.count++;
    }

    tracked_.erase(it);
}

void FeeEstimator::process_block(
    int height, const std::vector<rnet::uint256>& confirmed_txids)
{
    for (const auto& txid : confirmed_txids) {
        confirm_tx(txid, height);
    }
}

primitives::CFeeRate FeeEstimator::estimate_fee(int target_blocks) const {
    LOCK(mutex_);

    if (target_blocks < 1) target_blocks = 1;
    if (target_blocks > MAX_TARGET) target_blocks = MAX_TARGET;

    // Look at buckets from target down to 1 to find sufficient data
    int64_t total_rate = 0;
    int64_t total_count = 0;

    for (int i = 0; i < target_blocks && i < MAX_TARGET; ++i) {
        total_rate += buckets_[static_cast<size_t>(i)].total_fee_rate;
        total_count += buckets_[static_cast<size_t>(i)].count;
    }

    if (total_count > 0) {
        int64_t avg_rate = total_rate / total_count;
        // Convert fee rate to per-kVB
        return primitives::CFeeRate(avg_rate);
    }

    // Fallback: use median of recent fee rates
    if (!recent_fee_rates_.empty()) {
        std::vector<int64_t> sorted(recent_fee_rates_.begin(),
                                    recent_fee_rates_.end());
        int64_t med = median(sorted);
        return primitives::CFeeRate(med);
    }

    // No data: return minimum relay fee
    return primitives::MIN_RELAY_TX_FEE;
}

primitives::FeeEstimation FeeEstimator::estimate(
    int target_blocks, primitives::FeeEstimateTarget mode) const
{
    primitives::FeeEstimation result;
    result.target_blocks = target_blocks;
    result.mode = mode;

    auto rate = estimate_fee(target_blocks);

    // Apply mode adjustments
    switch (mode) {
        case primitives::FeeEstimateTarget::ECONOMICAL:
            // Use as-is
            break;
        case primitives::FeeEstimateTarget::CONSERVATIVE:
            // Add 10% safety margin
            rate = primitives::CFeeRate(
                rate.get_fee_per_kvb() + rate.get_fee_per_kvb() / 10);
            break;
        case primitives::FeeEstimateTarget::HIGH_PRIORITY:
            // Double the estimate for next-block confirmation
            rate = primitives::CFeeRate(rate.get_fee_per_kvb() * 2);
            break;
    }

    result.rate = rate;
    return result;
}

primitives::CFeeRate FeeEstimator::min_relay_fee() const {
    return primitives::MIN_RELAY_TX_FEE;
}

void FeeEstimator::clear() {
    LOCK(mutex_);
    tracked_.clear();
    recent_fee_rates_.clear();
    buckets_.fill({0, 0});
}

int64_t FeeEstimator::median(std::vector<int64_t>& values) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    size_t mid = values.size() / 2;
    if (values.size() % 2 == 0) {
        return (values[mid - 1] + values[mid]) / 2;
    }
    return values[mid];
}

}  // namespace rnet::mempool

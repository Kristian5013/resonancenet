#include "wallet/fees.h"

#include <algorithm>
#include <numeric>

namespace rnet::wallet {

FeeEstimator::FeeEstimator()
    : fallback_fee_(primitives::CFeeRate(10000))  // 10 res/vB default
    , buckets_(MAX_TARGET_BLOCKS) {
}

void FeeEstimator::record_fee(int64_t fee, size_t vsize, int32_t block_height) {
    if (vsize == 0) return;
    LOCK(mutex_);

    int64_t rate = fee * 1000 / static_cast<int64_t>(vsize);

    // Record in all applicable buckets
    for (int i = 0; i < MAX_TARGET_BLOCKS && i < static_cast<int>(buckets_.size()); ++i) {
        auto& bucket = buckets_[static_cast<size_t>(i)];
        bucket.rates.push_back(rate);
        bucket.last_height = block_height;

        // Keep only the last 1000 observations per bucket
        if (bucket.rates.size() > 1000) {
            bucket.rates.erase(bucket.rates.begin(),
                               bucket.rates.begin() +
                                   static_cast<int64_t>(bucket.rates.size() - 1000));
        }
    }
}

primitives::CFeeRate FeeEstimator::estimate_fee(int32_t target_blocks) const {
    LOCK(mutex_);

    if (target_blocks <= 0) target_blocks = 1;
    if (target_blocks > MAX_TARGET_BLOCKS) target_blocks = MAX_TARGET_BLOCKS;

    size_t bucket_idx = static_cast<size_t>(target_blocks - 1);
    if (bucket_idx >= buckets_.size() || buckets_[bucket_idx].rates.empty()) {
        return fallback_fee_;
    }

    const auto& rates = buckets_[bucket_idx].rates;

    // Use median fee rate
    auto sorted = rates;
    std::sort(sorted.begin(), sorted.end());
    int64_t median = sorted[sorted.size() / 2];

    // Ensure at least minimum relay fee
    int64_t min_rate = primitives::MIN_RELAY_TX_FEE.get_fee_per_kvb();
    if (median < min_rate) {
        median = min_rate;
    }

    return primitives::CFeeRate(median);
}

primitives::FeeEstimation FeeEstimator::estimate(
    primitives::FeeEstimateTarget target) const {

    primitives::FeeEstimation result;
    result.mode = target;

    switch (target) {
        case primitives::FeeEstimateTarget::HIGH_PRIORITY:
            result.target_blocks = 1;
            break;
        case primitives::FeeEstimateTarget::CONSERVATIVE:
            result.target_blocks = 6;
            break;
        case primitives::FeeEstimateTarget::ECONOMICAL:
            result.target_blocks = 25;
            break;
    }

    result.rate = estimate_fee(result.target_blocks);
    return result;
}

primitives::CFeeRate FeeEstimator::get_min_fee() const {
    return primitives::MIN_RELAY_TX_FEE;
}

primitives::CFeeRate FeeEstimator::get_fallback_fee() const {
    LOCK(mutex_);
    return fallback_fee_;
}

void FeeEstimator::set_fallback_fee(const primitives::CFeeRate& rate) {
    LOCK(mutex_);
    fallback_fee_ = rate;
}

int64_t FeeEstimator::get_fee_for_size(size_t vsize,
                                       primitives::FeeEstimateTarget target) const {
    auto est = estimate(target);
    return est.rate.get_fee(vsize);
}

}  // namespace rnet::wallet

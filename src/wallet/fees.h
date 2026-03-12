#pragma once

#include <cstdint>
#include <vector>

#include "core/error.h"
#include "core/sync.h"
#include "primitives/fees.h"

namespace rnet::wallet {

/// FeeEstimator: estimates appropriate fee rates for transactions.
/// Uses a simple bucket-based approach tracking recent block fee rates.
class FeeEstimator {
public:
    FeeEstimator();

    /// Record a confirmed transaction's fee rate (called when blocks arrive).
    void record_fee(int64_t fee, size_t vsize, int32_t block_height);

    /// Estimate fee rate for confirmation within target_blocks.
    primitives::CFeeRate estimate_fee(int32_t target_blocks) const;

    /// Estimate fee for a specific target type.
    primitives::FeeEstimation estimate(
        primitives::FeeEstimateTarget target) const;

    /// Get the minimum relay fee rate.
    primitives::CFeeRate get_min_fee() const;

    /// Get the fallback fee rate (used when no data is available).
    primitives::CFeeRate get_fallback_fee() const;

    /// Set the fallback fee rate.
    void set_fallback_fee(const primitives::CFeeRate& rate);

    /// Get the fee for a transaction of a given virtual size.
    int64_t get_fee_for_size(size_t vsize,
                             primitives::FeeEstimateTarget target =
                                 primitives::FeeEstimateTarget::CONSERVATIVE) const;

private:
    mutable core::Mutex mutex_;
    primitives::CFeeRate fallback_fee_;

    /// Recent fee rate observations bucketed by confirmation target.
    struct FeeBucket {
        std::vector<int64_t> rates;  ///< Fee rates in res/kvB
        int32_t last_height = 0;
    };

    static constexpr int MAX_TARGET_BLOCKS = 144;  // ~1 day
    std::vector<FeeBucket> buckets_;
};

}  // namespace rnet::wallet

#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "core/sync.h"
#include "core/types.h"
#include "primitives/fees.h"

namespace rnet::mempool {

/// FeeEstimator — estimates appropriate fee rates based on
/// recent transaction confirmation times.
class FeeEstimator {
public:
    FeeEstimator();
    ~FeeEstimator() = default;

    /// Record a transaction entering the mempool
    void track_tx(const rnet::uint256& txid, int64_t fee_rate,
                  int entry_height);

    /// Record a transaction being confirmed
    void confirm_tx(const rnet::uint256& txid, int confirm_height);

    /// Record a block being connected (update internal state)
    void process_block(int height,
                       const std::vector<rnet::uint256>& confirmed_txids);

    /// Estimate fee rate for confirmation within target_blocks
    primitives::CFeeRate estimate_fee(int target_blocks) const;

    /// Estimate fee rate with a given estimation mode
    primitives::FeeEstimation estimate(
        int target_blocks,
        primitives::FeeEstimateTarget mode =
            primitives::FeeEstimateTarget::CONSERVATIVE) const;

    /// Get the minimum relay fee rate
    primitives::CFeeRate min_relay_fee() const;

    /// Clear all tracked data
    void clear();

private:
    mutable core::Mutex mutex_;

    struct TrackedTx {
        int64_t fee_rate = 0;
        int entry_height = 0;
    };

    /// Recent confirmed transactions with their confirmation times
    struct ConfirmationBucket {
        int64_t total_fee_rate = 0;
        int64_t count = 0;
    };

    /// Buckets for different confirmation targets (1-block, 2-block, ..., 25-block)
    static constexpr int MAX_TARGET = 25;
    std::array<ConfirmationBucket, MAX_TARGET> buckets_{};

    /// Tracked but unconfirmed transactions
    std::unordered_map<rnet::uint256, TrackedTx> tracked_;

    /// Recent fee rate history for fallback estimation
    std::deque<int64_t> recent_fee_rates_;
    static constexpr size_t MAX_HISTORY = 1000;

    /// Get the median of a sorted range
    static int64_t median(std::vector<int64_t>& values);
};

}  // namespace rnet::mempool

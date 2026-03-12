#pragma once

#include <cstdint>
#include <string>

#include "core/serialize.h"
#include "primitives/amount.h"

namespace rnet::primitives {

/// CFeeRate — fee rate in resonances per virtual byte (res/vB).
class CFeeRate {
public:
    CFeeRate() = default;

    /// Construct from resonances per 1000 virtual bytes.
    explicit CFeeRate(int64_t resonances_per_kvb)
        : resonances_per_kvb_(resonances_per_kvb) {}

    /// Construct from total fee and transaction virtual size.
    CFeeRate(int64_t fee, size_t vsize) {
        if (vsize > 0) {
            resonances_per_kvb_ = fee * 1000 / static_cast<int64_t>(vsize);
        } else {
            resonances_per_kvb_ = 0;
        }
    }

    /// Get the fee for a transaction of the given virtual size (bytes).
    int64_t get_fee(size_t vsize) const {
        int64_t fee = resonances_per_kvb_ * static_cast<int64_t>(vsize) / 1000;
        // Minimum fee of 1 resonance if rate is nonzero and size > 0
        if (fee == 0 && resonances_per_kvb_ > 0 && vsize > 0) {
            fee = 1;
        }
        return fee;
    }

    /// Get the raw rate in resonances per 1000 virtual bytes.
    int64_t get_fee_per_kvb() const { return resonances_per_kvb_; }

    /// Human-readable (e.g., "10.00 res/vB")
    std::string to_string() const;

    bool operator==(const CFeeRate& other) const {
        return resonances_per_kvb_ == other.resonances_per_kvb_;
    }
    bool operator!=(const CFeeRate& other) const {
        return resonances_per_kvb_ != other.resonances_per_kvb_;
    }
    bool operator<(const CFeeRate& other) const {
        return resonances_per_kvb_ < other.resonances_per_kvb_;
    }
    bool operator<=(const CFeeRate& other) const {
        return resonances_per_kvb_ <= other.resonances_per_kvb_;
    }
    bool operator>(const CFeeRate& other) const {
        return resonances_per_kvb_ > other.resonances_per_kvb_;
    }
    bool operator>=(const CFeeRate& other) const {
        return resonances_per_kvb_ >= other.resonances_per_kvb_;
    }

    CFeeRate operator+(const CFeeRate& other) const {
        return CFeeRate(resonances_per_kvb_ + other.resonances_per_kvb_);
    }

    SERIALIZE_METHODS(
        READWRITE(self.resonances_per_kvb_);
    )

private:
    int64_t resonances_per_kvb_ = 0;  ///< Resonances per 1000 virtual bytes
};

/// Minimum relay fee rate
static const CFeeRate MIN_RELAY_TX_FEE{1000};  // 1 res/vB

/// Dust threshold: minimum output value to not be considered dust
/// at the given fee rate. An output is dust if its value is less than
/// the cost of spending it.
int64_t get_dust_threshold(size_t script_size, const CFeeRate& fee_rate);

/// Check if an output value is dust at the given fee rate.
bool is_dust(int64_t value, size_t script_size, const CFeeRate& fee_rate);

/// Fee estimation target types
enum class FeeEstimateTarget {
    ECONOMICAL,     ///< Lowest fee for eventual confirmation
    CONSERVATIVE,   ///< Higher fee for faster confirmation
    HIGH_PRIORITY,  ///< Overpay for next-block confirmation
};

/// Fee estimation result
struct FeeEstimation {
    CFeeRate rate;
    int32_t target_blocks = 0;  ///< Target confirmation blocks
    FeeEstimateTarget mode = FeeEstimateTarget::CONSERVATIVE;
};

}  // namespace rnet::primitives

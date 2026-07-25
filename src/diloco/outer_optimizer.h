// The outer optimizer: Nesterov momentum over aggregated pseudo-gradients.
//
// DiLoCo has two loops. Inside, each worker runs an ordinary optimizer for many
// steps. Outside, the network takes ONE step per round using the averaged
// pseudo-gradient. Momentum on that outer step is what recovers most of the
// quality lost by synchronising rarely.
//
// The state is fixed point, not floating point. Momentum accumulates across
// rounds, so a last-bit difference between two implementations would compound
// instead of cancelling, and every node must hold the identical buffer to agree
// on the next checkpoint. Coefficients are Q16 (value * 65536), and every
// operation is integer.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "diloco/aggregation.h"
#include "util/result.h"

namespace rnet::diloco {

inline constexpr int64_t kQ16One = 1 << 16;

// 0.9 and 0.7 in Q16 — the DiLoCo defaults (Nesterov momentum, outer LR).
inline constexpr int64_t kDefaultMomentumQ16 = 58982;   // 0.90
inline constexpr int64_t kDefaultOuterLrQ16 = 45875;    // 0.70

struct OuterOptimizerConfig {
    int64_t momentum_q16{kDefaultMomentumQ16};
    int64_t outer_lr_q16{kDefaultOuterLrQ16};
    bool nesterov{true};

    Status Validate() const;
};

// Holds the momentum buffer between rounds. All values are in units of
// 2^-scale_exp, and the scale is pinned at construction: a round that arrives at
// a different scale is rescaled into these units before use, so the buffer's
// meaning never silently changes.
class OuterOptimizer {
public:
    OuterOptimizer(uint64_t n_params, int32_t scale_exp, OuterOptimizerConfig config);

    // Consumes one aggregated pseudo-gradient and returns the weight update to
    // apply, in the same fixed-point units. Sign convention: the pseudo-gradient
    // is (weights_before - weights_after), i.e. it already points downhill, so the
    // returned delta is SUBTRACTED from the weights exactly like a gradient.
    Result<std::vector<int64_t>> Step(const Aggregate& aggregate);

    const std::vector<int64_t>& momentum() const { return momentum_; }
    int32_t scale_exp() const { return scale_exp_; }
    uint64_t steps_taken() const { return steps_taken_; }

    // Fixed-point state hashes into the checkpoint chain, so a resuming or joining
    // node can confirm it holds the same optimizer state as everyone else.
    crypto::Hash256 StateHash() const;

private:
    uint64_t n_params_;
    int32_t scale_exp_;
    OuterOptimizerConfig config_;
    std::vector<int64_t> momentum_;
    uint64_t steps_taken_{0};
};

// Q16 multiply with rounding, kept in one place because the rounding rule is
// consensus-relevant. Uses a 128-bit intermediate so large buffers cannot
// overflow mid-multiply.
int64_t MulQ16(int64_t value, int64_t q16);

// Converts a Q16 coefficient from a decimal string like "0.9". Used by tools and
// configuration so operators never write raw fixed-point integers by hand.
Result<int64_t> ParseQ16(std::string_view text);

}  // namespace rnet::diloco

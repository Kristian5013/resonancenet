// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "miner/difficulty.h"

#include "consensus/params.h"

#include <algorithm>
#include <cmath>

namespace rnet::miner {

// ---------------------------------------------------------------------------
// Design note — PoT difficulty adjustment algorithm (step count bounds,
//               growth-aware)
//
// In Proof-of-Training there is no numeric hash target.  "Difficulty" is
// the probability that a miner achieves val_loss < parent_val_loss within
// a bounded number of training steps.
//
// The core model is exponential:
//
//     P(improve) = 1 - exp(-lambda * n_steps)
//
// where lambda captures how easy it is to find an improvement:
//
//     lambda = k * loss^2 * (384 / d_model)
//
//   - Higher current loss  -> larger lambda -> improvement is easier.
//   - Larger model (d_model) -> smaller lambda -> more steps needed.
//   - k = 0.005 is an empirical base-rate constant.
//
// suggest_step_count() inverts the formula to find the step count that
// yields a target 60 % improvement probability, then clamps to the
// consensus [min_steps, max_steps] window.
//
// estimate_attempt_time() gives a rough wall-clock estimate so that the
// UI can display expected mining duration.  It assumes:
//   - param_count ~ d_model^2 * n_layers * 10
//   - FLOPs/step  ~ 6 * params * seq_len * batch_size
//   - GPU throughput ~ 100 TFLOPS (BF16)
//
// effective_difficulty() returns 1/P(improve) — a human-readable number
// analogous to Bitcoin's difficulty (higher = harder).
// ---------------------------------------------------------------------------

/// Empirical base-rate constant for the improvement-probability model.
constexpr double kBaseRateK = 0.005;

/// Reference model dimension used to normalise the model-size factor.
constexpr double kRefDModel = 384.0;

/// Target improvement probability for suggest_step_count().
constexpr double kTargetProbability = 0.6;  // 60 %

/// Assumed sequence length for time estimation (tokens).
constexpr double kSeqLen = 2'048.0;

/// Assumed micro-batch size for time estimation.
constexpr double kBatchSize = 4.0;

/// Assumed effective GPU throughput (FLOP/s, BF16).
constexpr double kGpuFlops = 100.0e12;  // 100 TFLOPS

/// Difficulty ceiling when improvement probability is zero.
constexpr double kMaxDifficulty = 1.0e18;

// ---------------------------------------------------------------------------
// estimate_improvement_probability
// ---------------------------------------------------------------------------

double estimate_improvement_probability(float current_loss,
                                        int n_steps,
                                        uint32_t d_model) {
    // 1. Guard against degenerate inputs.
    if (current_loss <= 0.0f || n_steps <= 0 || d_model == 0) {
        return 0.0;
    }

    // 2. Loss factor: higher loss -> easier improvement (quadratic).
    double loss_factor = static_cast<double>(current_loss)
                       * static_cast<double>(current_loss);

    // 3. Model-size penalty: larger models converge more slowly.
    double model_factor = kRefDModel / static_cast<double>(d_model);

    // 4. Compute lambda and the resulting probability.
    double lambda = kBaseRateK * loss_factor * model_factor;
    double prob   = 1.0 - std::exp(-lambda * static_cast<double>(n_steps));

    return std::clamp(prob, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// suggest_step_count
// ---------------------------------------------------------------------------

int suggest_step_count(float current_loss,
                       uint32_t d_model,
                       const consensus::ConsensusParams& params) {
    // 1. Degenerate inputs -> fall back to minimum steps.
    if (current_loss <= 0.0f || d_model == 0) {
        return params.min_steps_per_block;
    }

    // 2. Compute lambda (same formula as the probability model).
    double loss_factor  = static_cast<double>(current_loss)
                        * static_cast<double>(current_loss);
    double model_factor = kRefDModel / static_cast<double>(d_model);
    double lambda       = kBaseRateK * loss_factor * model_factor;

    if (lambda <= 0.0) {
        return params.max_steps_per_block;
    }

    // 3. Invert the exponential model:
    //    1 - exp(-lambda * n) = target  =>  n = -ln(1 - target) / lambda
    double n = -std::log(1.0 - kTargetProbability) / lambda;

    // 4. Clamp to consensus bounds [min_steps, max_steps].
    int steps = static_cast<int>(std::ceil(n));
    steps = std::clamp(steps, params.min_steps_per_block,
                              params.max_steps_per_block);

    return steps;
}

// ---------------------------------------------------------------------------
// estimate_attempt_time
// ---------------------------------------------------------------------------

double estimate_attempt_time(int n_steps,
                             uint32_t d_model,
                             uint32_t n_layers) {
    // 1. Rough parameter count: ~(d_model^2 * n_layers * 10).
    double params_est = static_cast<double>(d_model)
                      * static_cast<double>(d_model)
                      * static_cast<double>(n_layers) * 10.0;

    // 2. FLOPs per training step (simplified transformer estimate).
    double flops_per_step = 6.0 * params_est * kSeqLen * kBatchSize;

    // 3. Wall-clock time = total FLOPs / GPU throughput.
    double time_per_step = flops_per_step / kGpuFlops;
    return time_per_step * static_cast<double>(n_steps);
}

// ---------------------------------------------------------------------------
// effective_difficulty
// ---------------------------------------------------------------------------

double effective_difficulty(float current_loss,
                            int n_steps,
                            uint32_t d_model) {
    // 1. Compute the improvement probability.
    double prob = estimate_improvement_probability(current_loss, n_steps, d_model);

    // 2. Difficulty is the reciprocal; cap at a finite ceiling.
    if (prob <= 0.0) {
        return kMaxDifficulty;
    }
    return 1.0 / prob;
}

} // namespace rnet::miner

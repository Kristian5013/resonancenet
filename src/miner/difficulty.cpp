#include "miner/difficulty.h"

#include <algorithm>
#include <cmath>

namespace rnet::miner {

double estimate_improvement_probability(float current_loss,
                                        int n_steps,
                                        uint32_t d_model) {
    // Heuristic model:
    //
    // At high loss (early training), almost any training will improve.
    // At low loss (late training), improvement becomes exponentially harder.
    //
    // We model P(improve) = 1 - exp(-lambda * n_steps)
    // where lambda depends on loss and model size.
    //
    // lambda = k / (d_model * loss^(-2))
    // Intuition: lower loss → smaller lambda → harder to improve.
    //            larger model → smaller lambda → needs more steps.

    if (current_loss <= 0.0f || n_steps <= 0 || d_model == 0) {
        return 0.0;
    }

    // Scale factor: higher loss → easier improvement
    double loss_factor = static_cast<double>(current_loss) * static_cast<double>(current_loss);

    // Model size penalty: larger models need more steps per improvement
    double model_factor = 384.0 / static_cast<double>(d_model);

    // Base rate constant (tuned empirically)
    constexpr double k = 0.005;

    double lambda = k * loss_factor * model_factor;
    double prob = 1.0 - std::exp(-lambda * static_cast<double>(n_steps));

    return std::clamp(prob, 0.0, 1.0);
}

int suggest_step_count(float current_loss,
                       uint32_t d_model,
                       const consensus::ConsensusParams& params) {
    // Target ~60% improvement probability
    constexpr double target_prob = 0.6;

    if (current_loss <= 0.0f || d_model == 0) {
        return params.min_steps_per_block;
    }

    double loss_factor = static_cast<double>(current_loss) * static_cast<double>(current_loss);
    double model_factor = 384.0 / static_cast<double>(d_model);
    constexpr double k = 0.005;
    double lambda = k * loss_factor * model_factor;

    if (lambda <= 0.0) {
        return params.max_steps_per_block;
    }

    // Solve: 1 - exp(-lambda * n) = target_prob
    // n = -ln(1 - target_prob) / lambda
    double n = -std::log(1.0 - target_prob) / lambda;

    int steps = static_cast<int>(std::ceil(n));
    steps = std::clamp(steps, params.min_steps_per_block, params.max_steps_per_block);

    return steps;
}

double estimate_attempt_time(int n_steps,
                             uint32_t d_model,
                             uint32_t n_layers) {
    // Very rough estimate based on FLOPs per step.
    //
    // FLOPs per step ~ 6 * n_params * seq_len * batch_size
    // Assume seq_len=2048, batch_size=4
    // Modern GPU does ~100 TFLOPS BF16
    //
    // For a 384-dim, 6-layer model (~30M params):
    //   FLOPs/step ~ 6 * 30e6 * 2048 * 4 ~ 1.5e12
    //   Time/step  ~ 1.5e12 / 100e12 = 0.015s
    //   n_steps=1000 → 15 seconds

    // Rough param count estimate: ~(d_model^2 * n_layers * 10)
    double params_est = static_cast<double>(d_model) * static_cast<double>(d_model) *
                        static_cast<double>(n_layers) * 10.0;

    // FLOPs per step (simplified)
    constexpr double seq_len = 2048.0;
    constexpr double batch_size = 4.0;
    double flops_per_step = 6.0 * params_est * seq_len * batch_size;

    // Assume 100 TFLOPS effective throughput
    constexpr double gpu_tflops = 100.0e12;

    double time_per_step = flops_per_step / gpu_tflops;
    return time_per_step * static_cast<double>(n_steps);
}

double effective_difficulty(float current_loss,
                            int n_steps,
                            uint32_t d_model) {
    double prob = estimate_improvement_probability(current_loss, n_steps, d_model);
    if (prob <= 0.0) {
        return 1e18;  // Effectively infinite difficulty
    }
    return 1.0 / prob;
}

}  // namespace rnet::miner

#include "training/lr_schedule.h"

#include <algorithm>
#include <numbers>

namespace rnet::training {

LRSchedule::LRSchedule(const Config& config)
    : config_(config) {}

float LRSchedule::get_lr(int step) const {
    if (step < 0) return config_.min_lr;

    // Linear warmup phase
    if (step < config_.warmup_steps) {
        if (config_.warmup_steps <= 0) return config_.peak_lr;
        float t = static_cast<float>(step) / static_cast<float>(config_.warmup_steps);
        return config_.min_lr + t * (config_.peak_lr - config_.min_lr);
    }

    // Past total steps — return minimum
    if (step >= config_.total_steps) return config_.min_lr;

    // Cosine annealing phase
    int decay_steps = config_.total_steps - config_.warmup_steps;
    if (decay_steps <= 0) return config_.peak_lr;

    float progress = static_cast<float>(step - config_.warmup_steps)
                   / static_cast<float>(decay_steps);
    float cosine_factor = 0.5f * (1.0f + std::cos(static_cast<float>(std::numbers::pi) * progress));

    return config_.min_lr + cosine_factor * (config_.peak_lr - config_.min_lr);
}

}  // namespace rnet::training

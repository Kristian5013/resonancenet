// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

// Own header.
#include "training/lr_schedule.h"

// Standard library.
#include <algorithm>
#include <numbers>

namespace rnet::training {

// ---------------------------------------------------------------------------
// LRSchedule (constructor)
// ---------------------------------------------------------------------------
LRSchedule::LRSchedule(const Config& config)
    : config_(config) {}

// ---------------------------------------------------------------------------
// get_lr
// ---------------------------------------------------------------------------
// Returns the learning rate for the given training step.  Three phases:
// linear warmup from min_lr to peak_lr, then cosine annealing back down
// to min_lr, then constant min_lr.
// ---------------------------------------------------------------------------
float LRSchedule::get_lr(int step) const
{
    if (step < 0) return config_.min_lr;

    // 1. Linear warmup phase.
    if (step < config_.warmup_steps) {
        if (config_.warmup_steps <= 0) return config_.peak_lr;
        float t = static_cast<float>(step) / static_cast<float>(config_.warmup_steps);
        return config_.min_lr + t * (config_.peak_lr - config_.min_lr);
    }

    // 2. Past total steps -- return minimum.
    if (step >= config_.total_steps) return config_.min_lr;

    // 3. Cosine annealing phase.
    int decay_steps = config_.total_steps - config_.warmup_steps;
    if (decay_steps <= 0) return config_.peak_lr;

    float progress = static_cast<float>(step - config_.warmup_steps)
                   / static_cast<float>(decay_steps);
    float cosine_factor = 0.5f * (1.0f + std::cos(static_cast<float>(std::numbers::pi) * progress));

    return config_.min_lr + cosine_factor * (config_.peak_lr - config_.min_lr);
}

} // namespace rnet::training

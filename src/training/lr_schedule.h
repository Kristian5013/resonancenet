#pragma once

#include <cmath>
#include <cstdint>

namespace rnet::training {

/// Learning rate scheduler with linear warmup and cosine annealing.
class LRSchedule {
public:
    struct Config {
        float peak_lr = 3e-4f;
        float min_lr = 1e-5f;
        int warmup_steps = 100;
        int total_steps = 10000;
    };

    explicit LRSchedule(const Config& config);

    /// Get the learning rate for a given step.
    /// - Steps [0, warmup_steps): linear warmup from min_lr to peak_lr.
    /// - Steps [warmup_steps, total_steps]: cosine annealing from peak_lr to min_lr.
    /// - Steps > total_steps: returns min_lr.
    float get_lr(int step) const;

private:
    Config config_;
};

}  // namespace rnet::training

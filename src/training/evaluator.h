#pragma once

#include <filesystem>

#include "core/error.h"
#include "training/data_loader.h"

namespace rnet::gpu {
class GpuBackend;
}

namespace rnet::training {

/// Deterministic model evaluator.
/// Given the same checkpoint and the same data, produces the same loss
/// (within the consensus tolerance of 2%).
class Evaluator {
public:
    explicit Evaluator(rnet::gpu::GpuBackend& backend);

    /// Evaluate a checkpoint on validation data.
    /// Loads the checkpoint, runs n_batches forward passes, and returns
    /// the average cross-entropy loss.
    Result<float> evaluate(const std::filesystem::path& checkpoint,
                            DataLoader& val_data,
                            int n_batches);

private:
    rnet::gpu::GpuBackend& backend_;
};

}  // namespace rnet::training

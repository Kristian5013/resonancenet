#pragma once

#include <cstddef>
#include <vector>

#include "training/model_config.h"

namespace rnet::inference {

/// O(1) recurrent state per token — NO KV cache, NO quadratic growth.
/// MinGRU hidden states + causal convolution ring buffers.
/// Total: ~2MB for 34M model, ~96MB for 3B model.
struct InferenceState {
    /// MinGRU hidden states: [n_layers][d_model]
    std::vector<std::vector<float>> h_states;

    /// Conv ring buffers: [n_layers][n_branches][max_kernel_size][d_model]
    std::vector<std::vector<std::vector<std::vector<float>>>> conv_buffers;

    /// Conv ring buffer write positions: [n_layers][n_branches]
    std::vector<std::vector<int>> conv_positions;

    /// Number of tokens processed so far (for conv buffer fill tracking)
    uint64_t tokens_processed = 0;

    /// Create a fresh zero-initialized state for the given model config.
    static InferenceState create(const training::ModelConfig& config);

    /// Reset all state to zero (reuse allocations).
    void reset();

    /// Total memory footprint in bytes.
    size_t memory_bytes() const;

    /// Whether state has been initialized (non-empty).
    bool is_valid() const { return !h_states.empty(); }
};

}  // namespace rnet::inference

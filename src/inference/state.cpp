// Copyright (c) 2025-present The ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "inference/state.h"

#include <algorithm>
#include <cstring>
#include <numeric>

namespace rnet::inference {

// ---------------------------------------------------------------------------
// create
// ---------------------------------------------------------------------------
// Allocate O(1)-per-token inference state from model config:
//   - MinGRU hidden states: h[layer] in R^d_model, initialized to zero
//   - Conv ring buffers: buf[layer][branch][kernel_pos] in R^d_model --
//     stores last K inputs for each causal conv branch
//   - This is the key innovation: O(1) state per token (no KV cache),
//     constant memory regardless of sequence length
// ---------------------------------------------------------------------------
InferenceState InferenceState::create(const training::ModelConfig& config) {
    InferenceState state;

    const int n_layers = static_cast<int>(config.n_layers);
    const int d_model = static_cast<int>(config.d_model);
    const int n_branches = static_cast<int>(config.n_conv_branches);

    // 1. Allocate MinGRU hidden states: [n_layers][d_model], all zeros
    state.h_states.resize(n_layers);
    for (int l = 0; l < n_layers; ++l) {
        state.h_states[l].assign(d_model, 0.0f);
    }

    // 2. Allocate conv ring buffers: [n_layers][n_branches][kernel_size][d_model]
    state.conv_buffers.resize(n_layers);
    state.conv_positions.resize(n_layers);
    for (int l = 0; l < n_layers; ++l) {
        state.conv_buffers[l].resize(n_branches);
        state.conv_positions[l].assign(n_branches, 0);
        for (int b = 0; b < n_branches; ++b) {
            int k = config.kernel_sizes[b];
            if (k == 0) {
                // 3. Inactive branch — skip allocation
                continue;
            }
            state.conv_buffers[l][b].resize(k);
            for (int i = 0; i < k; ++i) {
                state.conv_buffers[l][b][i].assign(d_model, 0.0f);
            }
        }
    }

    // 4. Initialize token counter
    state.tokens_processed = 0;
    return state;
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------
// Zero all state tensors, reset token counter -- used between inference
// requests to reuse allocated memory without re-allocation.
// ---------------------------------------------------------------------------
void InferenceState::reset() {
    // 1. Zero all MinGRU hidden states
    for (auto& layer_h : h_states) {
        std::fill(layer_h.begin(), layer_h.end(), 0.0f);
    }

    // 2. Zero all conv ring buffers and reset write positions
    for (size_t l = 0; l < conv_buffers.size(); ++l) {
        for (size_t b = 0; b < conv_buffers[l].size(); ++b) {
            for (auto& slot : conv_buffers[l][b]) {
                std::fill(slot.begin(), slot.end(), 0.0f);
            }
        }
        if (l < conv_positions.size()) {
            std::fill(conv_positions[l].begin(), conv_positions[l].end(), 0);
        }
    }

    // 3. Reset token counter
    tokens_processed = 0;
}

// ---------------------------------------------------------------------------
// memory_bytes
// ---------------------------------------------------------------------------
// Total memory footprint:
//   sum(h_states) + sum(conv_buffers) + sum(conv_positions)
//   + sizeof(InferenceState)
// ---------------------------------------------------------------------------
size_t InferenceState::memory_bytes() const {
    size_t total = 0;

    // 1. MinGRU hidden states
    for (const auto& layer_h : h_states) {
        total += layer_h.size() * sizeof(float);
    }

    // 2. Conv ring buffers
    for (const auto& layer_bufs : conv_buffers) {
        for (const auto& branch_buf : layer_bufs) {
            for (const auto& slot : branch_buf) {
                total += slot.size() * sizeof(float);
            }
        }
    }

    // 3. Conv write positions
    for (const auto& layer_pos : conv_positions) {
        total += layer_pos.size() * sizeof(int);
    }

    // 4. Struct overhead
    total += sizeof(InferenceState);
    return total;
}

} // namespace rnet::inference

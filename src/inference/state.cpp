#include "inference/state.h"

#include <algorithm>
#include <cstring>
#include <numeric>

namespace rnet::inference {

InferenceState InferenceState::create(const training::ModelConfig& config) {
    InferenceState state;

    const int n_layers = static_cast<int>(config.n_layers);
    const int d_model = static_cast<int>(config.d_model);
    const int n_branches = static_cast<int>(config.n_conv_branches);

    // Allocate MinGRU hidden states: [n_layers][d_model], all zeros
    state.h_states.resize(n_layers);
    for (int l = 0; l < n_layers; ++l) {
        state.h_states[l].assign(d_model, 0.0f);
    }

    // Allocate conv ring buffers: [n_layers][n_branches][kernel_size][d_model]
    state.conv_buffers.resize(n_layers);
    state.conv_positions.resize(n_layers);
    for (int l = 0; l < n_layers; ++l) {
        state.conv_buffers[l].resize(n_branches);
        state.conv_positions[l].assign(n_branches, 0);
        for (int b = 0; b < n_branches; ++b) {
            int k = config.kernel_sizes[b];
            if (k == 0) {
                // Inactive branch
                continue;
            }
            state.conv_buffers[l][b].resize(k);
            for (int i = 0; i < k; ++i) {
                state.conv_buffers[l][b][i].assign(d_model, 0.0f);
            }
        }
    }

    state.tokens_processed = 0;
    return state;
}

void InferenceState::reset() {
    for (auto& layer_h : h_states) {
        std::fill(layer_h.begin(), layer_h.end(), 0.0f);
    }
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
    tokens_processed = 0;
}

size_t InferenceState::memory_bytes() const {
    size_t total = 0;

    // h_states
    for (const auto& layer_h : h_states) {
        total += layer_h.size() * sizeof(float);
    }

    // conv_buffers
    for (const auto& layer_bufs : conv_buffers) {
        for (const auto& branch_buf : layer_bufs) {
            for (const auto& slot : branch_buf) {
                total += slot.size() * sizeof(float);
            }
        }
    }

    // conv_positions
    for (const auto& layer_pos : conv_positions) {
        total += layer_pos.size() * sizeof(int);
    }

    total += sizeof(InferenceState);
    return total;
}

}  // namespace rnet::inference

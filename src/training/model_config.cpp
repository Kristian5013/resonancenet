// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

// Own header.
#include "training/model_config.h"

namespace rnet::training {

// ---------------------------------------------------------------------------
// ModelConfig::param_count
// ---------------------------------------------------------------------------
// Computes the total number of trainable parameters from the model
// dimensions: embedding table + per-layer (rmsnorm, conv, minGRU, slots,
// FFN) + output projection.
// ---------------------------------------------------------------------------
uint64_t ModelConfig::param_count() const
{
    // 1. Embedding table.
    uint64_t embedding = static_cast<uint64_t>(vocab_size) * d_model;

    // 2. Per-layer parameters.
    uint64_t conv_weights = 0;
    for (int i = 0; i < n_conv_branches; ++i) {
        if (kernel_sizes[i] > 0) {
            conv_weights += static_cast<uint64_t>(kernel_sizes[i]) * d_model;
        }
    }

    uint64_t per_layer =
        d_model +                                              // rmsnorm scale
        conv_weights +                                         // conv weights
        static_cast<uint64_t>(d_model) * d_model * 2 +        // mingru Wz, Wh
        static_cast<uint64_t>(d_model) * n_slots +            // slot keys
        static_cast<uint64_t>(n_slots) * d_model +            // slot values
        d_model +                                              // rmsnorm2 scale
        static_cast<uint64_t>(d_model) * d_ff * 3;            // ffn W_up, W_gate, W_down

    // 3. Output projection.
    uint64_t output_proj = static_cast<uint64_t>(d_model) * vocab_size;

    return embedding + static_cast<uint64_t>(n_layers) * per_layer + output_proj;
}

// ---------------------------------------------------------------------------
// ModelConfig::checkpoint_bytes
// ---------------------------------------------------------------------------
// BF16 storage: 2 bytes per parameter.
// ---------------------------------------------------------------------------
uint64_t ModelConfig::checkpoint_bytes() const
{
    return 2 * param_count();
}

// ---------------------------------------------------------------------------
// ModelConfig::from_block_header
// ---------------------------------------------------------------------------
// Extracts a ModelConfig from the training-related fields in a block header.
// ---------------------------------------------------------------------------
ModelConfig ModelConfig::from_block_header(const primitives::CBlockHeader& header)
{
    ModelConfig cfg;
    cfg.d_model = header.d_model;
    cfg.n_layers = header.n_layers;
    cfg.n_slots = header.n_slots;
    cfg.d_ff = header.d_ff;
    cfg.vocab_size = header.vocab_size;
    cfg.max_seq_len = header.max_seq_len;
    cfg.n_conv_branches = header.n_conv_branches;
    cfg.kernel_sizes = header.kernel_sizes;
    return cfg;
}

// ---------------------------------------------------------------------------
// ModelConfig::genesis
// ---------------------------------------------------------------------------
// Returns the genesis model configuration (defaults match genesis values).
// ---------------------------------------------------------------------------
ModelConfig ModelConfig::genesis()
{
    return ModelConfig{};
}

} // namespace rnet::training

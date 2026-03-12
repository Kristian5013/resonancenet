#pragma once

#include <array>
#include <cstdint>

#include "primitives/block_header.h"

namespace rnet::training {

/// V5 neural network model configuration.
/// Defines all hyperparameters for a given block height.
struct ModelConfig {
    uint32_t d_model = 384;
    uint32_t n_layers = 6;
    uint32_t n_slots = 64;
    uint32_t d_ff = 768;            // = 2 * d_model
    uint32_t vocab_size = 50257;
    uint32_t max_seq_len = 2048;
    uint8_t  n_conv_branches = 5;
    std::array<uint8_t, 8> kernel_sizes = {3, 7, 15, 31, 63, 0, 0, 0};

    /// Exact parameter count for this model configuration.
    ///
    /// Formula:
    ///   embedding     = vocab_size * d_model
    ///   per_layer     = d_model                                    (rmsnorm scale)
    ///                 + sum(kernel * d_model for active kernels)   (conv weights)
    ///                 + d_model * d_model * 2                      (mingru Wz, Wh)
    ///                 + d_model * n_slots + n_slots * d_model      (slot keys + values)
    ///                 + d_model                                    (rmsnorm2 scale)
    ///                 + d_model * d_ff * 3                         (ffn W_up, W_gate, W_down)
    ///   output_proj   = d_model * vocab_size
    ///   total         = embedding + n_layers * per_layer + output_proj
    uint64_t param_count() const;

    /// Checkpoint size in bytes (BF16 = 2 bytes per parameter).
    uint64_t checkpoint_bytes() const;

    /// Derive config from a parent block header's model fields.
    static ModelConfig from_block_header(const primitives::CBlockHeader& header);

    /// Genesis model configuration.
    static ModelConfig genesis();

    bool operator==(const ModelConfig&) const = default;
};

}  // namespace rnet::training

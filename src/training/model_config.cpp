#include "training/model_config.h"

namespace rnet::training {

uint64_t ModelConfig::param_count() const {
    // Embedding table
    uint64_t embedding = static_cast<uint64_t>(vocab_size) * d_model;

    // Per-layer parameters
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

    // Output projection
    uint64_t output_proj = static_cast<uint64_t>(d_model) * vocab_size;

    return embedding + static_cast<uint64_t>(n_layers) * per_layer + output_proj;
}

uint64_t ModelConfig::checkpoint_bytes() const {
    return 2 * param_count();  // BF16 = 2 bytes per parameter
}

ModelConfig ModelConfig::from_block_header(const primitives::CBlockHeader& header) {
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

ModelConfig ModelConfig::genesis() {
    return ModelConfig{};  // defaults match genesis values
}

}  // namespace rnet::training

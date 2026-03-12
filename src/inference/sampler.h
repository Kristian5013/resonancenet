#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace rnet::inference {

/// Sampling configuration for text generation.
struct SamplerConfig {
    float temperature = 1.0f;      ///< Temperature scaling (0 = greedy)
    int top_k = 50;                ///< Top-k filtering (0 = disabled)
    float top_p = 0.9f;            ///< Nucleus (top-p) filtering (1.0 = disabled)
    float repetition_penalty = 1.0f; ///< Repetition penalty (1.0 = disabled)
    int repetition_window = 64;    ///< How many recent tokens to penalize
    uint64_t seed = 0;             ///< RNG seed (0 = use system random)
};

/// Sampler: applies temperature, top-k, top-p, repetition penalty, then samples.
class Sampler {
public:
    explicit Sampler(const SamplerConfig& config = {});

    /// Set or update the configuration.
    void set_config(const SamplerConfig& config);

    /// Sample a single token from logits.
    /// @param logits  Raw logits of shape [vocab_size]
    /// @param recent_tokens  Recent token history for repetition penalty
    /// @return Sampled token index
    int sample(std::span<const float> logits,
               std::span<const int> recent_tokens = {});

    /// Apply repetition penalty in-place to logits.
    static void apply_repetition_penalty(std::span<float> logits,
                                         std::span<const int> recent_tokens,
                                         float penalty);

    /// Apply temperature scaling in-place.
    static void apply_temperature(std::span<float> logits, float temperature);

    /// Apply top-k filtering: set logits outside top-k to -inf.
    static void apply_top_k(std::span<float> logits, int k);

    /// Apply nucleus (top-p) filtering: set logits outside cumulative p to -inf.
    static void apply_top_p(std::span<float> logits, float p);

    /// Softmax in-place.
    static void softmax(std::span<float> logits);

    /// Greedy (argmax) selection.
    static int argmax(std::span<const float> logits);

private:
    SamplerConfig config_;
    uint64_t rng_state_[4] = {};

    /// Xoshiro256** PRNG
    uint64_t rng_next();
    double rng_uniform();  ///< Returns [0, 1)
    void rng_seed(uint64_t seed);
};

}  // namespace rnet::inference

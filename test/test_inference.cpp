// Tests for inference: sampler, state management

#include "test_framework.h"

#include "inference/sampler.h"
#include "inference/state.h"
#include "primitives/block_header.h"
#include "training/model_config.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using namespace rnet;
using namespace rnet::inference;
using namespace rnet::training;

// ─── Sampler static methods ─────────────────────────────────────────

TEST(sampler_argmax_basic) {
    std::vector<float> logits = {1.0f, 3.0f, 2.0f, 0.5f};
    int idx = Sampler::argmax(logits);
    ASSERT_EQ(idx, 1);
}

TEST(sampler_argmax_first_element) {
    std::vector<float> logits = {10.0f, 1.0f, 2.0f};
    ASSERT_EQ(Sampler::argmax(logits), 0);
}

TEST(sampler_argmax_last_element) {
    std::vector<float> logits = {1.0f, 2.0f, 3.0f};
    ASSERT_EQ(Sampler::argmax(logits), 2);
}

TEST(sampler_argmax_negative) {
    std::vector<float> logits = {-3.0f, -1.0f, -2.0f};
    ASSERT_EQ(Sampler::argmax(logits), 1);
}

TEST(sampler_softmax_sums_to_one) {
    std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f};
    Sampler::softmax(logits);

    float sum = 0.0f;
    for (float p : logits) sum += p;
    ASSERT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(sampler_softmax_preserves_order) {
    std::vector<float> logits = {1.0f, 3.0f, 2.0f};
    Sampler::softmax(logits);

    // After softmax, larger logit should give larger probability
    ASSERT_TRUE(logits[1] > logits[2]);
    ASSERT_TRUE(logits[2] > logits[0]);
}

TEST(sampler_softmax_all_equal) {
    std::vector<float> logits = {0.0f, 0.0f, 0.0f, 0.0f};
    Sampler::softmax(logits);

    // All should be equal = 0.25
    for (float p : logits) {
        ASSERT_NEAR(p, 0.25f, 1e-5f);
    }
}

TEST(sampler_temperature_zero_is_greedy) {
    std::vector<float> logits = {1.0f, 5.0f, 2.0f};

    SamplerConfig config;
    config.temperature = 0.0f;
    config.seed = 12345;

    Sampler sampler(config);
    int token = sampler.sample(logits);
    // Temperature 0 = greedy = argmax
    ASSERT_EQ(token, 1);
}

TEST(sampler_apply_temperature) {
    std::vector<float> logits = {1.0f, 2.0f, 3.0f};
    std::vector<float> original = logits;

    Sampler::apply_temperature(logits, 2.0f);

    // Temperature > 1 flattens the distribution
    // All values should be divided by 2
    ASSERT_NEAR(logits[0], 0.5f, 1e-5f);
    ASSERT_NEAR(logits[1], 1.0f, 1e-5f);
    ASSERT_NEAR(logits[2], 1.5f, 1e-5f);
}

TEST(sampler_apply_top_k) {
    std::vector<float> logits = {1.0f, 5.0f, 3.0f, 2.0f, 4.0f};
    Sampler::apply_top_k(logits, 2);

    // Only top-2 (indices 1 and 4, values 5 and 4) should be kept
    // Others should be -inf
    ASSERT_TRUE(logits[1] > -1e30f);  // kept (5.0)
    ASSERT_TRUE(logits[4] > -1e30f);  // kept (4.0)
    ASSERT_TRUE(logits[0] < -1e30f);  // removed (1.0)
    ASSERT_TRUE(logits[3] < -1e30f);  // removed (2.0)
}

TEST(sampler_apply_top_p) {
    std::vector<float> logits = {0.0f, 0.0f, 0.0f, 0.0f};
    // First apply softmax to get uniform distribution
    Sampler::softmax(logits);
    // Then convert back to logits (all should be ~-1.386 for uniform)
    // Instead, just test with already-softmaxed values treated as logits

    // Better test: use logits where top-p will trim
    std::vector<float> logits2 = {10.0f, 1.0f, 0.1f, 0.01f};
    Sampler::apply_top_p(logits2, 0.9f);
    // The dominant token (10.0) should survive, and maybe the second
    ASSERT_TRUE(logits2[0] > -1e30f);
}

TEST(sampler_apply_repetition_penalty) {
    std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int> recent = {0, 2};  // Token 0 and 2 were recently seen

    Sampler::apply_repetition_penalty(logits, recent, 1.5f);

    // Positive logits of recent tokens should be divided by penalty
    ASSERT_NEAR(logits[0], 1.0f / 1.5f, 1e-5f);
    ASSERT_NEAR(logits[2], 3.0f / 1.5f, 1e-5f);
    // Non-recent tokens unchanged
    ASSERT_NEAR(logits[1], 2.0f, 1e-5f);
    ASSERT_NEAR(logits[3], 4.0f, 1e-5f);
}

TEST(sampler_repetition_penalty_disabled) {
    std::vector<float> logits = {1.0f, 2.0f, 3.0f};
    std::vector<float> original = logits;
    std::vector<int> recent = {0, 1};

    // Penalty 1.0 = disabled, should not change logits
    Sampler::apply_repetition_penalty(logits, recent, 1.0f);
    ASSERT_EQ(logits, original);
}

// ─── Sampler integration ────────────────────────────────────────────

TEST(sampler_greedy_sample) {
    SamplerConfig config;
    config.temperature = 0.0f;  // greedy
    config.seed = 42;

    Sampler sampler(config);
    std::vector<float> logits = {0.1f, 0.5f, 10.0f, 0.3f};
    int token = sampler.sample(logits);
    ASSERT_EQ(token, 2);
}

TEST(sampler_deterministic_with_seed) {
    SamplerConfig config;
    config.temperature = 1.0f;
    config.top_k = 0;
    config.top_p = 1.0f;
    config.seed = 12345;

    // Same seed should produce same sequence
    Sampler s1(config);
    Sampler s2(config);

    std::vector<float> logits = {1.0f, 1.0f, 1.0f, 1.0f};
    int t1 = s1.sample(logits);
    int t2 = s2.sample(logits);
    ASSERT_EQ(t1, t2);
}

TEST(sampler_different_seeds_diverge) {
    std::vector<float> logits(100, 1.0f);  // Uniform distribution, many tokens

    SamplerConfig cfg1;
    cfg1.temperature = 1.0f;
    cfg1.top_k = 0;
    cfg1.top_p = 1.0f;
    cfg1.seed = 111;
    Sampler s1(cfg1);

    SamplerConfig cfg2;
    cfg2.temperature = 1.0f;
    cfg2.top_k = 0;
    cfg2.top_p = 1.0f;
    cfg2.seed = 222;
    Sampler s2(cfg2);

    // With different seeds and many tokens, samples should differ
    // (very high probability, but not guaranteed)
    bool all_same = true;
    for (int i = 0; i < 10; ++i) {
        std::vector<float> l(100, 1.0f);
        int a = s1.sample(l);
        std::vector<float> l2(100, 1.0f);
        int b = s2.sample(l2);
        if (a != b) all_same = false;
    }
    // Probabilistically, not all 10 samples should be the same
    ASSERT_FALSE(all_same);
}

// ─── InferenceState tests ───────────────────────────────────────────

TEST(inference_state_create) {
    ModelConfig config = ModelConfig::genesis();
    auto state = InferenceState::create(config);

    ASSERT_TRUE(state.is_valid());
    ASSERT_EQ(state.tokens_processed, uint64_t(0));
}

TEST(inference_state_h_states_shape) {
    ModelConfig config;
    config.d_model = 384;
    config.n_layers = 6;

    auto state = InferenceState::create(config);
    ASSERT_EQ(state.h_states.size(), size_t(6));
    for (const auto& h : state.h_states) {
        ASSERT_EQ(h.size(), size_t(384));
    }
}

TEST(inference_state_zero_initialized) {
    ModelConfig config = ModelConfig::genesis();
    auto state = InferenceState::create(config);

    // All hidden states should be zero-initialized
    for (const auto& layer_h : state.h_states) {
        for (float v : layer_h) {
            ASSERT_NEAR(v, 0.0f, 1e-10f);
        }
    }
}

TEST(inference_state_reset) {
    ModelConfig config = ModelConfig::genesis();
    auto state = InferenceState::create(config);

    // Modify some state
    state.h_states[0][0] = 1.0f;
    state.tokens_processed = 100;

    state.reset();
    ASSERT_EQ(state.tokens_processed, uint64_t(0));
    ASSERT_NEAR(state.h_states[0][0], 0.0f, 1e-10f);
}

TEST(inference_state_memory_positive) {
    ModelConfig config = ModelConfig::genesis();
    auto state = InferenceState::create(config);
    ASSERT_TRUE(state.memory_bytes() > 0);
}

TEST(inference_state_invalid_default) {
    InferenceState state;
    ASSERT_FALSE(state.is_valid());
}

// ─── ModelConfig tests ──────────────────────────────────────────────

TEST(model_config_genesis) {
    auto config = ModelConfig::genesis();
    ASSERT_EQ(config.d_model, uint32_t(384));
    ASSERT_EQ(config.n_layers, uint32_t(6));
    ASSERT_EQ(config.n_slots, uint32_t(64));
    ASSERT_EQ(config.d_ff, uint32_t(768));
    ASSERT_EQ(config.vocab_size, uint32_t(50257));
}

TEST(model_config_param_count) {
    auto config = ModelConfig::genesis();
    auto params = config.param_count();
    // Should be in the tens of millions
    ASSERT_TRUE(params > 1'000'000);
    ASSERT_TRUE(params < 100'000'000);
}

TEST(model_config_checkpoint_bytes) {
    auto config = ModelConfig::genesis();
    auto bytes = config.checkpoint_bytes();
    // BF16: 2 bytes per param
    ASSERT_EQ(bytes, config.param_count() * 2);
}

TEST(model_config_from_block_header) {
    primitives::CBlockHeader header;
    header.d_model = 512;
    header.n_layers = 8;
    header.n_slots = 64;
    header.d_ff = 1024;
    header.vocab_size = 50257;
    header.max_seq_len = 2048;
    header.n_conv_branches = 5;
    header.kernel_sizes = {3, 7, 15, 31, 63, 0, 0, 0};

    auto config = ModelConfig::from_block_header(header);
    ASSERT_EQ(config.d_model, uint32_t(512));
    ASSERT_EQ(config.n_layers, uint32_t(8));
    ASSERT_EQ(config.d_ff, uint32_t(1024));
}

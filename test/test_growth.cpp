// Tests for growth algorithm exactness: stagnation, d_model growth, layer addition

#include "test_framework.h"

#include "consensus/growth_policy.h"
#include "primitives/block_header.h"

using namespace rnet::consensus;
using namespace rnet::primitives;

// ─── Basic growth computation ───────────────────────────────────────

TEST(growth_no_improvement) {
    GrowthState state;
    state.d_model = 384;
    state.n_layers = 6;
    state.stagnation = 0;
    state.last_loss = 5.0f;

    auto result = GrowthPolicy::compute_growth(state, false);

    // No improvement: stagnation increments, no growth
    ASSERT_EQ(result.new_d_model, uint32_t(384));
    ASSERT_EQ(result.new_n_layers, uint32_t(6));
    ASSERT_EQ(result.delta_d_model, uint32_t(0));
    ASSERT_FALSE(result.layer_added);
    ASSERT_EQ(result.new_stagnation, uint32_t(1));
}

TEST(growth_with_improvement) {
    GrowthState state;
    state.d_model = 384;
    state.n_layers = 6;
    state.stagnation = 0;
    state.last_loss = 5.0f;

    auto result = GrowthPolicy::compute_growth(state, true);

    // Improvement: stagnation resets, model grows by BASE_GROWTH
    ASSERT_EQ(result.new_d_model, uint32_t(384 + GrowthPolicy::BASE_GROWTH));
    ASSERT_EQ(result.delta_d_model, GrowthPolicy::BASE_GROWTH);
    ASSERT_EQ(result.new_stagnation, uint32_t(0));
}

TEST(growth_stagnation_accumulates) {
    GrowthState state;
    state.d_model = 384;
    state.n_layers = 6;
    state.stagnation = 5;
    state.last_loss = 5.0f;

    auto result = GrowthPolicy::compute_growth(state, false);
    ASSERT_EQ(result.new_stagnation, uint32_t(6));
    ASSERT_EQ(result.new_d_model, uint32_t(384));
}

TEST(growth_stagnation_resets_on_improvement) {
    GrowthState state;
    state.d_model = 400;
    state.n_layers = 6;
    state.stagnation = 8;
    state.last_loss = 4.0f;

    auto result = GrowthPolicy::compute_growth(state, true);
    ASSERT_EQ(result.new_stagnation, uint32_t(0));
}

// ─── d_model growth ─────────────────────────────────────────────────

TEST(growth_d_model_increase) {
    GrowthState state;
    state.d_model = 500;
    state.n_layers = 6;
    state.stagnation = 0;
    state.last_loss = 3.0f;

    auto result = GrowthPolicy::compute_growth(state, true);
    ASSERT_EQ(result.new_d_model, uint32_t(500 + GrowthPolicy::BASE_GROWTH));
    ASSERT_EQ(result.new_d_ff, result.new_d_model * 2);
}

TEST(growth_d_model_cap_at_max) {
    GrowthState state;
    state.d_model = GrowthPolicy::MAX_D_MODEL;
    state.n_layers = 6;
    state.stagnation = 0;
    state.last_loss = 2.0f;

    auto result = GrowthPolicy::compute_growth(state, true);
    // Should not exceed max
    ASSERT_EQ(result.new_d_model, GrowthPolicy::MAX_D_MODEL);
    ASSERT_EQ(result.delta_d_model, uint32_t(0));
}

TEST(growth_d_model_near_max) {
    GrowthState state;
    state.d_model = GrowthPolicy::MAX_D_MODEL - 1;
    state.n_layers = 6;
    state.stagnation = 0;
    state.last_loss = 2.0f;

    auto result = GrowthPolicy::compute_growth(state, true);
    // Should clamp to MAX_D_MODEL
    ASSERT_EQ(result.new_d_model, GrowthPolicy::MAX_D_MODEL);
}

// ─── Layer addition ─────────────────────────────────────────────────

TEST(growth_layer_addition_at_threshold) {
    GrowthState state;
    // d_model crosses a LAYER_THRESHOLD boundary
    state.d_model = GrowthPolicy::GENESIS_D_MODEL + GrowthPolicy::LAYER_THRESHOLD - GrowthPolicy::BASE_GROWTH;
    state.n_layers = GrowthPolicy::GENESIS_N_LAYERS;
    state.stagnation = 0;
    state.last_loss = 3.0f;

    auto result = GrowthPolicy::compute_growth(state, true);

    // When d_model crosses the next LAYER_THRESHOLD multiple,
    // a layer should be added
    uint32_t new_d = state.d_model + GrowthPolicy::BASE_GROWTH;
    uint32_t prev_layers_from_d = (state.d_model - GrowthPolicy::GENESIS_D_MODEL) / GrowthPolicy::LAYER_THRESHOLD;
    uint32_t new_layers_from_d = (new_d - GrowthPolicy::GENESIS_D_MODEL) / GrowthPolicy::LAYER_THRESHOLD;

    if (new_layers_from_d > prev_layers_from_d) {
        ASSERT_TRUE(result.layer_added);
        ASSERT_EQ(result.new_n_layers, state.n_layers + 1);
    } else {
        ASSERT_FALSE(result.layer_added);
        ASSERT_EQ(result.new_n_layers, state.n_layers);
    }
}

TEST(growth_layer_cap_at_max) {
    GrowthState state;
    state.d_model = 2000;
    state.n_layers = GrowthPolicy::MAX_LAYERS;
    state.stagnation = 0;
    state.last_loss = 1.0f;

    auto result = GrowthPolicy::compute_growth(state, true);
    // Layers should not exceed max
    ASSERT_TRUE(result.new_n_layers <= GrowthPolicy::MAX_LAYERS);
}

// ─── Growth verification ────────────────────────────────────────────

TEST(growth_verify_correct_header) {
    CBlockHeader parent;
    parent.d_model = 384;
    parent.n_layers = 6;
    parent.d_ff = 768;
    parent.stagnation_count = 0;
    parent.val_loss = 5.0f;

    // Simulate child with improvement
    auto expected = GrowthPolicy::compute_growth(
        {parent.d_model, parent.n_layers, parent.stagnation_count, parent.val_loss},
        true);

    CBlockHeader child;
    child.d_model = expected.new_d_model;
    child.n_layers = expected.new_n_layers;
    child.d_ff = expected.new_d_ff;
    child.stagnation_count = expected.new_stagnation;
    child.growth_delta = expected.delta_d_model;
    child.val_loss = 4.5f;  // improved
    child.prev_val_loss = parent.val_loss;

    ASSERT_TRUE(GrowthPolicy::verify_growth(child, parent));
}

TEST(growth_d_ff_is_twice_d_model) {
    GrowthState state;
    state.d_model = 500;
    state.n_layers = 6;
    state.stagnation = 0;
    state.last_loss = 3.0f;

    auto result = GrowthPolicy::compute_growth(state, true);
    ASSERT_EQ(result.new_d_ff, result.new_d_model * 2);
}

// ─── Genesis model config ───────────────────────────────────────────

TEST(growth_genesis_constants) {
    ASSERT_EQ(GrowthPolicy::GENESIS_D_MODEL, uint32_t(384));
    ASSERT_EQ(GrowthPolicy::GENESIS_N_LAYERS, uint32_t(6));
    ASSERT_EQ(GrowthPolicy::BASE_GROWTH, uint32_t(2));
    ASSERT_EQ(GrowthPolicy::PATIENCE, uint32_t(10));
    ASSERT_EQ(GrowthPolicy::MAX_D_MODEL, uint32_t(4096));
    ASSERT_EQ(GrowthPolicy::MAX_LAYERS, uint32_t(48));
    ASSERT_EQ(GrowthPolicy::LAYER_THRESHOLD, uint32_t(128));
}

// ─── Progression test ───────────────────────────────────────────────

TEST(growth_progressive_series) {
    // Simulate 50 blocks of continuous improvement
    GrowthState state;
    state.d_model = GrowthPolicy::GENESIS_D_MODEL;
    state.n_layers = GrowthPolicy::GENESIS_N_LAYERS;
    state.stagnation = 0;
    state.last_loss = 10.0f;

    uint32_t initial_d = state.d_model;
    uint32_t initial_l = state.n_layers;

    for (int i = 0; i < 50; ++i) {
        auto result = GrowthPolicy::compute_growth(state, true);
        state.d_model = result.new_d_model;
        state.n_layers = result.new_n_layers;
        state.stagnation = result.new_stagnation;
        state.last_loss -= 0.1f;
    }

    // After 50 blocks of improvement, d_model should have grown
    ASSERT_TRUE(state.d_model > initial_d);
    ASSERT_EQ(state.d_model, initial_d + 50 * GrowthPolicy::BASE_GROWTH);
    // Stagnation should be 0 throughout
    ASSERT_EQ(state.stagnation, uint32_t(0));
}

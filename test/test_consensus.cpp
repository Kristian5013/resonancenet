// Tests for consensus: block reward, genesis block, consensus params

#include "test_framework.h"

#include "consensus/block_reward.h"
#include "consensus/genesis.h"
#include "consensus/params.h"
#include "primitives/amount.h"

using namespace rnet;
using namespace rnet::consensus;
using namespace rnet::primitives;

// ─── ConsensusParams tests ──────────────────────────────────────────

TEST(consensus_params_mainnet) {
    auto params = ConsensusParams::mainnet();
    ASSERT_EQ(params.default_port, uint16_t(9555));
    ASSERT_EQ(params.rpc_port, uint16_t(9554));
    ASSERT_EQ(params.lightning_port, uint16_t(9556));
    ASSERT_EQ(params.bech32_hrp, std::string("rn"));
    ASSERT_EQ(params.initial_reward, int64_t(50) * COIN);
    ASSERT_EQ(params.max_supply, int64_t(21'000'000) * COIN);
    ASSERT_EQ(params.genesis_d_model, uint32_t(384));
    ASSERT_EQ(params.genesis_n_layers, uint32_t(6));
}

TEST(consensus_params_testnet) {
    auto params = ConsensusParams::testnet();
    // Testnet should have different port numbers
    ASSERT_NE(params.default_port, uint16_t(9555));
}

TEST(consensus_params_regtest) {
    auto params = ConsensusParams::regtest();
    // Regtest should have minimal consensus constraints for fast testing
    ASSERT_TRUE(params.min_steps_per_block <= params.max_steps_per_block);
}

TEST(consensus_params_magic_bytes) {
    auto params = ConsensusParams::mainnet();
    // "RNET" = 0x52, 0x4E, 0x45, 0x54
    ASSERT_EQ(params.magic[0], uint8_t(0x52));
    ASSERT_EQ(params.magic[1], uint8_t(0x4E));
    ASSERT_EQ(params.magic[2], uint8_t(0x45));
    ASSERT_EQ(params.magic[3], uint8_t(0x54));
}

TEST(consensus_params_halving_thresholds) {
    auto params = ConsensusParams::mainnet();
    ASSERT_FALSE(params.halving_thresholds.empty());
    // Each threshold should be increasing
    for (size_t i = 1; i < params.halving_thresholds.size(); ++i) {
        ASSERT_TRUE(params.halving_thresholds[i] > params.halving_thresholds[i - 1]);
    }
}

// ─── Block reward tests ─────────────────────────────────────────────

TEST(block_reward_initial) {
    auto params = ConsensusParams::mainnet();
    EmissionState state;
    state.total_minted = 0;
    state.effective_supply = 0;
    state.estimated_lost = 0;

    auto base = get_base_reward(state, params);
    ASSERT_EQ(base, params.initial_reward);
}

TEST(block_reward_halving) {
    auto params = ConsensusParams::mainnet();

    // Before first halving threshold
    EmissionState state;
    state.effective_supply = params.halving_thresholds[0] - 1;
    auto reward_before = get_base_reward(state, params);
    ASSERT_EQ(reward_before, params.initial_reward);

    // After first halving threshold
    state.effective_supply = params.halving_thresholds[0];
    auto reward_after = get_base_reward(state, params);
    ASSERT_EQ(reward_after, params.initial_reward / 2);
}

TEST(block_reward_multiple_halvings) {
    auto params = ConsensusParams::mainnet();
    EmissionState state;

    // After all halving thresholds
    state.effective_supply = params.halving_thresholds.back();
    auto reward = get_base_reward(state, params);
    int64_t expected = params.initial_reward;
    for (size_t i = 0; i < params.halving_thresholds.size(); ++i) {
        expected /= 2;
    }
    ASSERT_EQ(reward, expected);
}

TEST(block_reward_compute_with_bonus) {
    auto params = ConsensusParams::mainnet();
    EmissionState state;
    state.total_minted = 0;
    state.effective_supply = 0;

    // With 5% improvement, should get a bonus
    auto reward = compute_block_reward(1, 0.05f, state, params);
    ASSERT_TRUE(reward.base > 0);
    ASSERT_TRUE(reward.total() >= reward.base);
}

TEST(block_reward_no_improvement) {
    auto params = ConsensusParams::mainnet();
    EmissionState state;
    state.total_minted = 0;
    state.effective_supply = 0;

    // 0 improvement = no bonus
    auto reward = compute_block_reward(1, 0.0f, state, params);
    ASSERT_TRUE(reward.base > 0);
    ASSERT_EQ(reward.bonus, int64_t(0));
}

TEST(block_reward_total) {
    auto params = ConsensusParams::mainnet();
    EmissionState state{};
    auto reward = compute_block_reward(1, 0.1f, state, params);
    ASSERT_EQ(reward.total(), reward.base + reward.bonus + reward.recovered);
}

// ─── Genesis block tests ────────────────────────────────────────────

TEST(genesis_block_creation) {
    auto params = ConsensusParams::mainnet();
    auto genesis = create_genesis_block(params);

    ASSERT_TRUE(genesis.is_genesis());
    ASSERT_EQ(genesis.height, uint64_t(0));
    ASSERT_TRUE(genesis.prev_hash.is_zero());
}

TEST(genesis_block_has_coinbase) {
    auto params = ConsensusParams::mainnet();
    auto genesis = create_genesis_block(params);

    ASSERT_FALSE(genesis.vtx.empty());
    ASSERT_TRUE(genesis.get_coinbase() != nullptr);
    ASSERT_TRUE(genesis.get_coinbase()->is_coinbase());
}

TEST(genesis_block_model_config) {
    auto params = ConsensusParams::mainnet();
    auto genesis = create_genesis_block(params);

    ASSERT_EQ(genesis.d_model, params.genesis_d_model);
    ASSERT_EQ(genesis.n_layers, params.genesis_n_layers);
    ASSERT_EQ(genesis.vocab_size, params.genesis_vocab_size);
}

TEST(genesis_block_val_loss) {
    auto params = ConsensusParams::mainnet();
    auto genesis = create_genesis_block(params);

    // Genesis val_loss should be high (10.0)
    ASSERT_NEAR(genesis.val_loss, 10.0f, 0.01f);
}

TEST(genesis_block_deterministic) {
    auto params = ConsensusParams::mainnet();
    auto g1 = create_genesis_block(params);
    auto g2 = create_genesis_block(params);
    ASSERT_EQ(g1.hash(), g2.hash());
}

TEST(genesis_message) {
    ASSERT_TRUE(std::string(GENESIS_MESSAGE).find("ResonanceNet") != std::string::npos);
}

// Tests for miner: coinbase creation, block template

#include "test_framework.h"

#include "consensus/block_reward.h"
#include "consensus/growth_policy.h"
#include "consensus/params.h"
#include "crypto/ed25519.h"
#include "miner/block_template.h"
#include "primitives/amount.h"
#include "primitives/block.h"
#include "primitives/block_header.h"
#include "primitives/transaction.h"

#include <cstring>
#include <vector>

using namespace rnet;
using namespace rnet::primitives;
using namespace rnet::miner;

// ─── Coinbase transaction tests ─────────────────────────────────────

TEST(miner_coinbase_script_format) {
    auto kp = crypto::ed25519_generate().value();
    auto script = crypto::ed25519_coinbase_script(kp.public_key);

    // Format: [0x20][32-byte pubkey][0xAC]
    ASSERT_EQ(script.size(), size_t(34));
    ASSERT_EQ(script[0], uint8_t(0x20));    // OP_PUSH32
    ASSERT_EQ(script[33], uint8_t(0xAC));   // OP_CHECKSIG
    ASSERT_EQ(std::memcmp(script.data() + 1, kp.public_key.data.data(), 32), 0);
}

TEST(miner_coinbase_tx_is_valid) {
    auto kp = crypto::ed25519_generate().value();
    auto cb_script = crypto::ed25519_coinbase_script(kp.public_key);

    CMutableTransaction cb;
    cb.version = TX_VERSION_DEFAULT;

    COutPoint null_op;
    null_op.set_null();

    // BIP34-style height encoding in scriptSig
    std::vector<uint8_t> script_sig = {0x01, 0x01};  // height = 1
    cb.vin.emplace_back(null_op, script_sig);
    cb.vout.emplace_back(50 * COIN, cb_script);

    ASSERT_TRUE(cb.is_coinbase());
    ASSERT_EQ(cb.vout[0].value, int64_t(50) * COIN);
}

TEST(miner_coinbase_immutable) {
    auto kp = crypto::ed25519_generate().value();
    auto cb_script = crypto::ed25519_coinbase_script(kp.public_key);

    CMutableTransaction cb;
    cb.version = TX_VERSION_DEFAULT;
    COutPoint null_op;
    null_op.set_null();
    cb.vin.emplace_back(null_op);
    cb.vout.emplace_back(50 * COIN, cb_script);

    CTransaction tx(cb);
    ASSERT_TRUE(tx.is_coinbase());
    ASSERT_FALSE(tx.txid().is_zero());
    ASSERT_EQ(tx.get_value_out(), int64_t(50) * COIN);
}

// ─── Block template tests ───────────────────────────────────────────

TEST(miner_block_template_creation) {
    auto params = consensus::ConsensusParams::mainnet();
    auto kp = crypto::ed25519_generate().value();

    // Create parent header (genesis-like)
    CBlockHeader parent;
    parent.height = 0;
    parent.version = 1;
    parent.d_model = params.genesis_d_model;
    parent.n_layers = params.genesis_n_layers;
    parent.d_ff = params.genesis_d_model * 2;
    parent.n_slots = params.genesis_n_slots;
    parent.vocab_size = params.genesis_vocab_size;
    parent.val_loss = 10.0f;
    parent.timestamp = 1700000000;

    consensus::EmissionState emission;
    emission.total_minted = 0;
    emission.effective_supply = 0;

    // No extra transactions
    std::vector<CTransactionRef> txs;
    std::vector<int64_t> fees;

    auto tmpl = create_block_template(
        parent, txs, fees, kp.public_key, emission, params);

    // Template should have a block with height = parent + 1
    ASSERT_EQ(tmpl.block.height, uint64_t(1));

    // Should have at least the coinbase
    ASSERT_FALSE(tmpl.block.vtx.empty());
    ASSERT_TRUE(tmpl.block.get_coinbase() != nullptr);
    ASSERT_TRUE(tmpl.block.get_coinbase()->is_coinbase());

    // Reward should be positive
    ASSERT_TRUE(tmpl.reward.base > 0);
    ASSERT_TRUE(tmpl.reward.total() > 0);
}

TEST(miner_block_template_with_txs) {
    auto params = consensus::ConsensusParams::mainnet();
    auto kp = crypto::ed25519_generate().value();

    CBlockHeader parent;
    parent.height = 100;
    parent.d_model = 400;
    parent.n_layers = 6;
    parent.d_ff = 800;
    parent.n_slots = 64;
    parent.vocab_size = 50257;
    parent.val_loss = 4.0f;
    parent.stagnation_count = 0;
    parent.timestamp = 1700000100;

    consensus::EmissionState emission;
    emission.total_minted = 5000 * COIN;
    emission.effective_supply = 5000 * COIN;

    // Create some transactions
    std::vector<CTransactionRef> txs;
    std::vector<int64_t> fees;

    for (int i = 0; i < 3; ++i) {
        CMutableTransaction mtx;
        mtx.version = TX_VERSION_DEFAULT;
        auto hash = uint256::from_hex(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        hash[31] = static_cast<uint8_t>(i);
        mtx.vin.emplace_back(COutPoint(hash, 0));
        mtx.vout.emplace_back(COIN, std::vector<uint8_t>{0x00, 0x14});
        txs.push_back(MakeTransactionRef(std::move(mtx)));
        fees.push_back(10000 + i * 1000);
    }

    auto tmpl = create_block_template(
        parent, txs, fees, kp.public_key, emission, params);

    // coinbase + 3 user txs = 4 total
    ASSERT_EQ(tmpl.block.vtx.size(), size_t(4));
    ASSERT_EQ(tmpl.tx_count(), size_t(3));
    ASSERT_TRUE(tmpl.total_fees > 0);
}

TEST(miner_block_template_growth) {
    auto params = consensus::ConsensusParams::mainnet();
    auto kp = crypto::ed25519_generate().value();

    CBlockHeader parent;
    parent.height = 50;
    parent.d_model = 384;
    parent.n_layers = 6;
    parent.d_ff = 768;
    parent.stagnation_count = 0;
    parent.val_loss = 5.0f;
    parent.timestamp = 1700000000;

    consensus::EmissionState emission{};

    auto tmpl = create_block_template(
        parent, {}, {}, kp.public_key, emission, params);

    // Growth result should be set
    // Since the template prepares for potential improvement,
    // the growth fields should match expectations
    ASSERT_TRUE(tmpl.growth.new_d_model >= parent.d_model);
}

TEST(miner_block_template_reward_breakdown) {
    auto params = consensus::ConsensusParams::mainnet();
    auto kp = crypto::ed25519_generate().value();

    CBlockHeader parent;
    parent.height = 0;
    parent.d_model = 384;
    parent.n_layers = 6;
    parent.d_ff = 768;
    parent.val_loss = 10.0f;
    parent.timestamp = 1700000000;

    consensus::EmissionState emission{};

    auto tmpl = create_block_template(
        parent, {}, {}, kp.public_key, emission, params);

    ASSERT_EQ(tmpl.reward.total(),
              tmpl.reward.base + tmpl.reward.recovered);
}

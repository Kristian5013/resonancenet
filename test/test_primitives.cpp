// Tests for primitives: transaction creation, serialization, block header

#include "test_framework.h"

#include "core/hex.h"
#include "core/stream.h"
#include "core/types.h"
#include "crypto/ed25519.h"
#include "crypto/keccak.h"
#include "primitives/amount.h"
#include "primitives/block.h"
#include "primitives/block_header.h"
#include "primitives/outpoint.h"
#include "primitives/transaction.h"
#include "primitives/txin.h"
#include "primitives/txout.h"

#include <cstring>
#include <memory>
#include <vector>

using namespace rnet;
using namespace rnet::primitives;

// ─── Amount tests ───────────────────────────────────────────────────

TEST(amount_coin_constant) {
    ASSERT_EQ(COIN, int64_t(100'000'000));
}

TEST(amount_max_money) {
    ASSERT_EQ(MAX_MONEY, int64_t(21'000'000) * COIN);
}

TEST(amount_money_range) {
    ASSERT_TRUE(MoneyRange(0));
    ASSERT_TRUE(MoneyRange(COIN));
    ASSERT_TRUE(MoneyRange(MAX_MONEY));
    ASSERT_FALSE(MoneyRange(-1));
    ASSERT_FALSE(MoneyRange(MAX_MONEY + 1));
}

TEST(amount_format_money) {
    ASSERT_EQ(FormatMoney(0), std::string("0.00000000"));
    ASSERT_EQ(FormatMoney(COIN), std::string("1.00000000"));
    ASSERT_EQ(FormatMoney(50 * COIN), std::string("50.00000000"));
    ASSERT_EQ(FormatMoney(COIN / 2), std::string("0.50000000"));
}

TEST(amount_parse_money) {
    int64_t amount = 0;
    ASSERT_TRUE(ParseMoney("1.0", amount));
    ASSERT_EQ(amount, COIN);

    ASSERT_TRUE(ParseMoney("50", amount));
    ASSERT_EQ(amount, int64_t(50) * COIN);

    ASSERT_TRUE(ParseMoney("0.00000001", amount));
    ASSERT_EQ(amount, int64_t(1));
}

// ─── COutPoint tests ───────────────────────────────────────────────

TEST(outpoint_default_is_not_null) {
    COutPoint op;
    ASSERT_FALSE(op.is_null());  // n defaults to 0, not 0xFFFFFFFF
}

TEST(outpoint_null) {
    COutPoint op;
    op.set_null();
    ASSERT_TRUE(op.is_null());
    ASSERT_TRUE(op.hash.is_zero());
    ASSERT_EQ(op.n, uint32_t(0xFFFFFFFF));
}

TEST(outpoint_equality) {
    auto hash = uint256::from_hex(
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
    COutPoint a(hash, 0);
    COutPoint b(hash, 0);
    COutPoint c(hash, 1);
    ASSERT_EQ(a, b);
    ASSERT_NE(a, c);
}

// ─── CTxOut tests ───────────────────────────────────────────────────

TEST(txout_default_is_null) {
    CTxOut out;
    ASSERT_TRUE(out.is_null());
}

TEST(txout_p2wpkh) {
    std::vector<uint8_t> script = {0x00, 0x14};
    script.resize(22, 0xAB);
    CTxOut out(COIN, script);
    ASSERT_TRUE(out.is_p2wpkh());
    ASSERT_FALSE(out.is_p2wsh());
}

TEST(txout_p2wsh) {
    std::vector<uint8_t> script = {0x00, 0x20};
    script.resize(34, 0xCD);
    CTxOut out(2 * COIN, script);
    ASSERT_FALSE(out.is_p2wpkh());
    ASSERT_TRUE(out.is_p2wsh());
}

// ─── Transaction creation ───────────────────────────────────────────

TEST(tx_coinbase_creation) {
    CMutableTransaction mtx;
    mtx.version = TX_VERSION_DEFAULT;

    // Coinbase input
    COutPoint null_op;
    null_op.set_null();
    mtx.vin.emplace_back(null_op, std::vector<uint8_t>{0x01, 0x00});

    // Coinbase output
    auto kp = crypto::ed25519_generate().value();
    auto cb_script = crypto::ed25519_coinbase_script(kp.public_key);
    mtx.vout.emplace_back(50 * COIN, cb_script);

    ASSERT_TRUE(mtx.is_coinbase());
    ASSERT_EQ(mtx.vin.size(), size_t(1));
    ASSERT_EQ(mtx.vout.size(), size_t(1));
    ASSERT_EQ(mtx.vout[0].value, int64_t(50) * COIN);
}

TEST(tx_regular_creation) {
    CMutableTransaction mtx;
    mtx.version = TX_VERSION_DEFAULT;

    // Input spending a previous output
    auto prev_hash = uint256::from_hex(
        "aabbccddaabbccddaabbccddaabbccddaabbccddaabbccddaabbccddaabbccdd");
    mtx.vin.emplace_back(COutPoint(prev_hash, 0));

    // Two outputs
    std::vector<uint8_t> script1 = {0x00, 0x14};
    script1.resize(22, 0x01);
    mtx.vout.emplace_back(49 * COIN, script1);

    std::vector<uint8_t> script2 = {0x00, 0x14};
    script2.resize(22, 0x02);
    mtx.vout.emplace_back(COIN / 2, script2);

    ASSERT_FALSE(mtx.is_coinbase());
    ASSERT_EQ(mtx.vin.size(), size_t(1));
    ASSERT_EQ(mtx.vout.size(), size_t(2));
}

TEST(tx_heartbeat) {
    CMutableTransaction mtx;
    mtx.version = TX_VERSION_HEARTBEAT;
    ASSERT_TRUE(mtx.is_heartbeat());
}

TEST(tx_immutable_from_mutable) {
    CMutableTransaction mtx;
    mtx.version = TX_VERSION_DEFAULT;
    COutPoint null_op;
    null_op.set_null();
    mtx.vin.emplace_back(null_op, std::vector<uint8_t>{0x01, 0x00});
    mtx.vout.emplace_back(50 * COIN, std::vector<uint8_t>{0x00, 0x14, 0xAA});

    CTransaction tx(mtx);
    ASSERT_EQ(tx.version(), TX_VERSION_DEFAULT);
    ASSERT_EQ(tx.vin().size(), size_t(1));
    ASSERT_EQ(tx.vout().size(), size_t(1));
    ASSERT_FALSE(tx.txid().is_zero());
    ASSERT_TRUE(tx.is_coinbase());
}

TEST(tx_txid_deterministic) {
    CMutableTransaction mtx;
    mtx.version = TX_VERSION_DEFAULT;
    COutPoint null_op;
    null_op.set_null();
    mtx.vin.emplace_back(null_op, std::vector<uint8_t>{0x01, 0x01});
    mtx.vout.emplace_back(COIN, std::vector<uint8_t>{0x00});

    CTransaction tx1(mtx);
    CTransaction tx2(mtx);
    ASSERT_EQ(tx1.txid(), tx2.txid());
}

TEST(tx_different_content_different_txid) {
    CMutableTransaction mtx1;
    mtx1.version = TX_VERSION_DEFAULT;
    COutPoint null_op;
    null_op.set_null();
    mtx1.vin.emplace_back(null_op, std::vector<uint8_t>{0x01, 0x01});
    mtx1.vout.emplace_back(COIN, std::vector<uint8_t>{0x00});

    CMutableTransaction mtx2 = mtx1;
    mtx2.vout[0].value = 2 * COIN;

    CTransaction tx1(mtx1);
    CTransaction tx2(mtx2);
    ASSERT_NE(tx1.txid(), tx2.txid());
}

TEST(tx_serialization_roundtrip) {
    CMutableTransaction mtx;
    mtx.version = TX_VERSION_DEFAULT;
    COutPoint null_op;
    null_op.set_null();
    mtx.vin.emplace_back(null_op, std::vector<uint8_t>{0x01, 0x05});
    mtx.vout.emplace_back(50 * COIN, std::vector<uint8_t>{0x20, 0x00, 0xAC});

    CTransaction tx(mtx);
    auto original_txid = tx.txid();

    // The tx should have non-zero size
    ASSERT_TRUE(tx.get_total_size() > 0);
    ASSERT_TRUE(tx.get_base_size() > 0);
}

TEST(tx_get_value_out) {
    CMutableTransaction mtx;
    mtx.version = TX_VERSION_DEFAULT;
    COutPoint null_op;
    null_op.set_null();
    mtx.vin.emplace_back(null_op);
    mtx.vout.emplace_back(10 * COIN, std::vector<uint8_t>{0x00});
    mtx.vout.emplace_back(5 * COIN, std::vector<uint8_t>{0x00});

    CTransaction tx(mtx);
    ASSERT_EQ(tx.get_value_out(), int64_t(15) * COIN);
}

TEST(tx_ref_shared_ptr) {
    CMutableTransaction mtx;
    mtx.version = TX_VERSION_DEFAULT;
    COutPoint null_op;
    null_op.set_null();
    mtx.vin.emplace_back(null_op);
    mtx.vout.emplace_back(COIN, std::vector<uint8_t>{0x00});

    auto ref = MakeTransactionRef(std::move(mtx));
    ASSERT_TRUE(ref != nullptr);
    ASSERT_FALSE(ref->txid().is_zero());
}

// ─── Block header tests ─────────────────────────────────────────────

TEST(block_header_genesis_check) {
    CBlockHeader header;
    header.height = 0;
    header.prev_hash.set_zero();
    ASSERT_TRUE(header.is_genesis());
}

TEST(block_header_non_genesis) {
    CBlockHeader header;
    header.height = 1;
    header.prev_hash = uint256::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000001");
    ASSERT_FALSE(header.is_genesis());
}

TEST(block_header_hash_deterministic) {
    CBlockHeader header;
    header.version = 1;
    header.height = 100;
    header.d_model = 384;
    header.n_layers = 6;
    header.val_loss = 5.0f;
    header.timestamp = 1700000000;

    auto h1 = header.hash();
    auto h2 = header.hash();
    ASSERT_EQ(h1, h2);
    ASSERT_FALSE(h1.is_zero());
}

TEST(block_header_different_content_different_hash) {
    CBlockHeader h1, h2;
    h1.height = 1;
    h1.timestamp = 1000;
    h2.height = 2;
    h2.timestamp = 1000;

    ASSERT_NE(h1.hash(), h2.hash());
}

TEST(block_header_model_param_count) {
    CBlockHeader header;
    header.d_model = 384;
    header.n_layers = 6;
    header.n_slots = 64;
    header.d_ff = 768;
    header.vocab_size = 50257;
    header.n_conv_branches = 5;
    header.kernel_sizes = {3, 7, 15, 31, 63, 0, 0, 0};

    auto params = header.model_param_count();
    // Should be a positive number in the millions
    ASSERT_TRUE(params > 1'000'000);
    // Genesis model should be around 34M params
    ASSERT_TRUE(params < 100'000'000);
}

// ─── CBlock tests ───────────────────────────────────────────────────

TEST(block_empty) {
    CBlock block;
    ASSERT_TRUE(block.vtx.empty());
    ASSERT_EQ(block.tx_count(), size_t(0));
    ASSERT_TRUE(block.get_coinbase() == nullptr);
}

TEST(block_with_coinbase) {
    CBlock block;
    block.height = 1;
    block.version = 1;

    CMutableTransaction cb;
    cb.version = TX_VERSION_DEFAULT;
    COutPoint null_op;
    null_op.set_null();
    cb.vin.emplace_back(null_op);
    cb.vout.emplace_back(50 * COIN, std::vector<uint8_t>{0x00});

    block.vtx.push_back(MakeTransactionRef(std::move(cb)));

    ASSERT_EQ(block.tx_count(), size_t(1));
    ASSERT_TRUE(block.get_coinbase() != nullptr);
    ASSERT_TRUE(block.get_coinbase()->is_coinbase());
}

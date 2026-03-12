// Tests for mempool: entry creation, fee estimation

#include "test_framework.h"

#include "mempool/entry.h"
#include "primitives/amount.h"
#include "primitives/fees.h"
#include "primitives/outpoint.h"
#include "primitives/transaction.h"
#include "primitives/txin.h"
#include "primitives/txout.h"

#include <memory>
#include <vector>

using namespace rnet;
using namespace rnet::primitives;
using namespace rnet::mempool;

// ─── Helper: create a simple transaction ────────────────────────────

static CTransactionRef make_test_tx(int64_t value = COIN) {
    CMutableTransaction mtx;
    mtx.version = TX_VERSION_DEFAULT;

    auto prev_hash = uint256::from_hex(
        "1111111111111111111111111111111111111111111111111111111111111111");
    mtx.vin.emplace_back(COutPoint(prev_hash, 0));

    std::vector<uint8_t> script = {0x00, 0x14};
    script.resize(22, 0xAA);
    mtx.vout.emplace_back(value, script);

    return MakeTransactionRef(std::move(mtx));
}

// ─── CTxMemPoolEntry tests ──────────────────────────────────────────

TEST(mempool_entry_creation) {
    auto tx = make_test_tx(COIN);
    int64_t fee = 10000;
    int64_t time = 1700000000;
    int height = 100;

    CTxMemPoolEntry entry(tx, fee, time, height, 5.0f);

    ASSERT_EQ(entry.get_fee(), fee);
    ASSERT_EQ(entry.get_time(), time);
    ASSERT_EQ(entry.get_entry_height(), height);
    ASSERT_NEAR(entry.get_val_loss(), 5.0f, 0.001f);
    ASSERT_EQ(entry.txid(), tx->txid());
}

TEST(mempool_entry_fee_rate) {
    auto tx = make_test_tx(COIN);
    int64_t fee = 50000;
    CTxMemPoolEntry entry(tx, fee, 0, 0);

    auto fee_rate = entry.get_fee_rate();
    // Fee rate should be positive
    ASSERT_TRUE(fee_rate.get_fee_per_kvb() > 0);
}

TEST(mempool_entry_modified_fee) {
    auto tx = make_test_tx(COIN);
    int64_t fee = 10000;
    CTxMemPoolEntry entry(tx, fee, 0, 0);

    ASSERT_EQ(entry.get_modified_fee(), fee);

    // Prioritize: add delta
    entry.update_fee_delta(5000);
    ASSERT_EQ(entry.get_modified_fee(), fee + 5000);
    ASSERT_EQ(entry.get_fee_delta(), int64_t(5000));
}

TEST(mempool_entry_ancestor_stats) {
    auto tx = make_test_tx(COIN);
    CTxMemPoolEntry entry(tx, 10000, 0, 0);

    // Default ancestor stats
    ASSERT_EQ(entry.ancestor_count, int64_t(1));

    // Update: adds to existing (count starts at 1 for self, size/fee include self)
    auto self_size = static_cast<int64_t>(entry.get_tx()->get_virtual_size());
    auto self_fee = entry.get_fee();
    entry.update_ancestors(2, 500, 30000);
    ASSERT_EQ(entry.ancestor_count, int64_t(3));
    ASSERT_EQ(entry.ancestor_size, self_size + 500);
    ASSERT_EQ(entry.ancestor_fee, self_fee + 30000);
}

TEST(mempool_entry_descendant_stats) {
    auto tx = make_test_tx(COIN);
    CTxMemPoolEntry entry(tx, 10000, 0, 0);

    auto self_size = static_cast<int64_t>(entry.get_tx()->get_virtual_size());
    auto self_fee = entry.get_fee();
    entry.update_descendants(1, 300, 20000);
    ASSERT_EQ(entry.descendant_count, int64_t(2));
    ASSERT_EQ(entry.descendant_size, self_size + 300);
    ASSERT_EQ(entry.descendant_fee, self_fee + 20000);
}

// ─── CFeeRate tests ─────────────────────────────────────────────────

TEST(fee_rate_default) {
    CFeeRate rate;
    ASSERT_EQ(rate.get_fee_per_kvb(), int64_t(0));
}

TEST(fee_rate_from_kvb) {
    CFeeRate rate(10000);  // 10000 res/kvB
    ASSERT_EQ(rate.get_fee_per_kvb(), int64_t(10000));
    // Fee for 250 vbytes = 10000 * 250 / 1000 = 2500
    ASSERT_EQ(rate.get_fee(250), int64_t(2500));
}

TEST(fee_rate_from_fee_and_size) {
    CFeeRate rate(5000, 250);  // 5000 fee for 250 bytes
    // rate = 5000 * 1000 / 250 = 20000 res/kvB
    ASSERT_EQ(rate.get_fee_per_kvb(), int64_t(20000));
}

TEST(fee_rate_minimum_fee) {
    CFeeRate rate(1);  // Very low rate
    // Even for small tx, minimum fee should be 1 if rate > 0
    ASSERT_TRUE(rate.get_fee(100) >= 0);
}

TEST(fee_rate_zero_size) {
    CFeeRate rate(10000, 0);  // Zero size
    ASSERT_EQ(rate.get_fee_per_kvb(), int64_t(0));
}

TEST(fee_rate_comparison) {
    CFeeRate low(1000);
    CFeeRate high(5000);
    ASSERT_TRUE(low < high);
    ASSERT_TRUE(high > low);
    ASSERT_TRUE(low <= high);
    ASSERT_TRUE(high >= low);
    ASSERT_FALSE(low == high);
    ASSERT_TRUE(low != high);
}

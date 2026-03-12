// Tests for lightning: channel state machine, invoice encode/decode, HTLC

#include "test_framework.h"

#include "core/types.h"
#include "crypto/ed25519.h"
#include "crypto/keccak.h"
#include "lightning/channel.h"
#include "lightning/channel_state.h"
#include "lightning/htlc.h"
#include "lightning/invoice.h"
#include "primitives/amount.h"
#include "primitives/outpoint.h"

#include <cstring>
#include <string>
#include <vector>

using namespace rnet;
using namespace rnet::lightning;
using namespace rnet::primitives;

// ─── Channel state constants ────────────────────────────────────────

TEST(lightning_constants) {
    ASSERT_EQ(LIGHTNING_PORT_MAINNET, uint16_t(9556));
    ASSERT_EQ(LIGHTNING_PORT_TESTNET, uint16_t(19556));
    ASSERT_EQ(LIGHTNING_PORT_REGTEST, uint16_t(29556));
    ASSERT_EQ(MIN_CHANNEL_CAPACITY, int64_t(100'000));
    ASSERT_TRUE(MAX_CHANNEL_CAPACITY > MIN_CHANNEL_CAPACITY);
    ASSERT_EQ(MAX_HTLCS_PER_CHANNEL, uint32_t(483));
}

TEST(lightning_channel_state_names) {
    ASSERT_FALSE(channel_state_name(ChannelState::PREOPENING).empty());
    ASSERT_FALSE(channel_state_name(ChannelState::NORMAL).empty());
    ASSERT_FALSE(channel_state_name(ChannelState::CLOSED).empty());
}

// ─── Channel ID ─────────────────────────────────────────────────────

TEST(lightning_channel_id_from_outpoint) {
    auto txid = uint256::from_hex(
        "aabbccddaabbccddaabbccddaabbccddaabbccddaabbccddaabbccddaabbccdd");
    COutPoint outpoint(txid, 0);

    auto channel_id = make_channel_id(outpoint);
    ASSERT_FALSE(channel_id.is_zero());
}

TEST(lightning_channel_id_different_index) {
    auto txid = uint256::from_hex(
        "1111111111111111111111111111111111111111111111111111111111111111");
    auto id0 = make_channel_id(COutPoint(txid, 0));
    auto id1 = make_channel_id(COutPoint(txid, 1));
    // Different output indices should produce different channel IDs
    ASSERT_NE(id0, id1);
}

// ─── Channel config ─────────────────────────────────────────────────

TEST(lightning_channel_config_defaults) {
    ChannelConfig config;
    ASSERT_EQ(config.dust_limit, LIGHTNING_DUST_LIMIT);
    ASSERT_EQ(config.csv_delay, DEFAULT_CSV_DELAY);
    ASSERT_EQ(config.max_accepted_htlcs, MAX_HTLCS_PER_CHANNEL);
    ASSERT_EQ(config.min_htlc_value, MIN_HTLC_VALUE);
}

// ─── Channel balance ────────────────────────────────────────────────

TEST(lightning_channel_balance) {
    ChannelBalance bal;
    bal.local = 5 * COIN;
    bal.remote = 3 * COIN;
    ASSERT_EQ(bal.total(), int64_t(8) * COIN);
}

// ─── Channel creation ───────────────────────────────────────────────

TEST(lightning_create_outbound_channel) {
    auto local_kp = crypto::ed25519_generate().value();
    auto remote_pk = crypto::ed25519_generate().value().public_key;

    ChannelConfig config;
    config.channel_reserve = COIN / 100;

    auto result = LightningChannel::create_outbound(
        local_kp, remote_pk, 10 * COIN, 0, config);

    ASSERT_TRUE(result.is_ok());
    auto channel = std::move(result.value());
    ASSERT_EQ(channel.state(), ChannelState::PREOPENING);
}

TEST(lightning_create_inbound_channel) {
    auto local_kp = crypto::ed25519_generate().value();
    auto remote_pk = crypto::ed25519_generate().value().public_key;

    ChannelConfig local_config, remote_config;
    local_config.channel_reserve = COIN / 100;
    remote_config.channel_reserve = COIN / 100;

    auto result = LightningChannel::create_inbound(
        local_kp, remote_pk, 5 * COIN, 0, local_config, remote_config);

    ASSERT_TRUE(result.is_ok());
    auto channel = std::move(result.value());
    ASSERT_EQ(channel.state(), ChannelState::PREOPENING);
}

// ─── HTLC tests ─────────────────────────────────────────────────────

TEST(lightning_htlc_payment_hash) {
    auto preimage = generate_preimage();
    ASSERT_FALSE(preimage.is_zero());

    auto payment_hash = compute_payment_hash(preimage);
    ASSERT_FALSE(payment_hash.is_zero());
    ASSERT_NE(preimage, payment_hash);
}

TEST(lightning_htlc_verify_preimage) {
    auto preimage = generate_preimage();
    auto payment_hash = compute_payment_hash(preimage);
    ASSERT_TRUE(verify_preimage(preimage, payment_hash));

    // Wrong preimage should fail
    auto wrong = generate_preimage();
    ASSERT_FALSE(verify_preimage(wrong, payment_hash));
}

TEST(lightning_htlc_fee_computation) {
    int64_t amount = 1 * COIN;
    int64_t base_fee = 1000;
    int64_t fee_rate_ppm = 100;  // 100 ppm = 0.01%

    auto fee = compute_htlc_fee(amount, base_fee, fee_rate_ppm);
    // fee = 1000 + (100000000 * 100 / 1000000) = 1000 + 10000 = 11000
    ASSERT_EQ(fee, int64_t(1000 + 10000));
}

TEST(lightning_htlc_dust_check) {
    Htlc htlc;
    htlc.amount = 500;
    ASSERT_TRUE(htlc.is_dust(546));
    ASSERT_FALSE(htlc.is_dust(500));
    ASSERT_FALSE(htlc.is_dust(100));
}

TEST(lightning_htlc_expiry) {
    Htlc htlc;
    htlc.cltv_expiry = 1000;
    ASSERT_FALSE(htlc.is_expired(999));
    ASSERT_TRUE(htlc.is_expired(1000));
    ASSERT_TRUE(htlc.is_expired(1001));
}

TEST(lightning_htlc_set_add) {
    HtlcSet set;
    ASSERT_EQ(set.pending_count(), uint32_t(0));

    Htlc htlc;
    htlc.direction = HtlcDirection::OFFERED;
    htlc.amount = 10000;
    htlc.payment_hash = generate_preimage();  // just need non-zero hash
    htlc.cltv_expiry = 1000;

    auto result = set.add(htlc);
    ASSERT_TRUE(result.is_ok());
    ASSERT_EQ(set.pending_count(), uint32_t(1));
}

TEST(lightning_htlc_set_fulfill) {
    HtlcSet set;
    auto preimage = generate_preimage();
    auto payment_hash = compute_payment_hash(preimage);

    Htlc htlc;
    htlc.direction = HtlcDirection::RECEIVED;
    htlc.amount = 50000;
    htlc.payment_hash = payment_hash;
    htlc.cltv_expiry = 2000;

    set.add(htlc);
    ASSERT_EQ(set.pending_count(), uint32_t(1));

    auto fulfill_result = set.fulfill(0, preimage);
    ASSERT_TRUE(fulfill_result.is_ok());

    // After fulfillment, the HTLC state should be FULFILLED
    auto* found = set.find(0);
    ASSERT_TRUE(found != nullptr);
    ASSERT_EQ(found->state, HtlcState::FULFILLED);
}

TEST(lightning_htlc_set_fail) {
    HtlcSet set;
    Htlc htlc;
    htlc.direction = HtlcDirection::OFFERED;
    htlc.amount = 5000;
    htlc.payment_hash = generate_preimage();
    htlc.cltv_expiry = 3000;

    set.add(htlc);
    auto fail_result = set.fail(0);
    ASSERT_TRUE(fail_result.is_ok());

    auto* found = set.find(0);
    ASSERT_TRUE(found != nullptr);
    ASSERT_EQ(found->state, HtlcState::FAILED);
}

TEST(lightning_htlc_set_expire) {
    HtlcSet set;

    Htlc h1;
    h1.direction = HtlcDirection::OFFERED;
    h1.amount = 1000;
    h1.payment_hash = generate_preimage();
    h1.cltv_expiry = 100;
    set.add(h1);

    Htlc h2;
    h2.direction = HtlcDirection::OFFERED;
    h2.amount = 2000;
    h2.payment_hash = generate_preimage();
    h2.cltv_expiry = 200;
    set.add(h2);

    // Expire at height 150: only h1 should expire
    uint32_t expired_count = set.mark_expired(150);
    ASSERT_EQ(expired_count, uint32_t(1));
}

// ─── Invoice tests ──────────────────────────────────────────────────

TEST(lightning_invoice_build) {
    auto kp = crypto::ed25519_generate().value();
    auto preimage = generate_preimage();
    auto payment_hash = compute_payment_hash(preimage);

    Invoice inv;
    inv.set_payment_hash(payment_hash)
       .set_amount(COIN)
       .set_description("Test payment")
       .set_payee(kp.public_key)
       .set_expiry(3600)
       .set_timestamp(1700000000);

    ASSERT_EQ(inv.payment_hash(), payment_hash);
    ASSERT_TRUE(inv.amount().has_value());
    ASSERT_EQ(inv.amount().value(), COIN);
    ASSERT_EQ(inv.description(), std::string("Test payment"));
    ASSERT_EQ(inv.expiry(), uint32_t(3600));
}

TEST(lightning_invoice_encode_decode_roundtrip) {
    auto kp = crypto::ed25519_generate().value();
    auto preimage = generate_preimage();
    auto payment_hash = compute_payment_hash(preimage);

    Invoice inv;
    inv.set_payment_hash(payment_hash)
       .set_amount(5 * COIN)
       .set_description("Roundtrip test")
       .set_payee(kp.public_key)
       .set_expiry(7200)
       .set_timestamp(1700000000);

    auto encode_result = inv.encode(kp.secret);
    ASSERT_TRUE(encode_result.is_ok());
    auto encoded = encode_result.value();
    ASSERT_FALSE(encoded.empty());

    auto decode_result = Invoice::decode(encoded);
    ASSERT_TRUE(decode_result.is_ok());
    auto decoded = decode_result.value();

    ASSERT_EQ(decoded.payment_hash(), payment_hash);
    ASSERT_TRUE(decoded.amount().has_value());
    ASSERT_EQ(decoded.amount().value(), int64_t(5) * COIN);
    ASSERT_EQ(decoded.description(), std::string("Roundtrip test"));
    ASSERT_EQ(decoded.payee(), kp.public_key);
    ASSERT_EQ(decoded.expiry(), uint32_t(7200));
}

TEST(lightning_invoice_testnet) {
    auto kp = crypto::ed25519_generate().value();

    Invoice inv;
    inv.set_payment_hash(generate_preimage())
       .set_amount(COIN)
       .set_description("testnet invoice")
       .set_payee(kp.public_key)
       .set_testnet(true)
       .set_timestamp(1700000000);

    ASSERT_TRUE(inv.is_testnet());
    ASSERT_EQ(inv.hrp(), std::string_view(INVOICE_HRP_TESTNET));
}

TEST(lightning_invoice_expiry) {
    Invoice inv;
    inv.set_timestamp(1000)
       .set_expiry(3600);

    ASSERT_FALSE(inv.is_expired(4000));  // 1000 + 3600 = 4600 > 4000
    ASSERT_TRUE(inv.is_expired(5000));   // 1000 + 3600 = 4600 < 5000
}

TEST(lightning_invoice_verify_signature) {
    auto kp = crypto::ed25519_generate().value();

    Invoice inv;
    inv.set_payment_hash(generate_preimage())
       .set_amount(COIN / 2)
       .set_description("verify sig test")
       .set_payee(kp.public_key)
       .set_timestamp(1700000000);

    auto encoded = inv.encode(kp.secret).value();
    auto decoded = Invoice::decode(encoded).value();

    ASSERT_TRUE(decoded.verify_signature());
}

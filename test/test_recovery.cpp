// Tests for recovery: recovery script creation, heartbeat verification

#include "test_framework.h"

#include "core/hex.h"
#include "core/types.h"
#include "crypto/hash.h"
#include "crypto/ed25519.h"
#include "script/recovery_script.h"

#include <cstring>
#include <vector>

using namespace rnet;
using namespace rnet::script;

// ─── Recovery script construction ───────────────────────────────────

TEST(recovery_heartbeat_script) {
    std::vector<uint8_t> owner_hash(20, 0xAA);

    HeartbeatPolicy policy;
    policy.interval = 100000;
    policy.recovery_pubkey_hash = std::vector<uint8_t>(20, 0xBB);

    auto script = build_recovery_script(
        owner_hash, RecoveryType::HEARTBEAT, policy);

    // Script should not be empty
    ASSERT_FALSE(script.empty());
}

TEST(recovery_heartbeat_parse_roundtrip) {
    std::vector<uint8_t> owner_hash(20, 0xAA);

    HeartbeatPolicy policy;
    policy.interval = 50000;
    policy.recovery_pubkey_hash = std::vector<uint8_t>(20, 0xCC);

    auto script = build_recovery_script(
        owner_hash, RecoveryType::HEARTBEAT, policy);

    RecoveryType parsed_type;
    RecoveryPolicy parsed_policy;
    bool ok = parse_recovery_script(script, parsed_type, parsed_policy);
    ASSERT_TRUE(ok);
    ASSERT_EQ(static_cast<uint8_t>(parsed_type),
              static_cast<uint8_t>(RecoveryType::HEARTBEAT));

    auto& hp = std::get<HeartbeatPolicy>(parsed_policy);
    ASSERT_EQ(hp.interval, uint64_t(50000));
    ASSERT_EQ(hp.recovery_pubkey_hash.size(), size_t(20));
}

TEST(recovery_social_script) {
    std::vector<uint8_t> owner_hash(20, 0x11);

    SocialPolicy policy;
    policy.threshold = 2;
    policy.waiting_period = 1000;
    // 3 guardian pubkeys
    for (int i = 0; i < 3; ++i) {
        std::array<uint8_t, 32> pk{};
        pk.fill(static_cast<uint8_t>(i + 1));
        policy.guardian_pubkeys.push_back(pk);
    }

    auto script = build_recovery_script(
        owner_hash, RecoveryType::SOCIAL, policy);
    ASSERT_FALSE(script.empty());

    RecoveryType parsed_type;
    RecoveryPolicy parsed_policy;
    bool ok = parse_recovery_script(script, parsed_type, parsed_policy);
    ASSERT_TRUE(ok);
    ASSERT_EQ(static_cast<uint8_t>(parsed_type),
              static_cast<uint8_t>(RecoveryType::SOCIAL));

    auto& sp = std::get<SocialPolicy>(parsed_policy);
    ASSERT_EQ(sp.threshold, uint8_t(2));
    ASSERT_EQ(sp.guardian_pubkeys.size(), size_t(3));
    ASSERT_EQ(sp.waiting_period, uint64_t(1000));
}

TEST(recovery_emission_script) {
    std::vector<uint8_t> owner_hash(20, 0x22);

    EmissionPolicy policy;
    policy.inactivity_period = 200000;

    auto script = build_recovery_script(
        owner_hash, RecoveryType::EMISSION, policy);
    ASSERT_FALSE(script.empty());

    RecoveryType parsed_type;
    RecoveryPolicy parsed_policy;
    bool ok = parse_recovery_script(script, parsed_type, parsed_policy);
    ASSERT_TRUE(ok);
    ASSERT_EQ(static_cast<uint8_t>(parsed_type),
              static_cast<uint8_t>(RecoveryType::EMISSION));

    auto& ep = std::get<EmissionPolicy>(parsed_policy);
    ASSERT_EQ(ep.inactivity_period, uint64_t(200000));
}

// ─── P2WSH wrapping ────────────────────────────────────────────────

TEST(recovery_p2wsh) {
    std::vector<uint8_t> owner_hash(20, 0xAA);

    HeartbeatPolicy policy;
    policy.interval = 100000;
    policy.recovery_pubkey_hash = std::vector<uint8_t>(20, 0xBB);

    auto recovery_script = build_recovery_script(
        owner_hash, RecoveryType::HEARTBEAT, policy);

    auto p2wsh = build_recovery_p2wsh(recovery_script);

    // P2WSH: [0x00][0x20][32-byte hash] = 34 bytes
    ASSERT_EQ(p2wsh.size(), size_t(34));
    ASSERT_EQ(p2wsh[0], uint8_t(0x00));
    ASSERT_EQ(p2wsh[1], uint8_t(0x20));
}

// ─── Witness stack construction ─────────────────────────────────────

TEST(recovery_owner_spend_witness) {
    std::vector<uint8_t> owner_hash(20, 0xAA);

    HeartbeatPolicy policy;
    policy.interval = 100000;
    policy.recovery_pubkey_hash = std::vector<uint8_t>(20, 0xBB);

    auto recovery_script = build_recovery_script(
        owner_hash, RecoveryType::HEARTBEAT, policy);

    std::vector<uint8_t> fake_sig(64, 0xDD);
    auto witness = build_owner_spend_witness(fake_sig, recovery_script);

    // Witness should have items: signature, TRUE, script
    ASSERT_TRUE(witness.size() >= 2);
}

TEST(recovery_heartbeat_recovery_witness) {
    std::vector<uint8_t> owner_hash(20, 0xAA);

    HeartbeatPolicy policy;
    policy.interval = 100000;
    policy.recovery_pubkey_hash = std::vector<uint8_t>(20, 0xBB);

    auto recovery_script = build_recovery_script(
        owner_hash, RecoveryType::HEARTBEAT, policy);

    std::vector<uint8_t> fake_sig(64, 0xEE);
    auto witness = build_heartbeat_recovery_witness(fake_sig, recovery_script);

    // Witness should have items: signature, FALSE, script
    ASSERT_TRUE(witness.size() >= 2);
}

TEST(recovery_social_witness) {
    std::vector<uint8_t> owner_hash(20, 0x11);

    SocialPolicy policy;
    policy.threshold = 2;
    policy.waiting_period = 1000;
    for (int i = 0; i < 3; ++i) {
        std::array<uint8_t, 32> pk{};
        pk.fill(static_cast<uint8_t>(i + 1));
        policy.guardian_pubkeys.push_back(pk);
    }

    auto recovery_script = build_recovery_script(
        owner_hash, RecoveryType::SOCIAL, policy);

    std::vector<std::vector<uint8_t>> sigs;
    sigs.push_back(std::vector<uint8_t>(64, 0xA1));
    sigs.push_back(std::vector<uint8_t>(64, 0xA2));

    auto witness = build_social_recovery_witness(sigs, recovery_script);
    ASSERT_TRUE(witness.size() >= 3);
}

// ─── Heartbeat interval edge cases ─────────────────────────────────

TEST(recovery_heartbeat_min_interval) {
    std::vector<uint8_t> owner_hash(20, 0xAA);
    HeartbeatPolicy policy;
    policy.interval = 1;  // Minimum: heartbeat every block
    policy.recovery_pubkey_hash = std::vector<uint8_t>(20, 0xBB);

    auto script = build_recovery_script(
        owner_hash, RecoveryType::HEARTBEAT, policy);
    ASSERT_FALSE(script.empty());

    RecoveryType parsed_type;
    RecoveryPolicy parsed_policy;
    ASSERT_TRUE(parse_recovery_script(script, parsed_type, parsed_policy));
    ASSERT_EQ(std::get<HeartbeatPolicy>(parsed_policy).interval, uint64_t(1));
}

TEST(recovery_heartbeat_large_interval) {
    std::vector<uint8_t> owner_hash(20, 0xAA);
    HeartbeatPolicy policy;
    policy.interval = 500000;
    policy.recovery_pubkey_hash = std::vector<uint8_t>(20, 0xBB);

    auto script = build_recovery_script(
        owner_hash, RecoveryType::HEARTBEAT, policy);
    ASSERT_FALSE(script.empty());

    RecoveryType parsed_type;
    RecoveryPolicy parsed_policy;
    ASSERT_TRUE(parse_recovery_script(script, parsed_type, parsed_policy));
    ASSERT_EQ(std::get<HeartbeatPolicy>(parsed_policy).interval, uint64_t(500000));
}

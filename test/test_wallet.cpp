// Tests for wallet module: key generation, HD derivation, address creation

#include "test_framework.h"

#include "core/bech32.h"
#include "core/hex.h"
#include "core/types.h"
#include "crypto/bip39.h"
#include "crypto/ed25519.h"
#include "crypto/hash.h"
#include "primitives/address.h"
#include "wallet/hd.h"
#include "wallet/keys.h"

#include <cstring>
#include <string>

using namespace rnet;
using namespace rnet::wallet;
using namespace rnet::crypto;

// ─── Key generation tests ───────────────────────────────────────────

TEST(wallet_keygen_unique) {
    auto kp1 = ed25519_generate().value();
    auto kp2 = ed25519_generate().value();
    // Two random keypairs should be different
    ASSERT_NE(kp1.public_key, kp2.public_key);
}

TEST(wallet_keygen_pubkey_valid) {
    auto kp = ed25519_generate().value();
    ASSERT_FALSE(kp.public_key.is_zero());
    ASSERT_TRUE(ed25519_is_valid_pubkey(kp.public_key.data));
}

TEST(wallet_keygen_from_seed_deterministic) {
    auto seed = core::from_hex(
        "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef");
    auto kp1 = ed25519_from_seed(seed).value();
    auto kp2 = ed25519_from_seed(seed).value();
    ASSERT_EQ(kp1.public_key, kp2.public_key);
}

TEST(wallet_secret_key_wipe) {
    auto kp = ed25519_generate().value();
    // Store original pubkey
    auto original_pk = kp.public_key;
    kp.wipe();
    // After wipe, secret should be zero
    bool all_zero = true;
    for (auto b : kp.secret.data) {
        if (b != 0) { all_zero = false; break; }
    }
    ASSERT_TRUE(all_zero);
}

// ─── HD derivation tests ────────────────────────────────────────────

TEST(wallet_hd_constants) {
    ASSERT_EQ(HD_PURPOSE, uint32_t(44));
    ASSERT_EQ(HD_COIN_TYPE, uint32_t(9555));
}

TEST(wallet_hd_create) {
    HDKeyManager hd;
    ASSERT_FALSE(hd.is_initialized());

    auto result = hd.create("");
    ASSERT_TRUE(result.is_ok());
    ASSERT_TRUE(hd.is_initialized());

    // Should return a 24-word mnemonic
    auto mnemonic = result.value();
    int word_count = 1;
    for (char c : mnemonic) {
        if (c == ' ') word_count++;
    }
    ASSERT_EQ(word_count, 24);
}

TEST(wallet_hd_restore) {
    HDKeyManager hd1;
    auto mnemonic = hd1.create("").value();

    HDKeyManager hd2;
    auto restore_result = hd2.restore(mnemonic, "");
    ASSERT_TRUE(restore_result.is_ok());
    ASSERT_TRUE(hd2.is_initialized());
}

TEST(wallet_hd_get_mnemonic) {
    HDKeyManager hd;
    auto mnemonic = hd.create("").value();
    auto retrieved = hd.get_mnemonic();
    ASSERT_TRUE(retrieved.is_ok());
    ASSERT_EQ(retrieved.value(), mnemonic);
}

// ─── Address creation tests ─────────────────────────────────────────

TEST(address_network_types) {
    ASSERT_EQ(primitives::get_bech32_hrp(primitives::NetworkType::MAINNET),
              std::string_view("rn"));
    ASSERT_EQ(primitives::get_bech32_hrp(primitives::NetworkType::TESTNET),
              std::string_view("trn"));
    ASSERT_EQ(primitives::get_bech32_hrp(primitives::NetworkType::REGTEST),
              std::string_view("rnrt"));
}

TEST(address_p2wpkh_from_pubkey) {
    // Generate a keypair, compute Hash160, create a P2WPKH address
    auto kp = ed25519_generate().value();
    auto h160 = hash160(std::span<const uint8_t>(kp.public_key.data));

    // P2WPKH script: [0x00][0x14][20-byte hash]
    std::vector<uint8_t> script = {0x00, 0x14};
    script.insert(script.end(), h160.begin(), h160.end());
    ASSERT_EQ(script.size(), size_t(22));
    ASSERT_EQ(script[0], uint8_t(0x00));
    ASSERT_EQ(script[1], uint8_t(0x14));
}

TEST(address_bech32_encoding) {
    // Create a bech32 address from a pubkey hash
    auto kp = ed25519_generate().value();
    auto h160 = hash160(std::span<const uint8_t>(kp.public_key.data));

    auto addr = core::encode_segwit_addr("rn", 0,
        std::span<const uint8_t>(h160.data(), 20));
    ASSERT_FALSE(addr.empty());
    // Should start with "rn1"
    ASSERT_TRUE(addr.substr(0, 3) == "rn1");
}

TEST(address_bech32_roundtrip) {
    auto kp = ed25519_generate().value();
    auto h160 = hash160(std::span<const uint8_t>(kp.public_key.data));
    std::vector<uint8_t> prog(h160.begin(), h160.end());

    auto addr = core::encode_segwit_addr("rn", 0, prog);
    auto decoded = core::decode_segwit_addr("rn", addr);

    ASSERT_TRUE(decoded.valid);
    ASSERT_EQ(decoded.witness_version, 0);
    ASSERT_EQ(decoded.witness_program, prog);
}

TEST(address_different_keys_different_addresses) {
    auto kp1 = ed25519_generate().value();
    auto kp2 = ed25519_generate().value();

    auto h1 = hash160(std::span<const uint8_t>(kp1.public_key.data));
    auto h2 = hash160(std::span<const uint8_t>(kp2.public_key.data));

    auto addr1 = core::encode_segwit_addr("rn", 0,
        std::span<const uint8_t>(h1.data(), 20));
    auto addr2 = core::encode_segwit_addr("rn", 0,
        std::span<const uint8_t>(h2.data(), 20));

    ASSERT_NE(addr1, addr2);
}

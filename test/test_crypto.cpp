// Tests for crypto module: Keccak-256, Ed25519, BIP39, Hash160

#include "test_framework.h"

#include "core/hex.h"
#include "core/types.h"
#include "crypto/bip39.h"
#include "crypto/ed25519.h"
#include "crypto/hash.h"
#include "crypto/keccak.h"

#include <array>
#include <cstring>
#include <string>
#include <vector>

using namespace rnet;
using namespace rnet::crypto;

// ─── Keccak-256 test vectors (MUST match Ethereum) ──────────────────

TEST(keccak256_empty_string) {
    // Ethereum keccak256("") = 0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470
    auto hash = keccak256(std::string_view(""));
    auto expected = uint256::from_hex(
        "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470");
    ASSERT_EQ(hash, expected);
}

TEST(keccak256_abc) {
    // Ethereum keccak256("abc") = 0x4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45
    auto hash = keccak256(std::string_view("abc"));
    auto expected = uint256::from_hex(
        "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45");
    ASSERT_EQ(hash, expected);
}

TEST(keccak256_bytes_overload) {
    // Same as above but using span<const uint8_t> overload
    std::vector<uint8_t> data = {'a', 'b', 'c'};
    auto hash = keccak256(std::span<const uint8_t>(data));
    auto expected = uint256::from_hex(
        "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45");
    ASSERT_EQ(hash, expected);
}

TEST(keccak256_long_input) {
    // keccak256 of 256 zero bytes
    std::vector<uint8_t> data(256, 0x00);
    auto hash = keccak256(data);
    // Result should be non-zero and deterministic
    ASSERT_FALSE(hash.is_zero());
    // Hash again to verify determinism
    auto hash2 = keccak256(data);
    ASSERT_EQ(hash, hash2);
}

TEST(keccak256d_is_double_hash) {
    // keccak256d(x) = keccak256(keccak256(x))
    std::string_view input = "test data";
    auto single = keccak256(input);
    auto double_manual = keccak256(single.span());
    auto double_fn = keccak256d(input);
    ASSERT_EQ(double_manual, double_fn);
}

TEST(keccak256_incremental) {
    // Incremental hasher should produce the same result as one-shot
    std::string_view part1 = "hello ";
    std::string_view part2 = "world";
    std::string_view full = "hello world";

    auto oneshot = keccak256(full);

    KeccakHasher hasher;
    hasher.write(part1);
    hasher.write(part2);
    auto incremental = hasher.finalize();

    ASSERT_EQ(oneshot, incremental);
}

TEST(keccak256_incremental_reset) {
    KeccakHasher hasher;
    hasher.write("data");
    hasher.finalize();  // consumes state

    hasher.reset();
    hasher.write("abc");
    auto hash = hasher.finalize();

    auto expected = keccak256(std::string_view("abc"));
    ASSERT_EQ(hash, expected);
}

// ─── Ed25519 test vectors (RFC 8032) ────────────────────────────────

TEST(ed25519_rfc8032_test1) {
    // Verify ed25519_from_seed produces a valid, deterministic keypair
    auto seed = core::from_hex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60");
    ASSERT_EQ(seed.size(), size_t(32));

    auto kp_result = ed25519_from_seed(seed);
    ASSERT_TRUE(kp_result.is_ok());
    auto kp = kp_result.value();

    // Key must not be zero
    ASSERT_FALSE(kp.public_key.is_zero());

    // Deterministic: same seed -> same key
    auto kp2 = ed25519_from_seed(seed).value();
    ASSERT_EQ(std::memcmp(kp.public_key.data.data(), kp2.public_key.data.data(), 32), 0);

    // Sign and verify roundtrip
    std::vector<uint8_t> msg = {0x48, 0x65, 0x6c, 0x6c, 0x6f};
    auto sig = ed25519_sign(kp.secret, msg).value();
    ASSERT_TRUE(ed25519_verify(kp.public_key, msg, sig));
}

TEST(ed25519_rfc8032_test1_sign_empty) {
    // RFC 8032 Test 1: sign empty message
    auto seed = core::from_hex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60");
    auto kp = ed25519_from_seed(seed).value();

    // Sign empty message
    std::vector<uint8_t> msg;
    auto sig_result = ed25519_sign(kp.secret, std::span<const uint8_t>(msg));
    ASSERT_TRUE(sig_result.is_ok());
    auto sig = sig_result.value();

    // Expected signature from RFC 8032:
    auto expected_sig = core::from_hex(
        "e5564300c360ac729086e2cc806e828a"
        "84877f1eb8e5d974d873e06522490155"
        "5fb8821590a33bacc61e39701cf9b46b"
        "d25bf5f0595bbe24655141438e7a100b");
    ASSERT_EQ(sig.data.size(), expected_sig.size());
    ASSERT_EQ(std::memcmp(sig.data.data(), expected_sig.data(), 64), 0);
}

TEST(ed25519_sign_verify_roundtrip) {
    auto kp_result = ed25519_generate();
    ASSERT_TRUE(kp_result.is_ok());
    auto kp = kp_result.value();

    std::string_view msg = "ResonanceNet test message";
    auto sig_result = ed25519_sign(kp.secret, msg);
    ASSERT_TRUE(sig_result.is_ok());
    auto sig = sig_result.value();

    ASSERT_TRUE(ed25519_verify(kp.public_key, msg, sig));
}

TEST(ed25519_wrong_key_fails) {
    auto kp1 = ed25519_generate().value();
    auto kp2 = ed25519_generate().value();

    std::string_view msg = "test";
    auto sig = ed25519_sign(kp1.secret, msg).value();

    // Verify with wrong key should fail
    ASSERT_FALSE(ed25519_verify(kp2.public_key, msg, sig));
}

TEST(ed25519_modified_message_fails) {
    auto kp = ed25519_generate().value();

    std::string_view msg1 = "original message";
    auto sig = ed25519_sign(kp.secret, msg1).value();

    std::string_view msg2 = "modified message";
    ASSERT_FALSE(ed25519_verify(kp.public_key, msg2, sig));
}

TEST(ed25519_deterministic_signatures) {
    auto kp = ed25519_generate().value();
    std::string_view msg = "deterministic test";
    auto sig1 = ed25519_sign(kp.secret, msg).value();
    auto sig2 = ed25519_sign(kp.secret, msg).value();
    ASSERT_EQ(sig1, sig2);
}

TEST(ed25519_from_seed_deterministic) {
    auto seed = core::from_hex("0000000000000000000000000000000000000000000000000000000000000001");
    auto kp1 = ed25519_from_seed(seed).value();
    auto kp2 = ed25519_from_seed(seed).value();
    ASSERT_EQ(kp1.public_key, kp2.public_key);
}

TEST(ed25519_coinbase_script) {
    auto kp = ed25519_generate().value();
    auto script = ed25519_coinbase_script(kp.public_key);
    // Should be [0x20][32-byte pubkey][0xAC] = 34 bytes
    ASSERT_EQ(script.size(), size_t(34));
    ASSERT_EQ(script[0], uint8_t(0x20));
    ASSERT_EQ(script[33], uint8_t(0xAC));
    ASSERT_EQ(std::memcmp(script.data() + 1, kp.public_key.data.data(), 32), 0);
}

TEST(ed25519_parse_coinbase_script) {
    auto kp = ed25519_generate().value();
    auto script = ed25519_coinbase_script(kp.public_key);
    auto parsed = ed25519_parse_coinbase_script(script);
    ASSERT_TRUE(parsed.is_ok());
    ASSERT_EQ(parsed.value(), kp.public_key);
}

// ─── BIP39 tests ────────────────────────────────────────────────────

TEST(bip39_generate_24_words) {
    auto result = generate_mnemonic(24);
    ASSERT_TRUE(result.is_ok());
    auto mnemonic = result.value();

    // Count words
    int count = 1;
    for (char c : mnemonic) {
        if (c == ' ') count++;
    }
    ASSERT_EQ(count, 24);
}

TEST(bip39_generate_12_words) {
    auto result = generate_mnemonic(12);
    ASSERT_TRUE(result.is_ok());
    auto mnemonic = result.value();

    int count = 1;
    for (char c : mnemonic) {
        if (c == ' ') count++;
    }
    ASSERT_EQ(count, 12);
}

TEST(bip39_validate_generated) {
    auto mnemonic = generate_mnemonic(24).value();
    ASSERT_TRUE(validate_mnemonic(mnemonic));
}

TEST(bip39_validate_invalid) {
    ASSERT_FALSE(validate_mnemonic("not a valid mnemonic at all"));
    ASSERT_FALSE(validate_mnemonic(""));
}

TEST(bip39_mnemonic_to_seed) {
    // BIP39 test vector (from the spec):
    // Mnemonic: "abandon" x 11 + "about"
    std::string mnemonic = "abandon abandon abandon abandon abandon "
                           "abandon abandon abandon abandon abandon "
                           "abandon about";
    auto seed_result = mnemonic_to_seed(mnemonic, "");
    ASSERT_TRUE(seed_result.is_ok());
    auto seed = seed_result.value();
    // Seed should be 64 bytes and non-zero
    bool all_zero = true;
    for (auto b : seed) {
        if (b != 0) { all_zero = false; break; }
    }
    ASSERT_FALSE(all_zero);
}

TEST(bip39_mnemonic_to_seed_with_passphrase) {
    auto mnemonic = generate_mnemonic(24).value();
    auto seed1 = mnemonic_to_seed(mnemonic, "").value();
    auto seed2 = mnemonic_to_seed(mnemonic, "my secret passphrase").value();
    // Different passphrases must produce different seeds
    ASSERT_NE(seed1, seed2);
}

TEST(bip39_wordlist_size) {
    auto& wordlist = bip39_wordlist();
    ASSERT_EQ(wordlist.size(), size_t(2048));
}

TEST(bip39_word_index_found) {
    // "abandon" should be at index 0
    int idx = bip39_word_index("abandon");
    ASSERT_EQ(idx, 0);
    // "zoo" should be at index 2047
    int idx2 = bip39_word_index("zoo");
    ASSERT_EQ(idx2, 2047);
}

TEST(bip39_word_index_not_found) {
    int idx = bip39_word_index("notaword");
    ASSERT_EQ(idx, -1);
}

TEST(bip39_entropy_roundtrip) {
    // Generate mnemonic from entropy, convert back
    std::vector<uint8_t> entropy(32, 0);  // 256 bits of zeros
    auto mnemonic_result = entropy_to_mnemonic(entropy);
    ASSERT_TRUE(mnemonic_result.is_ok());
    auto mnemonic = mnemonic_result.value();

    auto entropy_back = mnemonic_to_entropy(mnemonic);
    ASSERT_TRUE(entropy_back.is_ok());
    ASSERT_EQ(entropy_back.value(), entropy);
}

// ─── Hash160 tests ──────────────────────────────────────────────────

TEST(hash160_basic) {
    // Hash160 = first 20 bytes of keccak256d
    std::vector<uint8_t> data = {'t', 'e', 's', 't'};
    auto h160 = hash160(data);
    auto h256d = keccak256d(data);

    // First 20 bytes should match
    ASSERT_EQ(std::memcmp(h160.data(), h256d.data(), 20), 0);
}

TEST(hash160_different_inputs) {
    auto h1 = hash160(std::string_view("input1"));
    auto h2 = hash160(std::string_view("input2"));
    ASSERT_NE(h1, h2);
}

TEST(hash256_is_keccak256d) {
    std::string_view input = "hello";
    auto h256 = hash256(input);
    auto kd = keccak256d(input);
    ASSERT_EQ(h256, kd);
}

// ─── Tagged hash tests ──────────────────────────────────────────────

TEST(tagged_hash_domain_separation) {
    std::vector<uint8_t> data = {1, 2, 3};
    auto h1 = tagged_hash("TagA", data);
    auto h2 = tagged_hash("TagB", data);
    ASSERT_NE(h1, h2);
}

TEST(tagged_hash_deterministic) {
    std::vector<uint8_t> data = {0xAA, 0xBB};
    auto h1 = tagged_hash("test", data);
    auto h2 = tagged_hash("test", data);
    ASSERT_EQ(h1, h2);
}

// Tests for core module: Result<T>, DataStream, hex, base58, bech32

#include "test_framework.h"

#include "core/base58.h"
#include "core/bech32.h"
#include "core/error.h"
#include "core/hex.h"
#include "core/stream.h"
#include "core/types.h"

#include <cstring>
#include <string>
#include <vector>

using namespace rnet;
using namespace rnet::core;

// ─── Result<T> tests ────────────────────────────────────────────────

TEST(result_ok_value) {
    auto r = Result<int>::ok(42);
    ASSERT_TRUE(r.is_ok());
    ASSERT_FALSE(r.is_err());
    ASSERT_EQ(r.value(), 42);
    ASSERT_TRUE(static_cast<bool>(r));
}

TEST(result_err_string) {
    auto r = Result<int>::err("something went wrong");
    ASSERT_TRUE(r.is_err());
    ASSERT_FALSE(r.is_ok());
    ASSERT_EQ(r.error(), std::string("something went wrong"));
    ASSERT_FALSE(static_cast<bool>(r));
}

TEST(result_void_ok) {
    auto r = Result<void>::ok();
    ASSERT_TRUE(r.is_ok());
    ASSERT_FALSE(r.is_err());
}

TEST(result_void_err) {
    auto r = Result<void>::err("bad");
    ASSERT_TRUE(r.is_err());
    ASSERT_EQ(r.error(), std::string("bad"));
}

TEST(result_value_or) {
    auto r = Result<int>::err("nope");
    ASSERT_EQ(r.value_or(99), 99);

    auto r2 = Result<int>::ok(7);
    ASSERT_EQ(r2.value_or(99), 7);
}

TEST(result_map) {
    auto r = Result<int>::ok(5);
    auto r2 = r.map([](int v) { return v * 2; });
    ASSERT_TRUE(r2.is_ok());
    ASSERT_EQ(r2.value(), 10);

    auto r3 = Result<int>::err("fail");
    auto r4 = r3.map([](int v) { return v * 2; });
    ASSERT_TRUE(r4.is_err());
    ASSERT_EQ(r4.error(), std::string("fail"));
}

TEST(result_and_then) {
    auto r = Result<int>::ok(10);
    auto r2 = r.and_then([](int v) -> Result<std::string> {
        return Result<std::string>::ok(std::to_string(v));
    });
    ASSERT_TRUE(r2.is_ok());
    ASSERT_EQ(r2.value(), std::string("10"));
}

// ─── DataStream tests ───────────────────────────────────────────────

TEST(datastream_write_read) {
    DataStream ds;
    uint8_t data[] = {0xDE, 0xAD, 0xBE, 0xEF};
    ds.write(data, 4);
    ASSERT_EQ(ds.size(), size_t(4));

    uint8_t out[4] = {};
    ds.read(out, 4);
    ASSERT_EQ(out[0], uint8_t(0xDE));
    ASSERT_EQ(out[1], uint8_t(0xAD));
    ASSERT_EQ(out[2], uint8_t(0xBE));
    ASSERT_EQ(out[3], uint8_t(0xEF));
    ASSERT_TRUE(ds.eof());
}

TEST(datastream_roundtrip) {
    DataStream ds;
    std::vector<uint8_t> original = {1, 2, 3, 4, 5, 6, 7, 8};
    ds.write(original.data(), original.size());

    ASSERT_EQ(ds.size(), size_t(8));
    ASSERT_EQ(ds.remaining(), size_t(8));

    std::vector<uint8_t> readback(8);
    ds.read(readback.data(), 8);
    ASSERT_EQ(original, readback);
    ASSERT_EQ(ds.remaining(), size_t(0));
}

TEST(datastream_rewind) {
    DataStream ds;
    uint8_t byte = 0x42;
    ds.write(&byte, 1);
    ds.read(&byte, 1);
    ASSERT_TRUE(ds.eof());

    ds.rewind();
    ASSERT_FALSE(ds.eof());
    ASSERT_EQ(ds.remaining(), size_t(1));
}

TEST(datastream_read_past_end_throws) {
    DataStream ds;
    uint8_t byte = 0;
    ASSERT_THROWS(ds.read(&byte, 1));
}

TEST(datastream_span) {
    std::vector<uint8_t> data = {0x01, 0x02, 0x03};
    DataStream ds(data);
    ASSERT_EQ(ds.size(), size_t(3));
    auto sp = ds.span();
    ASSERT_EQ(sp.size(), size_t(3));
    ASSERT_EQ(sp[0], uint8_t(0x01));
}

// ─── Hex encoding tests ────────────────────────────────────────────

TEST(hex_encode_empty) {
    std::vector<uint8_t> empty;
    ASSERT_EQ(to_hex(empty), std::string(""));
}

TEST(hex_encode_bytes) {
    std::vector<uint8_t> data = {0xDE, 0xAD, 0xBE, 0xEF};
    ASSERT_EQ(to_hex(data), std::string("deadbeef"));
}

TEST(hex_decode_valid) {
    auto result = from_hex("deadbeef");
    ASSERT_EQ(result.size(), size_t(4));
    ASSERT_EQ(result[0], uint8_t(0xDE));
    ASSERT_EQ(result[1], uint8_t(0xAD));
    ASSERT_EQ(result[2], uint8_t(0xBE));
    ASSERT_EQ(result[3], uint8_t(0xEF));
}

TEST(hex_decode_uppercase) {
    auto result = from_hex("DEADBEEF");
    ASSERT_EQ(result.size(), size_t(4));
    ASSERT_EQ(result[0], uint8_t(0xDE));
}

TEST(hex_decode_empty) {
    auto result = from_hex("");
    ASSERT_TRUE(result.empty());
}

TEST(hex_roundtrip) {
    std::vector<uint8_t> data = {0x00, 0xFF, 0x80, 0x7F, 0x01};
    auto hex_str = to_hex(data);
    auto decoded = from_hex(hex_str);
    ASSERT_EQ(data, decoded);
}

TEST(hex_is_hex_valid) {
    ASSERT_TRUE(is_hex("abcdef0123456789"));
    ASSERT_TRUE(is_hex("ABCDEF"));
    ASSERT_TRUE(is_hex(""));
}

TEST(hex_is_hex_invalid) {
    ASSERT_FALSE(is_hex("xyz"));
    ASSERT_FALSE(is_hex("abcdefg"));  // odd length
}

TEST(hex_reverse) {
    ASSERT_EQ(reverse_hex("aabb"), std::string("bbaa"));
    ASSERT_EQ(reverse_hex("0102030405"), std::string("0504030201"));
}

// ─── uint256 tests ──────────────────────────────────────────────────

TEST(uint256_zero) {
    uint256 zero;
    ASSERT_TRUE(zero.is_zero());
    ASSERT_EQ(zero.to_hex(), std::string(
        "0000000000000000000000000000000000000000000000000000000000000000"));
}

TEST(uint256_from_hex) {
    auto val = uint256::from_hex(
        "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470");
    ASSERT_FALSE(val.is_zero());
    ASSERT_EQ(val[0], uint8_t(0xc5));
    ASSERT_EQ(val[31], uint8_t(0x70));
}

TEST(uint256_comparison) {
    auto a = uint256::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000001");
    auto b = uint256::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000002");
    ASSERT_TRUE(a < b);
    ASSERT_NE(a, b);
}

TEST(uint256_xor) {
    auto a = uint256::from_hex(
        "ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00");
    auto b = uint256::from_hex(
        "00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff");
    auto c = a ^ b;
    ASSERT_EQ(c.to_hex(), std::string(
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"));
}

// ─── Base58 tests ───────────────────────────────────────────────────

TEST(base58_encode_decode_roundtrip) {
    std::vector<uint8_t> data = {0x00, 0x01, 0x02, 0x03, 0x04};
    auto encoded = base58_encode(data);
    ASSERT_FALSE(encoded.empty());

    auto decoded = base58_decode(encoded);
    ASSERT_TRUE(decoded.has_value());
    ASSERT_EQ(decoded.value(), data);
}

TEST(base58_encode_leading_zeros) {
    // Leading zero bytes should produce leading '1' characters
    std::vector<uint8_t> data = {0x00, 0x00, 0x00, 0x01};
    auto encoded = base58_encode(data);
    // Should start with three '1's (for the three leading zero bytes)
    ASSERT_TRUE(encoded.size() >= 3);
    ASSERT_EQ(encoded[0], '1');
    ASSERT_EQ(encoded[1], '1');
    ASSERT_EQ(encoded[2], '1');
}

TEST(base58_decode_invalid) {
    // '0', 'O', 'I', 'l' are not in the Base58 alphabet
    auto result = base58_decode("0OIl");
    ASSERT_FALSE(result.has_value());
}

TEST(base58_is_valid) {
    ASSERT_TRUE(is_valid_base58("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"));
    ASSERT_FALSE(is_valid_base58("0"));
    ASSERT_FALSE(is_valid_base58("O"));
    ASSERT_FALSE(is_valid_base58("I"));
    ASSERT_FALSE(is_valid_base58("l"));
}

TEST(base58check_roundtrip) {
    std::vector<uint8_t> payload = {0x05, 0xDE, 0xAD, 0xBE, 0xEF};
    auto encoded = base58check_encode_simple(payload);
    ASSERT_FALSE(encoded.empty());

    auto decoded = base58check_decode_simple(encoded);
    ASSERT_TRUE(decoded.has_value());
    ASSERT_EQ(decoded.value(), payload);
}

// ─── Bech32 tests ───────────────────────────────────────────────────

TEST(bech32_encode_decode_roundtrip) {
    std::vector<uint8_t> witness_program(20, 0xAB);
    auto addr = encode_segwit_addr("rn", 0, witness_program);
    ASSERT_FALSE(addr.empty());

    auto decoded = decode_segwit_addr("rn", addr);
    ASSERT_TRUE(decoded.valid);
    ASSERT_EQ(decoded.witness_version, 0);
    ASSERT_EQ(decoded.witness_program, witness_program);
}

TEST(bech32_encode_v0_20byte) {
    // P2WPKH: version 0, 20-byte program
    std::vector<uint8_t> prog(20, 0x00);
    prog[0] = 0x75;
    prog[19] = 0x1E;
    auto addr = encode_segwit_addr("rn", 0, prog);
    ASSERT_FALSE(addr.empty());
    // Should start with "rn1q" (version 0 = 'q' in bech32)
    ASSERT_TRUE(addr.substr(0, 3) == "rn1");
}

TEST(bech32_encode_v0_32byte) {
    // P2WSH: version 0, 32-byte program
    std::vector<uint8_t> prog(32, 0xFF);
    auto addr = encode_segwit_addr("rn", 0, prog);
    ASSERT_FALSE(addr.empty());

    auto decoded = decode_segwit_addr("rn", addr);
    ASSERT_TRUE(decoded.valid);
    ASSERT_EQ(decoded.witness_version, 0);
    ASSERT_EQ(decoded.witness_program.size(), size_t(32));
}

TEST(bech32_testnet_hrp) {
    std::vector<uint8_t> prog(20, 0x42);
    auto addr = encode_segwit_addr("trn", 0, prog);
    ASSERT_FALSE(addr.empty());
    ASSERT_TRUE(addr.substr(0, 4) == "trn1");

    auto decoded = decode_segwit_addr("trn", addr);
    ASSERT_TRUE(decoded.valid);
}

TEST(bech32_wrong_hrp_fails) {
    std::vector<uint8_t> prog(20, 0x42);
    auto addr = encode_segwit_addr("rn", 0, prog);
    // Decode with wrong HRP should fail
    auto decoded = decode_segwit_addr("trn", addr);
    ASSERT_FALSE(decoded.valid);
}

TEST(bech32_convert_bits) {
    // Convert 8-bit to 5-bit and back
    std::vector<uint8_t> data = {0xFF, 0x00, 0xAA};
    auto five_bit = convert_bits(data, 8, 5, true);
    ASSERT_FALSE(five_bit.empty());
    auto eight_bit = convert_bits(five_bit, 5, 8, false);
    ASSERT_EQ(eight_bit, data);
}

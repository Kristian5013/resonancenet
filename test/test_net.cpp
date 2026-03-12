// Tests for net module: protocol message serialization, network addresses

#include "test_framework.h"

#include "core/stream.h"
#include "core/types.h"
#include "net/protocol.h"

#include <cstring>
#include <string>

using namespace rnet;
using namespace rnet::net;

// ─── Protocol constants ─────────────────────────────────────────────

TEST(net_magic_bytes) {
    ASSERT_EQ(NETWORK_MAGIC[0], uint8_t(0x52));  // 'R'
    ASSERT_EQ(NETWORK_MAGIC[1], uint8_t(0x4E));  // 'N'
    ASSERT_EQ(NETWORK_MAGIC[2], uint8_t(0x45));  // 'E'
    ASSERT_EQ(NETWORK_MAGIC[3], uint8_t(0x54));  // 'T'
}

TEST(net_default_port) {
    ASSERT_EQ(DEFAULT_PORT, uint16_t(9555));
}

TEST(net_protocol_version) {
    ASSERT_TRUE(PROTOCOL_VERSION >= MIN_PROTOCOL_VERSION);
    ASSERT_EQ(PROTOCOL_VERSION, int32_t(70100));
}

TEST(net_max_message_size) {
    // 512 MB
    ASSERT_EQ(MAX_MESSAGE_SIZE, uint32_t(512 * 1024 * 1024));
}

// ─── Message command strings ────────────────────────────────────────

TEST(net_msg_commands) {
    ASSERT_EQ(std::string(msg::VERSION), std::string("version"));
    ASSERT_EQ(std::string(msg::VERACK), std::string("verack"));
    ASSERT_EQ(std::string(msg::ADDR), std::string("addr"));
    ASSERT_EQ(std::string(msg::INV), std::string("inv"));
    ASSERT_EQ(std::string(msg::GETDATA), std::string("getdata"));
    ASSERT_EQ(std::string(msg::TX), std::string("tx"));
    ASSERT_EQ(std::string(msg::BLOCK), std::string("block"));
    ASSERT_EQ(std::string(msg::HEADERS), std::string("headers"));
    ASSERT_EQ(std::string(msg::PING), std::string("ping"));
    ASSERT_EQ(std::string(msg::PONG), std::string("pong"));
    ASSERT_EQ(std::string(msg::CHECKPOINT), std::string("checkpoint"));
}

TEST(net_msg_command_size) {
    // All command strings should fit in COMMAND_SIZE bytes
    ASSERT_TRUE(std::strlen(msg::VERSION) <= COMMAND_SIZE);
    ASSERT_TRUE(std::strlen(msg::TRAININGSTATUS) <= COMMAND_SIZE);
    ASSERT_TRUE(std::strlen(msg::GETCHECKPOINT) <= COMMAND_SIZE);
}

// ─── InvType tests ──────────────────────────────────────────────────

TEST(net_inv_types) {
    ASSERT_EQ(static_cast<uint32_t>(InvType::INV_ERROR), uint32_t(0));
    ASSERT_EQ(static_cast<uint32_t>(InvType::INV_TX), uint32_t(1));
    ASSERT_EQ(static_cast<uint32_t>(InvType::INV_BLOCK), uint32_t(2));
    ASSERT_EQ(static_cast<uint32_t>(InvType::INV_CHECKPOINT), uint32_t(5));
}

TEST(net_inv_witness_types) {
    ASSERT_EQ(static_cast<uint32_t>(InvType::INV_WITNESS_TX), uint32_t(0x40000001));
    ASSERT_EQ(static_cast<uint32_t>(InvType::INV_WITNESS_BLOCK), uint32_t(0x40000002));
}

// ─── Service flags ──────────────────────────────────────────────────

TEST(net_service_flags) {
    ASSERT_EQ(static_cast<uint64_t>(NODE_NONE), uint64_t(0));
    ASSERT_EQ(static_cast<uint64_t>(NODE_NETWORK), uint64_t(1));
    ASSERT_EQ(static_cast<uint64_t>(NODE_TRAINING), uint64_t(2));
    ASSERT_EQ(static_cast<uint64_t>(NODE_CHECKPOINT), uint64_t(4));
}

TEST(net_service_flags_bitwise) {
    uint64_t flags = NODE_NETWORK | NODE_TRAINING | NODE_WITNESS;
    ASSERT_TRUE((flags & NODE_NETWORK) != 0);
    ASSERT_TRUE((flags & NODE_TRAINING) != 0);
    ASSERT_TRUE((flags & NODE_WITNESS) != 0);
    ASSERT_FALSE((flags & NODE_LIGHTNING) != 0);
}

// ─── CInv tests ─────────────────────────────────────────────────────

TEST(net_inv_creation) {
    auto hash = uint256::from_hex(
        "aabbccddaabbccddaabbccddaabbccddaabbccddaabbccddaabbccddaabbccdd");
    CInv inv;
    inv.type = InvType::INV_TX;
    inv.hash = hash;

    ASSERT_EQ(static_cast<uint32_t>(inv.type), uint32_t(1));
    ASSERT_EQ(inv.hash, hash);
}

TEST(net_inv_block) {
    auto hash = uint256::from_hex(
        "1111111111111111111111111111111111111111111111111111111111111111");
    CInv inv;
    inv.type = InvType::INV_BLOCK;
    inv.hash = hash;
    ASSERT_EQ(static_cast<uint32_t>(inv.type), uint32_t(2));
}

// ─── Message header serialization ───────────────────────────────────

TEST(net_message_header_serialize) {
    // Construct a message header manually and verify field placement
    core::DataStream ds;

    // Write magic bytes
    ds.write(NETWORK_MAGIC.data(), 4);

    // Write command (12 bytes, zero-padded)
    char cmd[COMMAND_SIZE] = {};
    std::strncpy(cmd, msg::VERSION, COMMAND_SIZE);
    ds.write(cmd, COMMAND_SIZE);

    // Write payload size (4 bytes, little-endian)
    uint32_t payload_size = 100;
    core::ser_write_u32(ds, payload_size);

    // Write checksum (4 bytes)
    uint32_t checksum = 0xDEADBEEF;
    core::ser_write_u32(ds, checksum);

    // Header should be 4 + 12 + 4 + 4 = 24 bytes
    ASSERT_EQ(ds.size(), size_t(24));

    // Read back and verify
    ds.rewind();
    uint8_t magic_read[4];
    ds.read(magic_read, 4);
    ASSERT_EQ(magic_read[0], uint8_t(0x52));

    char cmd_read[COMMAND_SIZE];
    ds.read(cmd_read, COMMAND_SIZE);
    ASSERT_EQ(std::string(cmd_read), std::string("version"));

    uint32_t size_read = core::ser_read_u32(ds);
    ASSERT_EQ(size_read, uint32_t(100));

    uint32_t checksum_read = core::ser_read_u32(ds);
    ASSERT_EQ(checksum_read, uint32_t(0xDEADBEEF));
}

TEST(net_ping_pong_serialization) {
    // Ping/pong messages contain a 64-bit nonce
    core::DataStream ds;
    uint64_t nonce = 0x123456789ABCDEF0ULL;
    core::ser_write_u64(ds, nonce);
    ASSERT_EQ(ds.size(), size_t(8));

    ds.rewind();
    uint64_t nonce_read = core::ser_read_u64(ds);
    ASSERT_EQ(nonce_read, nonce);
}

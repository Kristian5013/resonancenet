#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

#include "core/serialize.h"
#include "core/stream.h"
#include "core/types.h"

namespace rnet::net {

/// Default P2P network magic bytes: "RNET" (mainnet)
/// The active magic is set at startup via set_network_magic() based on
/// the selected network (mainnet/testnet/regtest).
inline std::array<uint8_t, 4>& active_network_magic() {
    static std::array<uint8_t, 4> magic = {0x52, 0x4E, 0x45, 0x54};  // "RNET"
    return magic;
}

/// Set the active network magic (call once at startup before any P2P I/O).
inline void set_network_magic(const std::array<uint8_t, 4>& magic) {
    active_network_magic() = magic;
}

/// Convenience alias — returns the currently active magic bytes.
/// All serialization / parsing code should use this instead of a constant.
inline const std::array<uint8_t, 4>& NETWORK_MAGIC_REF() {
    return active_network_magic();
}

/// Legacy constant kept for backward-compatible default initialization.
static constexpr std::array<uint8_t, 4> MAINNET_MAGIC = {
    0x52, 0x4E, 0x45, 0x54  // "RNET"
};

/// Default P2P port
static constexpr uint16_t DEFAULT_PORT = 9555;

/// Maximum message payload size (512 MB for checkpoints)
static constexpr uint32_t MAX_MESSAGE_SIZE = 512 * 1024 * 1024;

/// Maximum protocol message command length
static constexpr size_t COMMAND_SIZE = 12;

/// Protocol version
static constexpr int32_t PROTOCOL_VERSION = 70100;
static constexpr int32_t MIN_PROTOCOL_VERSION = 70000;

/// Service flags
enum ServiceFlags : uint64_t {
    NODE_NONE          = 0,
    NODE_NETWORK       = (1 << 0),  ///< Full node with block data
    NODE_TRAINING      = (1 << 1),  ///< Can verify PoT training
    NODE_CHECKPOINT    = (1 << 2),  ///< Serves model checkpoints
    NODE_BLOOM         = (1 << 3),  ///< Supports bloom filters
    NODE_WITNESS       = (1 << 4),  ///< Supports witness data
    NODE_LIGHTNING     = (1 << 5),  ///< Lightning network node
};

/// Message command strings
namespace msg {
    inline constexpr const char* VERSION         = "version";
    inline constexpr const char* VERACK          = "verack";
    inline constexpr const char* ADDR            = "addr";
    inline constexpr const char* INV             = "inv";
    inline constexpr const char* GETDATA         = "getdata";
    inline constexpr const char* GETBLOCKS       = "getblocks";
    inline constexpr const char* GETHEADERS      = "getheaders";
    inline constexpr const char* TX              = "tx";
    inline constexpr const char* BLOCK           = "block";
    inline constexpr const char* HEADERS         = "headers";
    inline constexpr const char* PING            = "ping";
    inline constexpr const char* PONG            = "pong";
    inline constexpr const char* GETADDR         = "getaddr";
    inline constexpr const char* REJECT          = "reject";
    inline constexpr const char* CHECKPOINT      = "checkpoint";
    inline constexpr const char* GETCHECKPOINT   = "getchkpt";
    inline constexpr const char* TRAININGSTATUS  = "trainstatus";
    inline constexpr const char* GROWTHINFO      = "growthinfo";
    inline constexpr const char* SENDHEADERS     = "sendheaders";
    inline constexpr const char* MEMPOOL         = "mempool";
    inline constexpr const char* NOTFOUND        = "notfound";
}

/// Inventory type codes
enum class InvType : uint32_t {
    INV_ERROR       = 0,
    INV_TX          = 1,
    INV_BLOCK       = 2,
    INV_FILTERED_BLOCK = 3,
    INV_WITNESS_TX  = 0x40000001,
    INV_WITNESS_BLOCK = 0x40000002,
    INV_CHECKPOINT  = 5,
};

/// Inventory vector — identifies an object by type + hash
struct CInv {
    InvType type = InvType::INV_ERROR;
    rnet::uint256 hash;

    CInv() = default;
    CInv(InvType t, const rnet::uint256& h) : type(t), hash(h) {}

    std::string to_string() const;

    bool operator==(const CInv& other) const {
        return type == other.type && hash == other.hash;
    }

    template<typename Stream>
    void serialize(Stream& s) const {
        uint32_t type_val = static_cast<uint32_t>(type);
        core::Serialize(s, type_val);
        core::Serialize(s, hash);
    }

    template<typename Stream>
    void unserialize(Stream& s) {
        uint32_t type_val = 0;
        core::Unserialize(s, type_val);
        type = static_cast<InvType>(type_val);
        core::Unserialize(s, hash);
    }
};

/// Network address (IPv4/IPv6 + port + services)
struct CNetAddr {
    std::array<uint8_t, 16> ip{};   ///< IPv6-mapped address
    uint16_t port = DEFAULT_PORT;
    uint64_t services = NODE_NONE;
    int64_t time = 0;                ///< Last seen time

    /// Set from IPv4 address bytes
    void set_ipv4(uint8_t a, uint8_t b, uint8_t c, uint8_t d);

    /// Get IPv4 string if it is an IPv4-mapped address
    std::string to_string() const;

    /// Check if this is an IPv4-mapped address
    bool is_ipv4() const;

    /// Check if this is a routable address
    bool is_routable() const;

    /// Check if this is a local address
    bool is_local() const;

    bool operator==(const CNetAddr& other) const {
        return ip == other.ip && port == other.port;
    }

    SERIALIZE_METHODS(
        READWRITE(self.services);
        READWRITE(self.ip);
        READWRITE(self.port);
        READWRITE(self.time);
    )
};

/// Wire message header
/// Format: [4B magic] [12B command] [4B payload_size] [4B checksum]
struct MessageHeader {
    static constexpr size_t HEADER_SIZE = 4 + COMMAND_SIZE + 4 + 4;

    std::array<uint8_t, 4> magic = MAINNET_MAGIC;
    std::array<char, COMMAND_SIZE> command{};
    uint32_t payload_size = 0;
    uint32_t checksum = 0;

    /// Set the command field from a string
    void set_command(std::string_view cmd);

    /// Get the command as a string (trimmed)
    std::string get_command() const;

    /// Validate the magic bytes
    bool valid_magic() const;

    /// Serialize the header
    template<typename Stream>
    void serialize(Stream& s) const {
        s.write(magic.data(), 4);
        s.write(reinterpret_cast<const uint8_t*>(command.data()),
                COMMAND_SIZE);
        core::ser_write_u32(s, payload_size);
        core::ser_write_u32(s, checksum);
    }

    template<typename Stream>
    void unserialize(Stream& s) {
        s.read(magic.data(), 4);
        s.read(reinterpret_cast<uint8_t*>(command.data()), COMMAND_SIZE);
        payload_size = core::ser_read_u32(s);
        checksum = core::ser_read_u32(s);
    }
};

/// Version message payload
struct VersionMessage {
    int32_t version = PROTOCOL_VERSION;
    uint64_t services = NODE_NETWORK;
    int64_t timestamp = 0;
    CNetAddr addr_recv;
    CNetAddr addr_from;
    uint64_t nonce = 0;
    std::string user_agent;
    int32_t start_height = 0;
    bool relay = true;

    SERIALIZE_METHODS(
        READWRITE(self.version);
        READWRITE(self.services);
        READWRITE(self.timestamp);
        READWRITE(self.addr_recv);
        READWRITE(self.addr_from);
        READWRITE(self.nonce);
        READWRITE(self.user_agent);
        READWRITE(self.start_height);
        READWRITE(self.relay);
    )
};

/// Reject message payload
struct RejectMessage {
    std::string message;
    uint8_t code = 0;
    std::string reason;
    rnet::uint256 data;

    enum RejectCode : uint8_t {
        REJECT_MALFORMED       = 0x01,
        REJECT_INVALID         = 0x10,
        REJECT_OBSOLETE        = 0x11,
        REJECT_DUPLICATE       = 0x12,
        REJECT_NONSTANDARD     = 0x40,
        REJECT_DUST            = 0x41,
        REJECT_INSUFFICIENTFEE = 0x42,
        REJECT_CHECKPOINT      = 0x43,
    };
};

/// Convert InvType to string
std::string inv_type_string(InvType type);

}  // namespace rnet::net

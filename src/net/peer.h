#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <set>
#include <string>

#include "core/sync.h"
#include "core/time.h"
#include "core/types.h"
#include "net/protocol.h"

namespace rnet::net {

/// CPeer — represents the state of a P2P connection to a remote node.
class CPeer {
public:
    /// Unique peer identifier
    uint64_t id = 0;

    /// Remote address
    CNetAddr addr;

    /// Connection direction
    bool is_inbound = false;

    /// Versioning
    int32_t version = 0;
    uint64_t services = NODE_NONE;
    std::string user_agent;
    int32_t start_height = 0;
    bool relay = true;
    bool version_sent = false;
    bool version_received = false;
    bool handshake_complete = false;

    /// Timing
    int64_t connect_time = 0;
    int64_t last_send = 0;
    int64_t last_recv = 0;
    int64_t last_ping_time = 0;
    int64_t ping_wait = 0;

    /// Ping/pong tracking
    uint64_t ping_nonce_sent = 0;
    bool ping_outstanding = false;

    /// Bytes transferred
    std::atomic<uint64_t> bytes_sent{0};
    std::atomic<uint64_t> bytes_recv{0};

    /// Message counts
    std::atomic<uint64_t> messages_sent{0};
    std::atomic<uint64_t> messages_recv{0};

    /// Disconnect flags
    std::atomic<bool> disconnect_requested{false};

    /// Ban score (accumulated misbehavior points)
    std::atomic<int> ban_score{0};

    /// Headers sync state
    bool prefer_headers = false;
    int best_known_height = 0;
    rnet::uint256 best_known_hash;

    /// Known inventory (txs and blocks this peer has announced)
    mutable core::Mutex cs_inventory;
    std::set<rnet::uint256> known_txs;
    std::set<rnet::uint256> known_blocks;

    /// Socket file descriptor (platform-specific)
    int64_t socket_fd = -1;

    /// Transport layer encryption established
    bool transport_encrypted = false;

    CPeer() = default;

    /// Construct with id and address
    CPeer(uint64_t peer_id, const CNetAddr& address, bool inbound);

    /// Mark the handshake as complete
    void complete_handshake();

    /// Check if the peer supports a given service
    bool has_service(ServiceFlags flag) const;

    /// Add misbehavior points; returns true if peer should be banned
    bool misbehaving(int points, const std::string& reason);

    /// Check if this peer should be disconnected
    bool should_disconnect() const;

    /// Get latency (ms) based on last ping/pong
    int64_t get_ping_time() const;

    /// Human-readable
    std::string to_string() const;

    /// Ban score threshold
    static constexpr int BAN_THRESHOLD = 100;
};

}  // namespace rnet::net

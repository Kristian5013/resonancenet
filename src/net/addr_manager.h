#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/sync.h"
#include "core/types.h"
#include "net/protocol.h"

namespace rnet::net {

/// AddrInfo — extended address record with metadata
struct AddrInfo {
    CNetAddr addr;

    /// Source address (who told us about this peer)
    CNetAddr source;

    /// Number of successful connections
    int32_t success_count = 0;

    /// Number of connection attempts
    int32_t attempt_count = 0;

    /// Last successful connection time
    int64_t last_success = 0;

    /// Last connection attempt time
    int64_t last_attempt = 0;

    /// Time when we first learned about this address
    int64_t first_seen = 0;

    /// Whether this address is in the "tried" table
    bool is_tried = false;

    /// Bucket index in the new/tried table
    int32_t bucket = -1;
    int32_t bucket_position = -1;

    /// Chance weight for selection (higher = more likely to be selected)
    double get_chance() const;

    /// Key for hash map (IP:port string)
    std::string key() const { return addr.to_string(); }
};

/// AddrManager — maintains known peer addresses for discovery.
///
/// Implements a bucketed address manager similar to Bitcoin Core's AddrMan:
///
///   - "new" table: addresses we've learned about but never connected to
///   - "tried" table: addresses we've successfully connected to
///
/// Addresses are bucketed by source network group for eclipse attack
/// resistance. The selection algorithm prefers recently-seen addresses
/// and tried addresses over new ones.
///
/// The address database can be persisted to disk and loaded on startup.
class AddrManager {
public:
    /// Table sizes
    static constexpr int NEW_BUCKET_COUNT = 1024;
    static constexpr int NEW_BUCKET_SIZE = 64;
    static constexpr int TRIED_BUCKET_COUNT = 256;
    static constexpr int TRIED_BUCKET_SIZE = 64;

    /// Maximum age before an address is considered stale (30 days)
    static constexpr int64_t MAX_ADDR_AGE = 30 * 24 * 60 * 60;

    /// Maximum addresses to return for getaddr
    static constexpr int MAX_GETADDR_RETURN = 2500;

    /// Percentage of addresses to return for getaddr (23%)
    static constexpr int GETADDR_PERCENT = 23;

    AddrManager();
    ~AddrManager();

    // Non-copyable
    AddrManager(const AddrManager&) = delete;
    AddrManager& operator=(const AddrManager&) = delete;

    // ── Core operations ─────────────────────────────────────────────

    /// Add a new address, optionally with a source.
    /// Returns true if the address was added (not a duplicate).
    bool add(const CNetAddr& addr, const CNetAddr& source);

    /// Add multiple addresses from the same source
    int add(const std::vector<CNetAddr>& addrs, const CNetAddr& source);

    /// Mark an address as successfully connected.
    /// Moves it from the "new" table to the "tried" table.
    void mark_good(const CNetAddr& addr);

    /// Mark an address as failed to connect.
    /// Decreases its selection priority.
    void mark_failed(const CNetAddr& addr);

    /// Mark an address as attempted (regardless of success/failure).
    void mark_attempt(const CNetAddr& addr);

    /// Select a random address to connect to.
    /// Prefers tried addresses; uses weighted random selection.
    /// Returns a zero-port address if no addresses are available.
    CNetAddr select();

    /// Get a set of addresses for responding to a getaddr message.
    /// Returns up to GETADDR_PERCENT% of known addresses, capped
    /// at MAX_GETADDR_RETURN.
    std::vector<CNetAddr> get_addr(int max_count = 0) const;

    // ── Queries ─────────────────────────────────────────────────────

    /// Total number of known addresses
    size_t size() const;

    /// Number of addresses in the "new" table
    size_t new_count() const;

    /// Number of addresses in the "tried" table
    size_t tried_count() const;

    /// Check if an address is known
    bool contains(const CNetAddr& addr) const;

    /// Get info for a specific address (returns nullptr if not found)
    const AddrInfo* find(const CNetAddr& addr) const;

    // ── Persistence ─────────────────────────────────────────────────

    /// Save the address database to a file
    bool save(const std::string& path) const;

    /// Load the address database from a file
    bool load(const std::string& path);

    /// Clear all addresses
    void clear();

    // ── Seed addresses ──────────────────────────────────────────────

    /// Add DNS seed addresses
    void add_seed(const std::string& hostname, uint16_t port = DEFAULT_PORT);

    /// Add hardcoded seed addresses
    void add_seeds(const std::vector<std::string>& ips,
                   uint16_t port = DEFAULT_PORT);

private:
    mutable core::Mutex cs_addr_;

    /// All known addresses by key (IP:port string)
    std::unordered_map<std::string, AddrInfo> addr_map_;

    /// Count of addresses in new vs tried tables
    int32_t new_count_ = 0;
    int32_t tried_count_ = 0;

    /// Random nonce for bucket hashing (set once at construction)
    rnet::uint256 nonce_;

    /// Compute the "new" bucket for an address given its source
    int new_bucket(const CNetAddr& addr, const CNetAddr& source) const;

    /// Compute the "tried" bucket for an address
    int tried_bucket(const CNetAddr& addr) const;

    /// Get a network group identifier for eclipse attack resistance
    static std::vector<uint8_t> get_group(const CNetAddr& addr);

    /// Remove stale addresses (older than MAX_ADDR_AGE)
    void remove_stale();
};

}  // namespace rnet::net

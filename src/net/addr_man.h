#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/error.h"
#include "core/sync.h"
#include "net/protocol.h"

namespace rnet::net {

/// AddrInfo — extended address information for peer discovery
struct AddrInfo {
    CNetAddr addr;
    int64_t last_try = 0;         ///< Last connection attempt time
    int64_t last_success = 0;     ///< Last successful connection time
    int attempts = 0;              ///< Number of connection attempts
    int64_t last_count_attempt = 0;
    std::string source;            ///< How we learned about this address

    /// Check if this address is "terrible" (many failed attempts)
    bool is_terrible(int64_t now) const;

    /// Get a selection chance weight (higher = more likely to try)
    double get_chance(int64_t now) const;
};

/// AddrManager — manages known peer addresses for discovery.
/// Maintains a database of addresses learned from peers, DNS seeds,
/// and manual entries.
class AddrManager {
public:
    static constexpr size_t MAX_ADDRESSES = 65536;

    AddrManager();
    ~AddrManager() = default;

    // Non-copyable
    AddrManager(const AddrManager&) = delete;
    AddrManager& operator=(const AddrManager&) = delete;

    /// Add addresses learned from a peer
    void add(const std::vector<CNetAddr>& addrs,
             const std::string& source);

    /// Add a single address
    void add(const CNetAddr& addr, const std::string& source);

    /// Mark an address as successfully connected
    void mark_good(const CNetAddr& addr, int64_t time);

    /// Mark a connection attempt
    void mark_attempt(const CNetAddr& addr);

    /// Select an address to connect to
    /// Prefers addresses we haven't tried recently
    CNetAddr select() const;

    /// Select multiple addresses (for addr relay)
    std::vector<CNetAddr> get_addr(size_t max_count = 1000) const;

    /// Get the total number of known addresses
    size_t size() const;

    /// Check if we know about an address
    bool contains(const CNetAddr& addr) const;

    /// Save the address database to disk
    Result<void> save(const std::string& path) const;

    /// Load the address database from disk
    Result<void> load(const std::string& path);

    /// Clear all addresses
    void clear();

    /// Add DNS seed addresses
    void add_seed(const std::string& hostname, uint16_t port = DEFAULT_PORT);

    /// Get addresses for seeding (hardcoded seeds for the active network)
    static std::vector<CNetAddr> get_default_seeds();

    /// Get addresses for seeding a specific network type.
    /// Returns hardcoded seed node IPs for mainnet/testnet; empty for regtest.
    static std::vector<CNetAddr> get_default_seeds(const std::string& network);

private:
    mutable core::Mutex mutex_;

    /// All known addresses, keyed by address string
    std::unordered_map<std::string, AddrInfo> addrs_;

    /// Make a key from an address
    static std::string make_key(const CNetAddr& addr);
};

}  // namespace rnet::net

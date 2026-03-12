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

/// BanEntry — information about a banned address
struct BanEntry {
    std::string address;        ///< Address string (IP:port or subnet)
    int64_t ban_time = 0;       ///< When the ban was created
    int64_t ban_until = 0;      ///< When the ban expires (0 = permanent)
    std::string reason;         ///< Reason for the ban

    bool is_expired(int64_t now) const {
        return ban_until > 0 && ban_until < now;
    }
};

/// BanManager — manages the ban list for P2P peers.
/// Banned addresses are prevented from connecting.
class BanManager {
public:
    /// Default ban duration: 24 hours
    static constexpr int64_t DEFAULT_BAN_TIME = 24 * 60 * 60;

    BanManager() = default;
    ~BanManager() = default;

    // Non-copyable
    BanManager(const BanManager&) = delete;
    BanManager& operator=(const BanManager&) = delete;

    /// Ban an address for a duration (in seconds)
    void ban(const CNetAddr& addr, const std::string& reason,
             int64_t duration = DEFAULT_BAN_TIME);

    /// Ban an address string (IP or subnet)
    void ban(const std::string& addr_str, const std::string& reason,
             int64_t duration = DEFAULT_BAN_TIME);

    /// Unban an address
    bool unban(const CNetAddr& addr);
    bool unban(const std::string& addr_str);

    /// Check if an address is banned
    bool is_banned(const CNetAddr& addr) const;
    bool is_banned(const std::string& addr_str) const;

    /// Get all current bans
    std::vector<BanEntry> get_bans() const;

    /// Remove expired bans
    void sweep_expired();

    /// Clear all bans
    void clear();

    /// Save ban list to file
    Result<void> save(const std::string& path) const;

    /// Load ban list from file
    Result<void> load(const std::string& path);

    /// Number of active bans
    size_t size() const;

private:
    mutable core::Mutex mutex_;
    std::unordered_map<std::string, BanEntry> bans_;

    static std::string make_key(const CNetAddr& addr);
};

}  // namespace rnet::net

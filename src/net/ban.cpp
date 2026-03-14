// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "net/ban.h"

#include "core/logging.h"
#include "core/time.h"

namespace rnet::net {

// ===========================================================================
//  Banning
// ===========================================================================

// ---------------------------------------------------------------------------
// ban (CNetAddr)
// ---------------------------------------------------------------------------

void BanManager::ban(const CNetAddr& addr, const std::string& reason,
                     int64_t duration) {
    ban(make_key(addr), reason, duration);
}

// ---------------------------------------------------------------------------
// ban (string)
//
// Design: creates a BanEntry with the current time and an optional
// duration.  A duration of 0 means permanent ban.
// ---------------------------------------------------------------------------

void BanManager::ban(const std::string& addr_str,
                     const std::string& reason, int64_t duration) {
    LOCK(mutex_);

    BanEntry entry;
    entry.address = addr_str;
    entry.ban_time = core::get_time();
    entry.ban_until = (duration > 0) ? entry.ban_time + duration : 0;
    entry.reason = reason;

    bans_[addr_str] = entry;

    LogPrint(NET, "Banned %s: %s (duration=%llds)",
             addr_str.c_str(), reason.c_str(),
             static_cast<long long>(duration));
}

// ===========================================================================
//  Unbanning
// ===========================================================================

// ---------------------------------------------------------------------------
// unban (CNetAddr)
// ---------------------------------------------------------------------------

bool BanManager::unban(const CNetAddr& addr) {
    return unban(make_key(addr));
}

// ---------------------------------------------------------------------------
// unban (string)
// ---------------------------------------------------------------------------

bool BanManager::unban(const std::string& addr_str) {
    LOCK(mutex_);
    auto it = bans_.find(addr_str);
    if (it == bans_.end()) return false;
    bans_.erase(it);
    LogPrint(NET, "Unbanned %s", addr_str.c_str());
    return true;
}

// ===========================================================================
//  Queries
// ===========================================================================

// ---------------------------------------------------------------------------
// is_banned (CNetAddr)
// ---------------------------------------------------------------------------

bool BanManager::is_banned(const CNetAddr& addr) const {
    return is_banned(make_key(addr));
}

// ---------------------------------------------------------------------------
// is_banned (string)
//
// Design: performs lazy expiry -- removes the entry from the map if it
// has expired rather than returning a stale positive.
// ---------------------------------------------------------------------------

bool BanManager::is_banned(const std::string& addr_str) const {
    LOCK(mutex_);
    auto it = bans_.find(addr_str);
    if (it == bans_.end()) return false;

    int64_t now = core::get_time();
    if (it->second.is_expired(now)) {
        // 1. Lazy expiry cleanup
        const_cast<BanManager*>(this)->bans_.erase(addr_str);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// get_bans
//
// Design: returns only non-expired entries.
// ---------------------------------------------------------------------------

std::vector<BanEntry> BanManager::get_bans() const {
    LOCK(mutex_);
    std::vector<BanEntry> result;
    result.reserve(bans_.size());
    int64_t now = core::get_time();
    for (const auto& [key, entry] : bans_) {
        if (!entry.is_expired(now)) {
            result.push_back(entry);
        }
    }
    return result;
}

// ===========================================================================
//  Housekeeping
// ===========================================================================

// ---------------------------------------------------------------------------
// sweep_expired
//
// Design: eagerly removes all expired ban entries in a single pass.
// ---------------------------------------------------------------------------

void BanManager::sweep_expired() {
    LOCK(mutex_);
    int64_t now = core::get_time();
    for (auto it = bans_.begin(); it != bans_.end(); ) {
        if (it->second.is_expired(now)) {
            it = bans_.erase(it);
        } else {
            ++it;
        }
    }
}

// ---------------------------------------------------------------------------
// clear
// ---------------------------------------------------------------------------

void BanManager::clear() {
    LOCK(mutex_);
    bans_.clear();
}

// ===========================================================================
//  Persistence (stubs)
// ===========================================================================

// ---------------------------------------------------------------------------
// save
// ---------------------------------------------------------------------------

Result<void> BanManager::save(const std::string& /*path*/) const {
    // Stub: would serialize ban list to JSON
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------

Result<void> BanManager::load(const std::string& /*path*/) {
    // Stub: would deserialize ban list from JSON
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// size
// ---------------------------------------------------------------------------

size_t BanManager::size() const {
    LOCK(mutex_);
    return bans_.size();
}

// ===========================================================================
//  Internal helpers
// ===========================================================================

// ---------------------------------------------------------------------------
// make_key
// ---------------------------------------------------------------------------

std::string BanManager::make_key(const CNetAddr& addr) {
    return addr.to_string();
}

} // namespace rnet::net

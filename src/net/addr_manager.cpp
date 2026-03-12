#include "net/addr_manager.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>

#include "core/logging.h"
#include "core/random.h"
#include "core/serialize.h"
#include "core/stream.h"
#include "core/time.h"
#include "crypto/keccak.h"

namespace rnet::net {

// ---------------------------------------------------------------------------
// AddrInfo
// ---------------------------------------------------------------------------

double AddrInfo::get_chance() const {
    double chance = 1.0;

    int64_t now = core::get_time();
    int64_t since_last = now - last_success;

    // Reduce chance for addresses not seen recently
    if (since_last < 0) since_last = 0;
    if (since_last > 60 * 60 * 24) {
        // More than a day old: exponential decay
        chance *= std::pow(0.66, static_cast<double>(since_last) /
                                 (60.0 * 60.0 * 24.0));
    }

    // Boost tried addresses
    if (is_tried) {
        chance *= 2.0;
    }

    // Boost addresses with successful connections
    if (success_count > 0) {
        chance *= (1.0 + std::log(static_cast<double>(success_count)));
    }

    // Penalize addresses with many failed attempts
    if (attempt_count > success_count + 3) {
        int failures = attempt_count - success_count;
        chance *= std::pow(0.7, failures);
    }

    return (std::max)(chance, 0.01);
}

// ---------------------------------------------------------------------------
// AddrManager — Construction
// ---------------------------------------------------------------------------

AddrManager::AddrManager() {
    // Generate a random nonce for bucket hashing
    nonce_ = core::get_rand_hash();
}

AddrManager::~AddrManager() = default;

// ---------------------------------------------------------------------------
// Core operations
// ---------------------------------------------------------------------------

bool AddrManager::add(const CNetAddr& addr, const CNetAddr& source) {
    LOCK(cs_addr_);

    // Don't add local or unroutable addresses
    if (addr.is_local()) return false;
    if (addr.port == 0) return false;

    std::string key = addr.to_string();

    // Check if already known
    auto it = addr_map_.find(key);
    if (it != addr_map_.end()) {
        // Update last seen time
        it->second.addr.time = core::get_time();
        return false;
    }

    // Create new entry
    AddrInfo info;
    info.addr = addr;
    info.addr.time = core::get_time();
    info.source = source;
    info.first_seen = core::get_time();
    info.is_tried = false;
    info.bucket = new_bucket(addr, source);

    addr_map_[key] = info;
    ++new_count_;

    LogDebug(NET, "Added address %s (source=%s, new_count=%d)",
             key.c_str(), source.to_string().c_str(), new_count_);

    return true;
}

int AddrManager::add(const std::vector<CNetAddr>& addrs,
                     const CNetAddr& source) {
    int added = 0;
    for (const auto& addr : addrs) {
        if (add(addr, source)) {
            ++added;
        }
    }
    return added;
}

void AddrManager::mark_good(const CNetAddr& addr) {
    LOCK(cs_addr_);

    std::string key = addr.to_string();
    auto it = addr_map_.find(key);
    if (it == addr_map_.end()) return;

    auto& info = it->second;
    info.last_success = core::get_time();
    ++info.success_count;

    // Move from new to tried
    if (!info.is_tried) {
        info.is_tried = true;
        info.bucket = tried_bucket(addr);
        --new_count_;
        ++tried_count_;

        LogDebug(NET, "Promoted %s to tried table", key.c_str());
    }
}

void AddrManager::mark_failed(const CNetAddr& addr) {
    LOCK(cs_addr_);

    std::string key = addr.to_string();
    auto it = addr_map_.find(key);
    if (it == addr_map_.end()) return;

    ++it->second.attempt_count;
    it->second.last_attempt = core::get_time();
}

void AddrManager::mark_attempt(const CNetAddr& addr) {
    LOCK(cs_addr_);

    std::string key = addr.to_string();
    auto it = addr_map_.find(key);
    if (it == addr_map_.end()) return;

    ++it->second.attempt_count;
    it->second.last_attempt = core::get_time();
}

CNetAddr AddrManager::select() {
    LOCK(cs_addr_);

    if (addr_map_.empty()) return CNetAddr{};

    // Remove stale entries periodically
    remove_stale();

    if (addr_map_.empty()) return CNetAddr{};

    // Build weighted selection list
    std::vector<std::pair<std::string, double>> candidates;
    candidates.reserve(addr_map_.size());

    for (const auto& [key, info] : addr_map_) {
        double chance = info.get_chance();
        candidates.push_back({key, chance});
    }

    // Weighted random selection
    double total_weight = 0.0;
    for (const auto& [key, weight] : candidates) {
        total_weight += weight;
    }

    if (total_weight <= 0.0) {
        // Fallback: pick a random address
        auto idx = core::get_rand_range(addr_map_.size());
        auto it = addr_map_.begin();
        std::advance(it, static_cast<ptrdiff_t>(idx));
        return it->second.addr;
    }

    double target = static_cast<double>(core::get_rand_range(1000000)) /
                    1000000.0 * total_weight;
    double cumulative = 0.0;

    for (const auto& [key, weight] : candidates) {
        cumulative += weight;
        if (cumulative >= target) {
            auto it = addr_map_.find(key);
            if (it != addr_map_.end()) {
                return it->second.addr;
            }
        }
    }

    // Shouldn't reach here, but return first address as fallback
    return addr_map_.begin()->second.addr;
}

std::vector<CNetAddr> AddrManager::get_addr(int max_count) const {
    LOCK(cs_addr_);

    if (addr_map_.empty()) return {};

    int count = static_cast<int>(addr_map_.size());
    int num_return = count * GETADDR_PERCENT / 100;

    if (max_count > 0) {
        num_return = (std::min)(num_return, max_count);
    }
    num_return = (std::min)(num_return, MAX_GETADDR_RETURN);
    num_return = (std::max)(num_return, 1);

    // Collect all addresses
    std::vector<CNetAddr> all_addrs;
    all_addrs.reserve(addr_map_.size());
    for (const auto& [key, info] : addr_map_) {
        all_addrs.push_back(info.addr);
    }

    // Shuffle and take first N
    // (Fisher-Yates on the subset we'll return)
    for (int i = static_cast<int>(all_addrs.size()) - 1;
         i > 0 && i >= static_cast<int>(all_addrs.size()) - num_return;
         --i) {
        int j = static_cast<int>(core::get_rand_range(
            static_cast<uint64_t>(i + 1)));
        std::swap(all_addrs[static_cast<size_t>(i)],
                  all_addrs[static_cast<size_t>(j)]);
    }

    // Return the last num_return elements
    int start = static_cast<int>(all_addrs.size()) - num_return;
    if (start < 0) start = 0;

    return std::vector<CNetAddr>(
        all_addrs.begin() + start,
        all_addrs.end());
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

size_t AddrManager::size() const {
    LOCK(cs_addr_);
    return addr_map_.size();
}

size_t AddrManager::new_count() const {
    LOCK(cs_addr_);
    return static_cast<size_t>(new_count_);
}

size_t AddrManager::tried_count() const {
    LOCK(cs_addr_);
    return static_cast<size_t>(tried_count_);
}

bool AddrManager::contains(const CNetAddr& addr) const {
    LOCK(cs_addr_);
    return addr_map_.count(addr.to_string()) > 0;
}

const AddrInfo* AddrManager::find(const CNetAddr& addr) const {
    LOCK(cs_addr_);
    auto it = addr_map_.find(addr.to_string());
    if (it != addr_map_.end()) return &it->second;
    return nullptr;
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

bool AddrManager::save(const std::string& path) const {
    LOCK(cs_addr_);

    try {
        core::DataStream ss;

        // Write version
        core::ser_write_u32(ss, 1);  // format version

        // Write nonce
        nonce_.serialize(ss);

        // Write address count
        core::serialize_compact_size(ss, addr_map_.size());

        // Write each address
        for (const auto& [key, info] : addr_map_) {
            info.addr.serialize(ss);
            info.source.serialize(ss);
            core::ser_write_i32(ss, info.success_count);
            core::ser_write_i32(ss, info.attempt_count);
            core::ser_write_i64(ss, info.last_success);
            core::ser_write_i64(ss, info.last_attempt);
            core::ser_write_i64(ss, info.first_seen);
            core::ser_write_bool(ss, info.is_tried);
        }

        // Write to file
        std::ofstream file(path, std::ios::binary);
        if (!file.is_open()) {
            LogError("Failed to open %s for writing", path.c_str());
            return false;
        }

        file.write(reinterpret_cast<const char*>(ss.data()),
                   static_cast<std::streamsize>(ss.size()));
        file.close();

        LogPrint(NET, "Saved %zu addresses to %s",
                 addr_map_.size(), path.c_str());
        return true;
    } catch (const std::exception& e) {
        LogError("Failed to save addresses: %s", e.what());
        return false;
    }
}

bool AddrManager::load(const std::string& path) {
    LOCK(cs_addr_);

    try {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            LogDebug(NET, "No address file found at %s", path.c_str());
            return false;
        }

        auto file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<uint8_t> data(static_cast<size_t>(file_size));
        file.read(reinterpret_cast<char*>(data.data()),
                  static_cast<std::streamsize>(file_size));
        file.close();

        core::DataStream ss(std::move(data));

        // Read version
        uint32_t version = core::ser_read_u32(ss);
        if (version != 1) {
            LogError("Unknown address file version: %u", version);
            return false;
        }

        // Read nonce
        nonce_.unserialize(ss);

        // Read address count
        uint64_t count = core::unserialize_compact_size(ss);

        addr_map_.clear();
        new_count_ = 0;
        tried_count_ = 0;

        for (uint64_t i = 0; i < count; ++i) {
            AddrInfo info;
            info.addr.unserialize(ss);
            info.source.unserialize(ss);
            info.success_count = core::ser_read_i32(ss);
            info.attempt_count = core::ser_read_i32(ss);
            info.last_success = core::ser_read_i64(ss);
            info.last_attempt = core::ser_read_i64(ss);
            info.first_seen = core::ser_read_i64(ss);
            info.is_tried = core::ser_read_bool(ss);

            // Recompute bucket
            if (info.is_tried) {
                info.bucket = tried_bucket(info.addr);
                ++tried_count_;
            } else {
                info.bucket = new_bucket(info.addr, info.source);
                ++new_count_;
            }

            addr_map_[info.key()] = info;
        }

        LogPrintf("Loaded %zu addresses from %s (new=%d, tried=%d)",
                  addr_map_.size(), path.c_str(),
                  new_count_, tried_count_);
        return true;
    } catch (const std::exception& e) {
        LogError("Failed to load addresses: %s", e.what());
        return false;
    }
}

void AddrManager::clear() {
    LOCK(cs_addr_);
    addr_map_.clear();
    new_count_ = 0;
    tried_count_ = 0;
}

// ---------------------------------------------------------------------------
// Seed addresses
// ---------------------------------------------------------------------------

void AddrManager::add_seed(const std::string& hostname, uint16_t port) {
    // DNS resolution would happen here in a full implementation.
    // For now, just log it.
    LogPrint(NET, "DNS seed: %s:%d (resolution not implemented)",
             hostname.c_str(), static_cast<int>(port));
}

void AddrManager::add_seeds(const std::vector<std::string>& ips,
                            uint16_t port) {
    CNetAddr source;
    source.set_ipv4(0, 0, 0, 0);  // Local source

    for (const auto& ip_str : ips) {
        CNetAddr addr;
        addr.port = port;

        // Parse IPv4 address
        unsigned int a = 0, b = 0, c = 0, d = 0;
        if (std::sscanf(ip_str.c_str(), "%u.%u.%u.%u",
                        &a, &b, &c, &d) == 4) {
            addr.set_ipv4(static_cast<uint8_t>(a),
                          static_cast<uint8_t>(b),
                          static_cast<uint8_t>(c),
                          static_cast<uint8_t>(d));
            add(addr, source);
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

int AddrManager::new_bucket(const CNetAddr& addr,
                            const CNetAddr& source) const {
    // Hash: H(nonce || addr_group || source_group) mod NEW_BUCKET_COUNT
    auto addr_group = get_group(addr);
    auto source_group = get_group(source);

    core::DataStream ss;
    nonce_.serialize(ss);
    ss.write(addr_group.data(), addr_group.size());
    ss.write(source_group.data(), source_group.size());

    auto hash = crypto::keccak256(ss.span());
    uint32_t bucket_val = 0;
    std::memcpy(&bucket_val, hash.data(), 4);

    return static_cast<int>(bucket_val % NEW_BUCKET_COUNT);
}

int AddrManager::tried_bucket(const CNetAddr& addr) const {
    // Hash: H(nonce || addr_group) mod TRIED_BUCKET_COUNT
    auto addr_group = get_group(addr);

    core::DataStream ss;
    nonce_.serialize(ss);
    ss.write(addr_group.data(), addr_group.size());

    auto hash = crypto::keccak256(ss.span());
    uint32_t bucket_val = 0;
    std::memcpy(&bucket_val, hash.data(), 4);

    return static_cast<int>(bucket_val % TRIED_BUCKET_COUNT);
}

std::vector<uint8_t> AddrManager::get_group(const CNetAddr& addr) {
    // Group by /16 subnet for IPv4 (first two octets)
    // This prevents a single subnet from dominating the address tables
    std::vector<uint8_t> group;

    if (addr.is_ipv4()) {
        group.push_back(1);  // IPv4 marker
        group.push_back(addr.ip[12]);
        group.push_back(addr.ip[13]);
    } else {
        group.push_back(2);  // IPv6 marker
        // Use first 4 bytes of IPv6 address as group
        group.push_back(addr.ip[0]);
        group.push_back(addr.ip[1]);
        group.push_back(addr.ip[2]);
        group.push_back(addr.ip[3]);
    }

    return group;
}

void AddrManager::remove_stale() {
    // cs_addr_ must be held

    int64_t now = core::get_time();
    std::vector<std::string> to_remove;

    for (const auto& [key, info] : addr_map_) {
        // Remove addresses older than MAX_ADDR_AGE that have never
        // successfully connected
        if (info.success_count == 0 &&
            (now - info.first_seen) > MAX_ADDR_AGE) {
            to_remove.push_back(key);
        }
    }

    for (const auto& key : to_remove) {
        auto it = addr_map_.find(key);
        if (it != addr_map_.end()) {
            if (it->second.is_tried) {
                --tried_count_;
            } else {
                --new_count_;
            }
            addr_map_.erase(it);
        }
    }

    if (!to_remove.empty()) {
        LogDebug(NET, "Removed %zu stale addresses", to_remove.size());
    }
}

}  // namespace rnet::net

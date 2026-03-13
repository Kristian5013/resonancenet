#include "net/addr_man.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "core/random.h"
#include "core/time.h"

namespace rnet::net {

bool AddrInfo::is_terrible(int64_t now) const {
    // Not tried in over 30 days and never succeeded
    if (last_success == 0 && attempts >= 3 &&
        now - last_try > 30 * 24 * 60 * 60) {
        return true;
    }

    // Too many failed attempts
    if (attempts >= 10 && last_success == 0) {
        return true;
    }

    return false;
}

double AddrInfo::get_chance(int64_t now) const {
    double chance = 1.0;

    int64_t since_last_try = now - last_try;
    if (since_last_try < 0) since_last_try = 0;

    // Lower chance for recently tried addresses
    if (since_last_try < 60 * 10) {
        chance *= 0.01;
    }

    // Lower chance for many attempts
    chance *= std::pow(0.66, std::min(attempts, 8));

    // Higher chance for addresses that have worked before
    if (last_success > 0) {
        chance *= 2.0;
    }

    return chance;
}

AddrManager::AddrManager() = default;

void AddrManager::add(const std::vector<CNetAddr>& addrs,
                      const std::string& source) {
    LOCK(mutex_);
    for (const auto& addr : addrs) {
        if (addrs_.size() >= MAX_ADDRESSES) break;

        auto key = make_key(addr);
        if (addrs_.count(key)) continue;

        AddrInfo info;
        info.addr = addr;
        info.source = source;
        addrs_[key] = info;
    }
}

void AddrManager::add(const CNetAddr& addr, const std::string& source) {
    add(std::vector<CNetAddr>{addr}, source);
}

void AddrManager::mark_good(const CNetAddr& addr, int64_t time) {
    LOCK(mutex_);
    auto key = make_key(addr);
    auto it = addrs_.find(key);
    if (it != addrs_.end()) {
        it->second.last_success = time;
        it->second.last_try = time;
    }
}

void AddrManager::mark_attempt(const CNetAddr& addr) {
    LOCK(mutex_);
    auto key = make_key(addr);
    auto it = addrs_.find(key);
    if (it != addrs_.end()) {
        it->second.attempts++;
        it->second.last_try = core::get_time();
    }
}

CNetAddr AddrManager::select() const {
    LOCK(mutex_);

    if (addrs_.empty()) {
        return CNetAddr{};
    }

    int64_t now = core::get_time();

    // Weighted random selection
    std::vector<std::pair<std::string, double>> candidates;
    for (const auto& [key, info] : addrs_) {
        if (info.is_terrible(now)) continue;
        double chance = info.get_chance(now);
        if (chance > 0) {
            candidates.push_back({key, chance});
        }
    }

    if (candidates.empty()) {
        // Fall back to random selection
        auto rand_idx = core::get_rand_range(addrs_.size());
        auto it = addrs_.begin();
        std::advance(it, static_cast<ptrdiff_t>(rand_idx));
        return it->second.addr;
    }

    // Weighted selection
    double total_weight = 0;
    for (const auto& [key, weight] : candidates) {
        total_weight += weight;
    }

    double roll = static_cast<double>(core::get_rand_range(1000000)) /
                  1000000.0 * total_weight;
    double cumulative = 0;
    for (const auto& [key, weight] : candidates) {
        cumulative += weight;
        if (roll <= cumulative) {
            return addrs_.at(key).addr;
        }
    }

    return candidates.back().first.empty()
        ? CNetAddr{} : addrs_.at(candidates.back().first).addr;
}

std::vector<CNetAddr> AddrManager::get_addr(size_t max_count) const {
    LOCK(mutex_);

    std::vector<CNetAddr> result;
    result.reserve(std::min(max_count, addrs_.size()));

    for (const auto& [key, info] : addrs_) {
        if (result.size() >= max_count) break;
        if (info.last_success > 0 || info.attempts < 3) {
            result.push_back(info.addr);
        }
    }

    return result;
}

size_t AddrManager::size() const {
    LOCK(mutex_);
    return addrs_.size();
}

bool AddrManager::contains(const CNetAddr& addr) const {
    LOCK(mutex_);
    return addrs_.count(make_key(addr)) > 0;
}

Result<void> AddrManager::save(const std::string& /*path*/) const {
    // Stub: serialize to disk
    return Result<void>::ok();
}

Result<void> AddrManager::load(const std::string& /*path*/) {
    // Stub: deserialize from disk
    return Result<void>::ok();
}

void AddrManager::clear() {
    LOCK(mutex_);
    addrs_.clear();
}

void AddrManager::add_seed(const std::string& /*hostname*/, uint16_t /*port*/) {
    // Stub: DNS resolution would happen here
}

std::vector<CNetAddr> AddrManager::get_default_seeds() {
    // Delegate to mainnet seeds by default
    return get_default_seeds("mainnet");
}

std::vector<CNetAddr> AddrManager::get_default_seeds(
    const std::string& network)
{
    std::vector<std::string> seed_ips;

    if (network == "mainnet") {
        // Add mainnet seed IPs here
        // seed_ips.push_back("198.51.100.1");
    } else if (network == "testnet") {
        seed_ips.push_back("188.137.227.180");
    } else {
        // regtest — no seeds needed (local testing only)
        return {};
    }

    // Determine the default port for this network
    uint16_t port = DEFAULT_PORT;
    if (network == "testnet") {
        port = 19555;
    }

    std::vector<CNetAddr> result;
    result.reserve(seed_ips.size());
    for (const auto& ip_str : seed_ips) {
        CNetAddr addr;
        unsigned int a = 0, b = 0, c = 0, d = 0;
        if (std::sscanf(ip_str.c_str(), "%u.%u.%u.%u",
                        &a, &b, &c, &d) == 4) {
            addr.set_ipv4(static_cast<uint8_t>(a),
                          static_cast<uint8_t>(b),
                          static_cast<uint8_t>(c),
                          static_cast<uint8_t>(d));
            addr.port = port;
            result.push_back(addr);
        }
    }
    return result;
}

std::string AddrManager::make_key(const CNetAddr& addr) {
    return addr.to_string();
}

}  // namespace rnet::net

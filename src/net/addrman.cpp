#include "net/addrman.h"
#include "util/logging.h"
#include "util/fs.h"
#include <format>
#include <cstring>

#include <algorithm>
#include <limits>

#include "canon/canon.h"

namespace rnet::net {

namespace {

uint64_t Be64(const crypto::Hash256& h) {
    uint64_t v = 0;
    for (size_t i = 0; i < 8; ++i) v = (v << 8) | h[i];
    return v;
}

}  // namespace

std::array<uint8_t, 8> NetworkGroup(const NetAddress& address) {
    std::array<uint8_t, 8> group{};
    if (address.IsIPv4()) {
        // /16 — the granularity at which addresses are actually bought and routed.
        group[0] = 4;
        group[1] = address.ip[12];
        group[2] = address.ip[13];
        return group;
    }
    group[0] = 6;
    // /32 for IPv6: a single allocation is typically far larger than a /64, so
    // grouping any finer would let one allocation look like many networks.
    for (size_t i = 0; i < 4; ++i) group[1 + i] = address.ip[i];
    return group;
}

AddressManager::AddressManager(crypto::Hash256 key) : key_(key) {}

size_t AddressManager::NewBucketFor(const NetAddress& address, const NetAddress& source) const {
    // Bucket by the SOURCE's group as well as the address's: an attacker gossiping
    // from one network is confined to a slice of the table no matter how many
    // distinct addresses it invents.
    const auto addr_group = NetworkGroup(address);
    const auto source_group = NetworkGroup(source);

    canon::Writer w;
    w.Hash(key_);
    w.Bytes(std::span<const uint8_t>(source_group.data(), source_group.size()));
    w.Bytes(std::span<const uint8_t>(addr_group.data(), addr_group.size()));
    return static_cast<size_t>(Be64(crypto::Sha3_256(w.data())) % kNewBuckets);
}

size_t AddressManager::TriedBucketFor(const NetAddress& address) const {
    const auto group = NetworkGroup(address);
    canon::Writer w;
    w.Hash(key_);
    w.U8(0x01);   // domain-separated from the new-table derivation
    w.Bytes(std::span<const uint8_t>(group.data(), group.size()));
    return static_cast<size_t>(Be64(crypto::Sha3_256(w.data())) % kTriedBuckets);
}

void AddressManager::EvictFrom(std::vector<NetAddress>& bucket) {
    if (bucket.empty()) return;
    // Evict the least recently seen: deterministic, and it favours addresses that
    // are still being gossiped by the network over stale ones.
    auto victim = bucket.begin();
    int64_t oldest = std::numeric_limits<int64_t>::max();
    for (auto it = bucket.begin(); it != bucket.end(); ++it) {
        const auto entry = entries_.find(*it);
        if (entry == entries_.end()) { victim = it; break; }
        if (entry->second.last_seen < oldest) {
            oldest = entry->second.last_seen;
            victim = it;
        }
    }
    entries_.erase(*victim);
    bucket.erase(victim);
}

bool AddressManager::Add(const NetAddress& address, const NetAddress& source, int64_t now) {
    // An unroutable address can never be connected to; storing it would only
    // waste a slot and, later, a connection attempt.
    if (!address.IsRoutable()) return false;

    const auto existing = entries_.find(address);
    if (existing != entries_.end()) {
        existing->second.last_seen = std::max(existing->second.last_seen, now);
        return false;
    }

    const size_t bucket = NewBucketFor(address, source);
    if (new_buckets_[bucket].size() >= kEntriesPerBucket) EvictFrom(new_buckets_[bucket]);

    AddrInfo info;
    info.address = address;
    info.source = source;
    info.last_seen = now;
    entries_[address] = info;
    new_buckets_[bucket].push_back(address);
    return true;
}

size_t AddressManager::AddMany(const std::vector<NetAddress>& addresses, const NetAddress& source,
                               int64_t now) {
    size_t added = 0;
    for (const auto& address : addresses) {
        if (Add(address, source, now)) ++added;
    }
    return added;
}

void AddressManager::MarkGood(const NetAddress& address, int64_t now) {
    const auto it = entries_.find(address);
    if (it == entries_.end()) return;

    it->second.last_success = now;
    it->second.last_seen = now;
    it->second.last_attempt = now;
    it->second.attempts = 0;
    if (it->second.tried) return;

    // Promotion out of `new`: a connection actually worked, so this address has
    // earned a place gossip alone cannot buy.
    for (auto& bucket : new_buckets_) {
        const auto found = std::find(bucket.begin(), bucket.end(), address);
        if (found != bucket.end()) { bucket.erase(found); break; }
    }

    const size_t bucket = TriedBucketFor(address);
    if (tried_buckets_[bucket].size() >= kEntriesPerBucket) EvictFrom(tried_buckets_[bucket]);
    tried_buckets_[bucket].push_back(address);
    it->second.tried = true;
}

void AddressManager::MarkFailed(const NetAddress& address, int64_t now) {
    const auto it = entries_.find(address);
    if (it == entries_.end()) return;

    it->second.last_attempt = now;
    ++it->second.attempts;
    if (it->second.attempts < kMaxFailures) return;

    // Repeatedly unreachable: release the slot rather than retrying forever.
    // The two tables have different bucket counts, so they are searched
    // separately rather than through a common reference.
    auto drop_from = [&](auto& buckets) {
        for (auto& bucket : buckets) {
            const auto found = std::find(bucket.begin(), bucket.end(), address);
            if (found != bucket.end()) { bucket.erase(found); return true; }
        }
        return false;
    };
    if (it->second.tried) {
        drop_from(tried_buckets_);
    } else {
        drop_from(new_buckets_);
    }
    entries_.erase(it);
}

void AddressManager::Forget(const NetAddress& address) {
    const auto it = entries_.find(address);
    if (it == entries_.end()) return;

    auto drop_from = [&](auto& buckets) {
        for (auto& bucket : buckets) {
            const auto found = std::find(bucket.begin(), bucket.end(), address);
            if (found != bucket.end()) { bucket.erase(found); return true; }
        }
        return false;
    };
    if (it->second.tried) {
        drop_from(tried_buckets_);
    } else {
        drop_from(new_buckets_);
    }
    entries_.erase(it);
}

std::optional<AddrInfo> AddressManager::Select(uint64_t randomness, uint32_t bias_tried) const {
    const bool prefer_tried = (randomness % 100) < bias_tried;

    // Two passes: the preferred table, then the other one, so a node with only
    // new addresses still makes progress instead of returning nothing.
    for (int pass = 0; pass < 2; ++pass) {
        const bool use_tried = (pass == 0) ? prefer_tried : !prefer_tried;

        std::vector<NetAddress> candidates;
        if (use_tried) {
            for (const auto& bucket : tried_buckets_) {
                candidates.insert(candidates.end(), bucket.begin(), bucket.end());
            }
        } else {
            for (const auto& bucket : new_buckets_) {
                candidates.insert(candidates.end(), bucket.begin(), bucket.end());
            }
        }
        if (candidates.empty()) continue;

        const auto& chosen = candidates[(randomness / 100) % candidates.size()];
        const auto it = entries_.find(chosen);
        if (it != entries_.end()) return it->second;
    }
    return std::nullopt;
}

std::vector<NetAddress> AddressManager::GetAddresses(size_t max_count, uint64_t randomness) const {
    std::vector<NetAddress> all;
    all.reserve(entries_.size());
    for (const auto& [address, info] : entries_) all.push_back(address);

    if (all.size() <= max_count) return all;

    // Deterministic rotation rather than a sort: returning the same prefix every
    // time would make a node's view trivially fingerprintable.
    const size_t start = static_cast<size_t>(randomness % all.size());
    std::vector<NetAddress> out;
    out.reserve(max_count);
    for (size_t i = 0; i < max_count; ++i) out.push_back(all[(start + i) % all.size()]);
    return out;
}

size_t AddressManager::TriedCount() const {
    size_t count = 0;
    for (const auto& [address, info] : entries_) {
        if (info.tried) ++count;
    }
    return count;
}

bool AddressManager::Contains(const NetAddress& address) const {
    return entries_.find(address) != entries_.end();
}

namespace {

// Distinct from the wire magic: a peers file fed to a node as a message, or the
// reverse, must fail immediately rather than half-parse.
constexpr uint8_t kPeersMagic[4] = {'R', 'N', 'P', 'D'};
constexpr uint32_t kPeersVersion = 1;

// A file larger than this is not a peers file. Bounded before allocation, not
// after: the length prefix is attacker-controlled if the file is.
constexpr uint32_t kMaxPersistedEntries = kNewBuckets * kEntriesPerBucket +
                                          kTriedBuckets * kEntriesPerBucket;

void WriteAddress(canon::Writer& w, const NetAddress& address) {
    w.Bytes(address.ip);
    w.U16(address.port);
    w.U64(address.services);
    w.U64(static_cast<uint64_t>(address.last_seen));
}

Result<NetAddress> ReadAddress(canon::Reader& r) {
    NetAddress address;
    auto ip = r.Bytes(16);
    if (!ip) return Err(ip.error());
    std::copy(ip.value().begin(), ip.value().end(), address.ip.begin());
    auto port = r.U16();
    if (!port) return Err(port.error());
    address.port = port.value();
    auto services = r.U64();
    if (!services) return Err(services.error());
    address.services = services.value();
    auto last_seen = r.U64();
    if (!last_seen) return Err(last_seen.error());
    address.last_seen = static_cast<int64_t>(last_seen.value());
    return address;
}

}  // namespace

Status AddressManager::Save(const std::string& path) const {
    canon::Writer body;
    body.U32(static_cast<uint32_t>(entries_.size()));
    for (const auto& [address, info] : entries_) {
        WriteAddress(body, address);
        WriteAddress(body, info.source);
        body.U64(static_cast<uint64_t>(info.last_success));
        body.U64(static_cast<uint64_t>(info.last_attempt));
        body.U32(info.attempts);
        body.U8(info.tried ? 1 : 0);
    }

    canon::Writer file;
    file.Bytes(std::span<const uint8_t>(kPeersMagic, 4));
    file.U32(kPeersVersion);
    file.VarBytes(body.data());
    file.Hash(crypto::Sha3_256(body.data()));

    // Written atomically: a node killed mid-save must find either the old file or
    // the new one, never a truncated one it would then refuse to load.
    return util::WriteFileAtomic(path, file.data());
}

Status AddressManager::Load(const std::string& path) {
    auto raw = util::ReadFile(path);
    if (!raw) return Err(raw.error());

    canon::Reader r(raw.value());
    auto magic = r.Bytes(4);
    if (!magic) return Err(magic.error());
    if (std::memcmp(magic.value().data(), kPeersMagic, 4) != 0) {
        return Err("peers: not a peers file");
    }
    auto version = r.U32();
    if (!version) return Err(version.error());
    if (version.value() != kPeersVersion) {
        return Err(std::format("peers: unsupported version {}", version.value()));
    }
    auto body = r.VarBytes();
    if (!body) return Err(body.error());
    auto checksum = r.Hash();
    if (!checksum) return Err(checksum.error());
    if (auto st = r.ExpectExhausted(); !st) return st;
    if (crypto::Sha3_256(body.value()) != checksum.value()) {
        return Err("peers: checksum mismatch");
    }

    canon::Reader br(body.value());
    auto count = br.U32();
    if (!count) return Err(count.error());
    if (count.value() > kMaxPersistedEntries) {
        return Err(std::format("peers: {} entries exceeds the table capacity", count.value()));
    }

    // Rebuilt from scratch, and every entry re-bucketed with THIS node's key. A
    // stored bucket index would be an attacker's choice of placement; recomputing
    // it means a tampered file can make a node forget, but not aim it.
    entries_.clear();
    for (auto& bucket : new_buckets_) bucket.clear();
    for (auto& bucket : tried_buckets_) bucket.clear();

    size_t restored = 0;
    for (uint32_t i = 0; i < count.value(); ++i) {
        auto address = ReadAddress(br);
        if (!address) return Err(address.error());
        auto source = ReadAddress(br);
        if (!source) return Err(source.error());
        auto last_success = br.U64();
        if (!last_success) return Err(last_success.error());
        auto last_attempt = br.U64();
        if (!last_attempt) return Err(last_attempt.error());
        auto attempts = br.U32();
        if (!attempts) return Err(attempts.error());
        auto tried = br.U8();
        if (!tried) return Err(tried.error());

        // Goes in through the ordinary path, so an unroutable or duplicate entry
        // in the file is filtered exactly as it would be coming off the wire.
        if (!Add(address.value(), source.value(), address.value().last_seen)) continue;

        auto entry = entries_.find(address.value());
        if (entry == entries_.end()) continue;
        entry->second.last_success = static_cast<int64_t>(last_success.value());
        entry->second.last_attempt = static_cast<int64_t>(last_attempt.value());
        entry->second.attempts = attempts.value();
        // `tried` is restored only if this node had actually connected: the flag
        // means "we reached it ourselves", and a file cannot confer that.
        if (tried.value() != 0 && last_success.value() != 0) {
            MarkGood(address.value(), static_cast<int64_t>(last_success.value()));
        }
        ++restored;
    }
    if (auto st = br.ExpectExhausted(); !st) return st;

    RNET_LOG_INFO("peers: restored {} of {} addresses ({} tried)", restored, count.value(),
                  TriedCount());
    return Status::Ok();
}

}  // namespace rnet::net

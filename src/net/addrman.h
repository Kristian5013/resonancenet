// The address database.
//
// Its job is not storage — it is deciding who a node talks to, which makes it the
// target of the eclipse attack: surround a victim with peers you control and you
// choose what it sees. A node that only remembers "the last thousand addresses it
// heard" is trivially eclipsed by anyone willing to send a thousand addresses.
//
// Three defences, all structural:
//
//   * BUCKETING BY NETWORK GROUP. Addresses are placed by their /16 (IPv4) or /32
//     (IPv6) group, not individually. Ten thousand addresses from one hoster
//     occupy the space of one, so buying IPs stops buying influence.
//
//   * SEPARATE NEW AND TRIED TABLES. An address only reaches `tried` after this
//     node connected to it successfully. Gossip alone cannot displace a peer that
//     has actually worked.
//
//   * BOUNDED, DETERMINISTIC EVICTION. Every table has a hard cap and a defined
//     victim, so filling the database is a bounded, predictable operation rather
//     than an open door.
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <optional>
#include <vector>

#include "crypto/sha3.h"
#include "net/protocol.h"
#include "util/result.h"

namespace rnet::net {

inline constexpr size_t kNewBuckets = 256;
inline constexpr size_t kTriedBuckets = 64;
inline constexpr size_t kEntriesPerBucket = 32;

// After this many consecutive failures an address is dropped from `tried`: a peer
// that no longer answers should not hold a slot forever.
inline constexpr uint32_t kMaxFailures = 5;

struct AddrInfo {
    NetAddress address;
    NetAddress source;        // who told us — bucketing uses this for new entries
    int64_t last_seen{0};
    int64_t last_success{0};
    int64_t last_attempt{0};
    uint32_t attempts{0};
    bool tried{false};
};

// The group an address belongs to for bucketing purposes: /16 for IPv4, /32 for
// IPv6. Exposed because it is the heart of the eclipse defence and deserves a
// direct test.
std::array<uint8_t, 8> NetworkGroup(const NetAddress& address);

class AddressManager {
public:
    // `key` makes bucket placement unpredictable to an attacker: without it, an
    // adversary could compute which addresses land in which bucket and craft a
    // set that fills exactly the buckets a victim would select from.
    explicit AddressManager(crypto::Hash256 key);

    // Records an address heard from `source`. Unroutable addresses are ignored:
    // storing them would let a peer waste a node's connection slots.
    bool Add(const NetAddress& address, const NetAddress& source, int64_t now);
    size_t AddMany(const std::vector<NetAddress>& addresses, const NetAddress& source, int64_t now);

    // A connection succeeded: promote to `tried`.
    void MarkGood(const NetAddress& address, int64_t now);
    // A connection failed: count it, and drop the address once it is clearly dead.
    void MarkFailed(const NetAddress& address, int64_t now);

    // Picks an address to connect to. `bias_tried` is the percentage chance of
    // drawing from the tried table; the rest of the time a new address gets a
    // turn, so the node keeps discovering rather than talking to the same peers
    // forever.
    std::optional<AddrInfo> Select(uint64_t randomness, uint32_t bias_tried = 50) const;

    // Addresses to gossip. Caps the response so a `getaddr` cannot be used to
    // pull a node's entire view in one request.
    std::vector<NetAddress> GetAddresses(size_t max_count, uint64_t randomness) const;

    // Persistence. A node that forgets every address when it restarts has to
    // re-bootstrap from seeds every time — which hands an attacker a fresh eclipse
    // opportunity on every restart, and makes the seeds far more powerful than
    // they should be. What was learned by connecting is worth keeping.
    //
    // The file is not trusted on load: it is checksummed, and every entry is
    // re-bucketed with THIS node's key rather than restored to a stored index. A
    // tampered file can therefore make a node forget, but not make it place
    // addresses where an attacker wants them.
    Status Save(const std::string& path) const;
    Status Load(const std::string& path);

    size_t Size() const { return entries_.size(); }
    size_t TriedCount() const;
    size_t NewCount() const { return Size() - TriedCount(); }
    bool Contains(const NetAddress& address) const;

private:
    size_t NewBucketFor(const NetAddress& address, const NetAddress& source) const;
    size_t TriedBucketFor(const NetAddress& address) const;
    void EvictFrom(std::vector<NetAddress>& bucket);

    crypto::Hash256 key_;
    std::map<NetAddress, AddrInfo> entries_;
    std::array<std::vector<NetAddress>, kNewBuckets> new_buckets_;
    std::array<std::vector<NetAddress>, kTriedBuckets> tried_buckets_;
};

}  // namespace rnet::net

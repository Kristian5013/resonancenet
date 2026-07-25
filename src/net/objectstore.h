// What a node holds, what it wants, and who owes it.
//
// Object exchange follows announce → request → deliver, and every step is a place
// a peer can misbehave:
//
//   * ANNOUNCING is free. A peer can offer a million hashes it does not have, so
//     announcements are bounded and never cause storage on their own.
//
//   * REQUESTING must not be duplicated. Asking every peer for the same object
//     wastes the node's bandwidth in proportion to its peer count — which is a
//     way to make a well-connected node the easiest to exhaust.
//
//   * DELIVERING is what a peer owes after an announcement. A peer that announces
//     and never delivers stalls the node silently, so requests expire and are
//     retried elsewhere, and the peer that failed is remembered.
//
// Nothing enters storage unverified: an object is admitted only if its bytes hash
// to the identity it was requested under.
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <vector>

#include "crypto/sha3.h"
#include "net/protocol.h"
#include "util/result.h"

namespace rnet::net {

// How long a peer has to deliver what it announced before the request is retried
// elsewhere.
inline constexpr int64_t kRequestTimeoutMs = 60'000;

// Outstanding requests per peer. Unbounded, a peer could announce endlessly and
// make the node queue work in proportion.
inline constexpr size_t kMaxRequestsPerPeer = 128;

// Announcements remembered per peer. Bounds the memory one peer can occupy just
// by talking.
inline constexpr size_t kMaxAnnouncementsPerPeer = 5000;

struct StoredObject {
    InvType type{InvType::Contribution};
    crypto::Hash256 hash{};
    std::vector<uint8_t> data;
    int64_t stored_at{0};
};

struct PendingRequest {
    InvType type{InvType::Contribution};
    crypto::Hash256 hash{};
    uint64_t peer_id{0};
    int64_t requested_at{0};
    uint32_t attempts{0};
};

class ObjectStore {
public:
    explicit ObjectStore(size_t max_objects = 10000);

    // Admits an object only if its bytes hash to `hash`. A peer that delivers
    // something else is not merely unhelpful — it is trying to substitute data.
    Result<bool> Put(InvType type, const crypto::Hash256& hash, std::vector<uint8_t> data,
                     int64_t now_ms);

    bool Has(const crypto::Hash256& hash) const;

    // Borrows rather than copies. `Get` returns a copy and is fine for small
    // objects; for a contribution payload it would duplicate a gigabyte on every
    // request, so anything on a serving path uses this.
    const StoredObject* Peek(const crypto::Hash256& hash) const;
    Result<StoredObject> Get(const crypto::Hash256& hash) const;
    size_t Size() const { return objects_.size(); }

    // Records what a peer says it has. Returns the entries this node does not
    // already hold and has not already requested — the things actually worth
    // asking for.
    std::vector<InvEntry> NoteAnnouncement(uint64_t peer_id, const std::vector<InvEntry>& entries,
                                           int64_t now_ms);

    // Marks an object as requested from a peer. Refuses when that peer already
    // owes too much, or when the object is already being fetched from someone.
    Status NoteRequested(uint64_t peer_id, const InvEntry& entry, int64_t now_ms);

    // A delivery arrived: clears the request.
    void NoteDelivered(const crypto::Hash256& hash);

    // Requests whose deadline passed. The caller retries them with another peer
    // and scores the one that failed.
    std::vector<PendingRequest> ExpiredRequests(int64_t now_ms) const;
    void DropRequest(const crypto::Hash256& hash);

    // Peers that claimed to have this object, so a retry has somewhere to go.
    std::vector<uint64_t> PeersAnnouncing(const crypto::Hash256& hash) const;

    void ForgetPeer(uint64_t peer_id);

    size_t PendingCount() const { return requests_.size(); }
    size_t PendingFor(uint64_t peer_id) const;

private:
    void EvictOldest();

    size_t max_objects_;
    std::map<crypto::Hash256, StoredObject> objects_;
    std::map<crypto::Hash256, PendingRequest> requests_;
    std::map<uint64_t, std::set<crypto::Hash256>> announced_by_peer_;
    std::map<crypto::Hash256, std::set<uint64_t>> announcers_;
};

}  // namespace rnet::net

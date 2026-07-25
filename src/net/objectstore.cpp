#include "net/objectstore.h"

#include <algorithm>
#include <format>

#include "util/hex.h"

namespace rnet::net {

ObjectStore::ObjectStore(size_t max_objects) : max_objects_(max_objects == 0 ? 1 : max_objects) {}

Result<bool> ObjectStore::Put(InvType type, const crypto::Hash256& hash, std::vector<uint8_t> data,
                              int64_t now_ms) {
    if (data.empty()) return Err("store: refusing an empty object");

    // The identity is the hash of the bytes. A peer that delivers something else
    // is attempting substitution, not making a mistake.
    if (crypto::Sha3_256(data) != hash) {
        return Err("store: delivered bytes do not hash to the requested identity");
    }

    const auto existing = objects_.find(hash);
    if (existing != objects_.end()) {
        NoteDelivered(hash);
        return false;   // already held; not an error
    }

    if (objects_.size() >= max_objects_) EvictOldest();

    StoredObject object;
    object.type = type;
    object.hash = hash;
    object.data = std::move(data);
    object.stored_at = now_ms;
    objects_[hash] = std::move(object);
    NoteDelivered(hash);
    return true;
}

void ObjectStore::EvictOldest() {
    auto oldest = objects_.begin();
    for (auto it = objects_.begin(); it != objects_.end(); ++it) {
        if (it->second.stored_at < oldest->second.stored_at) oldest = it;
    }
    if (oldest != objects_.end()) objects_.erase(oldest);
}

bool ObjectStore::Has(const crypto::Hash256& hash) const {
    return objects_.find(hash) != objects_.end();
}

const StoredObject* ObjectStore::Peek(const crypto::Hash256& hash) const {
    const auto it = objects_.find(hash);
    return it == objects_.end() ? nullptr : &it->second;
}

Result<StoredObject> ObjectStore::Get(const crypto::Hash256& hash) const {
    const auto it = objects_.find(hash);
    if (it == objects_.end()) return Err("store: unknown object " + util::ToHex(hash));
    return it->second;
}

std::vector<InvEntry> ObjectStore::NoteAnnouncement(uint64_t peer_id,
                                                    const std::vector<InvEntry>& entries,
                                                    int64_t now_ms) {
    (void)now_ms;
    auto& announced = announced_by_peer_[peer_id];
    std::vector<InvEntry> wanted;

    for (const auto& entry : entries) {
        // Remembering an announcement costs memory, so one peer's talking is
        // bounded regardless of how much it says.
        if (announced.size() < kMaxAnnouncementsPerPeer) {
            announced.insert(entry.hash);
            announcers_[entry.hash].insert(peer_id);
        }
        if (Has(entry.hash)) continue;                       // already held
        if (requests_.find(entry.hash) != requests_.end()) continue;  // already being fetched
        wanted.push_back(entry);
    }
    return wanted;
}

Status ObjectStore::NoteRequested(uint64_t peer_id, const InvEntry& entry, int64_t now_ms) {
    if (Has(entry.hash)) return Err("store: already held");

    const auto existing = requests_.find(entry.hash);
    if (existing != requests_.end()) {
        // Asking several peers for the same object multiplies the node's own
        // bandwidth cost by its peer count.
        return Err("store: already requested from another peer");
    }
    if (PendingFor(peer_id) >= kMaxRequestsPerPeer) {
        return Err(std::format("store: peer {} already owes {} objects", peer_id,
                               kMaxRequestsPerPeer));
    }

    PendingRequest request;
    request.type = entry.type;
    request.hash = entry.hash;
    request.peer_id = peer_id;
    request.requested_at = now_ms;
    request.attempts = 1;
    requests_[entry.hash] = request;
    return Status::Ok();
}

void ObjectStore::NoteDelivered(const crypto::Hash256& hash) { requests_.erase(hash); }

std::vector<PendingRequest> ObjectStore::ExpiredRequests(int64_t now_ms) const {
    std::vector<PendingRequest> expired;
    for (const auto& [hash, request] : requests_) {
        if (now_ms - request.requested_at > kRequestTimeoutMs) expired.push_back(request);
    }
    return expired;
}

void ObjectStore::DropRequest(const crypto::Hash256& hash) { requests_.erase(hash); }

std::vector<uint64_t> ObjectStore::PeersAnnouncing(const crypto::Hash256& hash) const {
    const auto it = announcers_.find(hash);
    if (it == announcers_.end()) return {};
    return std::vector<uint64_t>(it->second.begin(), it->second.end());
}

void ObjectStore::ForgetPeer(uint64_t peer_id) {
    const auto announced = announced_by_peer_.find(peer_id);
    if (announced != announced_by_peer_.end()) {
        for (const auto& hash : announced->second) {
            const auto it = announcers_.find(hash);
            if (it == announcers_.end()) continue;
            it->second.erase(peer_id);
            if (it->second.empty()) announcers_.erase(it);
        }
        announced_by_peer_.erase(announced);
    }

    // A departed peer cannot deliver what it owed; release those requests so the
    // objects can be fetched elsewhere.
    for (auto it = requests_.begin(); it != requests_.end();) {
        it = (it->second.peer_id == peer_id) ? requests_.erase(it) : std::next(it);
    }
}

size_t ObjectStore::PendingFor(uint64_t peer_id) const {
    return static_cast<size_t>(std::count_if(
        requests_.begin(), requests_.end(),
        [peer_id](const auto& kv) { return kv.second.peer_id == peer_id; }));
}

}  // namespace rnet::net

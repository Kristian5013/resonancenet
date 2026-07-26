#include "diloco/producer.h"

#include <algorithm>

#include "canon/canon.h"

namespace rnet::diloco {

crypto::Hash256 EvidenceHashOf(std::span<const canon::ContributionHeader> contributions) {
    // Ascending worker order, so two nodes holding the same set agree regardless
    // of the order the network happened to deliver them in.
    std::vector<const canon::ContributionHeader*> ordered;
    ordered.reserve(contributions.size());
    for (const auto& c : contributions) ordered.push_back(&c);
    std::sort(ordered.begin(), ordered.end(),
              [](const auto* a, const auto* b) { return a->worker_id < b->worker_id; });

    canon::Writer w;
    w.U32(static_cast<uint32_t>(ordered.size()));
    for (const auto* c : ordered) w.Hash(c->Id());
    return crypto::Sha3_256(w.data());
}

namespace {

// The ordering nobody can steer: it depends on the parent checkpoint's identity,
// which was fixed before this round opened.
crypto::Hash256 ProducerTicket(const crypto::Hash256& parent, uint64_t outer_step,
                               uint64_t slot, uint64_t worker_id) {
    canon::Writer w;
    w.Hash(parent);
    w.U64(outer_step);
    w.U64(slot);
    w.U64(worker_id);
    return crypto::Sha3_256(w.data());
}

}  // namespace

Result<uint64_t> ElectProducer(const crypto::Hash256& parent, uint64_t outer_step,
                               std::span<const uint64_t> eligible, uint64_t slot) {
    if (eligible.empty()) {
        // A step with nobody entitled to publish is a stalled network. Saying so
        // is more useful than quietly electing worker 0, which is the unset value.
        return Err("producer: no eligible workers for this step");
    }

    uint64_t winner = 0;
    crypto::Hash256 best{};
    bool have_winner = false;
    for (uint64_t worker_id : eligible) {
        if (worker_id == 0) continue;   // the unset value is not a worker
        const auto ticket = ProducerTicket(parent, outer_step, slot, worker_id);
        // Ties are broken by worker id, so the result is total even in the
        // astronomically unlikely case of a hash collision.
        if (!have_winner || ticket < best || (ticket == best && worker_id < winner)) {
            best = ticket;
            winner = worker_id;
            have_winner = true;
        }
    }
    if (!have_winner) return Err("producer: the eligible set contained no valid worker ids");
    return winner;
}

bool IsProducer(const crypto::Hash256& parent, uint64_t outer_step,
                std::span<const uint64_t> eligible, uint64_t worker_id, uint64_t slot) {
    auto elected = ElectProducer(parent, outer_step, eligible, slot);
    return elected.ok() && elected.value() == worker_id;
}

}  // namespace rnet::diloco

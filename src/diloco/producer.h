// Who publishes the checkpoint for a round.
//
// Aggregation is the one step in the protocol that nobody currently checks. Every
// node holds a slightly different set of contributions — the network is
// asynchronous, so it cannot be otherwise — and if each node aggregated its own
// set, every node would produce a different checkpoint for the same step and
// there would be no fact of the matter about which was right.
//
// So one node per step PRODUCES and everyone else CHECKS:
//
//   * The producer is derived, not volunteered. It is the eligible worker
//     minimising SHA3(parent ‖ outer_step ‖ worker_id) — an ordering nobody can
//     steer, because it depends on the previous checkpoint's identity, which was
//     fixed before this round opened.
//
//   * ELIGIBILITY COMES FROM THE PREVIOUS CHECKPOINT, not from this round. If the
//     producer were drawn from the set it publishes, it could omit whoever would
//     have beaten it and elect itself. Fixing the pool one step earlier removes
//     that move entirely: the pool is settled before anyone knows what this
//     round's contributions will be.
//
//   * Checking is recomputation. A checkpoint names its evidence set, so any node
//     holding those payloads re-aggregates and compares the weights hash. A
//     producer that publishes something other than what the inputs imply is
//     caught by arithmetic rather than by opinion.
//
// The genesis step is the one place this is weaker: there is no previous evidence
// set, so eligibility falls back to the contributors of the round being closed,
// and a step-0 producer could in principle exclude a rival. That is stated rather
// than hidden — it lasts exactly one step, and from step 1 the pool is settled in
// advance.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "canon/objects.h"
#include "crypto/sha3.h"
#include "util/result.h"

namespace rnet::diloco {

// The evidence hash a checkpoint commits to: SHA3-256 over the contribution ids
// in ascending worker order. Order is fixed so two nodes holding the same set
// produce the same value regardless of arrival order.
crypto::Hash256 EvidenceHashOf(std::span<const canon::ContributionHeader> contributions);

// The worker that must publish the checkpoint for `outer_step`.
//
// `eligible` is the pool, which the caller draws from the PREVIOUS checkpoint's
// contributors. Fails on an empty pool rather than picking a default: a step with
// nobody entitled to publish is a stalled network, and saying so is more useful
// than quietly electing worker 0.
Result<uint64_t> ElectProducer(const crypto::Hash256& parent, uint64_t outer_step,
                               std::span<const uint64_t> eligible);

// True if `worker_id` is the producer for this step. Convenience for the common
// question, so callers do not each re-derive it slightly differently.
bool IsProducer(const crypto::Hash256& parent, uint64_t outer_step,
                std::span<const uint64_t> eligible, uint64_t worker_id);

}  // namespace rnet::diloco

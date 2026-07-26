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

// The worker that must publish the checkpoint for `outer_step`, in `slot`.
//
// `eligible` is the pool, which the caller draws from the PREVIOUS checkpoint's
// contributors. Fails on an empty pool rather than picking a default: a step with
// nobody entitled to publish is a stalled network, and saying so is more useful
// than quietly electing worker 0.
//
// `slot` exists because the other three inputs are frozen exactly when the network
// most needs them to move. The parent, the step and the pool all change only when
// the tip advances — so if the elected producer dies, every honest node re-elects
// the same corpse on every retry, forever, and the whole network burns GPU without
// completing a step. A worker running out of memory partway through applying a
// multi-gigabyte update is not an exotic failure; it is the expected one for that
// operation.
//   proven by: protocol_tests.cpp TheWinnerRotatesAcrossSlotsSoADeadOneIsReplaced
//
// The slot is absolute time divided by a consensus-fixed length, not a count of
// local retries. Rounds open when each node happens to open them, so a counter
// anchored to a local event gives each node a different answer and splits the
// election instead of rotating it.
//   proven by: protocol_tests.cpp TwoNodesWhoseRoundsOpenedApartStillAgree
//
// Nodes therefore agree whenever their clocks agree to better than a slot — twenty
// minutes on main. Near a boundary they briefly disagree and two checkpoints may be
// published, which is a fork the chain rule resolves; that is a better failure than
// a network that cannot advance at all.
Result<uint64_t> ElectProducer(const crypto::Hash256& parent, uint64_t outer_step,
                               std::span<const uint64_t> eligible, uint64_t slot = 0);

// True if `worker_id` is the producer for this step and slot. Convenience for the
// common question, so callers do not each re-derive it slightly differently.
bool IsProducer(const crypto::Hash256& parent, uint64_t outer_step,
                std::span<const uint64_t> eligible, uint64_t worker_id, uint64_t slot = 0);

}  // namespace rnet::diloco

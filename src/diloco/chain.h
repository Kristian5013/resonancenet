// The checkpoint chain.
//
// Each published checkpoint names its parent, so the sequence of models forms a
// hash chain back to genesis. A node can prove that a model descends from genesis
// by following 32-byte links — it does not need the intermediate weights.
//
// That distinction matters for what participants must store. A Bitcoin node keeps
// the whole chain because current state is derived by replaying history. Here the
// weights ARE the state: checkpoint N+1 is validated against N, not against
// N-1000. So the mandatory storage is the header chain (kilobytes), plus a few
// recent full checkpoints for rollback depth and for recomputation verification.
// Complete archives are useful and voluntary, like archival nodes elsewhere.
//
// Weight distribution is out of scope here on purpose: transport (swarm, mirrors)
// is interchangeable, and identity must not depend on it. A checkpoint is named by
// the SHA3-256 of its canonical weight bytes, so a node verifies what it received
// no matter where it came from.
#pragma once

#include <cstdint>
#include <deque>
#include <map>
#include <vector>

#include "canon/objects.h"
#include "util/result.h"

namespace rnet::diloco {

// How many full checkpoints a participating node keeps. Enough to serve a
// rollback and to hold the base weights a verifier needs for recomputation.
inline constexpr uint32_t kDefaultRetainedCheckpoints = 5;

struct ChainEntry {
    canon::CheckpointHeader header;
    crypto::Hash256 id{};
    uint64_t height{0};   // 0 = genesis
};

// Which of two checkpoints competing for the same height the network keeps.
//
// Two can exist without anybody cheating. The producer for a step is derived from a
// wall-clock slot, so two nodes whose clocks straddle a slot boundary elect
// different producers, and both publish something valid. Before this rule existed
// the chain took whichever arrived first and refused the other, so two honest nodes
// that received them in opposite orders held different tips and never converged —
// and, because the refusal was scored, banned each other while they were at it.
//
// "Longest chain" cannot separate the branches here: there is no proof of work, and
// heights advance in lockstep at one checkpoint per outer step, so competing
// branches are always the same length. The tie-break IS the fork-choice rule.
//
// Lower id wins. It is a total order every node computes from bytes it already
// holds, with nothing to agree on beforehand, and it is self-certifying: an id is
// the SHA3-256 of the header itself, so no one can claim a lower one without
// actually producing a header that hashes there and passes every other check. A
// producer could grind its aggregate for a lower id, but only in the rare step
// where a second producer was elected at all, and any header it grinds is still one
// a follower holding the same evidence set can refute.
//   proven by: protocol_tests.cpp SiblingCheckpointsConvergeRatherThanSplittingTheNetwork
bool PreferredOverIncumbent(const crypto::Hash256& candidate, const crypto::Hash256& incumbent);

class CheckpointChain {
public:
    explicit CheckpointChain(uint32_t retained_full = kDefaultRetainedCheckpoints);

    // Adds the genesis checkpoint. Only one is permitted, and it must be
    // parentless.
    Status SetGenesis(const canon::CheckpointHeader& header);

    // Appends a checkpoint. Rejects it unless the parent is the current tip and
    // the outer step advances by exactly one — a gap would mean the node is
    // missing evidence it cannot reconstruct.
    Status Append(const canon::CheckpointHeader& header);

    bool Has(const crypto::Hash256& id) const;
    Result<ChainEntry> Get(const crypto::Hash256& id) const;
    Result<ChainEntry> Tip() const;
    uint64_t Height() const;

    // Whatever currently occupies `height`, so a competing checkpoint for a height
    // the chain has already filled can be compared against the incumbent rather
    // than refused for arriving second.
    Result<ChainEntry> AtHeight(uint64_t height) const;

    // Walks parent links from `id` back to genesis. This is the audit path: it
    // needs only headers, never weights.
    Result<std::vector<crypto::Hash256>> AncestryOf(const crypto::Hash256& id) const;

    // Confirms `id` descends from genesis.
    bool DescendsFromGenesis(const crypto::Hash256& id) const;

    // Rolls back to `id`, discarding everything after it. Used when a later
    // measurement shows an accepted checkpoint regressed the model. Refuses to
    // roll back past the retention window, since those weights are gone.
    Status RollbackTo(const crypto::Hash256& id);

    // Which checkpoints a node must keep in full: the tip and the retention
    // window behind it. Everything older can be pruned to its header.
    std::vector<crypto::Hash256> RequiredFullCheckpoints() const;

    uint32_t retained_full() const { return retained_full_; }

private:
    uint32_t retained_full_;
    std::vector<crypto::Hash256> order_;                // genesis .. tip
    std::map<crypto::Hash256, ChainEntry> entries_;
};

// Total bytes a node must store to participate: the header chain in full, plus
// the retained full checkpoints. Exposed so the storage requirement is a computed
// number in the protocol rather than folklore.
struct StorageEstimate {
    uint64_t header_bytes{0};
    uint64_t full_checkpoint_bytes{0};
    uint64_t total_bytes{0};
};

StorageEstimate EstimateStorage(uint64_t chain_height, uint64_t n_params, uint8_t weight_dtype,
                                uint32_t retained_full);

}  // namespace rnet::diloco

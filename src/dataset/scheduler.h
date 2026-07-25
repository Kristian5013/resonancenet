// Deterministic batch derivation — the mechanism that closes data poisoning.
//
// A worker does not choose what it trains on. Its training windows are a pure
// function of protocol state:
//
//   offset_i = SHA3-256(canon(BatchSeed{dataset_root, round, worker, step, i})) mod W
//
// Two consequences, both load-bearing:
//
//   * A malicious worker cannot feed itself crafted data. The corpus is pinned by
//     dataset_root and the windows are dictated by the schedule. Our poison-gate
//     experiment showed a held-out loss gate does NOT catch a targeted backdoor
//     introduced through data, so the protocol removes the worker's ability to
//     choose data at all rather than trying to detect the result afterwards.
//
//   * A verifier can reproduce the exact batch a worker claims to have trained
//     on, which is what makes spot-recompute verification possible.
//
// Integer-only, no floating point, defined over canonical bytes — so C++ and the
// Python worker derive identical offsets.
#pragma once

#include <cstdint>
#include <vector>

#include "canon/objects.h"
#include "crypto/sha3.h"
#include "util/result.h"

namespace rnet::dataset {

// Number of valid window start positions in a corpus: a window needs seq_len
// inputs plus one shifted target token.
Result<uint64_t> WindowCount(uint64_t n_tokens, uint32_t seq_len);

// Start offset (in tokens) of a single training window.
//
// Modulo bias note: the hash is reduced from a uniform 64-bit value, so for any
// realistic corpus (< 2^40 windows) the bias is below 2^-24 and irrelevant to
// training. It is deliberately NOT rejection-sampled, because a rejection loop
// would make the derivation's cost data-dependent and harder to reimplement
// identically in another language.
Result<uint64_t> WindowOffset(const crypto::Hash256& dataset_root, uint64_t round_id,
                              uint64_t worker_id, uint64_t step, uint32_t index,
                              uint64_t n_tokens, uint32_t seq_len);

// All offsets for one training step (micro_batch windows).
Result<std::vector<uint64_t>> BatchOffsets(const crypto::Hash256& dataset_root, uint64_t round_id,
                                           uint64_t worker_id, uint64_t step, uint32_t micro_batch,
                                           uint64_t n_tokens, uint32_t seq_len);

}  // namespace rnet::dataset

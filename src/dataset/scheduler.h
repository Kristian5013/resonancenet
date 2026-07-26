// Deterministic batch derivation — the mechanism that closes data poisoning.
//
// A worker does not choose what it trains on. Its training windows are a pure
// function of protocol state:
//
//   seed_i   = SHA3-256(canon(BatchSeed{dataset_root, round, worker, step, i}))
//   chunk_i  = seed_i mod n_chunks
//   start_i  = SHA3-256("rnet/window-offset" ‖ seed_i) mod (tokens_in_chunk - seq_len)
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

// The seed a window is drawn from: a pure function of protocol state, so a
// verifier reconstructs it from the contribution header alone.
crypto::Hash256 WindowSeed(const crypto::Hash256& dataset_root, uint64_t round_id,
                           uint64_t worker_id, uint64_t step, uint32_t index);

// Which chunk of the corpus this window comes from.
//
// The quantity a worker cannot influence is now the chunk count rather than a
// token count: it does not know how many tokens the corpus holds, and neither
// does the node, because nothing has been tokenized.
Result<uint64_t> ChunkForWindow(const crypto::Hash256& seed, uint64_t n_chunks);

// Where inside that chunk's tokenization the window starts.
//
// Drawn from a SEPARATE hash of the same seed rather than from the same value
// used twice, so the chunk and the position within it are independent. Takes the
// chunk's token count, which only whoever tokenized the chunk knows — the node
// never calls this, the worker and the verifier both do, and they agree because
// they tokenized the same bytes with the same pinned artifact.
Result<uint64_t> OffsetInChunk(const crypto::Hash256& seed, uint64_t tokens_in_chunk,
                               uint32_t seq_len);

// Every chunk index for one inner step's micro-batch.
Result<std::vector<uint64_t>> BatchChunks(const crypto::Hash256& dataset_root, uint64_t round_id,
                                          uint64_t worker_id, uint64_t step, uint32_t micro_batch,
                                          uint64_t n_chunks);

}  // namespace rnet::dataset

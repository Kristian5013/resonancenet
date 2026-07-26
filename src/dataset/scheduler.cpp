#include "dataset/scheduler.h"

#include <format>

namespace rnet::dataset {

namespace {
constexpr uint32_t kMaxMicroBatch = 4096;

// Big-endian read of the first 8 bytes of a digest.
uint64_t Be64(const crypto::Hash256& h) {
    uint64_t v = 0;
    for (size_t i = 0; i < 8; ++i) v = (v << 8) | h[i];
    return v;
}
}  // namespace

crypto::Hash256 WindowSeed(const crypto::Hash256& dataset_root, uint64_t round_id,
                           uint64_t worker_id, uint64_t step, uint32_t index) {
    canon::BatchSeed seed;
    seed.dataset_root = dataset_root;
    seed.round_id = round_id;
    seed.worker_id = worker_id;
    seed.step = step;
    seed.index = index;
    return seed.Id();
}

Result<uint64_t> ChunkForWindow(const crypto::Hash256& seed, uint64_t n_chunks) {
    if (n_chunks == 0) return Err("schedule: corpus has no chunks");
    return Be64(seed) % n_chunks;
}

Result<uint64_t> OffsetInChunk(const crypto::Hash256& seed, uint64_t tokens_in_chunk,
                               uint32_t seq_len) {
    if (seq_len == 0) return Err("schedule: seq_len must be non-zero");
    // seq_len inputs and one more for the shifted target.
    const uint64_t need = static_cast<uint64_t>(seq_len) + 1;
    if (tokens_in_chunk < need) {
        return Err(std::format(
            "schedule: a chunk of {} tokens cannot hold a window of {}; the manifest should have "
            "refused this corpus when it was built",
            tokens_in_chunk, need));
    }
    // A second hash rather than the same one reduced twice: reusing a value would
    // correlate which chunk a worker gets with where in it the window falls, and
    // a correlation is something to grind against.
    canon::Writer w;
    w.String("rnet/window-offset");
    w.Hash(seed);
    return Be64(crypto::Sha3_256(w.data())) % (tokens_in_chunk - seq_len);
}

Result<std::vector<uint64_t>> BatchChunks(const crypto::Hash256& dataset_root, uint64_t round_id,
                                          uint64_t worker_id, uint64_t step, uint32_t micro_batch,
                                          uint64_t n_chunks) {
    std::vector<uint64_t> out;
    out.reserve(micro_batch);
    for (uint32_t i = 0; i < micro_batch; ++i) {
        auto c = ChunkForWindow(WindowSeed(dataset_root, round_id, worker_id, step, i), n_chunks);
        if (!c) return Err(c.error());
        out.push_back(c.value());
    }
    return out;
}

}  // namespace rnet::dataset

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

Result<uint64_t> WindowCount(uint64_t n_tokens, uint32_t seq_len) {
    if (seq_len == 0) return Err("schedule: seq_len must be non-zero");
    // A window consumes seq_len inputs and one extra token for the shifted target.
    if (n_tokens <= static_cast<uint64_t>(seq_len) + 1) {
        return Err(std::format("schedule: corpus too small ({} tokens) for seq_len {}", n_tokens,
                               seq_len));
    }
    return n_tokens - seq_len - 1;
}

Result<uint64_t> WindowOffset(const crypto::Hash256& dataset_root, uint64_t round_id,
                              uint64_t worker_id, uint64_t step, uint32_t index, uint64_t n_tokens,
                              uint32_t seq_len) {
    auto windows = WindowCount(n_tokens, seq_len);
    if (!windows) return Err(windows.error());

    canon::BatchSeed seed;
    seed.dataset_root = dataset_root;
    seed.round_id = round_id;
    seed.worker_id = worker_id;
    seed.step = step;
    seed.index = index;

    return Be64(seed.Id()) % windows.value();
}

Result<std::vector<uint64_t>> BatchOffsets(const crypto::Hash256& dataset_root, uint64_t round_id,
                                           uint64_t worker_id, uint64_t step, uint32_t micro_batch,
                                           uint64_t n_tokens, uint32_t seq_len) {
    if (micro_batch == 0) return Err("schedule: micro_batch must be non-zero");
    if (micro_batch > kMaxMicroBatch) return Err("schedule: micro_batch too large");

    std::vector<uint64_t> offsets;
    offsets.reserve(micro_batch);
    for (uint32_t i = 0; i < micro_batch; ++i) {
        auto off = WindowOffset(dataset_root, round_id, worker_id, step, i, n_tokens, seq_len);
        if (!off) return Err(off.error());
        offsets.push_back(off.value());
    }
    return offsets;
}

}  // namespace rnet::dataset

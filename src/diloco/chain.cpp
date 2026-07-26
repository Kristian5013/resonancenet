#include "diloco/chain.h"

#include <algorithm>
#include <format>

#include "util/hex.h"

namespace rnet::diloco {

namespace {
// A canonical checkpoint header serialises to a fixed 141 bytes; the container
// adds 46. Used only for the storage estimate.
constexpr uint64_t kHeaderContainerBytes = 187;
}  // namespace

CheckpointChain::CheckpointChain(uint32_t retained_full) : retained_full_(retained_full) {
    if (retained_full_ == 0) retained_full_ = 1;   // the tip is always retained
}

Status CheckpointChain::SetGenesis(const canon::CheckpointHeader& header) {
    if (!order_.empty()) return Err("chain: genesis already set");
    if (auto st = header.Validate(); !st) return st;
    if (!header.IsGenesis()) return Err("chain: genesis checkpoint must have no parent");

    const crypto::Hash256 id = header.Id();
    entries_[id] = ChainEntry{header, id, 0};
    order_.push_back(id);
    return Status::Ok();
}

Status CheckpointChain::Append(const canon::CheckpointHeader& header) {
    if (order_.empty()) return Err("chain: no genesis");
    if (auto st = header.Validate(); !st) return st;

    const crypto::Hash256& tip_id = order_.back();
    const ChainEntry& tip = entries_.at(tip_id);

    if (header.parent != tip_id) {
        return Err("chain: parent is not the current tip\n  parent: " +
                   util::ToHex(header.parent) + "\n  tip:    " + util::ToHex(tip_id));
    }
    if (header.outer_step != tip.header.outer_step + 1) {
        return Err(std::format("chain: outer step {} does not follow {}", header.outer_step,
                               tip.header.outer_step));
    }
    if (header.n_params != tip.header.n_params) {
        return Err("chain: parameter count changed without a new round");
    }

    const crypto::Hash256 id = header.Id();
    if (entries_.count(id) != 0) return Err("chain: duplicate checkpoint");

    entries_[id] = ChainEntry{header, id, tip.height + 1};
    order_.push_back(id);
    return Status::Ok();
}

bool CheckpointChain::Has(const crypto::Hash256& id) const { return entries_.count(id) != 0; }

Result<ChainEntry> CheckpointChain::Get(const crypto::Hash256& id) const {
    const auto it = entries_.find(id);
    if (it == entries_.end()) return Err("chain: unknown checkpoint " + util::ToHex(id));
    return it->second;
}

Result<ChainEntry> CheckpointChain::Tip() const {
    if (order_.empty()) return Err("chain: empty");
    return entries_.at(order_.back());
}

uint64_t CheckpointChain::Height() const {
    return order_.empty() ? 0 : entries_.at(order_.back()).height;
}

Result<std::vector<crypto::Hash256>> CheckpointChain::AncestryOf(const crypto::Hash256& id) const {
    auto entry = Get(id);
    if (!entry) return Err(entry.error());

    std::vector<crypto::Hash256> path;
    crypto::Hash256 cursor = id;
    while (true) {
        path.push_back(cursor);
        const auto it = entries_.find(cursor);
        if (it == entries_.end()) return Err("chain: broken ancestry at " + util::ToHex(cursor));
        if (it->second.header.IsGenesis()) break;
        cursor = it->second.header.parent;
    }
    std::reverse(path.begin(), path.end());   // genesis first
    return path;
}

bool CheckpointChain::DescendsFromGenesis(const crypto::Hash256& id) const {
    auto path = AncestryOf(id);
    if (!path) return false;
    return !path.value().empty() && entries_.at(path.value().front()).header.IsGenesis();
}

bool PreferredOverIncumbent(const crypto::Hash256& candidate, const crypto::Hash256& incumbent) {
    // Bytewise, most significant first — the same order the hex spelling reads in,
    // so a log line saying which won is checkable by eye.
    return std::lexicographical_compare(candidate.begin(), candidate.end(), incumbent.begin(),
                                        incumbent.end());
}

Result<ChainEntry> CheckpointChain::AtHeight(uint64_t height) const {
    if (height >= order_.size()) {
        return Err(std::format("chain: no checkpoint at height {} (tip is {})", height,
                               order_.empty() ? 0 : order_.size() - 1));
    }
    return entries_.at(order_[static_cast<size_t>(height)]);
}

Status CheckpointChain::RollbackTo(const crypto::Hash256& id) {
    const auto it = entries_.find(id);
    if (it == entries_.end()) return Err("chain: cannot roll back to unknown checkpoint");

    const auto pos = std::find(order_.begin(), order_.end(), id);
    if (pos == order_.end()) return Err("chain: checkpoint is not on the canonical chain");

    const uint64_t target_height = it->second.height;
    const uint64_t tip_height = Height();
    // Beyond the retention window the full weights have been pruned, so a rollback
    // could not actually be carried out — refusing is more honest than accepting a
    // header-only rewind the node cannot execute.
    if (tip_height - target_height > retained_full_) {
        return Err(std::format(
            "chain: rollback of {} checkpoints exceeds the retention window of {}",
            tip_height - target_height, retained_full_));
    }

    for (auto cursor = std::next(pos); cursor != order_.end(); ++cursor) entries_.erase(*cursor);
    order_.erase(std::next(pos), order_.end());
    return Status::Ok();
}

std::vector<crypto::Hash256> CheckpointChain::RequiredFullCheckpoints() const {
    std::vector<crypto::Hash256> out;
    if (order_.empty()) return out;

    const size_t count = std::min<size_t>(retained_full_, order_.size());
    out.reserve(count);
    for (size_t i = 0; i < count; ++i) out.push_back(order_[order_.size() - 1 - i]);
    return out;   // newest first
}

StorageEstimate EstimateStorage(uint64_t chain_height, uint64_t n_params, uint8_t weight_dtype,
                                uint32_t retained_full) {
    const uint64_t bytes_per_param = weight_dtype == 2 ? 4 : 2;   // fp32 : bf16
    StorageEstimate est;
    est.header_bytes = (chain_height + 1) * kHeaderContainerBytes;
    est.full_checkpoint_bytes = static_cast<uint64_t>(retained_full) * n_params * bytes_per_param;
    est.total_bytes = est.header_bytes + est.full_checkpoint_bytes;
    return est;
}

}  // namespace rnet::diloco

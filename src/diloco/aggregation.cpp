#include "diloco/aggregation.h"

#include <format>

#include "diloco/quantize.h"

namespace rnet::diloco {

namespace {

// Bound on how far a contribution may be shifted when aligning scales. A worker
// whose scale differs by more than this has produced a pseudo-gradient orders of
// magnitude away from its peers, which is a divergence signal, not a rounding
// detail — so it is rejected rather than silently dominating the sum.
constexpr int32_t kMaxAlignShift = 24;

// Largest number of contributions that cannot overflow int64 while accumulating
// aligned int8 values: 127 << 24 is about 2.1e9, so int64 leaves room for far
// more contributors than the network will ever have in one round.
constexpr uint64_t kMaxContributors = 1u << 20;

}  // namespace

int64_t RoundedDiv(int64_t numerator, int64_t denominator) {
    if (denominator == 0) return 0;
    const bool negative = (numerator < 0) != (denominator < 0);
    // Work in unsigned magnitudes so the rule is identical either side of zero and
    // does not depend on the platform's integer-division truncation behaviour.
    const uint64_t n = numerator < 0 ? (~static_cast<uint64_t>(numerator) + 1)
                                     : static_cast<uint64_t>(numerator);
    const uint64_t d = denominator < 0 ? (~static_cast<uint64_t>(denominator) + 1)
                                       : static_cast<uint64_t>(denominator);
    const uint64_t q = (n + d / 2) / d;
    return negative ? -static_cast<int64_t>(q) : static_cast<int64_t>(q);
}

Result<Aggregate> SumContributions(std::span<const Contribution> contributions, uint64_t n_params) {
    if (contributions.empty()) return Err("aggregate: no contributions");
    if (n_params == 0) return Err("aggregate: n_params must be non-zero");
    if (contributions.size() > kMaxContributors) return Err("aggregate: too many contributions");

    int32_t target_exp = kMinScaleExp - 1;
    for (const auto& c : contributions) {
        if (c.values.size() != n_params) {
            return Err(std::format("aggregate: worker {} sent {} values, expected {}", c.worker_id,
                                   c.values.size(), n_params));
        }
        if (c.scale_exp < kMinScaleExp || c.scale_exp > kMaxScaleExp) {
            return Err(std::format("aggregate: worker {} has scale_exp {} out of range",
                                   c.worker_id, c.scale_exp));
        }
        if (c.scale_exp > target_exp) target_exp = c.scale_exp;
    }

    // Align everything to the finest scale present: a contribution recorded at a
    // coarser scale is shifted left, which loses nothing.
    std::vector<int32_t> shifts;
    shifts.reserve(contributions.size());
    for (const auto& c : contributions) {
        const int32_t shift = target_exp - c.scale_exp;
        if (shift > kMaxAlignShift) {
            return Err(std::format(
                "aggregate: worker {} scale differs by 2^{} from the round — treating as divergent",
                c.worker_id, shift));
        }
        shifts.push_back(shift);
    }

    Aggregate out;
    out.scale_exp = target_exp;
    out.contributor_count = static_cast<uint32_t>(contributions.size());
    out.values.assign(static_cast<size_t>(n_params), 0);

    for (size_t c = 0; c < contributions.size(); ++c) {
        const auto& values = contributions[c].values;
        const int32_t shift = shifts[c];
        for (size_t i = 0; i < values.size(); ++i) {
            out.values[i] += static_cast<int64_t>(values[i]) << shift;
        }
    }
    return out;
}

Result<Aggregate> AverageContributions(std::span<const Contribution> contributions,
                                       uint64_t n_params) {
    auto sum = SumContributions(contributions, n_params);
    if (!sum) return sum;

    const int64_t count = static_cast<int64_t>(sum.value().contributor_count);
    for (auto& v : sum.value().values) v = RoundedDiv(v, count);
    return sum;
}

}  // namespace rnet::diloco

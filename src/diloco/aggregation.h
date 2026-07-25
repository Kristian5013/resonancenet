// Aggregating contributions into one outer-step update.
//
// EVERY contribution is averaged in. Nothing is selected, nothing is discarded —
// this is not a race where one participant wins, it is a sum in which each
// worker's local progress enters the shared model. That is what makes throughput
// scale with the number of machines: M workers each doing H steps advance the
// model by M*H steps' worth of data per round.
//
// The arithmetic is exact integer arithmetic, for two reasons:
//
//   * Order independence. Contributions arrive over a network in arbitrary order.
//     Integer addition is associative and commutative, so every node computes the
//     identical aggregate regardless of arrival order. Floating-point addition is
//     not associative and would make the result depend on packet timing.
//
//   * Reproducibility. Verification works by recomputing and comparing bit for
//     bit; an aggregate that differs in the last bit between implementations
//     would make honest nodes look dishonest.
//
// Contributions may carry different power-of-two scales. They are aligned to the
// finest scale present by shifting left, which is exact for integers.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "canon/objects.h"
#include "util/result.h"

namespace rnet::diloco {

// One worker's submission, already validated and payload-verified.
struct Contribution {
    uint64_t worker_id{};
    int32_t scale_exp{};
    std::span<const int8_t> values;
};

struct Aggregate {
    std::vector<int64_t> values;   // in units of 2^-scale_exp
    int32_t scale_exp{};
    uint32_t contributor_count{};
};

// Sum of the aligned contributions, without dividing. Exposed because the sum is
// what a verifier recomputes and compares.
Result<Aggregate> SumContributions(std::span<const Contribution> contributions, uint64_t n_params);

// Mean of the contributions: the sum divided by the contributor count, rounded
// half away from zero so the operation is symmetric about zero and cannot drift
// in one direction over many rounds.
Result<Aggregate> AverageContributions(std::span<const Contribution> contributions,
                                       uint64_t n_params);

// Integer division rounding halves away from zero. Exposed for testing because
// the exact rounding rule is consensus-relevant.
int64_t RoundedDiv(int64_t numerator, int64_t denominator);

}  // namespace rnet::diloco

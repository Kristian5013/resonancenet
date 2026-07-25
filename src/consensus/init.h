// Where training starts.
//
// Every node must begin from the same weights, byte for byte. Nodes that start
// from different states compute deltas that cannot be aggregated and cannot be
// verified — and the failure would look like every worker cheating at once, which
// is the hardest kind of bug to read.
//
// Distributing a multi-gigabyte weights file solves this badly: it has to be
// hosted, mirrored and trusted. So the initial weights are DERIVED instead, from
// the genesis round descriptor a node already verified. The descriptor is the
// seed; anyone holding it can reproduce the exact starting state offline.
//
// The arithmetic is chosen so two languages agree exactly:
//
//   * The random stream is COUNTER-BASED (SHA3 of seed ‖ tensor ‖ counter), not a
//     stateful generator. Any implementation reaches the same bytes for the same
//     position without replaying the ones before it, which also makes the whole
//     thing parallelisable and streamable.
//
//   * The distribution is UNIFORM, not Gaussian. Box–Muller needs log, sqrt and
//     cos, and only sqrt is required by IEEE-754 to be correctly rounded — the
//     others differ between libm implementations, so a Gaussian init would put a
//     one-ulp cross-platform disagreement at the very root of the chain.
//
//   * Every step is an IEEE-754 operation that is correctly rounded by
//     definition: an exact integer-to-float conversion, one multiply, one
//     subtract, one multiply by a bound computed with a single division and a
//     single sqrt. No library function whose result is implementation-defined
//     appears anywhere.
//
// The bound per tensor is 1/sqrt(fan_in), the usual scaling for transformers, and
// norm weights start at one.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "canon/objects.h"
#include "consensus/params.h"
#include "crypto/sha3.h"
#include "util/result.h"

namespace rnet::consensus {

// One parameter tensor in canonical order. The order is part of consensus: the
// weights hash is over the concatenation, so two nodes that lay tensors out
// differently produce different hashes from identical numbers.
struct TensorSpec {
    std::string name;
    uint64_t rows{0};      // fan_out
    uint64_t cols{0};      // fan_in; 1 for vectors
    bool is_norm{false};   // norm weights start at one rather than random

    uint64_t elements() const { return rows * cols; }
};

// The canonical tensor layout for a model. Deterministic, and derived only from
// the spec — no file, no naming convention borrowed from another project.
std::vector<TensorSpec> TensorLayout(const canon::ModelSpec& model);

// The seed for a round. Domain-separated so the same descriptor cannot produce
// colliding streams for two different purposes.
crypto::Hash256 InitSeed(const canon::RoundDescriptor& round);

// Fills `out` with the initial values of one tensor, in row-major order.
// `element_offset` allows a caller to generate a slice without generating what
// comes before it — which is what makes a 1B-parameter model initialisable in
// bounded memory.
Status InitTensorSlice(const crypto::Hash256& seed, uint32_t tensor_index,
                       const TensorSpec& tensor, uint64_t element_offset, std::span<float> out);

// SHA3-256 of the canonical initial weight bytes: every tensor in layout order,
// each element as big-endian bf16. This is the value pinned as a network's
// genesis weights anchor.
//
// Streams in bounded memory, so it runs on a 1B model without holding it.
Result<crypto::Hash256> GenesisWeightsHash(const canon::RoundDescriptor& round);

// Converts an IEEE-754 float to bfloat16 with round-half-to-even — the same
// rounding the hardware does, so a node that converts in software and one that
// converts on a GPU agree.
uint16_t FloatToBf16(float value);

}  // namespace rnet::consensus

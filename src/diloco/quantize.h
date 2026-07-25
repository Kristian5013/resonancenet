// Pseudo-gradient quantisation.
//
// A worker trains locally for `inner_steps`, then submits what changed:
//
//     pseudo_gradient = weights_before - weights_after
//
// That delta is quantised to int8 for transport — roughly a 4x reduction over
// bf16 on top of the ~500x saved by syncing rarely instead of every step. For
// RN-1B this is about 1 GB per outer step instead of 1 GB every 1.3 seconds.
//
// THE SCALE IS A POWER OF TWO, carried as an integer exponent:
//
//     quantised = round(delta * 2^scale_exp)      clamped to [-127, 127]
//     delta     = quantised * 2^-scale_exp
//
// This is deliberate. A floating-point scale would have to be transmitted as
// float bits and multiplied out, and two implementations could round the product
// differently in the last bit. A power of two is exact in binary floating point:
// scaling only changes the exponent field, never the mantissa, so C++ and Python
// agree bit for bit. Verification by recomputation depends on that agreement.
//
// -128 is excluded so the value range is symmetric; an asymmetric range would
// bias the average of many contributions.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "util/result.h"

namespace rnet::diloco {

inline constexpr int8_t kQuantMax = 127;
inline constexpr int8_t kQuantMin = -127;
inline constexpr int32_t kMaxScaleExp = 64;
inline constexpr int32_t kMinScaleExp = -64;

// Chooses the exponent that maps the largest magnitude in `deltas` as close to
// 127 as a power-of-two scale allows, without clipping it.
Result<int32_t> ChooseScaleExp(std::span<const float> deltas);

// Quantises with a given exponent. Values beyond the representable range are
// clamped, and the count of clamped elements is reported: silent clipping would
// hide a diverging worker, so the caller can see it.
struct QuantizeResult {
    std::vector<int8_t> values;
    uint64_t clamped{0};
};

Result<QuantizeResult> Quantize(std::span<const float> deltas, int32_t scale_exp);

// Exact inverse of the above for values that were not clamped.
Result<std::vector<float>> Dequantize(std::span<const int8_t> values, int32_t scale_exp);

// round(x) with ties away from zero, implemented without libm so that every
// implementation of the protocol rounds identically.
int32_t RoundTiesAwayFromZero(float x);

}  // namespace rnet::diloco

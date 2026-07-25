#include "diloco/quantize.h"

#include <cmath>
#include <format>
#include <limits>

namespace rnet::diloco {

namespace {

// 2^exp as a float, built by exponent arithmetic rather than pow(): exact, and
// free of any libm implementation difference.
float PowerOfTwo(int32_t exp) { return std::ldexp(1.0f, exp); }

}  // namespace

int32_t RoundTiesAwayFromZero(float x) {
    // std::round() has exactly these semantics and is not affected by the current
    // rounding mode, unlike nearbyint/rint.
    const float r = std::round(x);
    if (r > static_cast<float>(std::numeric_limits<int32_t>::max())) {
        return std::numeric_limits<int32_t>::max();
    }
    if (r < static_cast<float>(std::numeric_limits<int32_t>::min())) {
        return std::numeric_limits<int32_t>::min();
    }
    return static_cast<int32_t>(r);
}

Result<int32_t> ChooseScaleExp(std::span<const float> deltas) {
    if (deltas.empty()) return Err("quantize: empty delta vector");

    float max_abs = 0.0f;
    for (float v : deltas) {
        if (!std::isfinite(v)) {
            // A non-finite delta means the local run diverged. Refusing here keeps
            // a NaN from being quantised into a plausible-looking integer and
            // silently poisoning the average.
            return Err("quantize: non-finite value in pseudo-gradient");
        }
        const float a = std::fabs(v);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0.0f) return int32_t{0};   // no change at all: any exponent works

    // Largest exponent with max_abs * 2^e <= 127.
    int exp = 0;
    std::frexp(max_abs / static_cast<float>(kQuantMax), &exp);
    int32_t scale_exp = -exp;
    while (max_abs * PowerOfTwo(scale_exp) > static_cast<float>(kQuantMax)) --scale_exp;
    while (max_abs * PowerOfTwo(scale_exp + 1) <= static_cast<float>(kQuantMax)) ++scale_exp;

    if (scale_exp > kMaxScaleExp) scale_exp = kMaxScaleExp;
    if (scale_exp < kMinScaleExp) scale_exp = kMinScaleExp;
    return scale_exp;
}

Result<QuantizeResult> Quantize(std::span<const float> deltas, int32_t scale_exp) {
    if (scale_exp < kMinScaleExp || scale_exp > kMaxScaleExp) {
        return Err(std::format("quantize: scale_exp {} out of range", scale_exp));
    }
    const float scale = PowerOfTwo(scale_exp);

    QuantizeResult out;
    out.values.reserve(deltas.size());
    for (float v : deltas) {
        if (!std::isfinite(v)) return Err("quantize: non-finite value in pseudo-gradient");
        int32_t q = RoundTiesAwayFromZero(v * scale);
        if (q > kQuantMax) {
            q = kQuantMax;
            ++out.clamped;
        } else if (q < kQuantMin) {
            q = kQuantMin;
            ++out.clamped;
        }
        out.values.push_back(static_cast<int8_t>(q));
    }
    return out;
}

Result<std::vector<float>> Dequantize(std::span<const int8_t> values, int32_t scale_exp) {
    if (scale_exp < kMinScaleExp || scale_exp > kMaxScaleExp) {
        return Err(std::format("dequantize: scale_exp {} out of range", scale_exp));
    }
    const float inv = PowerOfTwo(-scale_exp);
    std::vector<float> out;
    out.reserve(values.size());
    for (int8_t q : values) out.push_back(static_cast<float>(q) * inv);
    return out;
}

}  // namespace rnet::diloco

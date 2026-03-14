// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "inference/quantize.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace rnet::inference {

// ===========================================================================
// Format Names
// ===========================================================================

// ---------------------------------------------------------------------------
// quant_format_name
//
// Returns a human-readable string for each QuantFormat enum value.
// Used in logging, serialization headers, and CLI output.
// ---------------------------------------------------------------------------
const char* quant_format_name(QuantFormat fmt) {
    switch (fmt) {
        case QuantFormat::BF16: return "bf16";
        case QuantFormat::INT8: return "int8";
        case QuantFormat::INT4: return "int4";
    }
    return "unknown";
}

// ===========================================================================
// BF16 Conversion
// ===========================================================================

// ---------------------------------------------------------------------------
// bf16_to_fp32
//
// BF16 format: [1 sign][8 exp][7 mantissa] in 16 bits.
// FP32 format: [1 sign][8 exp][23 mantissa] in 32 bits.
//
// Conversion:
//   fp32_bits = bf16_val << 16
//
// BF16 is simply the top 16 bits of FP32, so left-shifting by 16
// places the sign, exponent, and truncated mantissa into the correct
// FP32 bit positions with the lower 16 mantissa bits zeroed.
//
// Data is stored little-endian: low byte at [i*2], high byte at [i*2+1].
// ---------------------------------------------------------------------------
std::vector<float> bf16_to_fp32(std::span<const uint8_t> bf16_data) {
    // 1. Compute element count from byte length
    const size_t n = bf16_data.size() / 2;
    std::vector<float> result(n);

    for (size_t i = 0; i < n; ++i) {
        // 2. Reconstruct BF16 value from little-endian bytes
        uint16_t bf16_val = static_cast<uint16_t>(bf16_data[i * 2]) |
                            (static_cast<uint16_t>(bf16_data[i * 2 + 1]) << 8);
        // 3. Place in upper 16 bits of float32
        uint32_t fp32_bits = static_cast<uint32_t>(bf16_val) << 16;
        float f;
        std::memcpy(&f, &fp32_bits, sizeof(f));
        result[i] = f;
    }
    return result;
}

// ---------------------------------------------------------------------------
// fp32_to_bf16
//
// Round-to-nearest-even truncation from FP32 to BF16:
//   rounding_bias = (fp32_bits >> 16) & 1
//   fp32_bits += 0x7FFF + rounding_bias
//
// The 0x7FFF addend pushes bit 16 up when the lower 16 bits exceed
// the midpoint. Adding the LSB of the result (rounding_bias) implements
// round-to-even: ties round to the nearest value whose bit 16 is 0.
//
// After rounding, the upper 16 bits are the BF16 representation.
// Output is stored little-endian: low byte first, high byte second.
// ---------------------------------------------------------------------------
std::vector<uint8_t> fp32_to_bf16(std::span<const float> fp32_data) {
    // 1. Allocate output buffer (2 bytes per element)
    std::vector<uint8_t> result(fp32_data.size() * 2);

    for (size_t i = 0; i < fp32_data.size(); ++i) {
        // 2. Reinterpret float bits as uint32
        uint32_t fp32_bits;
        std::memcpy(&fp32_bits, &fp32_data[i], sizeof(fp32_bits));
        // 3. Round to nearest even: add 0x7FFF + bit 16 for round-to-even
        uint32_t rounding_bias = (fp32_bits >> 16) & 1;
        fp32_bits += 0x7FFF + rounding_bias;
        uint16_t bf16_val = static_cast<uint16_t>(fp32_bits >> 16);
        // 4. Store little-endian
        result[i * 2]     = static_cast<uint8_t>(bf16_val & 0xFF);
        result[i * 2 + 1] = static_cast<uint8_t>(bf16_val >> 8);
    }
    return result;
}

// ===========================================================================
// INT8 Quantization
// ===========================================================================

// ---------------------------------------------------------------------------
// quantize_int8
//
// Per-channel symmetric quantization:
//   scale = max(|x_channel|) / 127
//   q = clamp(round(x / scale), -127, 127)
//
// Each output channel (dimension 0) gets its own scale factor.
// The signed 8-bit range [-127, 127] is used (not [-128, 127]) to keep
// the zero-point exactly at 0 and maintain symmetry.
//
// Skip conditions — tensors kept as BF16 without quantization:
//   - 1D tensors (shape.size() < 2): biases, layer norms
//   - Small tensors (< 128 elements): overhead not worth it
// ---------------------------------------------------------------------------
Result<std::vector<QuantizedTensor>> quantize_int8(
    const std::vector<training::TensorEntry>& tensors) {

    // 1. Reserve output vector
    std::vector<QuantizedTensor> result;
    result.reserve(tensors.size());

    for (const auto& tensor : tensors) {
        QuantizedTensor qt;
        qt.name = tensor.name;
        qt.shape = tensor.shape;

        // 2. Convert BF16 source data to FP32 for quantization math
        std::vector<float> fp32 = bf16_to_fp32(tensor.data);

        if (fp32.empty()) {
            qt.format = QuantFormat::BF16;
            qt.data = tensor.data;
            result.push_back(std::move(qt));
            continue;
        }

        // 3. Skip 1D or small tensors — keep as BF16
        int64_t numel = static_cast<int64_t>(fp32.size());
        if (tensor.shape.size() < 2 || numel < 128) {
            qt.format = QuantFormat::BF16;
            qt.data = tensor.data;
            result.push_back(std::move(qt));
            continue;
        }

        qt.format = QuantFormat::INT8;

        // 4. Compute per-channel dimensions
        int64_t n_channels = tensor.shape[0];
        int64_t channel_size = numel / n_channels;

        qt.int8_params.resize(n_channels);
        qt.data.resize(numel);

        for (int64_t c = 0; c < n_channels; ++c) {
            const float* channel_data = fp32.data() + c * channel_size;

            // 5. Find max absolute value in channel
            float max_abs = 0.0f;
            for (int64_t i = 0; i < channel_size; ++i) {
                float absval = std::fabs(channel_data[i]);
                if (absval > max_abs) max_abs = absval;
            }

            // 6. Compute scale, guard against zero
            float scale = max_abs / 127.0f;
            if (scale == 0.0f) scale = 1.0f;  // Avoid division by zero

            qt.int8_params[c].scale = scale;

            // 7. Quantize each element: q = clamp(round(x / scale), -127, 127)
            float inv_scale = 1.0f / scale;
            for (int64_t i = 0; i < channel_size; ++i) {
                float val = channel_data[i] * inv_scale;
                val = std::clamp(val, -127.0f, 127.0f);
                int8_t q = static_cast<int8_t>(std::round(val));
                qt.data[c * channel_size + i] = static_cast<uint8_t>(q);
            }
        }

        result.push_back(std::move(qt));
    }

    return Result<std::vector<QuantizedTensor>>::ok(std::move(result));
}

// ===========================================================================
// INT4 Quantization
// ===========================================================================

// ---------------------------------------------------------------------------
// quantize_int4
//
// Per-group asymmetric quantization (GROUP_SIZE elements per group):
//   scale = (max - min) / 15
//   q = clamp(round((x - min) / scale), 0, 15)
//
// The unsigned 4-bit range [0, 15] is used with a zero-point offset
// (stored as the group minimum) to handle asymmetric distributions.
//
// Packing: two INT4 values per byte.
//   - Even indices → low nibble  (bits 0-3)
//   - Odd  indices → high nibble (bits 4-7)
//
// Skip conditions (same as INT8):
//   - 1D tensors (shape.size() < 2)
//   - Small tensors (< 128 elements)
// ---------------------------------------------------------------------------
Result<std::vector<QuantizedTensor>> quantize_int4(
    const std::vector<training::TensorEntry>& tensors) {

    // 1. Reserve output vector
    std::vector<QuantizedTensor> result;
    result.reserve(tensors.size());

    for (const auto& tensor : tensors) {
        QuantizedTensor qt;
        qt.name = tensor.name;
        qt.shape = tensor.shape;

        // 2. Convert BF16 source data to FP32
        std::vector<float> fp32 = bf16_to_fp32(tensor.data);

        if (fp32.empty()) {
            qt.format = QuantFormat::BF16;
            qt.data = tensor.data;
            result.push_back(std::move(qt));
            continue;
        }

        int64_t numel = static_cast<int64_t>(fp32.size());

        // 3. Skip 1D or small tensors — keep as BF16
        if (tensor.shape.size() < 2 || numel < 128) {
            qt.format = QuantFormat::BF16;
            qt.data = tensor.data;
            result.push_back(std::move(qt));
            continue;
        }

        qt.format = QuantFormat::INT4;

        // 4. Compute group count
        constexpr int GROUP_SIZE = Int4QuantParams::GROUP_SIZE;
        int64_t n_groups = (numel + GROUP_SIZE - 1) / GROUP_SIZE;

        qt.int4_params.resize(n_groups);
        // 5. Allocate packed output: 2 INT4 values per byte
        qt.data.resize((numel + 1) / 2, 0);

        for (int64_t g = 0; g < n_groups; ++g) {
            int64_t start = g * GROUP_SIZE;
            int64_t end = std::min(start + GROUP_SIZE, numel);

            // 6. Find min/max in group
            float min_val = fp32[start];
            float max_val = fp32[start];
            for (int64_t i = start + 1; i < end; ++i) {
                min_val = std::min(min_val, fp32[i]);
                max_val = std::max(max_val, fp32[i]);
            }

            // 7. Compute scale and zero-point
            float range = max_val - min_val;
            float scale = range / 15.0f;  // 4-bit unsigned: 0..15
            if (scale == 0.0f) scale = 1.0f;

            qt.int4_params[g].scale = scale;
            qt.int4_params[g].zero = min_val;

            // 8. Quantize and pack each element
            float inv_scale = 1.0f / scale;

            for (int64_t i = start; i < end; ++i) {
                float normalized = (fp32[i] - min_val) * inv_scale;
                normalized = std::clamp(normalized, 0.0f, 15.0f);
                uint8_t q4 = static_cast<uint8_t>(std::round(normalized));

                // 9. Pack: even indices in low nibble, odd in high nibble
                size_t byte_idx = i / 2;
                if (i % 2 == 0) {
                    qt.data[byte_idx] = (qt.data[byte_idx] & 0xF0) | (q4 & 0x0F);
                } else {
                    qt.data[byte_idx] = (qt.data[byte_idx] & 0x0F) | ((q4 & 0x0F) << 4);
                }
            }
        }

        result.push_back(std::move(qt));
    }

    return Result<std::vector<QuantizedTensor>>::ok(std::move(result));
}

// ===========================================================================
// Dequantization
// ===========================================================================

// ---------------------------------------------------------------------------
// dequantize_to_fp32
//
// Reverse operations for each format:
//   BF16: Direct conversion via bf16_to_fp32 (lossless bit-shift).
//   INT8: x = q * scale            (per-channel symmetric)
//   INT4: x = q * scale + zero     (per-group asymmetric)
//
// The INT8 path iterates per-channel (dimension 0), applying each
// channel's scale factor. The INT4 path iterates per-element, looking
// up group parameters and unpacking nibbles from the packed byte array.
// ---------------------------------------------------------------------------
Result<std::vector<float>> dequantize_to_fp32(const QuantizedTensor& tensor) {
    // 1. Compute total element count from shape
    int64_t numel = 1;
    for (auto d : tensor.shape) numel *= d;

    if (numel <= 0) {
        return Result<std::vector<float>>::ok({});
    }

    switch (tensor.format) {
        case QuantFormat::BF16: {
            // 2. BF16 — direct bit-shift conversion
            auto fp32 = bf16_to_fp32(tensor.data);
            return Result<std::vector<float>>::ok(std::move(fp32));
        }

        case QuantFormat::INT8: {
            // 3. INT8 — per-channel dequantization: x = q * scale
            if (tensor.shape.empty()) {
                return Result<std::vector<float>>::err("INT8 tensor has no shape");
            }
            int64_t n_channels = tensor.shape[0];
            int64_t channel_size = numel / n_channels;

            std::vector<float> result(numel);
            for (int64_t c = 0; c < n_channels; ++c) {
                float scale = tensor.int8_params[c].scale;
                for (int64_t i = 0; i < channel_size; ++i) {
                    int8_t q = static_cast<int8_t>(tensor.data[c * channel_size + i]);
                    result[c * channel_size + i] = static_cast<float>(q) * scale;
                }
            }
            return Result<std::vector<float>>::ok(std::move(result));
        }

        case QuantFormat::INT4: {
            // 4. INT4 — per-group dequantization: x = q * scale + zero
            constexpr int GROUP_SIZE = Int4QuantParams::GROUP_SIZE;
            std::vector<float> result(numel);

            for (int64_t i = 0; i < numel; ++i) {
                int64_t group = i / GROUP_SIZE;
                float scale = tensor.int4_params[group].scale;
                float zero = tensor.int4_params[group].zero;

                // 5. Unpack nibble: even → low, odd → high
                size_t byte_idx = i / 2;
                uint8_t q4;
                if (i % 2 == 0) {
                    q4 = tensor.data[byte_idx] & 0x0F;
                } else {
                    q4 = (tensor.data[byte_idx] >> 4) & 0x0F;
                }

                result[i] = static_cast<float>(q4) * scale + zero;
            }
            return Result<std::vector<float>>::ok(std::move(result));
        }
    }

    return Result<std::vector<float>>::err("Unknown quantization format");
}

// ===========================================================================
// Size Estimation
// ===========================================================================

// ---------------------------------------------------------------------------
// estimate_quantized_size
//
// Memory estimation per format:
//   BF16: 2 bytes per element (no overhead)
//   INT8: 1 byte per element + per-channel scale floats
//   INT4: 0.5 bytes per element + per-group (scale, zero) floats
//
// Small/1D tensors are assumed to remain BF16 (2 bytes each) regardless
// of the target format, matching the skip logic in quantize_int8/int4.
// ---------------------------------------------------------------------------
size_t estimate_quantized_size(const std::vector<training::TensorEntry>& tensors,
                               QuantFormat format) {
    size_t total = 0;
    for (const auto& tensor : tensors) {
        // 1. Compute element count
        int64_t numel = 1;
        for (auto d : tensor.shape) numel *= d;

        // 2. Check skip conditions
        bool is_small = (tensor.shape.size() < 2 || numel < 128);

        switch (format) {
            case QuantFormat::BF16:
                // 3. BF16: 2 bytes per element
                total += numel * 2;
                break;
            case QuantFormat::INT8:
                if (is_small) {
                    total += numel * 2;
                } else {
                    // 4. INT8: 1 byte per element + per-channel scales
                    total += numel;
                    total += tensor.shape[0] * sizeof(Int8QuantParams);
                }
                break;
            case QuantFormat::INT4:
                if (is_small) {
                    total += numel * 2;
                } else {
                    // 5. INT4: 0.5 bytes per element + per-group params
                    total += (numel + 1) / 2;
                    int64_t n_groups = (numel + Int4QuantParams::GROUP_SIZE - 1)
                                       / Int4QuantParams::GROUP_SIZE;
                    total += n_groups * sizeof(Int4QuantParams);
                }
                break;
        }
    }
    return total;
}

} // namespace rnet::inference

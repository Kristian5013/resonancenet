#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/error.h"
#include "training/checkpoint_io.h"

namespace rnet::inference {

/// Quantization target formats.
enum class QuantFormat : uint8_t {
    BF16 = 0,   ///< No quantization (original format)
    INT8 = 1,   ///< 8-bit per-channel symmetric quantization
    INT4 = 2,   ///< 4-bit per-group quantization (group size 128)
};

/// Returns human-readable name for a quantization format.
const char* quant_format_name(QuantFormat fmt);

/// Per-channel INT8 quantization parameters.
struct Int8QuantParams {
    float scale = 1.0f;   ///< scale factor: float_val = int8_val * scale
};

/// Per-group INT4 quantization parameters.
struct Int4QuantParams {
    static constexpr int GROUP_SIZE = 128;
    float scale = 1.0f;   ///< Per-group scale
    float zero = 0.0f;    ///< Per-group zero-point
};

/// A quantized tensor: metadata + compressed data.
struct QuantizedTensor {
    std::string name;
    std::vector<int64_t> shape;
    QuantFormat format = QuantFormat::BF16;
    std::vector<uint8_t> data;               ///< Quantized data bytes
    std::vector<Int8QuantParams> int8_params; ///< One per output channel (INT8)
    std::vector<Int4QuantParams> int4_params; ///< One per group (INT4)
};

/// Quantize a full checkpoint to INT8 format.
/// Uses per-channel symmetric quantization: scale = max(abs(channel)) / 127.
Result<std::vector<QuantizedTensor>> quantize_int8(
    const std::vector<training::TensorEntry>& tensors);

/// Quantize a full checkpoint to INT4 format.
/// Uses per-group asymmetric quantization with group size 128.
Result<std::vector<QuantizedTensor>> quantize_int4(
    const std::vector<training::TensorEntry>& tensors);

/// Dequantize a single tensor back to FP32.
Result<std::vector<float>> dequantize_to_fp32(const QuantizedTensor& tensor);

/// Convert BF16 raw bytes to FP32 vector.
std::vector<float> bf16_to_fp32(std::span<const uint8_t> bf16_data);

/// Convert FP32 vector to BF16 raw bytes.
std::vector<uint8_t> fp32_to_bf16(std::span<const float> fp32_data);

/// Estimate model size in bytes for a given quantization format.
size_t estimate_quantized_size(const std::vector<training::TensorEntry>& tensors,
                               QuantFormat format);

}  // namespace rnet::inference

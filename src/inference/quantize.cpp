#include "inference/quantize.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace rnet::inference {

const char* quant_format_name(QuantFormat fmt) {
    switch (fmt) {
        case QuantFormat::BF16: return "bf16";
        case QuantFormat::INT8: return "int8";
        case QuantFormat::INT4: return "int4";
    }
    return "unknown";
}

// BF16 is stored as: [1 sign][8 exponent][7 mantissa] in 16 bits
// FP32 is stored as: [1 sign][8 exponent][23 mantissa] in 32 bits
// BF16 is simply the top 16 bits of FP32 (with rounding).

std::vector<float> bf16_to_fp32(std::span<const uint8_t> bf16_data) {
    const size_t n = bf16_data.size() / 2;
    std::vector<float> result(n);

    for (size_t i = 0; i < n; ++i) {
        // BF16 stored little-endian
        uint16_t bf16_val = static_cast<uint16_t>(bf16_data[i * 2]) |
                            (static_cast<uint16_t>(bf16_data[i * 2 + 1]) << 8);
        // Place in upper 16 bits of float32
        uint32_t fp32_bits = static_cast<uint32_t>(bf16_val) << 16;
        float f;
        std::memcpy(&f, &fp32_bits, sizeof(f));
        result[i] = f;
    }
    return result;
}

std::vector<uint8_t> fp32_to_bf16(std::span<const float> fp32_data) {
    std::vector<uint8_t> result(fp32_data.size() * 2);

    for (size_t i = 0; i < fp32_data.size(); ++i) {
        uint32_t fp32_bits;
        std::memcpy(&fp32_bits, &fp32_data[i], sizeof(fp32_bits));
        // Round to nearest even: add 0x7FFF + bit 16 for round-to-even
        uint32_t rounding_bias = (fp32_bits >> 16) & 1;
        fp32_bits += 0x7FFF + rounding_bias;
        uint16_t bf16_val = static_cast<uint16_t>(fp32_bits >> 16);
        // Store little-endian
        result[i * 2]     = static_cast<uint8_t>(bf16_val & 0xFF);
        result[i * 2 + 1] = static_cast<uint8_t>(bf16_val >> 8);
    }
    return result;
}

Result<std::vector<QuantizedTensor>> quantize_int8(
    const std::vector<training::TensorEntry>& tensors) {

    std::vector<QuantizedTensor> result;
    result.reserve(tensors.size());

    for (const auto& tensor : tensors) {
        QuantizedTensor qt;
        qt.name = tensor.name;
        qt.shape = tensor.shape;

        // Convert BF16 to FP32
        std::vector<float> fp32 = bf16_to_fp32(tensor.data);

        if (fp32.empty()) {
            qt.format = QuantFormat::BF16;
            qt.data = tensor.data;
            result.push_back(std::move(qt));
            continue;
        }

        // 1D or small tensors: keep as BF16
        int64_t numel = static_cast<int64_t>(fp32.size());
        if (tensor.shape.size() < 2 || numel < 128) {
            qt.format = QuantFormat::BF16;
            qt.data = tensor.data;
            result.push_back(std::move(qt));
            continue;
        }

        qt.format = QuantFormat::INT8;

        // Per-channel quantization along the first dimension
        int64_t n_channels = tensor.shape[0];
        int64_t channel_size = numel / n_channels;

        qt.int8_params.resize(n_channels);
        qt.data.resize(numel);

        for (int64_t c = 0; c < n_channels; ++c) {
            const float* channel_data = fp32.data() + c * channel_size;

            // Find max absolute value in channel
            float max_abs = 0.0f;
            for (int64_t i = 0; i < channel_size; ++i) {
                float absval = std::fabs(channel_data[i]);
                if (absval > max_abs) max_abs = absval;
            }

            float scale = max_abs / 127.0f;
            if (scale == 0.0f) scale = 1.0f;  // Avoid division by zero

            qt.int8_params[c].scale = scale;

            // Quantize
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

Result<std::vector<QuantizedTensor>> quantize_int4(
    const std::vector<training::TensorEntry>& tensors) {

    std::vector<QuantizedTensor> result;
    result.reserve(tensors.size());

    for (const auto& tensor : tensors) {
        QuantizedTensor qt;
        qt.name = tensor.name;
        qt.shape = tensor.shape;

        std::vector<float> fp32 = bf16_to_fp32(tensor.data);

        if (fp32.empty()) {
            qt.format = QuantFormat::BF16;
            qt.data = tensor.data;
            result.push_back(std::move(qt));
            continue;
        }

        int64_t numel = static_cast<int64_t>(fp32.size());

        // 1D or small tensors: keep as BF16
        if (tensor.shape.size() < 2 || numel < 128) {
            qt.format = QuantFormat::BF16;
            qt.data = tensor.data;
            result.push_back(std::move(qt));
            continue;
        }

        qt.format = QuantFormat::INT4;

        constexpr int GROUP_SIZE = Int4QuantParams::GROUP_SIZE;
        int64_t n_groups = (numel + GROUP_SIZE - 1) / GROUP_SIZE;

        qt.int4_params.resize(n_groups);
        // Pack 2 INT4 values per byte
        qt.data.resize((numel + 1) / 2, 0);

        for (int64_t g = 0; g < n_groups; ++g) {
            int64_t start = g * GROUP_SIZE;
            int64_t end = std::min(start + GROUP_SIZE, numel);

            // Find min/max in group
            float min_val = fp32[start];
            float max_val = fp32[start];
            for (int64_t i = start + 1; i < end; ++i) {
                min_val = std::min(min_val, fp32[i]);
                max_val = std::max(max_val, fp32[i]);
            }

            float range = max_val - min_val;
            float scale = range / 15.0f;  // 4-bit unsigned: 0..15
            if (scale == 0.0f) scale = 1.0f;

            qt.int4_params[g].scale = scale;
            qt.int4_params[g].zero = min_val;

            float inv_scale = 1.0f / scale;

            for (int64_t i = start; i < end; ++i) {
                float normalized = (fp32[i] - min_val) * inv_scale;
                normalized = std::clamp(normalized, 0.0f, 15.0f);
                uint8_t q4 = static_cast<uint8_t>(std::round(normalized));

                // Pack: even indices in low nibble, odd in high nibble
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

Result<std::vector<float>> dequantize_to_fp32(const QuantizedTensor& tensor) {
    int64_t numel = 1;
    for (auto d : tensor.shape) numel *= d;

    if (numel <= 0) {
        return Result<std::vector<float>>::ok({});
    }

    switch (tensor.format) {
        case QuantFormat::BF16: {
            auto fp32 = bf16_to_fp32(tensor.data);
            return Result<std::vector<float>>::ok(std::move(fp32));
        }

        case QuantFormat::INT8: {
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
            constexpr int GROUP_SIZE = Int4QuantParams::GROUP_SIZE;
            std::vector<float> result(numel);

            for (int64_t i = 0; i < numel; ++i) {
                int64_t group = i / GROUP_SIZE;
                float scale = tensor.int4_params[group].scale;
                float zero = tensor.int4_params[group].zero;

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

size_t estimate_quantized_size(const std::vector<training::TensorEntry>& tensors,
                               QuantFormat format) {
    size_t total = 0;
    for (const auto& tensor : tensors) {
        int64_t numel = 1;
        for (auto d : tensor.shape) numel *= d;

        bool is_small = (tensor.shape.size() < 2 || numel < 128);

        switch (format) {
            case QuantFormat::BF16:
                total += numel * 2;  // 2 bytes per element
                break;
            case QuantFormat::INT8:
                if (is_small) {
                    total += numel * 2;
                } else {
                    total += numel;  // 1 byte per element
                    total += tensor.shape[0] * sizeof(Int8QuantParams);  // scales
                }
                break;
            case QuantFormat::INT4:
                if (is_small) {
                    total += numel * 2;
                } else {
                    total += (numel + 1) / 2;  // 0.5 bytes per element
                    int64_t n_groups = (numel + Int4QuantParams::GROUP_SIZE - 1)
                                       / Int4QuantParams::GROUP_SIZE;
                    total += n_groups * sizeof(Int4QuantParams);  // group params
                }
                break;
        }
    }
    return total;
}

}  // namespace rnet::inference

#pragma once

#include "backend.h"
#include <vector>
#include <cstdint>
#include <numeric>

namespace rnet::gpu {

/// Device memory wrapper with shape and dtype metadata.
class GpuTensor {
public:
    GpuTensor(GpuBackend& backend, std::vector<int64_t> shape, DType dtype);
    ~GpuTensor();

    void* data();
    const void* data() const;
    const std::vector<int64_t>& shape() const;
    int64_t numel() const;
    size_t size_bytes() const;
    DType dtype() const;

    void copy_from_host(const void* src);
    void copy_to_host(void* dst) const;

    GpuTensor(const GpuTensor&) = delete;
    GpuTensor& operator=(const GpuTensor&) = delete;
    GpuTensor(GpuTensor&& other) noexcept;
    GpuTensor& operator=(GpuTensor&& other) noexcept;

private:
    GpuBackend* backend_ = nullptr;
    void* data_ = nullptr;
    std::vector<int64_t> shape_;
    DType dtype_ = DType::FP32;
    int64_t numel_ = 0;
    size_t size_bytes_ = 0;
};

}  // namespace rnet::gpu

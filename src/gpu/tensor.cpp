#include "tensor.h"
#include <algorithm>
#include <utility>

namespace rnet::gpu {

GpuTensor::GpuTensor(GpuBackend& backend, std::vector<int64_t> shape, DType dtype)
    : backend_(&backend)
    , shape_(std::move(shape))
    , dtype_(dtype)
{
    numel_ = 1;
    for (auto dim : shape_) {
        numel_ *= dim;
    }
    size_bytes_ = static_cast<size_t>(numel_) * dtype_size(dtype_);
    if (size_bytes_ > 0) {
        data_ = backend_->alloc(size_bytes_);
    }
}

GpuTensor::~GpuTensor() {
    if (data_ && backend_) {
        backend_->free(data_);
    }
}

GpuTensor::GpuTensor(GpuTensor&& other) noexcept
    : backend_(other.backend_)
    , data_(other.data_)
    , shape_(std::move(other.shape_))
    , dtype_(other.dtype_)
    , numel_(other.numel_)
    , size_bytes_(other.size_bytes_)
{
    other.backend_ = nullptr;
    other.data_ = nullptr;
    other.numel_ = 0;
    other.size_bytes_ = 0;
}

GpuTensor& GpuTensor::operator=(GpuTensor&& other) noexcept {
    if (this != &other) {
        if (data_ && backend_) {
            backend_->free(data_);
        }
        backend_ = other.backend_;
        data_ = other.data_;
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        numel_ = other.numel_;
        size_bytes_ = other.size_bytes_;
        other.backend_ = nullptr;
        other.data_ = nullptr;
        other.numel_ = 0;
        other.size_bytes_ = 0;
    }
    return *this;
}

void* GpuTensor::data() { return data_; }
const void* GpuTensor::data() const { return data_; }
std::span<const int64_t> GpuTensor::shape() const { return shape_; }
int64_t GpuTensor::numel() const { return numel_; }
size_t GpuTensor::size_bytes() const { return size_bytes_; }
DType GpuTensor::dtype() const { return dtype_; }

void GpuTensor::copy_from_host(const void* src) {
    if (data_ && backend_ && size_bytes_ > 0) {
        backend_->copy_to_device(data_, src, size_bytes_);
    }
}

void GpuTensor::copy_to_host(void* dst) const {
    if (data_ && backend_ && size_bytes_ > 0) {
        backend_->copy_to_host(dst, data_, size_bytes_);
    }
}

}  // namespace rnet::gpu

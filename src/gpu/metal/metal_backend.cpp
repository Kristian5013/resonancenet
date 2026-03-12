#ifdef RNET_HAS_METAL

#include "metal_backend.h"
#include "../../core/logging.h"

namespace rnet::gpu {

MetalBackend::MetalBackend() {
    device_name_ = "Metal Device (stub)";
    LogPrintf("GPU: initialized Metal backend (stub) — %s", device_name_.c_str());
}

MetalBackend::~MetalBackend() = default;

std::string MetalBackend::device_name() const { return device_name_; }
size_t MetalBackend::total_memory() const { return 0; }
size_t MetalBackend::free_memory() const { return 0; }
GpuBackendType MetalBackend::type() const { return GpuBackendType::METAL; }

void* MetalBackend::alloc(size_t /*bytes*/) {
    LogError("Metal backend stub: alloc not implemented");
    return nullptr;
}

void MetalBackend::free(void* /*ptr*/) {
    LogError("Metal backend stub: free not implemented");
}

void MetalBackend::copy_to_device(void*, const void*, size_t) {
    LogError("Metal backend stub: copy_to_device not implemented");
}

void MetalBackend::copy_to_host(void*, const void*, size_t) {
    LogError("Metal backend stub: copy_to_host not implemented");
}

void MetalBackend::synchronize() {
    // No-op stub
}

#define METAL_KERNEL_STUB(name) \
    LogPrint(TRAINING, "Metal kernel stub: " #name " called — not yet implemented")

void MetalBackend::embedding_forward(void*, const void*, const int*, int, int, int) {
    METAL_KERNEL_STUB(embedding_forward);
}

void MetalBackend::rmsnorm_forward(void*, const void*, const void*, int, int, int, float) {
    METAL_KERNEL_STUB(rmsnorm_forward);
}

void MetalBackend::causal_conv_forward(void*, const void*, const void*, const int*, int, int, int, int) {
    METAL_KERNEL_STUB(causal_conv_forward);
}

void MetalBackend::mingru_forward(void*, void*, const void*, const void*, const void*, const void*, int, int, int) {
    METAL_KERNEL_STUB(mingru_forward);
}

void MetalBackend::slot_memory_forward(void*, const void*, const void*, const void*, int, int, int, int) {
    METAL_KERNEL_STUB(slot_memory_forward);
}

void MetalBackend::swiglu_forward(void*, const void*, const void*, const void*, const void*, int, int, int, int) {
    METAL_KERNEL_STUB(swiglu_forward);
}

void MetalBackend::cross_entropy_loss(float* loss_out, const void*, const int*, int, int, int) {
    METAL_KERNEL_STUB(cross_entropy_loss);
    if (loss_out) *loss_out = 0.0f;
}

void MetalBackend::adamw_step(void*, const void*, void*, void*, float, float, float, float, float, int, int) {
    METAL_KERNEL_STUB(adamw_step);
}

void MetalBackend::gemm(void*, const void*, const void*, int, int, int, float, float) {
    METAL_KERNEL_STUB(gemm);
}

void MetalBackend::mingru_step(void*, const void*, const void*, const void*, const void*, int) {
    METAL_KERNEL_STUB(mingru_step);
}

void MetalBackend::conv_step(void*, void*, const void*, const void*, const int*, int, int) {
    METAL_KERNEL_STUB(conv_step);
}

void MetalBackend::slot_query(void*, const void*, const void*, const void*, int, int) {
    METAL_KERNEL_STUB(slot_query);
}

#undef METAL_KERNEL_STUB

}  // namespace rnet::gpu

#endif  // RNET_HAS_METAL

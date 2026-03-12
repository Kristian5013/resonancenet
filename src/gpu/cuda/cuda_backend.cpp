#ifdef RNET_HAS_CUDA

#include "cuda_backend.h"
#include "../../core/logging.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace rnet::gpu {

CudaBackend::CudaBackend() {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        device_name_ = prop.name;
        total_mem_ = prop.totalGlobalMem;
    } else {
        device_name_ = "CUDA Device (init failed)";
    }
    LogPrintf("GPU: initialized CUDA backend — %s", device_name_.c_str());
}

CudaBackend::~CudaBackend() = default;

std::string CudaBackend::device_name() const { return device_name_; }
size_t CudaBackend::total_memory() const { return total_mem_; }

size_t CudaBackend::free_memory() const {
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    return free_bytes;
}

GpuBackendType CudaBackend::type() const { return GpuBackendType::CUDA; }

void* CudaBackend::alloc(size_t bytes) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        LogError("CUDA alloc failed: %s", cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

void CudaBackend::free(void* ptr) {
    if (ptr) cudaFree(ptr);
}

void CudaBackend::copy_to_device(void* dst, const void* src, size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}

void CudaBackend::copy_to_host(void* dst, const void* src, size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}

void CudaBackend::synchronize() {
    cudaDeviceSynchronize();
}

// ── Stub kernel implementations ───────────────────────────────────────
// All training/inference kernels log a warning — real .cu kernels added later.

#define CUDA_KERNEL_STUB(name) \
    LogPrint(TRAINING, "CUDA kernel stub: " #name " called — not yet implemented")

void CudaBackend::embedding_forward(void*, const void*, const int*, int, int, int) {
    CUDA_KERNEL_STUB(embedding_forward);
}

void CudaBackend::rmsnorm_forward(void*, const void*, const void*, int, int, int, float) {
    CUDA_KERNEL_STUB(rmsnorm_forward);
}

void CudaBackend::causal_conv_forward(void*, const void*, const void*, const int*, int, int, int, int) {
    CUDA_KERNEL_STUB(causal_conv_forward);
}

void CudaBackend::mingru_forward(void*, void*, const void*, const void*, const void*, const void*, int, int, int) {
    CUDA_KERNEL_STUB(mingru_forward);
}

void CudaBackend::slot_memory_forward(void*, const void*, const void*, const void*, int, int, int, int) {
    CUDA_KERNEL_STUB(slot_memory_forward);
}

void CudaBackend::swiglu_forward(void*, const void*, const void*, const void*, const void*, int, int, int, int) {
    CUDA_KERNEL_STUB(swiglu_forward);
}

void CudaBackend::cross_entropy_loss(float* loss_out, const void*, const int*, int, int, int) {
    CUDA_KERNEL_STUB(cross_entropy_loss);
    if (loss_out) *loss_out = 0.0f;
}

void CudaBackend::adamw_step(void*, const void*, void*, void*, float, float, float, float, float, int, int) {
    CUDA_KERNEL_STUB(adamw_step);
}

void CudaBackend::gemm(void*, const void*, const void*, int, int, int, float, float) {
    CUDA_KERNEL_STUB(gemm);
}

void CudaBackend::mingru_step(void*, const void*, const void*, const void*, const void*, int) {
    CUDA_KERNEL_STUB(mingru_step);
}

void CudaBackend::conv_step(void*, void*, const void*, const void*, const int*, int, int) {
    CUDA_KERNEL_STUB(conv_step);
}

void CudaBackend::slot_query(void*, const void*, const void*, const void*, int, int) {
    CUDA_KERNEL_STUB(slot_query);
}

#undef CUDA_KERNEL_STUB

}  // namespace rnet::gpu

#endif  // RNET_HAS_CUDA

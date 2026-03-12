#include "backend.h"
#include "../core/logging.h"

#ifdef RNET_HAS_CUDA
#include "cuda/cuda_backend.h"
#endif

#ifdef RNET_HAS_METAL
#include "metal/metal_backend.h"
#endif

#ifdef RNET_HAS_VULKAN
#include "vulkan/vulkan_backend.h"
#endif

#include "cpu/cpu_backend.h"

namespace rnet::gpu {

std::unique_ptr<GpuBackend> GpuBackend::create(GpuBackendType type) {
    switch (type) {
#ifdef RNET_HAS_CUDA
        case GpuBackendType::CUDA:
            return std::make_unique<CudaBackend>();
#endif
#ifdef RNET_HAS_METAL
        case GpuBackendType::METAL:
            return std::make_unique<MetalBackend>();
#endif
#ifdef RNET_HAS_VULKAN
        case GpuBackendType::VULKAN:
            return std::make_unique<VulkanBackend>();
#endif
        case GpuBackendType::CPU_FALLBACK:
            return std::make_unique<CpuFallbackBackend>();
        default:
            LogPrintf("GPU: requested backend %s not available, falling back to CPU",
                      backend_type_name(type));
            return std::make_unique<CpuFallbackBackend>();
    }
}

GpuBackendType GpuBackend::auto_detect() {
#ifdef RNET_HAS_CUDA
    return GpuBackendType::CUDA;
#elif defined(RNET_HAS_METAL)
    return GpuBackendType::METAL;
#elif defined(RNET_HAS_VULKAN)
    return GpuBackendType::VULKAN;
#else
    return GpuBackendType::CPU_FALLBACK;
#endif
}

std::unique_ptr<GpuBackend> GpuBackend::create_best() {
    return create(auto_detect());
}

}  // namespace rnet::gpu

// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

// Own header.
#include "backend.h"

// Project headers.
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

// ---------------------------------------------------------------------------
// GpuBackend::create
// ---------------------------------------------------------------------------
// Factory: instantiates the requested backend type, or falls back to CPU if
// the requested type was not compiled in.
// ---------------------------------------------------------------------------
std::unique_ptr<GpuBackend> GpuBackend::create(GpuBackendType type)
{
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

// ---------------------------------------------------------------------------
// GpuBackend::auto_detect
// ---------------------------------------------------------------------------
// Returns the best available backend type in priority order:
// CUDA > Metal > Vulkan > CPU fallback.
// ---------------------------------------------------------------------------
GpuBackendType GpuBackend::auto_detect()
{
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

// ---------------------------------------------------------------------------
// GpuBackend::create_best
// ---------------------------------------------------------------------------
std::unique_ptr<GpuBackend> GpuBackend::create_best()
{
    return create(auto_detect());
}

} // namespace rnet::gpu

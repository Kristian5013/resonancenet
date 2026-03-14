// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#ifdef RNET_HAS_CUDA

// Own header.
#include "cuda_context.h"

// Project headers.
#include "../../core/logging.h"

// CUDA runtime.
#include <cuda_runtime.h>

namespace rnet::gpu {

// ---------------------------------------------------------------------------
// CudaContext::enumerate
// ---------------------------------------------------------------------------
// Queries CUDA for all available devices and returns their properties
// (name, total VRAM, compute capability).
// ---------------------------------------------------------------------------
std::vector<GpuDeviceInfo> CudaContext::enumerate()
{
    std::vector<GpuDeviceInfo> devices;

    // 1. Query device count.
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        LogPrintf("CUDA: no devices found (%s)",
                  err != cudaSuccess ? cudaGetErrorString(err) : "count=0");
        return devices;
    }

    // 2. Collect properties for each device.
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) continue;

        GpuDeviceInfo info;
        info.name = prop.name;
        info.total_memory = prop.totalGlobalMem;
        info.free_memory = 0; // Would need cudaSetDevice + cudaMemGetInfo
        info.compute_capability_major = prop.major;
        info.compute_capability_minor = prop.minor;
        info.backend_type = GpuBackendType::CUDA;
        devices.push_back(std::move(info));
    }

    LogPrintf("CUDA: found %d device(s)", count);
    return devices;
}

} // namespace rnet::gpu

#endif // RNET_HAS_CUDA

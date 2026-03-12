#pragma once

#include "../context.h"
#include <vector>

#ifdef RNET_HAS_CUDA

namespace rnet::gpu {

/// CUDA device enumeration and context management.
class CudaContext {
public:
    /// Enumerate all available CUDA devices.
    static std::vector<GpuDeviceInfo> enumerate();
};

}  // namespace rnet::gpu

#endif  // RNET_HAS_CUDA

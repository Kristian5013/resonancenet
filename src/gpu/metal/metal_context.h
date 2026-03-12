#pragma once

#include "../context.h"
#include <vector>

#ifdef RNET_HAS_METAL

namespace rnet::gpu {

/// Metal device enumeration.
class MetalContext {
public:
    static std::vector<GpuDeviceInfo> enumerate();
};

}  // namespace rnet::gpu

#endif  // RNET_HAS_METAL

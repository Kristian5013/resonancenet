#pragma once

#include "../context.h"
#include <vector>

#ifdef RNET_HAS_VULKAN

namespace rnet::gpu {

/// Vulkan device enumeration.
class VulkanContext {
public:
    static std::vector<GpuDeviceInfo> enumerate();
};

}  // namespace rnet::gpu

#endif  // RNET_HAS_VULKAN

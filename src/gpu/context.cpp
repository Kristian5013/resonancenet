#include "context.h"
#include "../core/logging.h"

#ifdef RNET_HAS_CUDA
#include "cuda/cuda_context.h"
#endif

#ifdef RNET_HAS_METAL
#include "metal/metal_context.h"
#endif

#ifdef RNET_HAS_VULKAN
#include "vulkan/vulkan_context.h"
#endif

namespace rnet::gpu {

std::vector<GpuDeviceInfo> GpuContext::enumerate_devices() {
    std::vector<GpuDeviceInfo> devices;

#ifdef RNET_HAS_CUDA
    {
        auto cuda_devs = CudaContext::enumerate();
        devices.insert(devices.end(), cuda_devs.begin(), cuda_devs.end());
    }
#endif

#ifdef RNET_HAS_METAL
    {
        auto metal_devs = MetalContext::enumerate();
        devices.insert(devices.end(), metal_devs.begin(), metal_devs.end());
    }
#endif

#ifdef RNET_HAS_VULKAN
    {
        auto vk_devs = VulkanContext::enumerate();
        devices.insert(devices.end(), vk_devs.begin(), vk_devs.end());
    }
#endif

    // Always add a CPU fallback device
    GpuDeviceInfo cpu_info;
    cpu_info.name = "CPU Fallback";
    cpu_info.total_memory = 0;
    cpu_info.free_memory = 0;
    cpu_info.compute_capability_major = 0;
    cpu_info.compute_capability_minor = 0;
    cpu_info.backend_type = GpuBackendType::CPU_FALLBACK;
    devices.push_back(cpu_info);

    return devices;
}

std::unique_ptr<GpuBackend> GpuContext::create_for_device(int device_id) {
    auto devices = enumerate_devices();
    if (device_id < 0 || device_id >= static_cast<int>(devices.size())) {
        LogPrintf("GPU: device_id %d out of range (%d devices), using CPU fallback",
                  device_id, static_cast<int>(devices.size()));
        return GpuBackend::create(GpuBackendType::CPU_FALLBACK);
    }
    return GpuBackend::create(devices[device_id].backend_type);
}

}  // namespace rnet::gpu

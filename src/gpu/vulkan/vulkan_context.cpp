// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#ifdef RNET_HAS_VULKAN

// Own header.
#include "vulkan_context.h"

// Project headers.
#include "../../core/logging.h"

// Standard library.
#include <vulkan/vulkan.h>

namespace rnet::gpu {

// ---------------------------------------------------------------------------
// VulkanContext::enumerate
// ---------------------------------------------------------------------------
// Creates a temporary VkInstance to enumerate all physical devices, collects
// their names, VRAM sizes, and Vulkan API versions, then destroys the
// instance.
// ---------------------------------------------------------------------------
std::vector<GpuDeviceInfo> VulkanContext::enumerate()
{
    std::vector<GpuDeviceInfo> devices;

    // 1. Create a temporary Vulkan instance for enumeration.
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "ResonanceNet";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    VkInstance instance = VK_NULL_HANDLE;
    VkResult result = vkCreateInstance(&create_info, nullptr, &instance);
    if (result != VK_SUCCESS) {
        LogPrintf("Vulkan: failed to create instance for enumeration (VkResult=%d)",
                  static_cast<int>(result));
        return devices;
    }

    // 2. Enumerate physical devices.
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);

    if (device_count > 0) {
        std::vector<VkPhysicalDevice> phys_devices(device_count);
        vkEnumeratePhysicalDevices(instance, &device_count, phys_devices.data());

        for (uint32_t i = 0; i < device_count; ++i) {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(phys_devices[i], &props);

            GpuDeviceInfo info;
            info.name = props.deviceName;
            info.backend_type = GpuBackendType::VULKAN;
            info.compute_capability_major = VK_API_VERSION_MAJOR(props.apiVersion);
            info.compute_capability_minor = VK_API_VERSION_MINOR(props.apiVersion);

            // Sum device-local heap sizes for total memory.
            VkPhysicalDeviceMemoryProperties mem_props{};
            vkGetPhysicalDeviceMemoryProperties(phys_devices[i], &mem_props);
            info.total_memory = 0;
            for (uint32_t h = 0; h < mem_props.memoryHeapCount; ++h) {
                if (mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                    info.total_memory += mem_props.memoryHeaps[h].size;
                }
            }
            info.free_memory = 0;

            devices.push_back(std::move(info));
        }
    }

    // 3. Clean up.
    vkDestroyInstance(instance, nullptr);
    LogPrintf("Vulkan: found %u device(s)", device_count);
    return devices;
}

} // namespace rnet::gpu

#endif // RNET_HAS_VULKAN

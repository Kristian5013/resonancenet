#ifdef RNET_HAS_VULKAN

#include "vulkan_backend.h"
#include "../../core/logging.h"

#include <vulkan/vulkan.h>

namespace rnet::gpu {

VulkanBackend::VulkanBackend() {
    // Attempt to create a Vulkan instance and query the first physical device
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
        device_name_ = "Vulkan Device (init failed)";
        LogPrintf("GPU: Vulkan instance creation failed (VkResult=%d)", static_cast<int>(result));
        return;
    }

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    if (device_count > 0) {
        VkPhysicalDevice phys_device = VK_NULL_HANDLE;
        vkEnumeratePhysicalDevices(instance, &device_count, &phys_device);

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(phys_device, &props);
        device_name_ = props.deviceName;

        VkPhysicalDeviceMemoryProperties mem_props{};
        vkGetPhysicalDeviceMemoryProperties(phys_device, &mem_props);
        for (uint32_t i = 0; i < mem_props.memoryHeapCount; ++i) {
            if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                total_mem_ += mem_props.memoryHeaps[i].size;
            }
        }
    } else {
        device_name_ = "Vulkan (no physical devices)";
    }

    vkDestroyInstance(instance, nullptr);
    LogPrintf("GPU: initialized Vulkan backend (stub) — %s", device_name_.c_str());
}

VulkanBackend::~VulkanBackend() = default;

std::string VulkanBackend::device_name() const { return device_name_; }
size_t VulkanBackend::total_memory() const { return total_mem_; }
size_t VulkanBackend::free_memory() const { return 0; }  // Vulkan doesn't easily expose free mem
GpuBackendType VulkanBackend::type() const { return GpuBackendType::VULKAN; }

void* VulkanBackend::alloc(size_t /*bytes*/) {
    LogError("Vulkan backend stub: alloc not implemented (need real compute pipeline)");
    return nullptr;
}

void VulkanBackend::free(void* /*ptr*/) {
    LogError("Vulkan backend stub: free not implemented");
}

void VulkanBackend::copy_to_device(void*, const void*, size_t) {
    LogError("Vulkan backend stub: copy_to_device not implemented");
}

void VulkanBackend::copy_to_host(void*, const void*, size_t) {
    LogError("Vulkan backend stub: copy_to_host not implemented");
}

void VulkanBackend::synchronize() {
    // No-op stub
}

#define VULKAN_KERNEL_STUB(name) \
    LogPrint(TRAINING, "Vulkan kernel stub: " #name " called — not yet implemented")

void VulkanBackend::embedding_forward(void*, const void*, const int*, int, int, int) {
    VULKAN_KERNEL_STUB(embedding_forward);
}

void VulkanBackend::rmsnorm_forward(void*, const void*, const void*, int, int, int, float) {
    VULKAN_KERNEL_STUB(rmsnorm_forward);
}

void VulkanBackend::causal_conv_forward(void*, const void*, const void*, const int*, int, int, int, int) {
    VULKAN_KERNEL_STUB(causal_conv_forward);
}

void VulkanBackend::mingru_forward(void*, void*, const void*, const void*, const void*, const void*, int, int, int) {
    VULKAN_KERNEL_STUB(mingru_forward);
}

void VulkanBackend::slot_memory_forward(void*, const void*, const void*, const void*, int, int, int, int) {
    VULKAN_KERNEL_STUB(slot_memory_forward);
}

void VulkanBackend::swiglu_forward(void*, const void*, const void*, const void*, const void*, int, int, int, int) {
    VULKAN_KERNEL_STUB(swiglu_forward);
}

void VulkanBackend::cross_entropy_loss(float* loss_out, const void*, const int*, int, int, int) {
    VULKAN_KERNEL_STUB(cross_entropy_loss);
    if (loss_out) *loss_out = 0.0f;
}

void VulkanBackend::adamw_step(void*, const void*, void*, void*, float, float, float, float, float, int, int) {
    VULKAN_KERNEL_STUB(adamw_step);
}

void VulkanBackend::gemm(void*, const void*, const void*, int, int, int, float, float) {
    VULKAN_KERNEL_STUB(gemm);
}

void VulkanBackend::mingru_step(void*, const void*, const void*, const void*, const void*, int) {
    VULKAN_KERNEL_STUB(mingru_step);
}

void VulkanBackend::conv_step(void*, void*, const void*, const void*, const int*, int, int) {
    VULKAN_KERNEL_STUB(conv_step);
}

void VulkanBackend::slot_query(void*, const void*, const void*, const void*, int, int) {
    VULKAN_KERNEL_STUB(slot_query);
}

#undef VULKAN_KERNEL_STUB

}  // namespace rnet::gpu

#endif  // RNET_HAS_VULKAN

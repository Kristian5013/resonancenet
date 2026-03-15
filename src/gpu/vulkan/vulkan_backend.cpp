// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#ifdef RNET_HAS_VULKAN

// Own header.
#include "vulkan_backend.h"

// Project headers.
#include "../../core/logging.h"

// Standard library.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace rnet::gpu {

// ===========================================================================
//  Helpers
// ===========================================================================

// ---------------------------------------------------------------------------
// find_memory_type
// ---------------------------------------------------------------------------
// Scans the device's memory types for one matching both the required type
// filter mask and the desired property flags.  Returns UINT32_MAX on failure.
// ---------------------------------------------------------------------------
uint32_t VulkanBackend::find_memory_type(uint32_t type_filter,
                                          VkMemoryPropertyFlags properties) const
{
    VkPhysicalDeviceMemoryProperties mem_props{};
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_props);

    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((type_filter & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return UINT32_MAX;
}

// ===========================================================================
//  Constructor / Destructor
// ===========================================================================

// ---------------------------------------------------------------------------
// VulkanBackend (constructor)
// ---------------------------------------------------------------------------
// Six-phase Vulkan initialisation: instance, physical device selection,
// compute queue family lookup, logical device + queue, command pool, and
// host-visible memory type cache.
// ---------------------------------------------------------------------------
VulkanBackend::VulkanBackend()
{
    // 1. Create VkInstance.
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "ResonanceNet";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo inst_ci{};
    inst_ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_ci.pApplicationInfo = &app_info;

    VkResult res = vkCreateInstance(&inst_ci, nullptr, &instance_);
    if (res != VK_SUCCESS) {
        device_name_ = "Vulkan Device (instance creation failed)";
        LogPrintf("GPU: Vulkan instance creation failed (VkResult=%d)",
                  static_cast<int>(res));
        return;
    }

    // 2. Enumerate physical devices and pick the first one.
    uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(instance_, &dev_count, nullptr);
    if (dev_count == 0) {
        device_name_ = "Vulkan (no physical devices)";
        LogPrintf("GPU: Vulkan -- no physical devices found");
        return;
    }

    std::vector<VkPhysicalDevice> devices(dev_count);
    vkEnumeratePhysicalDevices(instance_, &dev_count, devices.data());

    // Prefer discrete GPU, fall back to first device.
    physical_device_ = devices[0];
    for (auto& pd : devices) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(pd, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physical_device_ = pd;
            break;
        }
    }

    // Cache device name and total memory.
    {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physical_device_, &props);
        device_name_ = props.deviceName;

        VkPhysicalDeviceMemoryProperties mem_props{};
        vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_props);
        for (uint32_t i = 0; i < mem_props.memoryHeapCount; ++i) {
            if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                total_mem_ += mem_props.memoryHeaps[i].size;
            }
        }
    }

    // 3. Find a queue family that supports compute.
    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qf_count, nullptr);
    std::vector<VkQueueFamilyProperties> qf_props(qf_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qf_count, qf_props.data());

    for (uint32_t i = 0; i < qf_count; ++i) {
        if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            compute_queue_family_ = i;
            break;
        }
    }

    if (compute_queue_family_ == UINT32_MAX) {
        device_name_ += " (no compute queue)";
        LogPrintf("GPU: Vulkan -- no compute-capable queue family on %s",
                  device_name_.c_str());
        return;
    }

    // 4. Create logical device + single compute queue.
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_ci{};
    queue_ci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_ci.queueFamilyIndex = compute_queue_family_;
    queue_ci.queueCount = 1;
    queue_ci.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo dev_ci{};
    dev_ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dev_ci.queueCreateInfoCount = 1;
    dev_ci.pQueueCreateInfos = &queue_ci;

    res = vkCreateDevice(physical_device_, &dev_ci, nullptr, &device_);
    if (res != VK_SUCCESS) {
        device_name_ += " (device creation failed)";
        LogPrintf("GPU: Vulkan device creation failed (VkResult=%d)",
                  static_cast<int>(res));
        return;
    }

    vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);

    // 5. Create command pool.
    VkCommandPoolCreateInfo pool_ci{};
    pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_ci.queueFamilyIndex = compute_queue_family_;

    res = vkCreateCommandPool(device_, &pool_ci, nullptr, &command_pool_);
    if (res != VK_SUCCESS) {
        LogPrintf("GPU: Vulkan command pool creation failed (VkResult=%d)",
                  static_cast<int>(res));
        // Non-fatal -- alloc/free still work, but compute dispatch would not.
    }

    // 6. Cache host-visible memory type.
    // We need a memory type with HOST_VISIBLE | HOST_COHERENT.  Create a
    // tiny throwaway buffer to query memory requirements.
    {
        VkBufferCreateInfo buf_ci{};
        buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buf_ci.size = 256;
        buf_ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer tmp_buf = VK_NULL_HANDLE;
        res = vkCreateBuffer(device_, &buf_ci, nullptr, &tmp_buf);
        if (res == VK_SUCCESS) {
            VkMemoryRequirements mem_req{};
            vkGetBufferMemoryRequirements(device_, tmp_buf, &mem_req);

            host_visible_memory_type_ = find_memory_type(
                mem_req.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

            vkDestroyBuffer(device_, tmp_buf, nullptr);
        }
    }

    LogPrintf("GPU: initialized Vulkan backend -- %s (%.0f MiB VRAM, queue family %u)",
              device_name_.c_str(),
              static_cast<double>(total_mem_) / (1024.0 * 1024.0),
              compute_queue_family_);
}

// ---------------------------------------------------------------------------
// ~VulkanBackend
// ---------------------------------------------------------------------------
// Tears down Vulkan resources in reverse creation order.
// ---------------------------------------------------------------------------
VulkanBackend::~VulkanBackend()
{
    // 1. Free any outstanding allocations.
    for (auto& [ptr, alloc] : allocations_) {
        if (device_ != VK_NULL_HANDLE) {
            vkUnmapMemory(device_, alloc.memory);
            vkDestroyBuffer(device_, alloc.buffer, nullptr);
            vkFreeMemory(device_, alloc.memory, nullptr);
        }
    }
    allocations_.clear();

    // 2. Destroy in reverse creation order.
    if (command_pool_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_, command_pool_, nullptr);
    }
    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
}

// ===========================================================================
//  Device Info
// ===========================================================================

// ---------------------------------------------------------------------------
// device_name / total_memory / free_memory / type
// ---------------------------------------------------------------------------
std::string VulkanBackend::device_name() const { return device_name_; }
size_t VulkanBackend::total_memory() const { return total_mem_; }

size_t VulkanBackend::free_memory() const
{
    // Vulkan has no standard API for querying free memory.  Return an
    // approximation based on what we have allocated ourselves.
    if (total_mem_ > allocated_bytes_) {
        return total_mem_ - allocated_bytes_;
    }
    return 0;
}

GpuBackendType VulkanBackend::type() const { return GpuBackendType::VULKAN; }

// ===========================================================================
//  Memory Management
// ===========================================================================

// ---------------------------------------------------------------------------
// alloc
// ---------------------------------------------------------------------------
// Six-step allocation: create VkBuffer, query memory requirements, find a
// HOST_VISIBLE|HOST_COHERENT memory type, allocate device memory, bind
// buffer to memory, and persistently map.
// ---------------------------------------------------------------------------
void* VulkanBackend::alloc(size_t bytes)
{
    if (device_ == VK_NULL_HANDLE) {
        LogError("Vulkan alloc: device not initialised");
        return nullptr;
    }
    if (bytes == 0) {
        return nullptr;
    }

    // 1. Create a VkBuffer.
    VkBufferCreateInfo buf_ci{};
    buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_ci.size = bytes;
    buf_ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = VK_NULL_HANDLE;
    VkResult res = vkCreateBuffer(device_, &buf_ci, nullptr, &buffer);
    if (res != VK_SUCCESS) {
        LogError("Vulkan alloc: vkCreateBuffer failed (VkResult=%d, size=%zu)",
                 static_cast<int>(res), bytes);
        return nullptr;
    }

    // 2. Query memory requirements (may differ from requested size due to
    //    alignment).
    VkMemoryRequirements mem_req{};
    vkGetBufferMemoryRequirements(device_, buffer, &mem_req);

    // 3. Find a HOST_VISIBLE | HOST_COHERENT memory type.
    uint32_t mem_type = find_memory_type(
        mem_req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (mem_type == UINT32_MAX) {
        LogError("Vulkan alloc: no suitable host-visible memory type");
        vkDestroyBuffer(device_, buffer, nullptr);
        return nullptr;
    }

    // 4. Allocate device memory.
    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_req.size;
    alloc_info.memoryTypeIndex = mem_type;

    VkDeviceMemory memory = VK_NULL_HANDLE;
    res = vkAllocateMemory(device_, &alloc_info, nullptr, &memory);
    if (res != VK_SUCCESS) {
        LogError("Vulkan alloc: vkAllocateMemory failed (VkResult=%d, size=%zu)",
                 static_cast<int>(res), static_cast<size_t>(mem_req.size));
        vkDestroyBuffer(device_, buffer, nullptr);
        return nullptr;
    }

    // 5. Bind buffer to memory.
    res = vkBindBufferMemory(device_, buffer, memory, 0);
    if (res != VK_SUCCESS) {
        LogError("Vulkan alloc: vkBindBufferMemory failed (VkResult=%d)",
                 static_cast<int>(res));
        vkFreeMemory(device_, memory, nullptr);
        vkDestroyBuffer(device_, buffer, nullptr);
        return nullptr;
    }

    // 6. Persistently map the memory.
    void* mapped = nullptr;
    res = vkMapMemory(device_, memory, 0, bytes, 0, &mapped);
    if (res != VK_SUCCESS) {
        LogError("Vulkan alloc: vkMapMemory failed (VkResult=%d)",
                 static_cast<int>(res));
        vkDestroyBuffer(device_, buffer, nullptr);
        vkFreeMemory(device_, memory, nullptr);
        return nullptr;
    }

    // Zero-initialise (matches CPU backend behaviour).
    std::memset(mapped, 0, bytes);

    // Track the allocation.
    allocations_[mapped] = Allocation{buffer, memory, bytes};
    allocated_bytes_ += bytes;

    return mapped;
}

// ---------------------------------------------------------------------------
// free
// ---------------------------------------------------------------------------
// Releases a previously allocated buffer/memory pair and updates the
// internal allocation tracker.
// ---------------------------------------------------------------------------
void VulkanBackend::free(void* ptr)
{
    if (!ptr) return;

    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
        LogError("Vulkan free: unknown pointer %p", ptr);
        return;
    }

    if (device_ != VK_NULL_HANDLE) {
        vkUnmapMemory(device_, it->second.memory);
        vkDestroyBuffer(device_, it->second.buffer, nullptr);
        vkFreeMemory(device_, it->second.memory, nullptr);
    }

    if (allocated_bytes_ >= it->second.size) {
        allocated_bytes_ -= it->second.size;
    } else {
        allocated_bytes_ = 0;
    }

    allocations_.erase(it);
}

// ---------------------------------------------------------------------------
// copy_to_device / copy_to_host / synchronize / memset_zero
// ---------------------------------------------------------------------------
// Memory is HOST_VISIBLE and persistently mapped -- a plain memcpy is
// sufficient and correct.  HOST_COHERENT guarantees visibility without
// explicit flush.
// ---------------------------------------------------------------------------
void VulkanBackend::copy_to_device(void* dst, const void* src, size_t bytes)
{
    std::memcpy(dst, src, bytes);
}

void VulkanBackend::copy_to_host(void* dst, const void* src, size_t bytes)
{
    std::memcpy(dst, src, bytes);
}

void VulkanBackend::synchronize()
{
    if (compute_queue_ != VK_NULL_HANDLE) {
        vkQueueWaitIdle(compute_queue_);
    }
}

void VulkanBackend::memset_zero(void* ptr, size_t bytes)
{
    if (ptr && bytes > 0) {
        std::memset(ptr, 0, bytes);
    }
}

// ===========================================================================
//  Training Kernels
// ===========================================================================
// These operate on persistently mapped HOST_VISIBLE memory.  The math is
// identical to CpuFallbackBackend so that numerical results match exactly.

// ---------------------------------------------------------------------------
// embedding_forward
// ---------------------------------------------------------------------------
// Copies rows from the embedding weight table into the output buffer,
// indexed by the token IDs.
// ---------------------------------------------------------------------------
void VulkanBackend::embedding_forward(void* out, const void* weight,
                                       const int* tokens, int batch, int seq, int d_model)
{
    auto* o = static_cast<float*>(out);
    auto* w = static_cast<const float*>(weight);

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            int tok = tokens[b * seq + s];
            const float* row = w + tok * d_model;
            float* dst = o + (b * seq + s) * d_model;
            std::memcpy(dst, row, d_model * sizeof(float));
        }
    }
}

// ---------------------------------------------------------------------------
// rmsnorm_forward
// ---------------------------------------------------------------------------
// Root-mean-square layer normalisation: out = x * rsqrt(mean(x^2) + eps) * scale.
// ---------------------------------------------------------------------------
void VulkanBackend::rmsnorm_forward(void* out, const void* x, const void* scale,
                                     int batch, int seq, int d, float eps)
{
    auto* o = static_cast<float*>(out);
    auto* xp = static_cast<const float*>(x);
    auto* sp = static_cast<const float*>(scale);

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            const float* row = xp + (b * seq + s) * d;
            float* dst = o + (b * seq + s) * d;

            // 1. Compute sum of squares.
            float ss = 0.0f;
            for (int i = 0; i < d; ++i) {
                ss += row[i] * row[i];
            }
            float rms = 1.0f / std::sqrt(ss / static_cast<float>(d) + eps);

            // 2. Normalise and scale.
            for (int i = 0; i < d; ++i) {
                dst[i] = row[i] * rms * sp[i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// causal_conv_forward
// ---------------------------------------------------------------------------
// Multi-branch depthwise causal convolution.  Each branch applies a
// different kernel size over the sequence dimension.
// ---------------------------------------------------------------------------
void VulkanBackend::causal_conv_forward(void* out, const void* x, const void* weights,
                                         const int* kernel_sizes, int n_branches,
                                         int batch, int seq, int d)
{
    auto* o = static_cast<float*>(out);
    auto* xp = static_cast<const float*>(x);
    auto* wp = static_cast<const float*>(weights);

    std::memset(o, 0, batch * seq * d * sizeof(float));

    // 1. Determine maximum kernel size for weight-offset calculation.
    int max_kernel = 0;
    for (int br = 0; br < n_branches; ++br) {
        if (kernel_sizes[br] > max_kernel) max_kernel = kernel_sizes[br];
    }

    // 2. Accumulate convolution across branches.
    for (int br = 0; br < n_branches; ++br) {
        int ks = kernel_sizes[br];
        const float* w_branch = wp + br * max_kernel * d;

        for (int b = 0; b < batch; ++b) {
            for (int s = 0; s < seq; ++s) {
                float* dst = o + (b * seq + s) * d;
                for (int k = 0; k < ks; ++k) {
                    int src_pos = s - k;
                    if (src_pos < 0) continue;
                    const float* src_row = xp + (b * seq + src_pos) * d;
                    const float* w_k = w_branch + k * d;
                    for (int i = 0; i < d; ++i) {
                        dst[i] += src_row[i] * w_k[i];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// mingru_forward
// ---------------------------------------------------------------------------
// Minimal Gated Recurrent Unit forward pass over a full sequence.
// Gate z = sigma(Wz @ x_t + h), candidate h_tilde = Wh @ (z * x_t),
// update h = (1 - z) * h + z * h_tilde.
// ---------------------------------------------------------------------------
void VulkanBackend::mingru_forward(void* h_out, void* state_out,
                                    const void* x, const void* h_prev,
                                    const void* Wz, const void* Wh,
                                    int batch, int seq, int d)
{
    auto* ho = static_cast<float*>(h_out);
    auto* so = static_cast<float*>(state_out);
    auto* xp = static_cast<const float*>(x);
    auto* hp = static_cast<const float*>(h_prev);
    auto* wz = static_cast<const float*>(Wz);
    auto* wh = static_cast<const float*>(Wh);

    std::vector<float> h_cur(batch * d);
    std::memcpy(h_cur.data(), hp, batch * d * sizeof(float));

    for (int s = 0; s < seq; ++s) {
        for (int b = 0; b < batch; ++b) {
            const float* x_t = xp + (b * seq + s) * d;
            float* h_c = h_cur.data() + b * d;
            float* h_o = ho + (b * seq + s) * d;

            for (int i = 0; i < d; ++i) {
                // 1. Gate: z = sigma(Wz @ x_t + h_c).
                float z_val = 0.0f;
                for (int j = 0; j < d; ++j) {
                    z_val += wz[i * d + j] * x_t[j];
                }
                z_val += h_c[i];
                z_val = 1.0f / (1.0f + std::exp(-z_val));

                // 2. Candidate: h_tilde = Wh @ (z * x_t).
                float h_tilde = 0.0f;
                for (int j = 0; j < d; ++j) {
                    h_tilde += wh[i * d + j] * (z_val * x_t[j]);
                }

                // 3. Update: h_new = (1 - z) * h_c + z * h_tilde.
                float h_new = (1.0f - z_val) * h_c[i] + z_val * h_tilde;
                h_o[i] = h_new;
                h_c[i] = h_new;
            }
        }
    }

    std::memcpy(so, h_cur.data(), batch * d * sizeof(float));
}

// ---------------------------------------------------------------------------
// slot_memory_forward
// ---------------------------------------------------------------------------
// Soft-attention lookup over fixed-size slot memory.  Computes scaled
// dot-product attention between query (from x) and slot keys, then
// returns the weighted sum of slot values.
// ---------------------------------------------------------------------------
void VulkanBackend::slot_memory_forward(void* out, const void* x,
                                         const void* slot_keys, const void* slot_values,
                                         int batch, int seq, int d, int n_slots)
{
    auto* o = static_cast<float*>(out);
    auto* xp = static_cast<const float*>(x);
    auto* sk = static_cast<const float*>(slot_keys);
    auto* sv = static_cast<const float*>(slot_values);

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            const float* query = xp + (b * seq + s) * d;
            float* dst = o + (b * seq + s) * d;

            // 1. Compute scaled dot-product scores.
            std::vector<float> scores(n_slots);
            float max_score = -1e30f;
            for (int slot = 0; slot < n_slots; ++slot) {
                float dot = 0.0f;
                for (int i = 0; i < d; ++i) {
                    dot += query[i] * sk[slot * d + i];
                }
                dot /= std::sqrt(static_cast<float>(d));
                scores[slot] = dot;
                if (dot > max_score) max_score = dot;
            }

            // 2. Softmax.
            float sum_exp = 0.0f;
            for (int slot = 0; slot < n_slots; ++slot) {
                scores[slot] = std::exp(scores[slot] - max_score);
                sum_exp += scores[slot];
            }
            for (int slot = 0; slot < n_slots; ++slot) {
                scores[slot] /= sum_exp;
            }

            // 3. Weighted sum of slot values.
            std::memset(dst, 0, d * sizeof(float));
            for (int slot = 0; slot < n_slots; ++slot) {
                for (int i = 0; i < d; ++i) {
                    dst[i] += scores[slot] * sv[slot * d + i];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// swiglu_forward
// ---------------------------------------------------------------------------
// SwiGLU feed-forward: up = x @ W_up, gate = x @ W_gate,
// hidden = up * silu(gate), out = hidden @ W_down.
// ---------------------------------------------------------------------------
void VulkanBackend::swiglu_forward(void* out, const void* x,
                                    const void* W_up, const void* W_gate, const void* W_down,
                                    int batch, int seq, int d_model, int d_ff)
{
    auto* o = static_cast<float*>(out);
    auto* xp = static_cast<const float*>(x);
    auto* wu = static_cast<const float*>(W_up);
    auto* wg = static_cast<const float*>(W_gate);
    auto* wd = static_cast<const float*>(W_down);

    std::vector<float> up(d_ff);
    std::vector<float> gate(d_ff);
    std::vector<float> hidden(d_ff);

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            const float* inp = xp + (b * seq + s) * d_model;
            float* dst = o + (b * seq + s) * d_model;

            // 1. Up and gate projections.
            for (int i = 0; i < d_ff; ++i) {
                float u = 0.0f, g = 0.0f;
                for (int j = 0; j < d_model; ++j) {
                    u += inp[j] * wu[j * d_ff + i];
                    g += inp[j] * wg[j * d_ff + i];
                }
                up[i] = u;
                gate[i] = g;
            }

            // 2. SiLU activation and element-wise multiply.
            for (int i = 0; i < d_ff; ++i) {
                float silu = gate[i] / (1.0f + std::exp(-gate[i]));
                hidden[i] = up[i] * silu;
            }

            // 3. Down projection.
            for (int i = 0; i < d_model; ++i) {
                float val = 0.0f;
                for (int j = 0; j < d_ff; ++j) {
                    val += hidden[j] * wd[j * d_model + i];
                }
                dst[i] = val;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// cross_entropy_loss
// ---------------------------------------------------------------------------
// Computes mean cross-entropy loss with numerically stable log-softmax.
// ---------------------------------------------------------------------------
void VulkanBackend::cross_entropy_loss(float* loss_out, const void* logits,
                                        const int* targets, int batch, int seq, int vocab)
{
    auto* lp = static_cast<const float*>(logits);
    float total_loss = 0.0f;
    int count = batch * seq;

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            const float* row = lp + (b * seq + s) * vocab;
            int tgt = targets[b * seq + s];

            // 1. Find max for numerical stability.
            float max_val = row[0];
            for (int v = 1; v < vocab; ++v) {
                if (row[v] > max_val) max_val = row[v];
            }

            // 2. Log-softmax at target index.
            float sum_exp = 0.0f;
            for (int v = 0; v < vocab; ++v) {
                sum_exp += std::exp(row[v] - max_val);
            }
            float log_softmax = row[tgt] - max_val - std::log(sum_exp);
            total_loss -= log_softmax;
        }
    }

    *loss_out = total_loss / static_cast<float>(count);
}

// ---------------------------------------------------------------------------
// adamw_step
// ---------------------------------------------------------------------------
// Decoupled weight-decay AdamW optimiser step with bias correction.
// ---------------------------------------------------------------------------
void VulkanBackend::adamw_step(void* params, const void* grads, void* m, void* v,
                                float lr, float beta1, float beta2, float eps,
                                float weight_decay, int step, int n_params)
{
    auto* p = static_cast<float*>(params);
    auto* g = static_cast<const float*>(grads);
    auto* mp = static_cast<float*>(m);
    auto* vp = static_cast<float*>(v);

    float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
    float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));

    for (int i = 0; i < n_params; ++i) {
        // 1. Weight decay (decoupled).
        p[i] -= lr * weight_decay * p[i];

        // 2. Momentum update.
        mp[i] = beta1 * mp[i] + (1.0f - beta1) * g[i];
        vp[i] = beta2 * vp[i] + (1.0f - beta2) * g[i] * g[i];

        // 3. Bias-corrected step.
        float m_hat = mp[i] / bc1;
        float v_hat = vp[i] / bc2;

        p[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

// ---------------------------------------------------------------------------
// gemm
// ---------------------------------------------------------------------------
// General matrix multiply: C = alpha * A @ B + beta * C.
// A [M,K], B [K,N], C [M,N] -- all row-major.
// ---------------------------------------------------------------------------
void VulkanBackend::gemm(void* C_ptr, const void* A_ptr, const void* B_ptr,
                          int M, int N, int K, float alpha, float beta_val)
{
    auto* C = static_cast<float*>(C_ptr);
    auto* A = static_cast<const float*>(A_ptr);
    auto* B = static_cast<const float*>(B_ptr);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta_val * C[i * N + j];
        }
    }
}

// ===========================================================================
//  Inference Kernels
// ===========================================================================

// ---------------------------------------------------------------------------
// mingru_step
// ---------------------------------------------------------------------------
// Single-timestep minGRU for autoregressive inference.
// ---------------------------------------------------------------------------
void VulkanBackend::mingru_step(void* h_out, const void* x, const void* h_prev,
                                 const void* Wz, const void* Wh, int d)
{
    auto* ho = static_cast<float*>(h_out);
    auto* xp = static_cast<const float*>(x);
    auto* hp = static_cast<const float*>(h_prev);
    auto* wz = static_cast<const float*>(Wz);
    auto* wh = static_cast<const float*>(Wh);

    for (int i = 0; i < d; ++i) {
        // 1. Gate.
        float z_val = 0.0f;
        for (int j = 0; j < d; ++j) {
            z_val += wz[i * d + j] * xp[j];
        }
        z_val += hp[i];
        z_val = 1.0f / (1.0f + std::exp(-z_val));

        // 2. Candidate.
        float h_tilde = 0.0f;
        for (int j = 0; j < d; ++j) {
            h_tilde += wh[i * d + j] * (z_val * xp[j]);
        }

        // 3. Update.
        ho[i] = (1.0f - z_val) * hp[i] + z_val * h_tilde;
    }
}

// ---------------------------------------------------------------------------
// conv_step
// ---------------------------------------------------------------------------
// Single-timestep causal convolution for autoregressive inference.
// Maintains a rolling buffer of recent inputs per branch.
// ---------------------------------------------------------------------------
void VulkanBackend::conv_step(void* out, void* buffer, const void* x, const void* weights,
                               const int* kernel_sizes, int n_branches, int d)
{
    auto* o = static_cast<float*>(out);
    auto* xp = static_cast<const float*>(x);
    auto* wp = static_cast<const float*>(weights);
    auto* buf = static_cast<float*>(buffer);

    // 1. Determine maximum kernel size for offset calculations.
    int max_kernel = 0;
    for (int br = 0; br < n_branches; ++br) {
        if (kernel_sizes[br] > max_kernel) max_kernel = kernel_sizes[br];
    }

    std::memset(o, 0, d * sizeof(float));

    // 2. Process each branch.
    for (int br = 0; br < n_branches; ++br) {
        int ks = kernel_sizes[br];
        float* br_buf = buf + br * max_kernel * d;
        const float* w_branch = wp + br * max_kernel * d;

        // Shift buffer and insert new input.
        if (ks > 1) {
            std::memmove(br_buf, br_buf + d, (ks - 1) * d * sizeof(float));
        }
        std::memcpy(br_buf + (ks - 1) * d, xp, d * sizeof(float));

        // Accumulate convolution.
        for (int k = 0; k < ks; ++k) {
            const float* b_k = br_buf + k * d;
            const float* w_k = w_branch + k * d;
            for (int i = 0; i < d; ++i) {
                o[i] += b_k[i] * w_k[i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// slot_query
// ---------------------------------------------------------------------------
// Single-vector slot memory query for inference.
// ---------------------------------------------------------------------------
void VulkanBackend::slot_query(void* out, const void* x, const void* slot_keys,
                                const void* slot_values, int d, int n_slots)
{
    auto* o = static_cast<float*>(out);
    auto* xp = static_cast<const float*>(x);
    auto* sk = static_cast<const float*>(slot_keys);
    auto* sv = static_cast<const float*>(slot_values);

    // 1. Compute scaled dot-product scores.
    std::vector<float> scores(n_slots);
    float max_score = -1e30f;
    for (int slot = 0; slot < n_slots; ++slot) {
        float dot = 0.0f;
        for (int i = 0; i < d; ++i) {
            dot += xp[i] * sk[slot * d + i];
        }
        dot /= std::sqrt(static_cast<float>(d));
        scores[slot] = dot;
        if (dot > max_score) max_score = dot;
    }

    // 2. Softmax.
    float sum_exp = 0.0f;
    for (int slot = 0; slot < n_slots; ++slot) {
        scores[slot] = std::exp(scores[slot] - max_score);
        sum_exp += scores[slot];
    }
    for (int slot = 0; slot < n_slots; ++slot) {
        scores[slot] /= sum_exp;
    }

    // 3. Weighted sum of slot values.
    std::memset(o, 0, d * sizeof(float));
    for (int slot = 0; slot < n_slots; ++slot) {
        for (int i = 0; i < d; ++i) {
            o[i] += scores[slot] * sv[slot * d + i];
        }
    }
}

// ===========================================================================
//  Extended GEMM
// ===========================================================================

// ---------------------------------------------------------------------------
// gemm_ex
// ---------------------------------------------------------------------------
// GEMM with optional transpose on A and/or B.
// C = alpha * op(A) @ op(B) + beta * C, row-major layout.
// ---------------------------------------------------------------------------
void VulkanBackend::gemm_ex(void* C_ptr, const void* A_ptr, const void* B_ptr,
                              int M, int N, int K,
                              bool transpose_a, bool transpose_b,
                              float alpha, float beta_val)
{
    auto* C = static_cast<float*>(C_ptr);
    auto* A = static_cast<const float*>(A_ptr);
    auto* B = static_cast<const float*>(B_ptr);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = transpose_a ? A[k * M + i] : A[i * K + k];
                float b_val = transpose_b ? B[j * K + k] : B[k * N + j];
                sum += a_val * b_val;
            }
            C[i * N + j] = alpha * sum + beta_val * C[i * N + j];
        }
    }
}

// ===========================================================================
//  Backward Kernels
// ===========================================================================

// ---------------------------------------------------------------------------
// cross_entropy_backward
// ---------------------------------------------------------------------------
void VulkanBackend::cross_entropy_backward(void* d_logits_ptr, const void* logits_ptr,
                                             const int* targets,
                                             int batch, int seq, int vocab)
{
    auto* d_logits = static_cast<float*>(d_logits_ptr);
    const auto* logits = static_cast<const float*>(logits_ptr);
    const int n_tokens = batch * seq;

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            const int idx = b * seq + s;
            const float* row = logits + idx * vocab;
            float* d_row = d_logits + idx * vocab;
            const int tgt = targets[idx];

            float max_val = row[0];
            for (int v = 1; v < vocab; ++v) {
                if (row[v] > max_val) max_val = row[v];
            }

            float sum_exp = 0.0f;
            for (int v = 0; v < vocab; ++v) {
                sum_exp += std::exp(row[v] - max_val);
            }

            for (int v = 0; v < vocab; ++v) {
                float softmax_v = std::exp(row[v] - max_val) / sum_exp;
                float indicator = (v == tgt) ? 1.0f : 0.0f;
                d_row[v] = (softmax_v - indicator) / static_cast<float>(n_tokens);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// embedding_backward
// ---------------------------------------------------------------------------
void VulkanBackend::embedding_backward(void* d_weight_ptr, const void* d_out_ptr,
                                         const int* tokens,
                                         int batch, int seq, int d_model, int vocab_size)
{
    auto* d_weight = static_cast<float*>(d_weight_ptr);
    const auto* d_out = static_cast<const float*>(d_out_ptr);
    (void)vocab_size;

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            const int idx = b * seq + s;
            const int tok = tokens[idx];
            float* d_row = d_weight + tok * d_model;
            const float* grad_row = d_out + idx * d_model;
            for (int j = 0; j < d_model; ++j) {
                d_row[j] += grad_row[j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// rmsnorm_backward
// ---------------------------------------------------------------------------
void VulkanBackend::rmsnorm_backward(void* d_x_ptr, void* d_scale_ptr,
                                       const void* d_out_ptr,
                                       const void* x_ptr, const void* scale_ptr,
                                       int batch, int seq, int d, float eps)
{
    auto* d_x = static_cast<float*>(d_x_ptr);
    auto* d_scale = static_cast<float*>(d_scale_ptr);
    const auto* d_out = static_cast<const float*>(d_out_ptr);
    const auto* x = static_cast<const float*>(x_ptr);
    const auto* scale = static_cast<const float*>(scale_ptr);

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            const int row_idx = b * seq + s;
            const float* x_row = x + row_idx * d;
            const float* d_out_row = d_out + row_idx * d;
            float* d_x_row = d_x + row_idx * d;

            float ss = 0.0f;
            for (int j = 0; j < d; ++j) {
                ss += x_row[j] * x_row[j];
            }
            ss /= static_cast<float>(d);
            float rms = 1.0f / std::sqrt(ss + eps);

            float ds = 0.0f;
            for (int j = 0; j < d; ++j) {
                ds += d_out_row[j] * x_row[j] * scale[j];
            }

            for (int j = 0; j < d; ++j) {
                d_x_row[j] = rms * (d_out_row[j] * scale[j] - x_row[j] * rms * rms * ds / static_cast<float>(d));
            }

            for (int j = 0; j < d; ++j) {
                d_scale[j] += d_out_row[j] * x_row[j] * rms;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// causal_conv_backward
// ---------------------------------------------------------------------------
void VulkanBackend::causal_conv_backward(void* d_x_ptr, void* d_weights_ptr,
                                           const void* d_out_ptr,
                                           const void* x_ptr, const void* fwd_weights_ptr,
                                           const int* kernel_sizes, int n_branches,
                                           int batch, int seq, int d)
{
    auto* d_x = static_cast<float*>(d_x_ptr);
    auto* d_w = static_cast<float*>(d_weights_ptr);
    const auto* d_out = static_cast<const float*>(d_out_ptr);
    const auto* x = static_cast<const float*>(x_ptr);
    const auto* fwd_w = static_cast<const float*>(fwd_weights_ptr);

    int max_kernel = 0;
    for (int br = 0; br < n_branches; ++br) {
        if (kernel_sizes[br] > max_kernel) max_kernel = kernel_sizes[br];
    }

    for (int b_idx = 0; b_idx < batch; ++b_idx) {
        for (int s = 0; s < seq; ++s) {
            float* d_x_row = d_x + (b_idx * seq + s) * d;
            for (int br = 0; br < n_branches; ++br) {
                int ks = kernel_sizes[br];
                const float* w_branch = fwd_w + br * max_kernel * d;
                for (int k = 0; k < ks; ++k) {
                    int dst_pos = s + k;
                    if (dst_pos >= seq) continue;
                    const float* d_out_row = d_out + (b_idx * seq + dst_pos) * d;
                    const float* w_k = w_branch + k * d;
                    for (int i = 0; i < d; ++i) {
                        d_x_row[i] += w_k[i] * d_out_row[i];
                    }
                }
            }
        }
    }

    for (int br = 0; br < n_branches; ++br) {
        int ks = kernel_sizes[br];
        float* d_w_branch = d_w + br * max_kernel * d;
        for (int b_idx = 0; b_idx < batch; ++b_idx) {
            for (int s = 0; s < seq; ++s) {
                const float* d_out_row = d_out + (b_idx * seq + s) * d;
                for (int k = 0; k < ks; ++k) {
                    int src_pos = s - k;
                    if (src_pos < 0) continue;
                    const float* x_row = x + (b_idx * seq + src_pos) * d;
                    float* d_w_k = d_w_branch + k * d;
                    for (int i = 0; i < d; ++i) {
                        d_w_k[i] += d_out_row[i] * x_row[i];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// mingru_backward  (BPTT)
// ---------------------------------------------------------------------------
void VulkanBackend::mingru_backward(void* d_x_ptr, void* d_Wz_ptr, void* d_Wh_ptr,
                                      const void* d_h_out_ptr, const void* x_ptr,
                                      const void* h_all_ptr, const void* h_init_ptr,
                                      const void* Wz_ptr, const void* Wh_ptr,
                                      int batch, int seq, int d)
{
    auto* d_x = static_cast<float*>(d_x_ptr);
    auto* d_Wz = static_cast<float*>(d_Wz_ptr);
    auto* d_Wh = static_cast<float*>(d_Wh_ptr);
    const auto* d_h_out = static_cast<const float*>(d_h_out_ptr);
    const auto* x = static_cast<const float*>(x_ptr);
    const auto* h_all = static_cast<const float*>(h_all_ptr);
    const auto* h_init = static_cast<const float*>(h_init_ptr);
    const auto* Wz = static_cast<const float*>(Wz_ptr);
    const auto* Wh = static_cast<const float*>(Wh_ptr);

    std::vector<float> d_h_next(batch * d, 0.0f);
    std::vector<float> d_h_tilde_vec(d);
    std::vector<float> d_pre_z_vec(d);
    std::vector<float> z_vec(d);

    for (int s = seq - 1; s >= 0; --s) {
        for (int b = 0; b < batch; ++b) {
            const float* x_t = x + (b * seq + s) * d;

            for (int i = 0; i < d; ++i) {
                float h_prev = (s > 0) ? h_all[(b * seq + s - 1) * d + i] : h_init[b * d + i];

                float pre_z = h_prev;
                for (int j = 0; j < d; ++j) {
                    pre_z += Wz[i * d + j] * x_t[j];
                }
                float z = 1.0f / (1.0f + std::exp(-pre_z));
                z_vec[i] = z;

                float h_tilde = 0.0f;
                for (int j = 0; j < d; ++j) {
                    h_tilde += Wh[i * d + j] * z * x_t[j];
                }

                float d_h = d_h_out[(b * seq + s) * d + i] + d_h_next[b * d + i];

                float d_h_tilde_i = d_h * z;
                float d_z_update = d_h * (h_tilde - h_prev);
                float d_h_prev_update = d_h * (1.0f - z);

                float wx_sum = 0.0f;
                for (int j = 0; j < d; ++j) {
                    wx_sum += Wh[i * d + j] * x_t[j];
                }
                float d_z_htilde = d_h_tilde_i * wx_sum;

                for (int j = 0; j < d; ++j) {
                    d_Wh[i * d + j] += d_h_tilde_i * z * x_t[j];
                }

                float d_z = d_z_update + d_z_htilde;
                float d_pre_z = d_z * z * (1.0f - z);

                for (int j = 0; j < d; ++j) {
                    d_Wz[i * d + j] += d_pre_z * x_t[j];
                }

                d_h_tilde_vec[i] = d_h_tilde_i;
                d_pre_z_vec[i] = d_pre_z;
                d_h_next[b * d + i] = d_h_prev_update + d_pre_z;
            }

            float* d_x_t = d_x + (b * seq + s) * d;
            for (int j = 0; j < d; ++j) {
                float contrib = 0.0f;
                for (int i = 0; i < d; ++i) {
                    contrib += d_h_tilde_vec[i] * Wh[i * d + j] * z_vec[i];
                    contrib += d_pre_z_vec[i] * Wz[i * d + j];
                }
                d_x_t[j] += contrib;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// slot_memory_backward
// ---------------------------------------------------------------------------
void VulkanBackend::slot_memory_backward(void* d_x_ptr, void* d_keys_ptr, void* d_values_ptr,
                                           const void* d_out_ptr, const void* x_ptr,
                                           const void* keys_ptr, const void* values_ptr,
                                           int batch, int seq, int d, int n_slots)
{
    auto* d_x = static_cast<float*>(d_x_ptr);
    auto* d_keys = static_cast<float*>(d_keys_ptr);
    auto* d_values = static_cast<float*>(d_values_ptr);
    const auto* d_out = static_cast<const float*>(d_out_ptr);
    const auto* x = static_cast<const float*>(x_ptr);
    const auto* keys = static_cast<const float*>(keys_ptr);
    const auto* values = static_cast<const float*>(values_ptr);

    float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            int row = b * seq + s;
            const float* x_row = x + row * d;
            const float* d_out_row = d_out + row * d;
            float* d_x_row = d_x + row * d;

            std::vector<float> scores(n_slots);
            float max_score = -1e30f;
            for (int slot = 0; slot < n_slots; ++slot) {
                float dot = 0.0f;
                for (int i = 0; i < d; ++i) {
                    dot += x_row[i] * keys[slot * d + i];
                }
                dot *= inv_sqrt_d;
                scores[slot] = dot;
                if (dot > max_score) max_score = dot;
            }
            float sum_exp = 0.0f;
            for (int slot = 0; slot < n_slots; ++slot) {
                scores[slot] = std::exp(scores[slot] - max_score);
                sum_exp += scores[slot];
            }
            for (int slot = 0; slot < n_slots; ++slot) {
                scores[slot] /= sum_exp;
            }

            std::vector<float> d_score(n_slots, 0.0f);
            for (int slot = 0; slot < n_slots; ++slot) {
                for (int i = 0; i < d; ++i) {
                    d_score[slot] += d_out_row[i] * values[slot * d + i];
                }
            }

            float dot_ds = 0.0f;
            for (int slot = 0; slot < n_slots; ++slot) {
                dot_ds += d_score[slot] * scores[slot];
            }
            std::vector<float> d_pre_softmax(n_slots);
            for (int slot = 0; slot < n_slots; ++slot) {
                d_pre_softmax[slot] = scores[slot] * (d_score[slot] - dot_ds) * inv_sqrt_d;
            }

            for (int i = 0; i < d; ++i) {
                float val = 0.0f;
                for (int slot = 0; slot < n_slots; ++slot) {
                    val += d_pre_softmax[slot] * keys[slot * d + i];
                }
                d_x_row[i] += val;
            }

            for (int slot = 0; slot < n_slots; ++slot) {
                for (int i = 0; i < d; ++i) {
                    d_keys[slot * d + i] += d_pre_softmax[slot] * x_row[i];
                }
            }

            for (int slot = 0; slot < n_slots; ++slot) {
                for (int i = 0; i < d; ++i) {
                    d_values[slot * d + i] += scores[slot] * d_out_row[i];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// swiglu_backward
// ---------------------------------------------------------------------------
void VulkanBackend::swiglu_backward(void* d_x_ptr, void* d_W_up_ptr,
                                      void* d_W_gate_ptr, void* d_W_down_ptr,
                                      const void* d_out_ptr, const void* x_ptr,
                                      const void* W_up_ptr, const void* W_gate_ptr,
                                      const void* W_down_ptr,
                                      int batch, int seq, int d_model, int d_ff)
{
    auto* d_x = static_cast<float*>(d_x_ptr);
    auto* d_W_up = static_cast<float*>(d_W_up_ptr);
    auto* d_W_gate = static_cast<float*>(d_W_gate_ptr);
    auto* d_W_down = static_cast<float*>(d_W_down_ptr);
    const auto* d_out = static_cast<const float*>(d_out_ptr);
    const auto* x = static_cast<const float*>(x_ptr);
    const auto* W_up = static_cast<const float*>(W_up_ptr);
    const auto* W_gate = static_cast<const float*>(W_gate_ptr);
    const auto* W_down = static_cast<const float*>(W_down_ptr);

    std::vector<float> up(d_ff), gate(d_ff), sigmoid_g(d_ff);
    std::vector<float> silu_g(d_ff), hidden(d_ff), d_hidden(d_ff);
    std::vector<float> d_up(d_ff), d_gate(d_ff);

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            int row = b * seq + s;
            const float* inp = x + row * d_model;
            const float* d_out_row = d_out + row * d_model;
            float* d_x_row = d_x + row * d_model;

            for (int j = 0; j < d_ff; ++j) {
                float u = 0.0f, g = 0.0f;
                for (int k = 0; k < d_model; ++k) {
                    u += inp[k] * W_up[k * d_ff + j];
                    g += inp[k] * W_gate[k * d_ff + j];
                }
                up[j] = u;
                gate[j] = g;
                sigmoid_g[j] = 1.0f / (1.0f + std::exp(-gate[j]));
                silu_g[j] = gate[j] * sigmoid_g[j];
                hidden[j] = up[j] * silu_g[j];
            }

            for (int j = 0; j < d_ff; ++j) {
                float val = 0.0f;
                for (int k = 0; k < d_model; ++k) {
                    val += d_out_row[k] * W_down[j * d_model + k];
                }
                d_hidden[j] = val;
            }

            for (int j = 0; j < d_ff; ++j) {
                for (int k = 0; k < d_model; ++k) {
                    d_W_down[j * d_model + k] += hidden[j] * d_out_row[k];
                }
            }

            for (int j = 0; j < d_ff; ++j) {
                d_up[j] = d_hidden[j] * silu_g[j];
                d_gate[j] = d_hidden[j] * up[j] * sigmoid_g[j] * (1.0f + gate[j] * (1.0f - sigmoid_g[j]));
            }

            for (int k = 0; k < d_model; ++k) {
                float val = 0.0f;
                for (int j = 0; j < d_ff; ++j) {
                    val += d_up[j] * W_up[k * d_ff + j];
                    val += d_gate[j] * W_gate[k * d_ff + j];
                }
                d_x_row[k] += val;
            }

            for (int k = 0; k < d_model; ++k) {
                for (int j = 0; j < d_ff; ++j) {
                    d_W_up[k * d_ff + j] += inp[k] * d_up[j];
                }
            }

            for (int k = 0; k < d_model; ++k) {
                for (int j = 0; j < d_ff; ++j) {
                    d_W_gate[k * d_ff + j] += inp[k] * d_gate[j];
                }
            }
        }
    }
}

} // namespace rnet::gpu

#endif // RNET_HAS_VULKAN

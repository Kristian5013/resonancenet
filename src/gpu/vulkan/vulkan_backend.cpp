#ifdef RNET_HAS_VULKAN

#include "vulkan_backend.h"
#include "../../core/logging.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

namespace rnet::gpu {

// ── Helpers ──────────────────────────────────────────────────────────────

uint32_t VulkanBackend::find_memory_type(uint32_t type_filter,
                                          VkMemoryPropertyFlags properties) const {
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

// ── Constructor / Destructor ─────────────────────────────────────────────

VulkanBackend::VulkanBackend() {
    // ---- 1. Create VkInstance ----
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

    // ---- 2. Enumerate physical devices & pick the first one ----
    uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(instance_, &dev_count, nullptr);
    if (dev_count == 0) {
        device_name_ = "Vulkan (no physical devices)";
        LogPrintf("GPU: Vulkan — no physical devices found");
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

    // Cache device name & total memory.
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

    // ---- 3. Find a queue family that supports compute ----
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
        LogPrintf("GPU: Vulkan — no compute-capable queue family on %s",
                  device_name_.c_str());
        return;
    }

    // ---- 4. Create logical device + single compute queue ----
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

    // ---- 5. Create command pool ----
    VkCommandPoolCreateInfo pool_ci{};
    pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_ci.queueFamilyIndex = compute_queue_family_;

    res = vkCreateCommandPool(device_, &pool_ci, nullptr, &command_pool_);
    if (res != VK_SUCCESS) {
        LogPrintf("GPU: Vulkan command pool creation failed (VkResult=%d)",
                  static_cast<int>(res));
        // Non-fatal — alloc/free still work, but compute dispatch would not.
    }

    // ---- 6. Cache host-visible memory type ----
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

    LogPrintf("GPU: initialized Vulkan backend — %s (%.0f MiB VRAM, queue family %u)",
              device_name_.c_str(),
              static_cast<double>(total_mem_) / (1024.0 * 1024.0),
              compute_queue_family_);
}

VulkanBackend::~VulkanBackend() {
    // Free any outstanding allocations.
    for (auto& [ptr, alloc] : allocations_) {
        if (device_ != VK_NULL_HANDLE) {
            vkUnmapMemory(device_, alloc.memory);
            vkDestroyBuffer(device_, alloc.buffer, nullptr);
            vkFreeMemory(device_, alloc.memory, nullptr);
        }
    }
    allocations_.clear();

    // Destroy in reverse creation order.
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

// ── Device Info ──────────────────────────────────────────────────────────

std::string VulkanBackend::device_name() const { return device_name_; }
size_t VulkanBackend::total_memory() const { return total_mem_; }

size_t VulkanBackend::free_memory() const {
    // Vulkan has no standard API for querying free memory.  Return an
    // approximation based on what we have allocated ourselves.
    if (total_mem_ > allocated_bytes_) {
        return total_mem_ - allocated_bytes_;
    }
    return 0;
}

GpuBackendType VulkanBackend::type() const { return GpuBackendType::VULKAN; }

// ── Memory Management ───────────────────────────────────────────────────

void* VulkanBackend::alloc(size_t bytes) {
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

void VulkanBackend::free(void* ptr) {
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

void VulkanBackend::copy_to_device(void* dst, const void* src, size_t bytes) {
    // Memory is HOST_VISIBLE and persistently mapped — a plain memcpy is
    // sufficient and correct.  HOST_COHERENT guarantees visibility without
    // explicit flush.
    std::memcpy(dst, src, bytes);
}

void VulkanBackend::copy_to_host(void* dst, const void* src, size_t bytes) {
    std::memcpy(dst, src, bytes);
}

void VulkanBackend::synchronize() {
    if (compute_queue_ != VK_NULL_HANDLE) {
        vkQueueWaitIdle(compute_queue_);
    }
}

// ── Training Kernels ─────────────────────────────────────────────────────
// These operate on persistently mapped HOST_VISIBLE memory.  The math is
// identical to CpuFallbackBackend so that numerical results match exactly.

void VulkanBackend::embedding_forward(void* out, const void* weight,
                                       const int* tokens, int batch, int seq, int d_model) {
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

void VulkanBackend::rmsnorm_forward(void* out, const void* x, const void* scale,
                                     int batch, int seq, int d, float eps) {
    auto* o = static_cast<float*>(out);
    auto* xp = static_cast<const float*>(x);
    auto* sp = static_cast<const float*>(scale);

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            const float* row = xp + (b * seq + s) * d;
            float* dst = o + (b * seq + s) * d;

            float ss = 0.0f;
            for (int i = 0; i < d; ++i) {
                ss += row[i] * row[i];
            }
            float rms = 1.0f / std::sqrt(ss / static_cast<float>(d) + eps);

            for (int i = 0; i < d; ++i) {
                dst[i] = row[i] * rms * sp[i];
            }
        }
    }
}

void VulkanBackend::causal_conv_forward(void* out, const void* x, const void* weights,
                                         const int* kernel_sizes, int n_branches,
                                         int batch, int seq, int d) {
    auto* o = static_cast<float*>(out);
    auto* xp = static_cast<const float*>(x);
    auto* wp = static_cast<const float*>(weights);

    std::memset(o, 0, batch * seq * d * sizeof(float));

    int max_kernel = 0;
    for (int br = 0; br < n_branches; ++br) {
        if (kernel_sizes[br] > max_kernel) max_kernel = kernel_sizes[br];
    }

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

void VulkanBackend::mingru_forward(void* h_out, void* state_out,
                                    const void* x, const void* h_prev,
                                    const void* Wz, const void* Wh,
                                    int batch, int seq, int d) {
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
                // Gate: z = sigma(Wz @ x_t + h_c)
                float z_val = 0.0f;
                for (int j = 0; j < d; ++j) {
                    z_val += wz[i * d + j] * x_t[j];
                }
                z_val += h_c[i];
                z_val = 1.0f / (1.0f + std::exp(-z_val));

                // Candidate: h_tilde = Wh @ (z * x_t)
                float h_tilde = 0.0f;
                for (int j = 0; j < d; ++j) {
                    h_tilde += wh[i * d + j] * (z_val * x_t[j]);
                }

                // Update: h_new = (1 - z) * h_c + z * h_tilde
                float h_new = (1.0f - z_val) * h_c[i] + z_val * h_tilde;
                h_o[i] = h_new;
                h_c[i] = h_new;
            }
        }
    }

    std::memcpy(so, h_cur.data(), batch * d * sizeof(float));
}

void VulkanBackend::slot_memory_forward(void* out, const void* x,
                                         const void* slot_keys, const void* slot_values,
                                         int batch, int seq, int d, int n_slots) {
    auto* o = static_cast<float*>(out);
    auto* xp = static_cast<const float*>(x);
    auto* sk = static_cast<const float*>(slot_keys);
    auto* sv = static_cast<const float*>(slot_values);

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            const float* query = xp + (b * seq + s) * d;
            float* dst = o + (b * seq + s) * d;

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

            float sum_exp = 0.0f;
            for (int slot = 0; slot < n_slots; ++slot) {
                scores[slot] = std::exp(scores[slot] - max_score);
                sum_exp += scores[slot];
            }
            for (int slot = 0; slot < n_slots; ++slot) {
                scores[slot] /= sum_exp;
            }

            std::memset(dst, 0, d * sizeof(float));
            for (int slot = 0; slot < n_slots; ++slot) {
                for (int i = 0; i < d; ++i) {
                    dst[i] += scores[slot] * sv[slot * d + i];
                }
            }
        }
    }
}

void VulkanBackend::swiglu_forward(void* out, const void* x,
                                    const void* W_up, const void* W_gate, const void* W_down,
                                    int batch, int seq, int d_model, int d_ff) {
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

            for (int i = 0; i < d_ff; ++i) {
                float u = 0.0f, g = 0.0f;
                for (int j = 0; j < d_model; ++j) {
                    u += inp[j] * wu[j * d_ff + i];
                    g += inp[j] * wg[j * d_ff + i];
                }
                up[i] = u;
                gate[i] = g;
            }

            for (int i = 0; i < d_ff; ++i) {
                float silu = gate[i] / (1.0f + std::exp(-gate[i]));
                hidden[i] = up[i] * silu;
            }

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

void VulkanBackend::cross_entropy_loss(float* loss_out, const void* logits,
                                        const int* targets, int batch, int seq, int vocab) {
    auto* lp = static_cast<const float*>(logits);
    float total_loss = 0.0f;
    int count = batch * seq;

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            const float* row = lp + (b * seq + s) * vocab;
            int tgt = targets[b * seq + s];

            float max_val = row[0];
            for (int v = 1; v < vocab; ++v) {
                if (row[v] > max_val) max_val = row[v];
            }
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

void VulkanBackend::adamw_step(void* params, const void* grads, void* m, void* v,
                                float lr, float beta1, float beta2, float eps,
                                float weight_decay, int step, int n_params) {
    auto* p = static_cast<float*>(params);
    auto* g = static_cast<const float*>(grads);
    auto* mp = static_cast<float*>(m);
    auto* vp = static_cast<float*>(v);

    float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
    float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));

    for (int i = 0; i < n_params; ++i) {
        p[i] -= lr * weight_decay * p[i];

        mp[i] = beta1 * mp[i] + (1.0f - beta1) * g[i];
        vp[i] = beta2 * vp[i] + (1.0f - beta2) * g[i] * g[i];

        float m_hat = mp[i] / bc1;
        float v_hat = vp[i] / bc2;

        p[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

void VulkanBackend::gemm(void* C_ptr, const void* A_ptr, const void* B_ptr,
                          int M, int N, int K, float alpha, float beta_val) {
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

// ── Inference Kernels ────────────────────────────────────────────────────

void VulkanBackend::mingru_step(void* h_out, const void* x, const void* h_prev,
                                 const void* Wz, const void* Wh, int d) {
    auto* ho = static_cast<float*>(h_out);
    auto* xp = static_cast<const float*>(x);
    auto* hp = static_cast<const float*>(h_prev);
    auto* wz = static_cast<const float*>(Wz);
    auto* wh = static_cast<const float*>(Wh);

    for (int i = 0; i < d; ++i) {
        float z_val = 0.0f;
        for (int j = 0; j < d; ++j) {
            z_val += wz[i * d + j] * xp[j];
        }
        z_val += hp[i];
        z_val = 1.0f / (1.0f + std::exp(-z_val));

        float h_tilde = 0.0f;
        for (int j = 0; j < d; ++j) {
            h_tilde += wh[i * d + j] * (z_val * xp[j]);
        }

        ho[i] = (1.0f - z_val) * hp[i] + z_val * h_tilde;
    }
}

void VulkanBackend::conv_step(void* out, void* buffer, const void* x, const void* weights,
                               const int* kernel_sizes, int n_branches, int d) {
    auto* o = static_cast<float*>(out);
    auto* xp = static_cast<const float*>(x);
    auto* wp = static_cast<const float*>(weights);
    auto* buf = static_cast<float*>(buffer);

    int max_kernel = 0;
    for (int br = 0; br < n_branches; ++br) {
        if (kernel_sizes[br] > max_kernel) max_kernel = kernel_sizes[br];
    }

    std::memset(o, 0, d * sizeof(float));

    for (int br = 0; br < n_branches; ++br) {
        int ks = kernel_sizes[br];
        float* br_buf = buf + br * max_kernel * d;
        const float* w_branch = wp + br * max_kernel * d;

        if (ks > 1) {
            std::memmove(br_buf, br_buf + d, (ks - 1) * d * sizeof(float));
        }
        std::memcpy(br_buf + (ks - 1) * d, xp, d * sizeof(float));

        for (int k = 0; k < ks; ++k) {
            const float* b_k = br_buf + k * d;
            const float* w_k = w_branch + k * d;
            for (int i = 0; i < d; ++i) {
                o[i] += b_k[i] * w_k[i];
            }
        }
    }
}

void VulkanBackend::slot_query(void* out, const void* x, const void* slot_keys,
                                const void* slot_values, int d, int n_slots) {
    auto* o = static_cast<float*>(out);
    auto* xp = static_cast<const float*>(x);
    auto* sk = static_cast<const float*>(slot_keys);
    auto* sv = static_cast<const float*>(slot_values);

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

    float sum_exp = 0.0f;
    for (int slot = 0; slot < n_slots; ++slot) {
        scores[slot] = std::exp(scores[slot] - max_score);
        sum_exp += scores[slot];
    }
    for (int slot = 0; slot < n_slots; ++slot) {
        scores[slot] /= sum_exp;
    }

    std::memset(o, 0, d * sizeof(float));
    for (int slot = 0; slot < n_slots; ++slot) {
        for (int i = 0; i < d; ++i) {
            o[i] += scores[slot] * sv[slot * d + i];
        }
    }
}

// ── Extended GEMM ────────────────────────────────────────────────────────

void VulkanBackend::gemm_ex(void* C_ptr, const void* A_ptr, const void* B_ptr,
                              int M, int N, int K,
                              bool transpose_a, bool transpose_b,
                              float alpha, float beta_val) {
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

// ── Backward Pass Stubs ─────────────────────────────────────────────────

void VulkanBackend::cross_entropy_backward(void* /*d_logits*/, const void* /*logits*/,
                                             const int* /*targets*/,
                                             int /*batch*/, int /*seq*/, int /*vocab*/) {
    // Stub — not yet implemented for Vulkan backend
}

void VulkanBackend::embedding_backward(void* /*d_weight*/, const void* /*d_out*/,
                                         const int* /*tokens*/,
                                         int /*batch*/, int /*seq*/, int /*d_model*/,
                                         int /*vocab_size*/) {
    // Stub — not yet implemented for Vulkan backend
}

void VulkanBackend::rmsnorm_backward(void* /*d_x*/, void* /*d_scale*/,
                                       const void* /*d_out*/,
                                       const void* /*x*/, const void* /*scale*/,
                                       int /*batch*/, int /*seq*/, int /*d*/, float /*eps*/) {
    // Stub — not yet implemented for Vulkan backend
}

void VulkanBackend::causal_conv_backward(void* /*d_x*/, void* /*d_weights*/,
                                           const void* /*d_out*/,
                                           const void* /*x*/, const void* /*fwd_weights*/,
                                           const int* /*kernel_sizes*/, int /*n_branches*/,
                                           int /*batch*/, int /*seq*/, int /*d*/) {
    // Stub — not yet implemented for Vulkan backend
}

void VulkanBackend::mingru_backward(void* /*d_x*/, void* /*d_Wz*/, void* /*d_Wh*/,
                                      const void* /*d_h_out*/, const void* /*x*/,
                                      const void* /*h_all*/, const void* /*h_init*/,
                                      const void* /*Wz*/, const void* /*Wh*/,
                                      int /*batch*/, int /*seq*/, int /*d*/) {
    // Stub — not yet implemented for Vulkan backend
}

void VulkanBackend::slot_memory_backward(void* /*d_x*/, void* /*d_keys*/, void* /*d_values*/,
                                           const void* /*d_out*/, const void* /*x*/,
                                           const void* /*slot_keys*/, const void* /*slot_values*/,
                                           int /*batch*/, int /*seq*/, int /*d*/, int /*n_slots*/) {
    // Stub — not yet implemented for Vulkan backend
}

void VulkanBackend::swiglu_backward(void* /*d_x*/, void* /*d_W_up*/, void* /*d_W_gate*/,
                                      void* /*d_W_down*/,
                                      const void* /*d_out*/, const void* /*x*/,
                                      const void* /*W_up*/, const void* /*W_gate*/,
                                      const void* /*W_down*/,
                                      int /*batch*/, int /*seq*/, int /*d_model*/, int /*d_ff*/) {
    // Stub — not yet implemented for Vulkan backend
}

}  // namespace rnet::gpu

#endif  // RNET_HAS_VULKAN

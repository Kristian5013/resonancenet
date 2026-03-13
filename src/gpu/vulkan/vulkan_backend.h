#pragma once

#include "../backend.h"
#include <string>

#ifdef RNET_HAS_VULKAN

#include <vulkan/vulkan.h>
#include <unordered_map>
#include <utility>

namespace rnet::gpu {

/// Vulkan compute backend using HOST_VISIBLE|HOST_COHERENT memory.
/// Buffers are persistently mapped — kernel math runs on the CPU through
/// mapped pointers, while Vulkan manages all resource lifetime and
/// synchronization.  A future iteration will replace the CPU math with
/// SPIR-V compute shaders for full GPU acceleration.
class VulkanBackend final : public GpuBackend {
public:
    VulkanBackend();
    ~VulkanBackend() override;

    // Non-copyable, non-movable (Vulkan handles)
    VulkanBackend(const VulkanBackend&) = delete;
    VulkanBackend& operator=(const VulkanBackend&) = delete;
    VulkanBackend(VulkanBackend&&) = delete;
    VulkanBackend& operator=(VulkanBackend&&) = delete;

    std::string device_name() const override;
    size_t total_memory() const override;
    size_t free_memory() const override;
    GpuBackendType type() const override;

    void* alloc(size_t bytes) override;
    void free(void* ptr) override;
    void copy_to_device(void* dst, const void* src, size_t bytes) override;
    void copy_to_host(void* dst, const void* src, size_t bytes) override;
    void synchronize() override;

    void embedding_forward(void* out, const void* weight,
                            const int* tokens, int batch, int seq, int d_model) override;
    void rmsnorm_forward(void* out, const void* x, const void* scale,
                          int batch, int seq, int d, float eps) override;
    void causal_conv_forward(void* out, const void* x, const void* weights,
                              const int* kernel_sizes, int n_branches,
                              int batch, int seq, int d) override;
    void mingru_forward(void* h_out, void* state_out,
                         const void* x, const void* h_prev,
                         const void* Wz, const void* Wh,
                         int batch, int seq, int d) override;
    void slot_memory_forward(void* out, const void* x,
                              const void* slot_keys, const void* slot_values,
                              int batch, int seq, int d, int n_slots) override;
    void swiglu_forward(void* out, const void* x,
                         const void* W_up, const void* W_gate, const void* W_down,
                         int batch, int seq, int d_model, int d_ff) override;
    void cross_entropy_loss(float* loss_out, const void* logits,
                             const int* targets, int batch, int seq, int vocab) override;
    void adamw_step(void* params, const void* grads, void* m, void* v,
                     float lr, float beta1, float beta2, float eps,
                     float weight_decay, int step, int n_params) override;
    void gemm(void* C, const void* A, const void* B,
               int M, int N, int K, float alpha, float beta) override;
    void gemm_ex(void* C, const void* A, const void* B,
                  int M, int N, int K,
                  bool transpose_a, bool transpose_b,
                  float alpha, float beta) override;

    void cross_entropy_backward(void* d_logits, const void* logits, const int* targets,
                                 int batch, int seq, int vocab) override;
    void embedding_backward(void* d_weight, const void* d_out, const int* tokens,
                             int batch, int seq, int d_model, int vocab_size) override;
    void rmsnorm_backward(void* d_x, void* d_scale, const void* d_out,
                           const void* x, const void* scale,
                           int batch, int seq, int d, float eps) override;
    void causal_conv_backward(void* d_x, void* d_weights, const void* d_out,
                               const void* x, const void* fwd_weights,
                               const int* kernel_sizes, int n_branches,
                               int batch, int seq, int d) override;
    void mingru_backward(void* d_x, void* d_Wz, void* d_Wh,
                          const void* d_out, const void* x,
                          const void* h_prev, const void* Wz, const void* Wh,
                          int batch, int seq, int d) override;
    void slot_memory_backward(void* d_x, void* d_keys, void* d_values,
                               const void* d_out, const void* x,
                               const void* slot_keys, const void* slot_values,
                               int batch, int seq, int d, int n_slots) override;
    void swiglu_backward(void* d_x, void* d_W_up, void* d_W_gate, void* d_W_down,
                          const void* d_out, const void* x,
                          const void* W_up, const void* W_gate, const void* W_down,
                          int batch, int seq, int d_model, int d_ff) override;

    void mingru_step(void* h_out, const void* x, const void* h_prev,
                      const void* Wz, const void* Wh, int d) override;
    void conv_step(void* out, void* buffer, const void* x, const void* weights,
                    const int* kernel_sizes, int n_branches, int d) override;
    void slot_query(void* out, const void* x, const void* slot_keys,
                     const void* slot_values, int d, int n_slots) override;

private:
    /// Per-allocation tracking record.
    struct Allocation {
        VkBuffer buffer;
        VkDeviceMemory memory;
        size_t size;
    };

    // Vulkan handles — created in constructor, destroyed in destructor.
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    uint32_t compute_queue_family_ = UINT32_MAX;

    // Device info cached at init time.
    std::string device_name_;
    size_t total_mem_ = 0;
    size_t allocated_bytes_ = 0;

    // Memory type index for HOST_VISIBLE | HOST_COHERENT allocations.
    uint32_t host_visible_memory_type_ = UINT32_MAX;

    // All live allocations keyed by their mapped pointer.
    std::unordered_map<void*, Allocation> allocations_;

    /// Find a suitable memory type index that satisfies \p type_filter and
    /// \p properties.  Returns UINT32_MAX on failure.
    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) const;
};

}  // namespace rnet::gpu

#endif  // RNET_HAS_VULKAN

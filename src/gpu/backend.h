#pragma once

#include <memory>
#include <string>
#include <cstddef>
#include <cstdint>

namespace rnet::gpu {

enum class GpuBackendType { CUDA, METAL, VULKAN, CPU_FALLBACK };

enum class DType { FP32, FP16, BF16, INT8, INT4 };

/// Returns byte size for one element of the given dtype.
inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::FP32: return 4;
        case DType::FP16: return 2;
        case DType::BF16: return 2;
        case DType::INT8: return 1;
        case DType::INT4: return 1;  // packed later; treat as 1 byte per element for now
    }
    return 4;
}

/// Returns a human-readable name for the dtype.
inline const char* dtype_name(DType dt) {
    switch (dt) {
        case DType::FP32: return "fp32";
        case DType::FP16: return "fp16";
        case DType::BF16: return "bf16";
        case DType::INT8: return "int8";
        case DType::INT4: return "int4";
    }
    return "unknown";
}

/// Returns a human-readable name for the backend type.
inline const char* backend_type_name(GpuBackendType t) {
    switch (t) {
        case GpuBackendType::CUDA:         return "CUDA";
        case GpuBackendType::METAL:        return "Metal";
        case GpuBackendType::VULKAN:       return "Vulkan";
        case GpuBackendType::CPU_FALLBACK: return "CPU_FALLBACK";
    }
    return "unknown";
}

class GpuBackend {
public:
    virtual ~GpuBackend() = default;

    // Factory
    static std::unique_ptr<GpuBackend> create(GpuBackendType type);
    static GpuBackendType auto_detect();
    static std::unique_ptr<GpuBackend> create_best();

    // Device info
    virtual std::string device_name() const = 0;
    virtual size_t total_memory() const = 0;
    virtual size_t free_memory() const = 0;
    virtual GpuBackendType type() const = 0;

    // Memory management
    virtual void* alloc(size_t bytes) = 0;
    virtual void free(void* ptr) = 0;
    virtual void copy_to_device(void* dst, const void* src, size_t bytes) = 0;
    virtual void copy_to_host(void* dst, const void* src, size_t bytes) = 0;
    virtual void synchronize() = 0;

    // === Training kernels ===
    virtual void embedding_forward(void* out, const void* weight,
                                    const int* tokens, int batch, int seq, int d_model) = 0;
    virtual void rmsnorm_forward(void* out, const void* x, const void* scale,
                                  int batch, int seq, int d, float eps = 1e-6f) = 0;
    virtual void causal_conv_forward(void* out, const void* x, const void* weights,
                                      const int* kernel_sizes, int n_branches,
                                      int batch, int seq, int d) = 0;
    virtual void mingru_forward(void* h_out, void* state_out,
                                 const void* x, const void* h_prev,
                                 const void* Wz, const void* Wh,
                                 int batch, int seq, int d) = 0;
    virtual void slot_memory_forward(void* out, const void* x,
                                      const void* slot_keys, const void* slot_values,
                                      int batch, int seq, int d, int n_slots) = 0;
    virtual void swiglu_forward(void* out, const void* x,
                                 const void* W_up, const void* W_gate, const void* W_down,
                                 int batch, int seq, int d_model, int d_ff) = 0;
    virtual void cross_entropy_loss(float* loss_out, const void* logits,
                                     const int* targets, int batch, int seq, int vocab) = 0;
    virtual void adamw_step(void* params, const void* grads, void* m, void* v,
                             float lr, float beta1, float beta2, float eps,
                             float weight_decay, int step, int n_params) = 0;
    virtual void gemm(void* C, const void* A, const void* B,
                       int M, int N, int K, float alpha = 1.0f, float beta = 0.0f) = 0;

    /// GEMM with transpose flags.
    /// C[M,N] = alpha * op(A) @ op(B) + beta * C
    /// If trans_a: A is stored as [K,M], used transposed as [M,K].
    /// If trans_b: B is stored as [N,K], used transposed as [K,N].
    virtual void gemm_ex(void* C, const void* A, const void* B,
                          int M, int N, int K,
                          bool trans_a, bool trans_b,
                          float alpha = 1.0f, float beta = 0.0f) = 0;

    /// Fill device memory with zeros.
    virtual void memset_zero(void* ptr, size_t bytes) = 0;

    // === Backward kernels (gradient computation) ===

    /// d_logits[i,v] = softmax(logits[i,:])[v] - (v == targets[i] ? 1 : 0), averaged over batch*seq.
    virtual void cross_entropy_backward(void* d_logits, const void* logits, const int* targets,
                                         int batch, int seq, int vocab) = 0;

    /// Scatter-add gradients to embedding weight matrix.
    /// d_weight[tokens[i], :] += d_out[i, :]
    virtual void embedding_backward(void* d_weight, const void* d_out, const int* tokens,
                                     int batch, int seq, int d_model, int vocab_size) = 0;

    /// Backward through RMSNorm.
    virtual void rmsnorm_backward(void* d_x, void* d_scale, const void* d_out,
                                   const void* x, const void* scale,
                                   int batch, int seq, int d, float eps = 1e-6f) = 0;

    /// Backward through multi-branch causal convolution.
    /// fwd_weights = the forward convolution weights (needed for d_x computation).
    virtual void causal_conv_backward(void* d_x, void* d_weights, const void* d_out,
                                       const void* x, const void* fwd_weights,
                                       const int* kernel_sizes, int n_branches,
                                       int batch, int seq, int d) = 0;

    /// Backward through MinGRU (BPTT).
    /// h_all[batch, seq, d] = all hidden states from forward pass.
    /// h_init[batch, d] = initial hidden state before first timestep.
    virtual void mingru_backward(void* d_x, void* d_Wz, void* d_Wh,
                                  const void* d_h_out, const void* x,
                                  const void* h_all, const void* h_init,
                                  const void* Wz, const void* Wh,
                                  int batch, int seq, int d) = 0;

    /// Backward through slot memory attention.
    virtual void slot_memory_backward(void* d_x, void* d_keys, void* d_values,
                                       const void* d_out, const void* x,
                                       const void* keys, const void* values,
                                       int batch, int seq, int d, int n_slots) = 0;

    /// Backward through SwiGLU FFN (recomputes intermediates internally).
    virtual void swiglu_backward(void* d_x, void* d_W_up, void* d_W_gate, void* d_W_down,
                                  const void* d_out, const void* x,
                                  const void* W_up, const void* W_gate, const void* W_down,
                                  int batch, int seq, int d_model, int d_ff) = 0;

    // === Inference-specific (single token) ===
    virtual void mingru_step(void* h_out, const void* x, const void* h_prev,
                              const void* Wz, const void* Wh, int d) = 0;
    virtual void conv_step(void* out, void* buffer, const void* x, const void* weights,
                            const int* kernel_sizes, int n_branches, int d) = 0;
    virtual void slot_query(void* out, const void* x, const void* slot_keys,
                             const void* slot_values, int d, int n_slots) = 0;
};

}  // namespace rnet::gpu

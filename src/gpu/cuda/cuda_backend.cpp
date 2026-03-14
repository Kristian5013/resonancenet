// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#ifdef RNET_HAS_CUDA

// Own header.
#include "cuda_backend.h"

// Project headers.
#include "../../core/logging.h"

// Standard library.
#include <cstring>
#include <vector>

// CUDA runtime and cuBLAS.
#include <cublas_v2.h>
#include <cuda_runtime.h>

// ===========================================================================
//  Extern "C" kernel launchers (defined in kernels.cu)
// ===========================================================================

extern "C" {
void launch_embedding_forward(float* out, const float* weight,
                               const int* tokens, int batch, int seq, int d_model);
void launch_rmsnorm_forward(float* out, const float* x, const float* scale,
                             int batch, int seq, int d, float eps);
void launch_causal_conv_forward(float* out, const float* x, const float* weights,
                                 const int* kernel_sizes_dev, int n_branches,
                                 int max_kernel, int batch, int seq, int d);
void launch_mingru_forward(float* h_out, float* state_out,
                            const float* x, const float* h_prev,
                            const float* Wz, const float* Wh,
                            int batch, int seq, int d);
void launch_slot_memory_forward(float* out, const float* x,
                                 const float* slot_keys, const float* slot_values,
                                 int batch, int seq, int d, int n_slots);
void launch_swiglu_activation(float* hidden, const float* up, const float* gate, int total);
void launch_cross_entropy_loss(float* loss_out_host, const float* logits,
                                const int* targets, int batch, int seq, int vocab);
void launch_adamw_step(float* params, const float* grads, float* m, float* v,
                        float lr, float beta1, float beta2, float eps,
                        float weight_decay, int step, int n_params);
void launch_mingru_step(float* h_out, const float* x, const float* h_prev,
                         const float* Wz, const float* Wh, int d);
void launch_conv_step(float* out, float* buffer, const float* x, const float* weights,
                       const int* kernel_sizes_dev, int n_branches, int max_kernel, int d);
void launch_slot_query(float* out, const float* x,
                        const float* slot_keys, const float* slot_values,
                        int d, int n_slots);

// Backward kernel launchers.
void launch_cross_entropy_backward(float* d_logits, const float* logits,
                                    const int* targets, int batch, int seq, int vocab);
void launch_embedding_backward(float* d_weight, const float* d_out,
                                const int* tokens, int batch, int seq, int d_model);
void launch_rmsnorm_backward(float* d_x, float* d_scale, const float* d_out,
                              const float* x, const float* scale,
                              int batch, int seq, int d, float eps);
void launch_causal_conv_backward(float* d_x, float* d_weights,
                                  const float* d_out, const float* x,
                                  const float* fwd_weights,
                                  const int* kernel_sizes_dev, int n_branches,
                                  int max_kernel,
                                  int batch, int seq, int d);
void launch_mingru_backward(float* d_x, float* d_Wz, float* d_Wh,
                              float* d_h_next,
                              const float* d_h_out, const float* x,
                              const float* h_all, const float* h_init,
                              const float* Wz, const float* Wh,
                              int batch, int seq, int d);
void launch_slot_memory_backward(float* d_x, float* d_keys, float* d_values,
                                  const float* d_out, const float* x,
                                  const float* keys, const float* values,
                                  int batch, int seq, int d, int n_slots);
void launch_swiglu_backward_activation(float* d_up, float* d_gate,
                                        const float* d_hidden,
                                        const float* up, const float* gate,
                                        int total);
} // extern "C"

namespace rnet::gpu {

// ===========================================================================
//  Static cuBLAS handle
// ===========================================================================

static cublasHandle_t g_cublas_handle = nullptr;

// ---------------------------------------------------------------------------
// get_cublas
// ---------------------------------------------------------------------------
// Lazily creates and returns the global cuBLAS handle.
// ---------------------------------------------------------------------------
static cublasHandle_t get_cublas()
{
    if (!g_cublas_handle) {
        cublasStatus_t st = cublasCreate(&g_cublas_handle);
        if (st != CUBLAS_STATUS_SUCCESS) {
            LogError("cuBLAS: failed to create handle (status %d)", static_cast<int>(st));
        }
    }
    return g_cublas_handle;
}

// ===========================================================================
//  Constructor / Destructor
// ===========================================================================

// ---------------------------------------------------------------------------
// CudaBackend
// ---------------------------------------------------------------------------
// Queries device properties and eagerly initialises the cuBLAS handle.
// ---------------------------------------------------------------------------
CudaBackend::CudaBackend()
{
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        device_name_ = prop.name;
        total_mem_ = prop.totalGlobalMem;
    } else {
        device_name_ = "CUDA Device (init failed)";
    }
    get_cublas();
    LogPrintf("GPU: initialized CUDA backend -- %s", device_name_.c_str());
}

// ---------------------------------------------------------------------------
// ~CudaBackend
// ---------------------------------------------------------------------------
CudaBackend::~CudaBackend()
{
    if (g_cublas_handle) {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = nullptr;
    }
}

// ===========================================================================
//  Device Info
// ===========================================================================

// ---------------------------------------------------------------------------
// device_name / total_memory / free_memory / type
// ---------------------------------------------------------------------------
std::string CudaBackend::device_name() const { return device_name_; }
size_t CudaBackend::total_memory() const { return total_mem_; }

size_t CudaBackend::free_memory() const
{
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    return free_bytes;
}

GpuBackendType CudaBackend::type() const { return GpuBackendType::CUDA; }

// ===========================================================================
//  Memory Management
// ===========================================================================

// ---------------------------------------------------------------------------
// alloc / free / copy_to_device / copy_to_host / synchronize
// ---------------------------------------------------------------------------
void* CudaBackend::alloc(size_t bytes)
{
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        LogError("CUDA alloc failed: %s", cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

void CudaBackend::free(void* ptr)
{
    if (ptr) cudaFree(ptr);
}

void CudaBackend::copy_to_device(void* dst, const void* src, size_t bytes)
{
    cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}

void CudaBackend::copy_to_host(void* dst, const void* src, size_t bytes)
{
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}

void CudaBackend::synchronize()
{
    cudaDeviceSynchronize();
}

// ===========================================================================
//  Training Kernels
// ===========================================================================

// ---------------------------------------------------------------------------
// embedding_forward
// ---------------------------------------------------------------------------
void CudaBackend::embedding_forward(void* out, const void* weight,
                                      const int* tokens, int batch, int seq, int d_model)
{
    launch_embedding_forward(static_cast<float*>(out),
                              static_cast<const float*>(weight),
                              tokens, batch, seq, d_model);
}

// ---------------------------------------------------------------------------
// rmsnorm_forward
// ---------------------------------------------------------------------------
void CudaBackend::rmsnorm_forward(void* out, const void* x, const void* scale,
                                    int batch, int seq, int d, float eps)
{
    launch_rmsnorm_forward(static_cast<float*>(out),
                            static_cast<const float*>(x),
                            static_cast<const float*>(scale),
                            batch, seq, d, eps);
}

// ---------------------------------------------------------------------------
// causal_conv_forward
// ---------------------------------------------------------------------------
// Copies host-side kernel_sizes to device memory before dispatching.
// ---------------------------------------------------------------------------
void CudaBackend::causal_conv_forward(void* out, const void* x, const void* weights,
                                        const int* kernel_sizes, int n_branches,
                                        int batch, int seq, int d)
{
    // 1. Compute max_kernel on host for weight offset calculation.
    int max_kernel = 0;
    for (int i = 0; i < n_branches; ++i) {
        if (kernel_sizes[i] > max_kernel) max_kernel = kernel_sizes[i];
    }

    // 2. Copy kernel_sizes to device.
    int* d_kernel_sizes;
    cudaMalloc(&d_kernel_sizes, n_branches * sizeof(int));
    cudaMemcpy(d_kernel_sizes, kernel_sizes, n_branches * sizeof(int), cudaMemcpyHostToDevice);

    // 3. Dispatch.
    launch_causal_conv_forward(static_cast<float*>(out),
                                static_cast<const float*>(x),
                                static_cast<const float*>(weights),
                                d_kernel_sizes, n_branches, max_kernel,
                                batch, seq, d);

    cudaFree(d_kernel_sizes);
}

// ---------------------------------------------------------------------------
// mingru_forward
// ---------------------------------------------------------------------------
void CudaBackend::mingru_forward(void* h_out, void* state_out,
                                   const void* x, const void* h_prev,
                                   const void* Wz, const void* Wh,
                                   int batch, int seq, int d)
{
    launch_mingru_forward(static_cast<float*>(h_out),
                           static_cast<float*>(state_out),
                           static_cast<const float*>(x),
                           static_cast<const float*>(h_prev),
                           static_cast<const float*>(Wz),
                           static_cast<const float*>(Wh),
                           batch, seq, d);
}

// ---------------------------------------------------------------------------
// slot_memory_forward
// ---------------------------------------------------------------------------
void CudaBackend::slot_memory_forward(void* out, const void* x,
                                        const void* slot_keys, const void* slot_values,
                                        int batch, int seq, int d, int n_slots)
{
    launch_slot_memory_forward(static_cast<float*>(out),
                                static_cast<const float*>(x),
                                static_cast<const float*>(slot_keys),
                                static_cast<const float*>(slot_values),
                                batch, seq, d, n_slots);
}

// ---------------------------------------------------------------------------
// swiglu_forward
// ---------------------------------------------------------------------------
// SwiGLU uses cuBLAS for the three GEMMs and a custom kernel for the
// activation.  Layout: x [BS, d_model], W_up/W_gate [d_model, d_ff],
// W_down [d_ff, d_model].
// ---------------------------------------------------------------------------
void CudaBackend::swiglu_forward(void* out, const void* x,
                                   const void* W_up, const void* W_gate, const void* W_down,
                                   int batch, int seq, int d_model, int d_ff)
{
    int BS = batch * seq;
    cublasHandle_t handle = get_cublas();

    // 1. Allocate temp buffers for up and gate projections.
    float* d_up;
    float* d_gate;
    float* d_hidden;
    cudaMalloc(&d_up, BS * d_ff * sizeof(float));
    cudaMalloc(&d_gate, BS * d_ff * sizeof(float));
    cudaMalloc(&d_hidden, BS * d_ff * sizeof(float));

    const float* d_x = static_cast<const float*>(x);
    const float* d_wu = static_cast<const float*>(W_up);
    const float* d_wg = static_cast<const float*>(W_gate);
    const float* d_wd = static_cast<const float*>(W_down);
    float* d_out = static_cast<float*>(out);

    float alpha = 1.0f, beta_zero = 0.0f;

    // cuBLAS uses column-major. Our matrices are row-major.
    // For row-major C = A @ B where A is [M,K] and B is [K,N]:
    //   cublasSgemm(handle, N, N, N, M, K, &alpha, B, N, A, K, &beta, C, N)

    // 2. up = x @ W_up : [BS, d_model] @ [d_model, d_ff] = [BS, d_ff].
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                d_ff, BS, d_model, &alpha,
                d_wu, d_ff,
                d_x, d_model,
                &beta_zero, d_up, d_ff);

    // 3. gate = x @ W_gate : [BS, d_model] @ [d_model, d_ff] = [BS, d_ff].
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                d_ff, BS, d_model, &alpha,
                d_wg, d_ff,
                d_x, d_model,
                &beta_zero, d_gate, d_ff);

    // 4. hidden = up * silu(gate).
    launch_swiglu_activation(d_hidden, d_up, d_gate, BS * d_ff);

    // 5. out = hidden @ W_down : [BS, d_ff] @ [d_ff, d_model] = [BS, d_model].
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                d_model, BS, d_ff, &alpha,
                d_wd, d_model,
                d_hidden, d_ff,
                &beta_zero, d_out, d_model);

    cudaFree(d_up);
    cudaFree(d_gate);
    cudaFree(d_hidden);
}

// ---------------------------------------------------------------------------
// cross_entropy_loss
// ---------------------------------------------------------------------------
void CudaBackend::cross_entropy_loss(float* loss_out, const void* logits,
                                       const int* targets, int batch, int seq, int vocab)
{
    launch_cross_entropy_loss(loss_out,
                               static_cast<const float*>(logits),
                               targets, batch, seq, vocab);
}

// ---------------------------------------------------------------------------
// adamw_step
// ---------------------------------------------------------------------------
void CudaBackend::adamw_step(void* params, const void* grads, void* m, void* v,
                               float lr, float beta1, float beta2, float eps,
                               float weight_decay, int step, int n_params)
{
    launch_adamw_step(static_cast<float*>(params),
                       static_cast<const float*>(grads),
                       static_cast<float*>(m),
                       static_cast<float*>(v),
                       lr, beta1, beta2, eps, weight_decay, step, n_params);
}

// ---------------------------------------------------------------------------
// gemm
// ---------------------------------------------------------------------------
// C = alpha * A @ B + beta * C.  Row-major to cuBLAS column-major
// conversion: cublasSgemm(N, M, K, alpha, B, N, A, K, beta, C, N).
// ---------------------------------------------------------------------------
void CudaBackend::gemm(void* C_ptr, const void* A_ptr, const void* B_ptr,
                         int M, int N, int K, float alpha, float beta_val)
{
    cublasHandle_t handle = get_cublas();

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha,
                static_cast<const float*>(B_ptr), N,
                static_cast<const float*>(A_ptr), K,
                &beta_val,
                static_cast<float*>(C_ptr), N);
}

// ===========================================================================
//  Inference Kernels
// ===========================================================================

// ---------------------------------------------------------------------------
// mingru_step
// ---------------------------------------------------------------------------
void CudaBackend::mingru_step(void* h_out, const void* x, const void* h_prev,
                                const void* Wz, const void* Wh, int d)
{
    launch_mingru_step(static_cast<float*>(h_out),
                        static_cast<const float*>(x),
                        static_cast<const float*>(h_prev),
                        static_cast<const float*>(Wz),
                        static_cast<const float*>(Wh), d);
}

// ---------------------------------------------------------------------------
// conv_step
// ---------------------------------------------------------------------------
// Copies host-side kernel_sizes to device, dispatches the kernel, and
// cleans up the temporary device buffer.
// ---------------------------------------------------------------------------
void CudaBackend::conv_step(void* out, void* buffer, const void* x, const void* weights,
                              const int* kernel_sizes, int n_branches, int d)
{
    // 1. Compute max_kernel and copy kernel_sizes to device.
    int max_kernel = 0;
    for (int i = 0; i < n_branches; ++i) {
        if (kernel_sizes[i] > max_kernel) max_kernel = kernel_sizes[i];
    }

    int* d_kernel_sizes;
    cudaMalloc(&d_kernel_sizes, n_branches * sizeof(int));
    cudaMemcpy(d_kernel_sizes, kernel_sizes, n_branches * sizeof(int), cudaMemcpyHostToDevice);

    // 2. Dispatch.
    launch_conv_step(static_cast<float*>(out),
                      static_cast<float*>(buffer),
                      static_cast<const float*>(x),
                      static_cast<const float*>(weights),
                      d_kernel_sizes, n_branches, max_kernel, d);

    cudaFree(d_kernel_sizes);
}

// ---------------------------------------------------------------------------
// slot_query
// ---------------------------------------------------------------------------
void CudaBackend::slot_query(void* out, const void* x, const void* slot_keys,
                               const void* slot_values, int d, int n_slots)
{
    launch_slot_query(static_cast<float*>(out),
                       static_cast<const float*>(x),
                       static_cast<const float*>(slot_keys),
                       static_cast<const float*>(slot_values),
                       d, n_slots);
}

// ===========================================================================
//  Extended GEMM and Utility
// ===========================================================================

// ---------------------------------------------------------------------------
// gemm_ex
// ---------------------------------------------------------------------------
// Row-major C[M,N] = alpha * op(A) @ op(B) + beta * C.
// For cuBLAS col-major: compute C^T = op(B)^T @ op(A)^T -- swap A/B and
// flip transpose flags for row-major to col-major conversion.
// ---------------------------------------------------------------------------
void CudaBackend::gemm_ex(void* C, const void* A, const void* B,
                            int M, int N, int K,
                            bool trans_a, bool trans_b,
                            float alpha, float beta_val)
{
    cublasHandle_t handle = get_cublas();

    int lda = trans_a ? M : K;
    int ldb = trans_b ? K : N;
    int ldc = N;

    cublasOperation_t cublas_opA = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t cublas_opB = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasSgemm(handle, cublas_opA, cublas_opB,
                N, M, K, &alpha,
                static_cast<const float*>(B), ldb,
                static_cast<const float*>(A), lda,
                &beta_val,
                static_cast<float*>(C), ldc);
}

// ---------------------------------------------------------------------------
// memset_zero
// ---------------------------------------------------------------------------
void CudaBackend::memset_zero(void* ptr, size_t bytes)
{
    cudaMemset(ptr, 0, bytes);
}

// ===========================================================================
//  Backward Kernels
// ===========================================================================

// ---------------------------------------------------------------------------
// cross_entropy_backward
// ---------------------------------------------------------------------------
void CudaBackend::cross_entropy_backward(void* d_logits, const void* logits,
                                           const int* targets,
                                           int batch, int seq, int vocab)
{
    launch_cross_entropy_backward(static_cast<float*>(d_logits),
                                   static_cast<const float*>(logits),
                                   targets, batch, seq, vocab);
}

// ---------------------------------------------------------------------------
// embedding_backward
// ---------------------------------------------------------------------------
void CudaBackend::embedding_backward(void* d_weight, const void* d_out,
                                       const int* tokens,
                                       int batch, int seq, int d_model, int vocab_size)
{
    (void)vocab_size; // d_weight is [vocab_size, d_model], zeroed by caller
    launch_embedding_backward(static_cast<float*>(d_weight),
                               static_cast<const float*>(d_out),
                               tokens, batch, seq, d_model);
}

// ---------------------------------------------------------------------------
// rmsnorm_backward
// ---------------------------------------------------------------------------
void CudaBackend::rmsnorm_backward(void* d_x, void* d_scale, const void* d_out,
                                     const void* x, const void* scale,
                                     int batch, int seq, int d, float eps)
{
    launch_rmsnorm_backward(static_cast<float*>(d_x),
                             static_cast<float*>(d_scale),
                             static_cast<const float*>(d_out),
                             static_cast<const float*>(x),
                             static_cast<const float*>(scale),
                             batch, seq, d, eps);
}

// ---------------------------------------------------------------------------
// causal_conv_backward
// ---------------------------------------------------------------------------
// Copies kernel_sizes to device, dispatches, and cleans up.
// ---------------------------------------------------------------------------
void CudaBackend::causal_conv_backward(void* d_x, void* d_weights, const void* d_out,
                                         const void* x, const void* fwd_weights,
                                         const int* kernel_sizes, int n_branches,
                                         int batch, int seq, int d)
{
    // 1. Compute max_kernel and copy kernel_sizes to device.
    int max_kernel = 0;
    for (int i = 0; i < n_branches; ++i) {
        if (kernel_sizes[i] > max_kernel) max_kernel = kernel_sizes[i];
    }

    int* d_kernel_sizes;
    cudaMalloc(&d_kernel_sizes, n_branches * sizeof(int));
    cudaMemcpy(d_kernel_sizes, kernel_sizes, n_branches * sizeof(int), cudaMemcpyHostToDevice);

    // 2. Dispatch.
    launch_causal_conv_backward(static_cast<float*>(d_x),
                                 static_cast<float*>(d_weights),
                                 static_cast<const float*>(d_out),
                                 static_cast<const float*>(x),
                                 static_cast<const float*>(fwd_weights),
                                 d_kernel_sizes, n_branches, max_kernel,
                                 batch, seq, d);

    cudaFree(d_kernel_sizes);
}

// ---------------------------------------------------------------------------
// mingru_backward
// ---------------------------------------------------------------------------
// Allocates a temporary d_h_next buffer for gradient propagation between
// timesteps, dispatches, and cleans up.
// ---------------------------------------------------------------------------
void CudaBackend::mingru_backward(void* d_x, void* d_Wz, void* d_Wh,
                                    const void* d_h_out, const void* x,
                                    const void* h_all, const void* h_init,
                                    const void* Wz, const void* Wh,
                                    int batch, int seq, int d)
{
    // 1. Allocate d_h_next buffer (propagated gradient between timesteps).
    float* d_h_next;
    cudaMalloc(&d_h_next, batch * d * sizeof(float));
    cudaMemset(d_h_next, 0, batch * d * sizeof(float));

    // 2. Dispatch.
    launch_mingru_backward(static_cast<float*>(d_x),
                            static_cast<float*>(d_Wz),
                            static_cast<float*>(d_Wh),
                            d_h_next,
                            static_cast<const float*>(d_h_out),
                            static_cast<const float*>(x),
                            static_cast<const float*>(h_all),
                            static_cast<const float*>(h_init),
                            static_cast<const float*>(Wz),
                            static_cast<const float*>(Wh),
                            batch, seq, d);

    cudaFree(d_h_next);
}

// ---------------------------------------------------------------------------
// slot_memory_backward
// ---------------------------------------------------------------------------
void CudaBackend::slot_memory_backward(void* d_x, void* d_keys, void* d_values,
                                         const void* d_out, const void* x,
                                         const void* keys, const void* values,
                                         int batch, int seq, int d, int n_slots)
{
    launch_slot_memory_backward(static_cast<float*>(d_x),
                                 static_cast<float*>(d_keys),
                                 static_cast<float*>(d_values),
                                 static_cast<const float*>(d_out),
                                 static_cast<const float*>(x),
                                 static_cast<const float*>(keys),
                                 static_cast<const float*>(values),
                                 batch, seq, d, n_slots);
}

// ---------------------------------------------------------------------------
// swiglu_backward
// ---------------------------------------------------------------------------
// Full SwiGLU backward: recomputes forward intermediates, then computes
// gradients for all three weight matrices and the input.
// ---------------------------------------------------------------------------
void CudaBackend::swiglu_backward(void* d_x, void* d_W_up, void* d_W_gate, void* d_W_down,
                                    const void* d_out, const void* x,
                                    const void* W_up, const void* W_gate, const void* W_down,
                                    int batch, int seq, int d_model, int d_ff)
{
    int BS = batch * seq;

    // 1. Allocate temporaries for recomputed forward intermediates.
    float* up_buf;
    float* gate_buf;
    float* d_hidden;
    cudaMalloc(&up_buf, BS * d_ff * sizeof(float));
    cudaMalloc(&gate_buf, BS * d_ff * sizeof(float));
    cudaMalloc(&d_hidden, BS * d_ff * sizeof(float));

    // 2. Recompute forward: up = x @ W_up, gate = x @ W_gate.
    gemm(up_buf, x, W_up, BS, d_ff, d_model, 1.0f, 0.0f);
    gemm(gate_buf, x, W_gate, BS, d_ff, d_model, 1.0f, 0.0f);

    // 3. d_hidden = d_out @ W_down^T : [BS, d_model] @ [d_model, d_ff] (transposed).
    gemm_ex(d_hidden, d_out, W_down, BS, d_ff, d_model, false, true, 1.0f, 0.0f);

    // 4. Compute forward hidden = up * silu(gate) for d_W_down gradient.
    float* hidden_fwd;
    cudaMalloc(&hidden_fwd, BS * d_ff * sizeof(float));
    launch_swiglu_activation(hidden_fwd, up_buf, gate_buf, BS * d_ff);

    // 5. d_W_down += hidden^T @ d_out : [d_ff, BS] @ [BS, d_model] = [d_ff, d_model].
    gemm_ex(d_W_down, hidden_fwd, d_out, d_ff, d_model, BS, true, false, 1.0f, 1.0f);
    cudaFree(hidden_fwd);

    // 6. Backward activation: compute d_up_grad, d_gate_grad from d_hidden, up, gate.
    float* d_up_grad;
    float* d_gate_grad;
    cudaMalloc(&d_up_grad, BS * d_ff * sizeof(float));
    cudaMalloc(&d_gate_grad, BS * d_ff * sizeof(float));
    launch_swiglu_backward_activation(d_up_grad, d_gate_grad, d_hidden, up_buf, gate_buf, BS * d_ff);

    // 7. d_x = d_up_grad @ W_up^T + d_gate_grad @ W_gate^T.
    gemm_ex(d_x, d_up_grad, W_up, BS, d_model, d_ff, false, true, 1.0f, 0.0f);
    gemm_ex(d_x, d_gate_grad, W_gate, BS, d_model, d_ff, false, true, 1.0f, 1.0f);

    // 8. d_W_up += x^T @ d_up_grad : [d_model, BS] @ [BS, d_ff] = [d_model, d_ff].
    gemm_ex(d_W_up, x, d_up_grad, d_model, d_ff, BS, true, false, 1.0f, 1.0f);

    // 9. d_W_gate += x^T @ d_gate_grad : [d_model, BS] @ [BS, d_ff] = [d_model, d_ff].
    gemm_ex(d_W_gate, x, d_gate_grad, d_model, d_ff, BS, true, false, 1.0f, 1.0f);

    cudaFree(up_buf);
    cudaFree(gate_buf);
    cudaFree(d_hidden);
    cudaFree(d_up_grad);
    cudaFree(d_gate_grad);
}

} // namespace rnet::gpu

#endif // RNET_HAS_CUDA

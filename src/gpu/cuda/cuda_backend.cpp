#ifdef RNET_HAS_CUDA

#include "cuda_backend.h"
#include "../../core/logging.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstring>
#include <vector>

// ── Extern "C" kernel launchers (defined in kernels.cu) ─────────────────────

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
}

namespace rnet::gpu {

// ── Static cuBLAS handle ────────────────────────────────────────────────────

static cublasHandle_t g_cublas_handle = nullptr;

static cublasHandle_t get_cublas() {
    if (!g_cublas_handle) {
        cublasStatus_t st = cublasCreate(&g_cublas_handle);
        if (st != CUBLAS_STATUS_SUCCESS) {
            LogError("cuBLAS: failed to create handle (status %d)", static_cast<int>(st));
        }
    }
    return g_cublas_handle;
}

// ── Constructor / Destructor ────────────────────────────────────────────────

CudaBackend::CudaBackend() {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        device_name_ = prop.name;
        total_mem_ = prop.totalGlobalMem;
    } else {
        device_name_ = "CUDA Device (init failed)";
    }
    get_cublas();  // Eagerly initialize cuBLAS
    LogPrintf("GPU: initialized CUDA backend — %s", device_name_.c_str());
}

CudaBackend::~CudaBackend() {
    if (g_cublas_handle) {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = nullptr;
    }
}

// ── Device Info ─────────────────────────────────────────────────────────────

std::string CudaBackend::device_name() const { return device_name_; }
size_t CudaBackend::total_memory() const { return total_mem_; }

size_t CudaBackend::free_memory() const {
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    return free_bytes;
}

GpuBackendType CudaBackend::type() const { return GpuBackendType::CUDA; }

// ── Memory Management ───────────────────────────────────────────────────────

void* CudaBackend::alloc(size_t bytes) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        LogError("CUDA alloc failed: %s", cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

void CudaBackend::free(void* ptr) {
    if (ptr) cudaFree(ptr);
}

void CudaBackend::copy_to_device(void* dst, const void* src, size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}

void CudaBackend::copy_to_host(void* dst, const void* src, size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}

void CudaBackend::synchronize() {
    cudaDeviceSynchronize();
}

// ── Training Kernels ────────────────────────────────────────────────────────

void CudaBackend::embedding_forward(void* out, const void* weight,
                                      const int* tokens, int batch, int seq, int d_model) {
    launch_embedding_forward(static_cast<float*>(out),
                              static_cast<const float*>(weight),
                              tokens, batch, seq, d_model);
}

void CudaBackend::rmsnorm_forward(void* out, const void* x, const void* scale,
                                    int batch, int seq, int d, float eps) {
    launch_rmsnorm_forward(static_cast<float*>(out),
                            static_cast<const float*>(x),
                            static_cast<const float*>(scale),
                            batch, seq, d, eps);
}

void CudaBackend::causal_conv_forward(void* out, const void* x, const void* weights,
                                        const int* kernel_sizes, int n_branches,
                                        int batch, int seq, int d) {
    // kernel_sizes is host memory (passed from caller). We need it on device for the kernel,
    // and we also need max_kernel on host for the weight offset calculation.
    int max_kernel = 0;
    for (int i = 0; i < n_branches; ++i) {
        if (kernel_sizes[i] > max_kernel) max_kernel = kernel_sizes[i];
    }

    // Copy kernel_sizes to device
    int* d_kernel_sizes;
    cudaMalloc(&d_kernel_sizes, n_branches * sizeof(int));
    cudaMemcpy(d_kernel_sizes, kernel_sizes, n_branches * sizeof(int), cudaMemcpyHostToDevice);

    launch_causal_conv_forward(static_cast<float*>(out),
                                static_cast<const float*>(x),
                                static_cast<const float*>(weights),
                                d_kernel_sizes, n_branches, max_kernel,
                                batch, seq, d);

    cudaFree(d_kernel_sizes);
}

void CudaBackend::mingru_forward(void* h_out, void* state_out,
                                   const void* x, const void* h_prev,
                                   const void* Wz, const void* Wh,
                                   int batch, int seq, int d) {
    launch_mingru_forward(static_cast<float*>(h_out),
                           static_cast<float*>(state_out),
                           static_cast<const float*>(x),
                           static_cast<const float*>(h_prev),
                           static_cast<const float*>(Wz),
                           static_cast<const float*>(Wh),
                           batch, seq, d);
}

void CudaBackend::slot_memory_forward(void* out, const void* x,
                                        const void* slot_keys, const void* slot_values,
                                        int batch, int seq, int d, int n_slots) {
    launch_slot_memory_forward(static_cast<float*>(out),
                                static_cast<const float*>(x),
                                static_cast<const float*>(slot_keys),
                                static_cast<const float*>(slot_values),
                                batch, seq, d, n_slots);
}

void CudaBackend::swiglu_forward(void* out, const void* x,
                                   const void* W_up, const void* W_gate, const void* W_down,
                                   int batch, int seq, int d_model, int d_ff) {
    // SwiGLU uses cuBLAS for the three GEMMs and a custom kernel for the activation.
    // Layout: x [BS, d_model], W_up [d_model, d_ff], W_gate [d_model, d_ff], W_down [d_ff, d_model]
    // BS = batch * seq

    int BS = batch * seq;
    cublasHandle_t handle = get_cublas();

    // Allocate temp buffers for up and gate projections
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
    //   cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N)

    // up = x @ W_up  : [BS, d_model] @ [d_model, d_ff] = [BS, d_ff]
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                d_ff, BS, d_model, &alpha,
                d_wu, d_ff,
                d_x, d_model,
                &beta_zero, d_up, d_ff);

    // gate = x @ W_gate : [BS, d_model] @ [d_model, d_ff] = [BS, d_ff]
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                d_ff, BS, d_model, &alpha,
                d_wg, d_ff,
                d_x, d_model,
                &beta_zero, d_gate, d_ff);

    // hidden = up * silu(gate)
    launch_swiglu_activation(d_hidden, d_up, d_gate, BS * d_ff);

    // out = hidden @ W_down : [BS, d_ff] @ [d_ff, d_model] = [BS, d_model]
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                d_model, BS, d_ff, &alpha,
                d_wd, d_model,
                d_hidden, d_ff,
                &beta_zero, d_out, d_model);

    cudaFree(d_up);
    cudaFree(d_gate);
    cudaFree(d_hidden);
}

void CudaBackend::cross_entropy_loss(float* loss_out, const void* logits,
                                       const int* targets, int batch, int seq, int vocab) {
    launch_cross_entropy_loss(loss_out,
                               static_cast<const float*>(logits),
                               targets, batch, seq, vocab);
}

void CudaBackend::adamw_step(void* params, const void* grads, void* m, void* v,
                               float lr, float beta1, float beta2, float eps,
                               float weight_decay, int step, int n_params) {
    launch_adamw_step(static_cast<float*>(params),
                       static_cast<const float*>(grads),
                       static_cast<float*>(m),
                       static_cast<float*>(v),
                       lr, beta1, beta2, eps, weight_decay, step, n_params);
}

void CudaBackend::gemm(void* C_ptr, const void* A_ptr, const void* B_ptr,
                         int M, int N, int K, float alpha, float beta_val) {
    // C = alpha * A @ B + beta * C
    // A [M, K], B [K, N], C [M, N]  (row-major)
    // cuBLAS column-major: cublasSgemm(N, M, K, alpha, B, N, A, K, beta, C, N)
    cublasHandle_t handle = get_cublas();

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha,
                static_cast<const float*>(B_ptr), N,
                static_cast<const float*>(A_ptr), K,
                &beta_val,
                static_cast<float*>(C_ptr), N);
}

// ── Inference Kernels ───────────────────────────────────────────────────────

void CudaBackend::mingru_step(void* h_out, const void* x, const void* h_prev,
                                const void* Wz, const void* Wh, int d) {
    launch_mingru_step(static_cast<float*>(h_out),
                        static_cast<const float*>(x),
                        static_cast<const float*>(h_prev),
                        static_cast<const float*>(Wz),
                        static_cast<const float*>(Wh), d);
}

void CudaBackend::conv_step(void* out, void* buffer, const void* x, const void* weights,
                              const int* kernel_sizes, int n_branches, int d) {
    // kernel_sizes is host memory. Compute max_kernel and copy to device.
    int max_kernel = 0;
    for (int i = 0; i < n_branches; ++i) {
        if (kernel_sizes[i] > max_kernel) max_kernel = kernel_sizes[i];
    }

    int* d_kernel_sizes;
    cudaMalloc(&d_kernel_sizes, n_branches * sizeof(int));
    cudaMemcpy(d_kernel_sizes, kernel_sizes, n_branches * sizeof(int), cudaMemcpyHostToDevice);

    launch_conv_step(static_cast<float*>(out),
                      static_cast<float*>(buffer),
                      static_cast<const float*>(x),
                      static_cast<const float*>(weights),
                      d_kernel_sizes, n_branches, max_kernel, d);

    cudaFree(d_kernel_sizes);
}

void CudaBackend::slot_query(void* out, const void* x, const void* slot_keys,
                               const void* slot_values, int d, int n_slots) {
    launch_slot_query(static_cast<float*>(out),
                       static_cast<const float*>(x),
                       static_cast<const float*>(slot_keys),
                       static_cast<const float*>(slot_values),
                       d, n_slots);
}

}  // namespace rnet::gpu

#endif  // RNET_HAS_CUDA

#ifdef RNET_HAS_CUDA

#include <cuda_runtime.h>
#include <cstdio>

// ============================================================================
// CUDA Kernels for ResonanceNet GPU Training/Inference
// All math is identical to cpu_backend.cpp, parallelized for GPU.
// ============================================================================

static constexpr int BLOCK_SIZE = 256;

#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA kernel error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
        } \
    } while (0)

// ============================================================================
//  1. Embedding Forward
//     out[batch*seq, d_model] = weight[tokens[i], :]
// ============================================================================

__global__ void kernel_embedding_forward(float* __restrict__ out,
                                          const float* __restrict__ weight,
                                          const int* __restrict__ tokens,
                                          int total_tokens, int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_tokens * d_model) return;
    int token_idx = idx / d_model;
    int dim_idx = idx % d_model;
    out[idx] = weight[tokens[token_idx] * d_model + dim_idx];
}

extern "C" void launch_embedding_forward(float* out, const float* weight,
                                          const int* tokens, int batch, int seq, int d_model) {
    int total = batch * seq * d_model;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_embedding_forward<<<blocks, BLOCK_SIZE>>>(out, weight, tokens, batch * seq, d_model);
    CUDA_CHECK_KERNEL();
}

// ============================================================================
//  2. RMSNorm Forward
//     rms = 1/sqrt(mean(x^2) + eps); out = x * rms * scale
//     One block per row, shared-memory reduction for sum-of-squares.
// ============================================================================

__global__ void kernel_rmsnorm_forward(float* __restrict__ out,
                                        const float* __restrict__ x,
                                        const float* __restrict__ scale,
                                        int n_rows, int d, float eps) {
    int row = blockIdx.x;
    if (row >= n_rows) return;

    const float* x_row = x + row * d;
    float* o_row = out + row * d;

    extern __shared__ float sdata[];

    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float val = x_row[i];
        local_ss += val * val;
    }
    sdata[threadIdx.x] = local_ss;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    float rms = rsqrtf(sdata[0] / static_cast<float>(d) + eps);

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        o_row[i] = x_row[i] * rms * scale[i];
    }
}

extern "C" void launch_rmsnorm_forward(float* out, const float* x, const float* scale,
                                        int batch, int seq, int d, float eps) {
    int n_rows = batch * seq;
    // Round thread count up to power of 2 for clean reduction
    int threads = 1;
    while (threads < BLOCK_SIZE && threads < d) threads <<= 1;
    if (threads > 1024) threads = 1024;

    kernel_rmsnorm_forward<<<n_rows, threads, threads * sizeof(float)>>>(
        out, x, scale, n_rows, d, eps);
    CUDA_CHECK_KERNEL();
}

// ============================================================================
//  3. Causal Conv Forward
//     Multi-branch causal 1D convolution, accumulated into output.
//     One thread per (batch, seq, dim) element.
// ============================================================================

__global__ void kernel_causal_conv_forward(float* __restrict__ out,
                                            const float* __restrict__ x,
                                            const float* __restrict__ weights,
                                            const int* __restrict__ kernel_sizes,
                                            int n_branches, int max_kernel,
                                            int batch, int seq, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * d;
    if (idx >= total) return;

    int i = idx % d;
    int tmp = idx / d;
    int s = tmp % seq;
    int b = tmp / seq;

    float sum = 0.0f;
    for (int br = 0; br < n_branches; ++br) {
        int ks = kernel_sizes[br];
        const float* w_branch = weights + br * max_kernel * d;
        for (int k = 0; k < ks; ++k) {
            int src_pos = s - k;
            if (src_pos >= 0) {
                sum += x[(b * seq + src_pos) * d + i] * w_branch[k * d + i];
            }
        }
    }
    out[idx] = sum;
}

extern "C" void launch_causal_conv_forward(float* out, const float* x, const float* weights,
                                            const int* kernel_sizes_dev, int n_branches,
                                            int max_kernel,
                                            int batch, int seq, int d) {
    int total = batch * seq * d;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_causal_conv_forward<<<blocks, BLOCK_SIZE>>>(
        out, x, weights, kernel_sizes_dev, n_branches, max_kernel, batch, seq, d);
    CUDA_CHECK_KERNEL();
}

// ============================================================================
//  4. MinGRU Forward (sequential over seq, parallel over batch*d)
//     z = sigmoid(Wz @ x_t + h_prev)
//     h_tilde = Wh @ (z * x_t)
//     h_new = (1 - z) * h_prev + z * h_tilde
// ============================================================================

__global__ void kernel_mingru_forward_step(float* __restrict__ h_out,
                                            float* __restrict__ h_state,
                                            const float* __restrict__ x,
                                            const float* __restrict__ Wz,
                                            const float* __restrict__ Wh,
                                            int batch, int seq, int d, int s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * d) return;

    int b = idx / d;
    int i = idx % d;

    const float* x_t = x + (b * seq + s) * d;
    float h_prev_i = h_state[b * d + i];

    // z = sigmoid(Wz[i,:] @ x_t + h_prev[i])
    float z_val = 0.0f;
    for (int j = 0; j < d; ++j) {
        z_val += Wz[i * d + j] * x_t[j];
    }
    z_val += h_prev_i;
    z_val = 1.0f / (1.0f + expf(-z_val));

    // h_tilde = Wh[i,:] @ (z * x_t)
    float h_tilde = 0.0f;
    for (int j = 0; j < d; ++j) {
        h_tilde += Wh[i * d + j] * (z_val * x_t[j]);
    }

    // h_new = (1 - z) * h_prev + z * h_tilde
    float h_new = (1.0f - z_val) * h_prev_i + z_val * h_tilde;

    h_out[(b * seq + s) * d + i] = h_new;
    h_state[b * d + i] = h_new;
}

extern "C" void launch_mingru_forward(float* h_out, float* state_out,
                                       const float* x, const float* h_prev,
                                       const float* Wz, const float* Wh,
                                       int batch, int seq, int d) {
    int total_bd = batch * d;
    int blocks = (total_bd + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Copy initial hidden state into state_out (working buffer)
    cudaMemcpy(state_out, h_prev, total_bd * sizeof(float), cudaMemcpyDeviceToDevice);

    // Sequential over timesteps, parallel over batch*d
    for (int s = 0; s < seq; ++s) {
        kernel_mingru_forward_step<<<blocks, BLOCK_SIZE>>>(
            h_out, state_out, x, Wz, Wh, batch, seq, d, s);
        cudaDeviceSynchronize();  // Sequential dependency between timesteps
    }
    CUDA_CHECK_KERNEL();
}

// ============================================================================
//  5. Slot Memory Forward
//     Softmax attention over slot keys, weighted sum of slot values.
//     One block per (batch, seq) row.
// ============================================================================

__global__ void kernel_slot_memory_forward(float* __restrict__ out,
                                            const float* __restrict__ x,
                                            const float* __restrict__ slot_keys,
                                            const float* __restrict__ slot_values,
                                            int n_rows, int d, int n_slots) {
    int row = blockIdx.x;
    if (row >= n_rows) return;

    const float* query = x + row * d;
    float* dst = out + row * d;

    extern __shared__ float shared[];
    float* scores = shared;  // n_slots + 1 floats

    float inv_sqrt_d = rsqrtf(static_cast<float>(d));

    // Compute dot products
    for (int slot = threadIdx.x; slot < n_slots; slot += blockDim.x) {
        float dot = 0.0f;
        for (int i = 0; i < d; ++i) {
            dot += query[i] * slot_keys[slot * d + i];
        }
        scores[slot] = dot * inv_sqrt_d;
    }
    __syncthreads();

    // Find max (thread 0)
    if (threadIdx.x == 0) {
        float max_s = scores[0];
        for (int slot = 1; slot < n_slots; ++slot) {
            if (scores[slot] > max_s) max_s = scores[slot];
        }
        shared[n_slots] = max_s;
    }
    __syncthreads();

    float max_s = shared[n_slots];

    // Exponentiate
    for (int slot = threadIdx.x; slot < n_slots; slot += blockDim.x) {
        scores[slot] = expf(scores[slot] - max_s);
    }
    __syncthreads();

    // Sum (thread 0)
    if (threadIdx.x == 0) {
        float sum_exp = 0.0f;
        for (int slot = 0; slot < n_slots; ++slot) {
            sum_exp += scores[slot];
        }
        shared[n_slots] = sum_exp;
    }
    __syncthreads();

    // Normalize
    float sum_exp = shared[n_slots];
    for (int slot = threadIdx.x; slot < n_slots; slot += blockDim.x) {
        scores[slot] /= sum_exp;
    }
    __syncthreads();

    // Weighted sum of values
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float val = 0.0f;
        for (int slot = 0; slot < n_slots; ++slot) {
            val += scores[slot] * slot_values[slot * d + i];
        }
        dst[i] = val;
    }
}

extern "C" void launch_slot_memory_forward(float* out, const float* x,
                                            const float* slot_keys, const float* slot_values,
                                            int batch, int seq, int d, int n_slots) {
    int n_rows = batch * seq;
    size_t smem = (n_slots + 1) * sizeof(float);
    kernel_slot_memory_forward<<<n_rows, BLOCK_SIZE, smem>>>(
        out, x, slot_keys, slot_values, n_rows, d, n_slots);
    CUDA_CHECK_KERNEL();
}

// ============================================================================
//  6. SwiGLU Activation (element-wise part)
//     hidden[i] = up[i] * silu(gate[i])   where silu(x) = x / (1 + exp(-x))
//     The GEMM parts (x@W_up, x@W_gate, hidden@W_down) use cuBLAS on host side.
// ============================================================================

__global__ void kernel_swiglu_activation(float* __restrict__ hidden,
                                          const float* __restrict__ up,
                                          const float* __restrict__ gate,
                                          int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float g = gate[idx];
    float silu = g / (1.0f + expf(-g));
    hidden[idx] = up[idx] * silu;
}

extern "C" void launch_swiglu_activation(float* hidden, const float* up, const float* gate,
                                          int total) {
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_swiglu_activation<<<blocks, BLOCK_SIZE>>>(hidden, up, gate, total);
    CUDA_CHECK_KERNEL();
}

// ============================================================================
//  7. Cross Entropy Loss
//     Per-token: loss = -(logit[target] - max - log(sum(exp(logit - max))))
//     Two-phase: per-token losses, then parallel reduction.
// ============================================================================

__global__ void kernel_cross_entropy_per_token(float* __restrict__ token_losses,
                                                const float* __restrict__ logits,
                                                const int* __restrict__ targets,
                                                int n_tokens, int vocab) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tokens) return;

    const float* row = logits + idx * vocab;
    int tgt = targets[idx];

    float max_val = row[0];
    for (int v = 1; v < vocab; ++v) {
        if (row[v] > max_val) max_val = row[v];
    }

    float sum_exp = 0.0f;
    for (int v = 0; v < vocab; ++v) {
        sum_exp += expf(row[v] - max_val);
    }

    float log_softmax = row[tgt] - max_val - logf(sum_exp);
    token_losses[idx] = -log_softmax;
}

__global__ void kernel_reduce_sum(float* __restrict__ out,
                                   const float* __restrict__ in, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0f;
    if (idx < n) sum += in[idx];
    if (idx + blockDim.x < n) sum += in[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

extern "C" void launch_cross_entropy_loss(float* loss_out_host, const float* logits,
                                           const int* targets, int batch, int seq, int vocab) {
    int n_tokens = batch * seq;

    float* d_token_losses;
    cudaMalloc(&d_token_losses, n_tokens * sizeof(float));

    // Phase 1: per-token cross entropy
    int blocks1 = (n_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_cross_entropy_per_token<<<blocks1, BLOCK_SIZE>>>(
        d_token_losses, logits, targets, n_tokens, vocab);

    // Phase 2: parallel reduction to get sum
    int max_blocks = (n_tokens + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    float* d_reduce_buf;
    cudaMalloc(&d_reduce_buf, max_blocks * sizeof(float));

    float* d_a = d_token_losses;
    float* d_b = d_reduce_buf;
    int remaining = n_tokens;

    while (remaining > 1) {
        int blocks_r = (remaining + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
        kernel_reduce_sum<<<blocks_r, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
            d_b, d_a, remaining);
        remaining = blocks_r;
        float* tmp = d_a; d_a = d_b; d_b = tmp;
    }

    float total_loss;
    cudaMemcpy(&total_loss, d_a, sizeof(float), cudaMemcpyDeviceToHost);
    *loss_out_host = total_loss / static_cast<float>(n_tokens);

    cudaFree(d_token_losses);
    cudaFree(d_reduce_buf);
    CUDA_CHECK_KERNEL();
}

// ============================================================================
//  8. AdamW Step
//     Decoupled weight decay, bias-corrected moment estimates.
// ============================================================================

__global__ void kernel_adamw_step(float* __restrict__ params,
                                   const float* __restrict__ grads,
                                   float* __restrict__ m, float* __restrict__ v,
                                   float lr, float beta1, float beta2, float eps,
                                   float weight_decay, float bc1, float bc2, int n_params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_params) return;

    float p = params[idx];
    float g = grads[idx];

    // Decoupled weight decay
    p -= lr * weight_decay * p;

    // Moment updates
    float mi = beta1 * m[idx] + (1.0f - beta1) * g;
    float vi = beta2 * v[idx] + (1.0f - beta2) * g * g;
    m[idx] = mi;
    v[idx] = vi;

    // Bias-corrected estimates and parameter update
    float m_hat = mi / bc1;
    float v_hat = vi / bc2;
    p -= lr * m_hat / (sqrtf(v_hat) + eps);

    params[idx] = p;
}

extern "C" void launch_adamw_step(float* params, const float* grads, float* m, float* v,
                                   float lr, float beta1, float beta2, float eps,
                                   float weight_decay, int step, int n_params) {
    float bc1 = 1.0f - powf(beta1, static_cast<float>(step));
    float bc2 = 1.0f - powf(beta2, static_cast<float>(step));

    int blocks = (n_params + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_adamw_step<<<blocks, BLOCK_SIZE>>>(
        params, grads, m, v, lr, beta1, beta2, eps, weight_decay, bc1, bc2, n_params);
    CUDA_CHECK_KERNEL();
}

// ============================================================================
//  9. MinGRU Step (single token, inference)
// ============================================================================

__global__ void kernel_mingru_step_single(float* __restrict__ h_out,
                                           const float* __restrict__ x,
                                           const float* __restrict__ h_prev,
                                           const float* __restrict__ Wz,
                                           const float* __restrict__ Wh, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;

    float z_val = 0.0f;
    for (int j = 0; j < d; ++j) {
        z_val += Wz[i * d + j] * x[j];
    }
    z_val += h_prev[i];
    z_val = 1.0f / (1.0f + expf(-z_val));

    float h_tilde = 0.0f;
    for (int j = 0; j < d; ++j) {
        h_tilde += Wh[i * d + j] * (z_val * x[j]);
    }

    h_out[i] = (1.0f - z_val) * h_prev[i] + z_val * h_tilde;
}

extern "C" void launch_mingru_step(float* h_out, const float* x, const float* h_prev,
                                    const float* Wz, const float* Wh, int d) {
    int blocks = (d + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_mingru_step_single<<<blocks, BLOCK_SIZE>>>(h_out, x, h_prev, Wz, Wh, d);
    CUDA_CHECK_KERNEL();
}

// ============================================================================
// 10. Conv Step (single token, inference)
//     Shift rolling buffer, insert new token, convolve.
// ============================================================================

__global__ void kernel_conv_step_shift(float* __restrict__ buffer,
                                        const float* __restrict__ x,
                                        const int* __restrict__ kernel_sizes,
                                        int n_branches, int max_kernel, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_branches * d) return;

    int br = idx / d;
    int i = idx % d;
    int ks = kernel_sizes[br];
    float* br_buf = buffer + br * max_kernel * d;

    // Shift down
    for (int k = 0; k < ks - 1; ++k) {
        br_buf[k * d + i] = br_buf[(k + 1) * d + i];
    }
    // Insert new token
    br_buf[(ks - 1) * d + i] = x[i];
}

__global__ void kernel_conv_step_convolve(float* __restrict__ out,
                                           const float* __restrict__ buffer,
                                           const float* __restrict__ weights,
                                           const int* __restrict__ kernel_sizes,
                                           int n_branches, int max_kernel, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;

    float sum = 0.0f;
    for (int br = 0; br < n_branches; ++br) {
        int ks = kernel_sizes[br];
        const float* br_buf = buffer + br * max_kernel * d;
        const float* w_branch = weights + br * max_kernel * d;
        for (int k = 0; k < ks; ++k) {
            sum += br_buf[k * d + i] * w_branch[k * d + i];
        }
    }
    out[i] = sum;
}

extern "C" void launch_conv_step(float* out, float* buffer, const float* x, const float* weights,
                                  const int* kernel_sizes_dev, int n_branches, int max_kernel,
                                  int d) {
    int blocks1 = (n_branches * d + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_conv_step_shift<<<blocks1, BLOCK_SIZE>>>(
        buffer, x, kernel_sizes_dev, n_branches, max_kernel, d);

    int blocks2 = (d + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_conv_step_convolve<<<blocks2, BLOCK_SIZE>>>(
        out, buffer, weights, kernel_sizes_dev, n_branches, max_kernel, d);
    CUDA_CHECK_KERNEL();
}

// ============================================================================
// 11. Slot Query (single token, inference)
// ============================================================================

__global__ void kernel_slot_query(float* __restrict__ out,
                                   const float* __restrict__ x,
                                   const float* __restrict__ slot_keys,
                                   const float* __restrict__ slot_values,
                                   int d, int n_slots) {
    extern __shared__ float shared[];
    float* scores = shared;

    float inv_sqrt_d = rsqrtf(static_cast<float>(d));

    for (int slot = threadIdx.x; slot < n_slots; slot += blockDim.x) {
        float dot = 0.0f;
        for (int i = 0; i < d; ++i) {
            dot += x[i] * slot_keys[slot * d + i];
        }
        scores[slot] = dot * inv_sqrt_d;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float max_s = scores[0];
        for (int slot = 1; slot < n_slots; ++slot) {
            if (scores[slot] > max_s) max_s = scores[slot];
        }
        shared[n_slots] = max_s;
    }
    __syncthreads();

    float max_s = shared[n_slots];
    for (int slot = threadIdx.x; slot < n_slots; slot += blockDim.x) {
        scores[slot] = expf(scores[slot] - max_s);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum_exp = 0.0f;
        for (int slot = 0; slot < n_slots; ++slot) sum_exp += scores[slot];
        shared[n_slots] = sum_exp;
    }
    __syncthreads();

    float sum_exp = shared[n_slots];
    for (int slot = threadIdx.x; slot < n_slots; slot += blockDim.x) {
        scores[slot] /= sum_exp;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float val = 0.0f;
        for (int slot = 0; slot < n_slots; ++slot) {
            val += scores[slot] * slot_values[slot * d + i];
        }
        out[i] = val;
    }
}

extern "C" void launch_slot_query(float* out, const float* x,
                                   const float* slot_keys, const float* slot_values,
                                   int d, int n_slots) {
    size_t smem = (n_slots + 1) * sizeof(float);
    kernel_slot_query<<<1, BLOCK_SIZE, smem>>>(out, x, slot_keys, slot_values, d, n_slots);
    CUDA_CHECK_KERNEL();
}

#endif  // RNET_HAS_CUDA

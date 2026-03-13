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

// ============================================================================
// ======================= BACKWARD (GRADIENT) KERNELS ========================
// ============================================================================

// ============================================================================
// 12. Cross Entropy Backward
//     d_logits[i,v] = (softmax(logits[i,:])[v] - one_hot(targets[i])[v]) / n_tokens
//     One block per token row, shared memory for max/sum reduction.
// ============================================================================

__global__ void kernel_cross_entropy_backward(float* __restrict__ d_logits,
                                               const float* __restrict__ logits,
                                               const int* __restrict__ targets,
                                               int n_tokens, int vocab) {
    int row = blockIdx.x;
    if (row >= n_tokens) return;

    const float* logit_row = logits + row * vocab;
    float* d_row = d_logits + row * vocab;
    int tgt = targets[row];
    float inv_n = 1.0f / static_cast<float>(n_tokens);

    extern __shared__ float sdata[];

    // Phase 1: find max across vocab (shared memory reduction)
    float local_max = -1e30f;
    for (int v = threadIdx.x; v < vocab; v += blockDim.x) {
        float val = logit_row[v];
        if (val > local_max) local_max = val;
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata[threadIdx.x + s] > sdata[threadIdx.x])
                sdata[threadIdx.x] = sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();  // ensure all threads read max_val before reusing sdata

    // Phase 2: compute sum of exp(logit - max)
    float local_sum = 0.0f;
    for (int v = threadIdx.x; v < vocab; v += blockDim.x) {
        local_sum += expf(logit_row[v] - max_val);
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    float sum_exp = sdata[0];
    __syncthreads();  // ensure all threads read sum_exp before writing output

    // Phase 3: write gradients
    for (int v = threadIdx.x; v < vocab; v += blockDim.x) {
        float softmax_v = expf(logit_row[v] - max_val) / sum_exp;
        float indicator = (v == tgt) ? 1.0f : 0.0f;
        d_row[v] = (softmax_v - indicator) * inv_n;
    }
}

extern "C" void launch_cross_entropy_backward(float* d_logits, const float* logits,
                                                const int* targets, int batch, int seq, int vocab) {
    int n_tokens = batch * seq;
    int threads = BLOCK_SIZE;
    if (threads > 1024) threads = 1024;
    kernel_cross_entropy_backward<<<n_tokens, threads, threads * sizeof(float)>>>(
        d_logits, logits, targets, n_tokens, vocab);
    CUDA_CHECK_KERNEL();
}

// ============================================================================
// 13. Embedding Backward
//     d_weight[tokens[i], j] += d_out[i, j]
//     One thread per (token_position * d_model). atomicAdd for collisions.
// ============================================================================

__global__ void kernel_embedding_backward(float* __restrict__ d_weight,
                                            const float* __restrict__ d_out,
                                            const int* __restrict__ tokens,
                                            int total_tokens, int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_tokens * d_model) return;

    int token_idx = idx / d_model;
    int dim_idx = idx % d_model;
    int token_id = tokens[token_idx];

    atomicAdd(&d_weight[token_id * d_model + dim_idx], d_out[idx]);
}

extern "C" void launch_embedding_backward(float* d_weight, const float* d_out,
                                            const int* tokens, int batch, int seq, int d_model) {
    int total = batch * seq * d_model;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_embedding_backward<<<blocks, BLOCK_SIZE>>>(d_weight, d_out, tokens, batch * seq, d_model);
    CUDA_CHECK_KERNEL();
}

// ============================================================================
// 14. RMSNorm Backward
//     One block per row. Shared memory reduction for ds.
//     d_x[j] = rms * (d_out[j] * scale[j] - x[j] * rms^2 * ds / d)
//     d_scale[j] += d_out[j] * x[j] * rms   (atomicAdd across rows)
// ============================================================================

__global__ void kernel_rmsnorm_backward(float* __restrict__ d_x,
                                         float* __restrict__ d_scale,
                                         const float* __restrict__ d_out,
                                         const float* __restrict__ x,
                                         const float* __restrict__ scale,
                                         int n_rows, int d, float eps) {
    int row = blockIdx.x;
    if (row >= n_rows) return;

    const float* x_row = x + row * d;
    const float* d_out_row = d_out + row * d;
    float* d_x_row = d_x + row * d;

    extern __shared__ float sdata[];

    // Compute sum of squares for this row
    float local_ss = 0.0f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float val = x_row[j];
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
    float ss = sdata[0] / static_cast<float>(d);
    float rms = rsqrtf(ss + eps);
    __syncthreads();  // ensure all threads read sdata[0] before reusing shared memory

    // Compute ds = sum_j(d_out[j] * x[j] * scale[j])
    float local_ds = 0.0f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        local_ds += d_out_row[j] * x_row[j] * scale[j];
    }
    sdata[threadIdx.x] = local_ds;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    float ds = sdata[0];

    // Write d_x and accumulate d_scale
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float d_x_val = rms * (d_out_row[j] * scale[j] - x_row[j] * rms * rms * ds / static_cast<float>(d));
        d_x_row[j] = d_x_val;
        atomicAdd(&d_scale[j], d_out_row[j] * x_row[j] * rms);
    }
}

extern "C" void launch_rmsnorm_backward(float* d_x, float* d_scale, const float* d_out,
                                          const float* x, const float* scale,
                                          int batch, int seq, int d, float eps) {
    int n_rows = batch * seq;
    int threads = 1;
    while (threads < BLOCK_SIZE && threads < d) threads <<= 1;
    if (threads > 1024) threads = 1024;

    kernel_rmsnorm_backward<<<n_rows, threads, threads * sizeof(float)>>>(
        d_x, d_scale, d_out, x, scale, n_rows, d, eps);
    CUDA_CHECK_KERNEL();
}

// ============================================================================
// 15. Causal Conv Backward
//     d_x: one thread per (batch * seq * d)
//     d_w: accumulated via atomicAdd
// ============================================================================

__global__ void kernel_causal_conv_backward_dx(float* __restrict__ d_x,
                                                const float* __restrict__ d_out,
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
            int dst_pos = s + k;
            if (dst_pos < seq) {
                sum += d_out[(b * seq + dst_pos) * d + i] * w_branch[k * d + i];
            }
        }
    }
    d_x[idx] = sum;
}

__global__ void kernel_causal_conv_backward_dw(float* __restrict__ d_weights,
                                                const float* __restrict__ d_out,
                                                const float* __restrict__ x,
                                                const int* __restrict__ kernel_sizes,
                                                int n_branches, int max_kernel,
                                                int batch, int seq, int d) {
    // One thread per (batch * seq * d), accumulate into d_weights via atomicAdd
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * d;
    if (idx >= total) return;

    int i = idx % d;
    int tmp = idx / d;
    int s = tmp % seq;
    int b = tmp / seq;

    float d_out_val = d_out[idx];

    for (int br = 0; br < n_branches; ++br) {
        int ks = kernel_sizes[br];
        float* dw_branch = d_weights + br * max_kernel * d;
        for (int k = 0; k < ks; ++k) {
            int src_pos = s - k;
            if (src_pos >= 0) {
                atomicAdd(&dw_branch[k * d + i], d_out_val * x[(b * seq + src_pos) * d + i]);
            }
        }
    }
}

extern "C" void launch_causal_conv_backward(float* d_x, float* d_weights,
                                              const float* d_out, const float* x,
                                              const float* weights,
                                              const int* kernel_sizes_dev, int n_branches,
                                              int max_kernel,
                                              int batch, int seq, int d) {
    int total = batch * seq * d;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // d_x needs the original forward weights
    kernel_causal_conv_backward_dx<<<blocks, BLOCK_SIZE>>>(
        d_x, d_out, weights, kernel_sizes_dev, n_branches, max_kernel, batch, seq, d);
    CUDA_CHECK_KERNEL();

    // d_weights accumulates weight gradients via atomicAdd (must be pre-zeroed by caller)
    kernel_causal_conv_backward_dw<<<blocks, BLOCK_SIZE>>>(
        d_weights, d_out, x, kernel_sizes_dev, n_branches, max_kernel, batch, seq, d);
    CUDA_CHECK_KERNEL();
}

// ============================================================================
// 16. MinGRU Backward (BPTT)
//     Sub-kernel A: per (b,i) — compute intermediates, atomicAdd to d_Wz/d_Wh,
//                   atomicAdd to d_x, update d_h_next.
//     Called once per timestep, looping backward on host.
// ============================================================================

__global__ void kernel_mingru_backward_step(float* __restrict__ d_x,
                                              float* __restrict__ d_Wz,
                                              float* __restrict__ d_Wh,
                                              float* __restrict__ d_h_next,
                                              const float* __restrict__ d_h_out,
                                              const float* __restrict__ x,
                                              const float* __restrict__ h_all,
                                              const float* __restrict__ h_init,
                                              const float* __restrict__ Wz,
                                              const float* __restrict__ Wh,
                                              int batch, int seq, int d, int s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * d) return;

    int b = idx / d;
    int i = idx % d;

    const float* x_t = x + (b * seq + s) * d;

    // Get h_prev for this timestep
    float h_prev_i = (s > 0) ? h_all[(b * seq + (s - 1)) * d + i] : h_init[b * d + i];

    // Recompute forward values
    float pre_z = h_prev_i;
    for (int j = 0; j < d; ++j) {
        pre_z += Wz[i * d + j] * x_t[j];
    }
    float z_i = 1.0f / (1.0f + expf(-pre_z));

    float h_tilde_i = 0.0f;
    for (int j = 0; j < d; ++j) {
        h_tilde_i += Wh[i * d + j] * (z_i * x_t[j]);
    }

    // Incoming gradient: from output + propagated from future timestep
    float d_h_i = d_h_out[(b * seq + s) * d + i] + d_h_next[b * d + i];

    // Through update: h = (1 - z) * h_prev + z * h_tilde
    float d_h_tilde_i = d_h_i * z_i;
    float d_z_update_i = d_h_i * (h_tilde_i - h_prev_i);
    float d_h_prev_i = d_h_i * (1.0f - z_i);

    // Through h_tilde = sum_j(Wh[i,j] * z * x[j])
    // d_z from h_tilde path: d_z_htilde = d_h_tilde * sum_j(Wh[i,j] * x[j])
    float wx_sum = 0.0f;
    for (int j = 0; j < d; ++j) {
        wx_sum += Wh[i * d + j] * x_t[j];
    }
    float d_z_htilde_i = d_h_tilde_i * wx_sum;

    // Accumulate d_Wh: d_Wh[i,j] += d_h_tilde * z * x[j]
    for (int j = 0; j < d; ++j) {
        atomicAdd(&d_Wh[i * d + j], d_h_tilde_i * z_i * x_t[j]);
    }

    // Total z gradient and sigmoid backward
    float d_z_total_i = d_z_update_i + d_z_htilde_i;
    float d_pre_z_i = d_z_total_i * z_i * (1.0f - z_i);

    // Accumulate d_Wz: d_Wz[i,j] += d_pre_z * x[j]
    for (int j = 0; j < d; ++j) {
        atomicAdd(&d_Wz[i * d + j], d_pre_z_i * x_t[j]);
    }

    // Accumulate d_x via atomicAdd (reduction over i dimension)
    // d_x[j] += d_h_tilde * Wh[i,j] * z + d_pre_z * Wz[i,j]
    float* d_x_t = d_x + (b * seq + s) * d;
    for (int j = 0; j < d; ++j) {
        float contrib = d_h_tilde_i * Wh[i * d + j] * z_i + d_pre_z_i * Wz[i * d + j];
        atomicAdd(&d_x_t[j], contrib);
    }

    // Propagate to previous timestep
    d_h_next[b * d + i] = d_h_prev_i + d_pre_z_i;
}

extern "C" void launch_mingru_backward(float* d_x, float* d_Wz, float* d_Wh,
                                         float* d_h_next,
                                         const float* d_h_out, const float* x,
                                         const float* h_all, const float* h_init,
                                         const float* Wz, const float* Wh,
                                         int batch, int seq, int d) {
    int total_bd = batch * d;
    int blocks = (total_bd + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Loop backward through timesteps (sequential dependency)
    for (int s = seq - 1; s >= 0; --s) {
        kernel_mingru_backward_step<<<blocks, BLOCK_SIZE>>>(
            d_x, d_Wz, d_Wh, d_h_next,
            d_h_out, x, h_all, h_init, Wz, Wh,
            batch, seq, d, s);
        cudaDeviceSynchronize();
    }
    CUDA_CHECK_KERNEL();
}

// ============================================================================
// 17. Slot Memory Backward
//     One block per (batch, seq) row.
//     Recomputes softmax scores, then backpropagates through attention.
// ============================================================================

__global__ void kernel_slot_memory_backward(float* __restrict__ d_x,
                                              float* __restrict__ d_keys,
                                              float* __restrict__ d_values,
                                              const float* __restrict__ d_out,
                                              const float* __restrict__ x,
                                              const float* __restrict__ keys,
                                              const float* __restrict__ values,
                                              int n_rows, int d, int n_slots) {
    int row = blockIdx.x;
    if (row >= n_rows) return;

    const float* query = x + row * d;
    const float* d_out_row = d_out + row * d;
    float* d_x_row = d_x + row * d;

    extern __shared__ float shared[];
    float* scores = shared;                     // n_slots floats
    float* d_pre_softmax = shared + n_slots;    // n_slots floats

    float inv_sqrt_d = rsqrtf(static_cast<float>(d));

    // Recompute attention scores
    for (int slot = threadIdx.x; slot < n_slots; slot += blockDim.x) {
        float dot = 0.0f;
        for (int i = 0; i < d; ++i) {
            dot += query[i] * keys[slot * d + i];
        }
        scores[slot] = dot * inv_sqrt_d;
    }
    __syncthreads();

    // Softmax: find max
    if (threadIdx.x == 0) {
        float max_s = scores[0];
        for (int slot = 1; slot < n_slots; ++slot) {
            if (scores[slot] > max_s) max_s = scores[slot];
        }
        shared[2 * n_slots] = max_s;  // store max in extra slot
    }
    __syncthreads();

    float max_s = shared[2 * n_slots];

    // Exponentiate
    for (int slot = threadIdx.x; slot < n_slots; slot += blockDim.x) {
        scores[slot] = expf(scores[slot] - max_s);
    }
    __syncthreads();

    // Sum
    if (threadIdx.x == 0) {
        float sum_exp = 0.0f;
        for (int slot = 0; slot < n_slots; ++slot) {
            sum_exp += scores[slot];
        }
        shared[2 * n_slots] = sum_exp;
    }
    __syncthreads();

    float sum_exp = shared[2 * n_slots];
    for (int slot = threadIdx.x; slot < n_slots; slot += blockDim.x) {
        scores[slot] /= sum_exp;
    }
    __syncthreads();

    // d_scores[slot] = sum_i(d_out[row, i] * values[slot, i])
    for (int slot = threadIdx.x; slot < n_slots; slot += blockDim.x) {
        float ds = 0.0f;
        for (int i = 0; i < d; ++i) {
            ds += d_out_row[i] * values[slot * d + i];
        }
        d_pre_softmax[slot] = ds;
    }
    __syncthreads();

    // Softmax backward: dot = sum_slot(d_scores[slot] * scores[slot])
    if (threadIdx.x == 0) {
        float dot = 0.0f;
        for (int slot = 0; slot < n_slots; ++slot) {
            dot += d_pre_softmax[slot] * scores[slot];
        }
        shared[2 * n_slots] = dot;
    }
    __syncthreads();

    float dot = shared[2 * n_slots];
    for (int slot = threadIdx.x; slot < n_slots; slot += blockDim.x) {
        d_pre_softmax[slot] = scores[slot] * (d_pre_softmax[slot] - dot) * inv_sqrt_d;
    }
    __syncthreads();

    // d_x[row, i] = sum_slot(d_pre_softmax[slot] * keys[slot, i])
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float val = 0.0f;
        for (int slot = 0; slot < n_slots; ++slot) {
            val += d_pre_softmax[slot] * keys[slot * d + i];
        }
        d_x_row[i] = val;
    }

    // d_keys[slot, i] += d_pre_softmax[slot] * query[i]  (atomicAdd across rows)
    for (int slot = threadIdx.x; slot < n_slots; slot += blockDim.x) {
        float dps = d_pre_softmax[slot];
        for (int i = 0; i < d; ++i) {
            atomicAdd(&d_keys[slot * d + i], dps * query[i]);
        }
    }

    // d_values[slot, i] += scores[slot] * d_out[row, i]  (atomicAdd across rows)
    for (int slot = threadIdx.x; slot < n_slots; slot += blockDim.x) {
        float sc = scores[slot];
        for (int i = 0; i < d; ++i) {
            atomicAdd(&d_values[slot * d + i], sc * d_out_row[i]);
        }
    }
}

extern "C" void launch_slot_memory_backward(float* d_x, float* d_keys, float* d_values,
                                              const float* d_out, const float* x,
                                              const float* keys, const float* values,
                                              int batch, int seq, int d, int n_slots) {
    int n_rows = batch * seq;
    // Need: 2 * n_slots for scores + d_pre_softmax, plus 1 extra for temporaries
    size_t smem = (2 * n_slots + 1) * sizeof(float);
    kernel_slot_memory_backward<<<n_rows, BLOCK_SIZE, smem>>>(
        d_x, d_keys, d_values, d_out, x, keys, values, n_rows, d, n_slots);
    CUDA_CHECK_KERNEL();
}

// ============================================================================
// 18. SwiGLU Backward Activation (element-wise)
//     Computes d_up and d_gate from d_hidden, up, gate.
//     The GEMM parts are handled by cuBLAS in cuda_backend.cpp.
// ============================================================================

__global__ void kernel_swiglu_backward_activation(float* __restrict__ d_up,
                                                    float* __restrict__ d_gate,
                                                    const float* __restrict__ d_hidden,
                                                    const float* __restrict__ up,
                                                    const float* __restrict__ gate,
                                                    int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float g = gate[idx];
    float sigmoid_g = 1.0f / (1.0f + expf(-g));
    float silu_g = g * sigmoid_g;

    d_up[idx] = d_hidden[idx] * silu_g;
    d_gate[idx] = d_hidden[idx] * up[idx] * sigmoid_g * (1.0f + g * (1.0f - sigmoid_g));
}

extern "C" void launch_swiglu_backward_activation(float* d_up, float* d_gate,
                                                     const float* d_hidden,
                                                     const float* up, const float* gate,
                                                     int total) {
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_swiglu_backward_activation<<<blocks, BLOCK_SIZE>>>(
        d_up, d_gate, d_hidden, up, gate, total);
    CUDA_CHECK_KERNEL();
}

#endif  // RNET_HAS_CUDA

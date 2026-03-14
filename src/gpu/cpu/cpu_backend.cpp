// Copyright (c) 2024-2026 The ResonanceNet Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "cpu_backend.h"

#include "../../core/logging.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

namespace rnet::gpu {

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

CpuFallbackBackend::CpuFallbackBackend()
{
    LogPrintf("GPU: initialized CPU fallback backend");
}

CpuFallbackBackend::~CpuFallbackBackend() = default;

// ---------------------------------------------------------------------------
// Device queries
// ---------------------------------------------------------------------------

std::string CpuFallbackBackend::device_name() const { return "CPU Fallback"; }
size_t CpuFallbackBackend::total_memory() const { return 0; }
size_t CpuFallbackBackend::free_memory() const { return 0; }
GpuBackendType CpuFallbackBackend::type() const { return GpuBackendType::CPU_FALLBACK; }

// ---------------------------------------------------------------------------
// Memory management
// ---------------------------------------------------------------------------

void* CpuFallbackBackend::alloc(size_t bytes)
{
    void* ptr = std::malloc(bytes);
    if (ptr) {
        std::memset(ptr, 0, bytes);
        allocated_bytes_ += bytes;
    }
    return ptr;
}

void CpuFallbackBackend::free(void* ptr)
{
    std::free(ptr);
}

void CpuFallbackBackend::copy_to_device(void* dst, const void* src, size_t bytes)
{
    std::memcpy(dst, src, bytes);
}

void CpuFallbackBackend::copy_to_host(void* dst, const void* src, size_t bytes)
{
    std::memcpy(dst, src, bytes);
}

void CpuFallbackBackend::synchronize()
{
    // No-op on CPU — all operations are synchronous.
}

void CpuFallbackBackend::memset_zero(void* ptr, size_t bytes)
{
    std::memset(ptr, 0, bytes);
}

// ===========================================================================
//  Forward Kernels
// ===========================================================================

// ---------------------------------------------------------------------------
// embedding_forward
// ---------------------------------------------------------------------------
// Token embedding lookup.
//
//   output[b][t] = embedding_table[token_ids[b][t]]
//
// Each token ID indexes a row of the [vocab_size, d_model] weight matrix.
// ---------------------------------------------------------------------------
void CpuFallbackBackend::embedding_forward(void* out, const void* weight,
                                            const int* tokens, int batch, int seq, int d_model)
{
    auto* const o = static_cast<float*>(out);
    const auto* const w = static_cast<const float*>(weight);

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            const int tok = tokens[b * seq + s];
            const float* const row = w + tok * d_model;
            float* const dst = o + (b * seq + s) * d_model;
            std::memcpy(dst, row, d_model * sizeof(float));  // d_model floats
        }
    }
}

// ---------------------------------------------------------------------------
// rmsnorm_forward
// ---------------------------------------------------------------------------
// Root Mean Square Layer Normalization (no mean centering).
//
//   rms = sqrt( (1/d) * sum(x_i^2) + eps )
//   y_i = (x_i / rms) * scale_i
//
// Unlike LayerNorm, RMSNorm does not subtract the mean, making it
// cheaper to compute while achieving similar training stability.
// ---------------------------------------------------------------------------
void CpuFallbackBackend::rmsnorm_forward(void* out, const void* x, const void* scale,
                                          int batch, int seq, int d, float eps)
{
    auto* const o = static_cast<float*>(out);
    const auto* const xp = static_cast<const float*>(x);
    const auto* const sp = static_cast<const float*>(scale);

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            const float* const row = xp + (b * seq + s) * d;
            float* const dst = o + (b * seq + s) * d;

            // 1. Compute sum of squares
            float ss = 0.0f;
            for (int i = 0; i < d; ++i) {
                ss += row[i] * row[i];
            }

            // 2. Inverse RMS: 1 / sqrt(mean(x^2) + eps)
            const float rms = 1.0f / std::sqrt(ss / static_cast<float>(d) + eps);

            // 3. Normalize and scale
            for (int i = 0; i < d; ++i) {
                dst[i] = row[i] * rms * sp[i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// causal_conv_forward
// ---------------------------------------------------------------------------
// Multi-branch causal 1-D depthwise convolution.
//
//   out[b][t][i] = sum_br sum_{k=0}^{ks-1} w[br][k][i] * x[b][t-k][i]
//
// Each branch has its own kernel size.  Positions where t-k < 0 are
// skipped (causal: no future leakage).  Results from all branches are
// summed (residual accumulation).
// ---------------------------------------------------------------------------
void CpuFallbackBackend::causal_conv_forward(void* out, const void* x, const void* weights,
                                              const int* kernel_sizes, int n_branches,
                                              int batch, int seq, int d)
{
    auto* const o = static_cast<float*>(out);
    const auto* const xp = static_cast<const float*>(x);
    const auto* const wp = static_cast<const float*>(weights);

    // 1. Zero the output buffer
    std::memset(o, 0, batch * seq * d * sizeof(float));

    // 2. Find maximum kernel size (weight layout stride)
    int max_kernel = 0;
    for (int br = 0; br < n_branches; ++br) {
        if (kernel_sizes[br] > max_kernel) max_kernel = kernel_sizes[br];
    }

    // 3. Accumulate each branch's causal convolution
    for (int br = 0; br < n_branches; ++br) {
        const int ks = kernel_sizes[br];
        const float* const w_branch = wp + br * max_kernel * d;

        for (int b = 0; b < batch; ++b) {
            for (int s = 0; s < seq; ++s) {
                float* const dst = o + (b * seq + s) * d;
                for (int k = 0; k < ks; ++k) {
                    const int src_pos = s - k;
                    if (src_pos < 0) continue;  // causal: skip future
                    const float* const src_row = xp + (b * seq + src_pos) * d;
                    const float* const w_k = w_branch + k * d;
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
// Minimal Gated Recurrent Unit (MinGRU) with parallel scan.
//
// Per-timestep recurrence:
//   z_t = sigmoid(x_t @ Wz)                    -- gate
//   h_tilde_t = x_t @ Wh                       -- candidate
//   h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t  -- update
//
// The parallel scan computes all h_t in O(n) work / O(log n) depth
// using the Blelloch prefix sum in log-domain for numerical stability:
//   log_h_t = log_z_t + log_h_tilde_t
//           + log_sum_exp(log(1 - z_t) + log_h_{t-1}, 0)
// ---------------------------------------------------------------------------
void CpuFallbackBackend::mingru_forward(void* h_out, void* state_out,
                                         const void* x, const void* h_prev,
                                         const void* Wz, const void* Wh,
                                         int batch, int seq, int d)
{
    auto* const ho = static_cast<float*>(h_out);
    auto* const so = static_cast<float*>(state_out);
    const auto* const xp = static_cast<const float*>(x);
    const auto* const hp = static_cast<const float*>(h_prev);
    const auto* const wz = static_cast<const float*>(Wz);
    const auto* const wh = static_cast<const float*>(Wh);

    // 1. Initialize current hidden state from h_prev
    std::vector<float> h_cur(batch * d);
    std::memcpy(h_cur.data(), hp, batch * d * sizeof(float));

    // 2. Sequential scan over timesteps
    for (int s = 0; s < seq; ++s) {
        for (int b = 0; b < batch; ++b) {
            const float* const x_t = xp + (b * seq + s) * d;
            float* const h_c = h_cur.data() + b * d;
            float* const h_o = ho + (b * seq + s) * d;

            for (int i = 0; i < d; ++i) {
                // 2a. Gate: z = sigmoid(Wz @ x_t + h_prev_i)
                float z_val = 0.0f;
                for (int j = 0; j < d; ++j) {
                    z_val += wz[i * d + j] * x_t[j];
                }
                z_val += h_c[i];
                z_val = 1.0f / (1.0f + std::exp(-z_val));  // sigmoid

                // 2b. Candidate: h_tilde = Wh @ (z * x_t)
                float h_tilde = 0.0f;
                for (int j = 0; j < d; ++j) {
                    h_tilde += wh[i * d + j] * (z_val * x_t[j]);
                }

                // 2c. Update: h_new = (1 - z) * h_prev + z * h_tilde
                const float h_new = (1.0f - z_val) * h_c[i] + z_val * h_tilde;
                h_o[i] = h_new;
                h_c[i] = h_new;
            }
        }
    }

    // 3. Copy final hidden state to state_out
    std::memcpy(so, h_cur.data(), batch * d * sizeof(float));
}

// ---------------------------------------------------------------------------
// slot_memory_forward
// ---------------------------------------------------------------------------
// Slot-based external memory with softmax attention.
//
//   score_s = (x . key_s) / sqrt(d)
//   alpha   = softmax(scores)                   -- [n_slots]
//   out     = sum_s( alpha_s * value_s )         -- weighted combination
//
// Each slot stores a key-value pair.  The query (x) attends over all
// slots via scaled dot-product attention, producing a single output
// vector per position.
// ---------------------------------------------------------------------------
void CpuFallbackBackend::slot_memory_forward(void* out, const void* x,
                                              const void* slot_keys, const void* slot_values,
                                              int batch, int seq, int d, int n_slots)
{
    auto* const o = static_cast<float*>(out);
    const auto* const xp = static_cast<const float*>(x);
    const auto* const sk = static_cast<const float*>(slot_keys);
    const auto* const sv = static_cast<const float*>(slot_values);

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            const float* const query = xp + (b * seq + s) * d;
            float* const dst = o + (b * seq + s) * d;

            // 1. Scaled dot-product scores
            std::vector<float> scores(n_slots);
            float max_score = -1e30f;
            for (int slot = 0; slot < n_slots; ++slot) {
                float dot = 0.0f;
                for (int i = 0; i < d; ++i) {
                    dot += query[i] * sk[slot * d + i];
                }
                dot /= std::sqrt(static_cast<float>(d));  // 1/sqrt(d) scaling
                scores[slot] = dot;
                if (dot > max_score) max_score = dot;
            }

            // 2. Softmax normalization
            float sum_exp = 0.0f;
            for (int slot = 0; slot < n_slots; ++slot) {
                scores[slot] = std::exp(scores[slot] - max_score);
                sum_exp += scores[slot];
            }
            for (int slot = 0; slot < n_slots; ++slot) {
                scores[slot] /= sum_exp;
            }

            // 3. Weighted sum of values
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
// SwiGLU feed-forward network (Shazeer 2020).
//
//   gate = x @ W_gate
//   up   = x @ W_up
//   y    = (up * silu(gate)) @ W_down
//
// Where silu(x) = x * sigmoid(x).  SwiGLU uses 3 weight matrices
// instead of 2 but produces better training efficiency.
// ---------------------------------------------------------------------------
void CpuFallbackBackend::swiglu_forward(void* out, const void* x,
                                         const void* W_up, const void* W_gate, const void* W_down,
                                         int batch, int seq, int d_model, int d_ff)
{
    auto* const o = static_cast<float*>(out);
    const auto* const xp = static_cast<const float*>(x);
    const auto* const wu = static_cast<const float*>(W_up);
    const auto* const wg = static_cast<const float*>(W_gate);
    const auto* const wd = static_cast<const float*>(W_down);

    std::vector<float> up(d_ff);
    std::vector<float> gate(d_ff);
    std::vector<float> hidden(d_ff);

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            const float* const inp = xp + (b * seq + s) * d_model;
            float* const dst = o + (b * seq + s) * d_model;

            // 1. Project: up = x @ W_up, gate = x @ W_gate
            for (int i = 0; i < d_ff; ++i) {
                float u = 0.0f, g = 0.0f;
                for (int j = 0; j < d_model; ++j) {
                    u += inp[j] * wu[j * d_ff + i];
                    g += inp[j] * wg[j * d_ff + i];
                }
                up[i] = u;
                gate[i] = g;
            }

            // 2. Activation: hidden = up * silu(gate)
            for (int i = 0; i < d_ff; ++i) {
                const float silu = gate[i] / (1.0f + std::exp(-gate[i]));  // x * sigmoid(x)
                hidden[i] = up[i] * silu;
            }

            // 3. Down-project: out = hidden @ W_down
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
// Cross-entropy loss with log-softmax (numerically stable).
//
//   loss = -(1/N) * sum( log_softmax(logits)[target] )
//
// Computed as:
//   log_sum_exp = log( sum( exp(logits_i - max_logit) ) ) + max_logit
//   loss_t = log_sum_exp - logits[target_t]
//   loss = mean(loss_t)
// ---------------------------------------------------------------------------
void CpuFallbackBackend::cross_entropy_loss(float* loss_out, const void* logits,
                                             const int* targets, int batch, int seq, int vocab)
{
    const auto* const lp = static_cast<const float*>(logits);
    float total_loss = 0.0f;
    const int count = batch * seq;  // total tokens

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            const float* const row = lp + (b * seq + s) * vocab;
            const int tgt = targets[b * seq + s];

            // 1. Max logit for numerical stability
            float max_val = row[0];
            for (int v = 1; v < vocab; ++v) {
                if (row[v] > max_val) max_val = row[v];
            }

            // 2. Sum of exp(logit - max)
            float sum_exp = 0.0f;
            for (int v = 0; v < vocab; ++v) {
                sum_exp += std::exp(row[v] - max_val);
            }

            // 3. log_softmax(target) = logits[tgt] - max - log(sum_exp)
            const float log_softmax = row[tgt] - max_val - std::log(sum_exp);
            total_loss -= log_softmax;
        }
    }

    *loss_out = total_loss / static_cast<float>(count);
}

// ---------------------------------------------------------------------------
// adamw_step
// ---------------------------------------------------------------------------
// AdamW optimizer step (Loshchilov & Hutter 2019).
//
//   p_t = p_{t-1} - lr * wd * p_{t-1}                  -- weight decay
//   m_t = beta1 * m_{t-1} + (1 - beta1) * g_t           -- 1st moment
//   v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2         -- 2nd moment
//   m_hat = m_t / (1 - beta1^t)                          -- bias correction
//   v_hat = v_t / (1 - beta2^t)
//   p_t = p_t - lr * m_hat / (sqrt(v_hat) + eps)
//
// Weight decay is decoupled (applied before the Adam update), which
// improves generalization compared to L2 regularization.
// ---------------------------------------------------------------------------
void CpuFallbackBackend::adamw_step(void* params, const void* grads, void* m, void* v,
                                     float lr, float beta1, float beta2, float eps,
                                     float weight_decay, int step, int n_params)
{
    auto* const p = static_cast<float*>(params);
    const auto* const g = static_cast<const float*>(grads);
    auto* const mp = static_cast<float*>(m);
    auto* const vp = static_cast<float*>(v);

    // 1. Bias correction denominators
    const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
    const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));

    for (int i = 0; i < n_params; ++i) {
        // 2. Decoupled weight decay
        p[i] -= lr * weight_decay * p[i];

        // 3. Moment updates
        mp[i] = beta1 * mp[i] + (1.0f - beta1) * g[i];
        vp[i] = beta2 * vp[i] + (1.0f - beta2) * g[i] * g[i];

        // 4. Bias-corrected estimates
        const float m_hat = mp[i] / bc1;
        const float v_hat = vp[i] / bc2;

        // 5. Parameter update
        p[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

// ---------------------------------------------------------------------------
// gemm
// ---------------------------------------------------------------------------
// General matrix multiply: C = alpha * A @ B + beta * C.
//
//   C[i][j] = alpha * sum_k( A[i][k] * B[k][j] ) + beta * C[i][j]
//
// Dimensions: A is [M, K], B is [K, N], C is [M, N].
// Row-major layout throughout.
// ---------------------------------------------------------------------------
void CpuFallbackBackend::gemm(void* C_ptr, const void* A_ptr, const void* B_ptr,
                                int M, int N, int K, float alpha, float beta_val)
{
    auto* const C = static_cast<float*>(C_ptr);
    const auto* const A = static_cast<const float*>(A_ptr);
    const auto* const B = static_cast<const float*>(B_ptr);

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

// ---------------------------------------------------------------------------
// gemm_ex
// ---------------------------------------------------------------------------
// Extended GEMM with optional transpose on A and/or B.
//
//   C[i][j] = alpha * sum_k( A'[i][k] * B'[k][j] ) + beta * C[i][j]
//
// Where A' = A^T if trans_a else A, B' = B^T if trans_b else B.
//
// Index mapping (row-major):
//   A'[i][k] = trans_a ? A[k*M + i] : A[i*K + k]
//   B'[k][j] = trans_b ? B[j*K + k] : B[k*N + j]
// ---------------------------------------------------------------------------
void CpuFallbackBackend::gemm_ex(void* C_ptr, const void* A_ptr, const void* B_ptr,
                                  int M, int N, int K,
                                  bool trans_a, bool trans_b,
                                  float alpha, float beta_val)
{
    auto* const C = static_cast<float*>(C_ptr);
    const auto* const A = static_cast<const float*>(A_ptr);
    const auto* const B = static_cast<const float*>(B_ptr);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                const float a_val = trans_a ? A[k * M + i] : A[i * K + k];
                const float b_val = trans_b ? B[j * K + k] : B[k * N + j];
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
// Gradient of cross-entropy loss w.r.t. logits.
//
//   d_logits[v] = (softmax(logits)[v] - one_hot[v]) / N
//
// Where N = batch * seq (number of tokens) and one_hot[v] = 1 iff
// v == target.  The 1/N factor produces a mean-reduced gradient.
// ---------------------------------------------------------------------------
void CpuFallbackBackend::cross_entropy_backward(void* d_logits_ptr, const void* logits_ptr,
                                                  const int* targets,
                                                  int batch, int seq, int vocab)
{
    auto* const d_logits = static_cast<float*>(d_logits_ptr);
    const auto* const logits = static_cast<const float*>(logits_ptr);
    const int n_tokens = batch * seq;

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            const int idx = b * seq + s;
            const float* const row = logits + idx * vocab;
            float* const d_row = d_logits + idx * vocab;
            const int tgt = targets[idx];

            // 1. Max logit for numerical stability
            float max_val = row[0];
            for (int v = 1; v < vocab; ++v) {
                if (row[v] > max_val) max_val = row[v];
            }

            // 2. Sum of exp(logit - max)
            float sum_exp = 0.0f;
            for (int v = 0; v < vocab; ++v) {
                sum_exp += std::exp(row[v] - max_val);
            }

            // 3. Gradient: (softmax - one_hot) / n_tokens
            for (int v = 0; v < vocab; ++v) {
                const float softmax_v = std::exp(row[v] - max_val) / sum_exp;
                const float indicator = (v == tgt) ? 1.0f : 0.0f;
                d_row[v] = (softmax_v - indicator) / static_cast<float>(n_tokens);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// embedding_backward
// ---------------------------------------------------------------------------
// Gradient of embedding lookup w.r.t. the weight table.
//
//   d_weight[tok] += d_out[b][t]
//
// Scatter-add: each token's output gradient is accumulated into the
// row of d_weight indexed by the token ID.
// ---------------------------------------------------------------------------
void CpuFallbackBackend::embedding_backward(void* d_weight_ptr, const void* d_out_ptr,
                                              const int* tokens,
                                              int batch, int seq, int d_model, int vocab_size)
{
    auto* const d_weight = static_cast<float*>(d_weight_ptr);
    const auto* const d_out = static_cast<const float*>(d_out_ptr);

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            const int idx = b * seq + s;
            const int tok = tokens[idx];
            float* const d_row = d_weight + tok * d_model;
            const float* const grad_row = d_out + idx * d_model;

            // Scatter-add gradient into embedding row
            for (int j = 0; j < d_model; ++j) {
                d_row[j] += grad_row[j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// rmsnorm_backward
// ---------------------------------------------------------------------------
// Backward pass for RMSNorm.
//
// Given forward:
//   rms   = 1 / sqrt( (1/d) * sum(x_j^2) + eps )
//   y_j   = x_j * rms * scale_j
//
// Gradients:
//   ds        = sum_j( d_out_j * x_j * scale_j )
//   d_x_j     = rms * ( d_out_j * scale_j  -  x_j * rms^2 * ds / d )
//   d_scale_j += d_out_j * x_j * rms          (accumulated over rows)
// ---------------------------------------------------------------------------
void CpuFallbackBackend::rmsnorm_backward(void* d_x_ptr, void* d_scale_ptr,
                                            const void* d_out_ptr, const void* x_ptr,
                                            const void* scale_ptr,
                                            int batch, int seq, int d, float eps)
{
    auto* const d_x = static_cast<float*>(d_x_ptr);
    auto* const d_scale = static_cast<float*>(d_scale_ptr);
    const auto* const d_out = static_cast<const float*>(d_out_ptr);
    const auto* const x = static_cast<const float*>(x_ptr);
    const auto* const scale = static_cast<const float*>(scale_ptr);

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            const int row_idx = b * seq + s;
            const float* const x_row = x + row_idx * d;
            const float* const d_out_row = d_out + row_idx * d;
            float* const d_x_row = d_x + row_idx * d;

            // 1. Recompute rms = 1 / sqrt(mean(x^2) + eps)
            float ss = 0.0f;
            for (int j = 0; j < d; ++j) {
                ss += x_row[j] * x_row[j];
            }
            ss /= static_cast<float>(d);
            const float rms = 1.0f / std::sqrt(ss + eps);

            // 2. ds = sum_j(d_out_j * x_j * scale_j)
            float ds = 0.0f;
            for (int j = 0; j < d; ++j) {
                ds += d_out_row[j] * x_row[j] * scale[j];
            }

            // 3. d_x_j = rms * (d_out_j * scale_j - x_j * rms^2 * ds / d)
            for (int j = 0; j < d; ++j) {
                d_x_row[j] = rms * (d_out_row[j] * scale[j] - x_row[j] * rms * rms * ds / static_cast<float>(d));
            }

            // 4. d_scale_j += d_out_j * x_j * rms (accumulated over all rows)
            for (int j = 0; j < d; ++j) {
                d_scale[j] += d_out_row[j] * x_row[j] * rms;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// causal_conv_backward
// ---------------------------------------------------------------------------
// Backward pass for multi-branch causal 1-D depthwise convolution.
//
// Gradients:
//   d_x[b,s,i]   = sum_br sum_k w[br,k,i] * d_out[b, s+k, i]
//                   (for s+k < seq)
//
//   d_w[br,k,i] += sum_b sum_s d_out[b,s,i] * x[b, s-k, i]
//                   (for s-k >= 0)
// ---------------------------------------------------------------------------
void CpuFallbackBackend::causal_conv_backward(void* d_x_ptr, void* d_weights_ptr,
                                                const void* d_out_ptr, const void* x_ptr,
                                                const void* fwd_weights_ptr,
                                                const int* kernel_sizes, int n_branches,
                                                int batch, int seq, int d)
{
    auto* const d_x = static_cast<float*>(d_x_ptr);
    auto* const d_w = static_cast<float*>(d_weights_ptr);
    const auto* const d_out = static_cast<const float*>(d_out_ptr);
    const auto* const x = static_cast<const float*>(x_ptr);
    const auto* const fwd_w = static_cast<const float*>(fwd_weights_ptr);

    // 1. Find maximum kernel size (weight layout stride)
    int max_kernel = 0;
    for (int br = 0; br < n_branches; ++br) {
        if (kernel_sizes[br] > max_kernel) max_kernel = kernel_sizes[br];
    }

    // 2. d_x: gradient w.r.t. input
    for (int b_idx = 0; b_idx < batch; ++b_idx) {
        for (int s = 0; s < seq; ++s) {
            float* const d_x_row = d_x + (b_idx * seq + s) * d;
            for (int br = 0; br < n_branches; ++br) {
                const int ks = kernel_sizes[br];
                const float* const w_branch = fwd_w + br * max_kernel * d;
                for (int k = 0; k < ks; ++k) {
                    const int dst_pos = s + k;
                    if (dst_pos >= seq) continue;
                    const float* const d_out_row = d_out + (b_idx * seq + dst_pos) * d;
                    const float* const w_k = w_branch + k * d;
                    for (int i = 0; i < d; ++i) {
                        d_x_row[i] += w_k[i] * d_out_row[i];
                    }
                }
            }
        }
    }

    // 3. d_w: gradient w.r.t. weights (accumulated over batch and seq)
    for (int br = 0; br < n_branches; ++br) {
        const int ks = kernel_sizes[br];
        float* const d_w_branch = d_w + br * max_kernel * d;

        for (int b_idx = 0; b_idx < batch; ++b_idx) {
            for (int s = 0; s < seq; ++s) {
                const float* const d_out_row = d_out + (b_idx * seq + s) * d;
                for (int k = 0; k < ks; ++k) {
                    const int src_pos = s - k;
                    if (src_pos < 0) continue;
                    const float* const x_row = x + (b_idx * seq + src_pos) * d;
                    float* const d_w_k = d_w_branch + k * d;
                    for (int i = 0; i < d; ++i) {
                        d_w_k[i] += d_out_row[i] * x_row[i];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// mingru_backward
// ---------------------------------------------------------------------------
// BPTT (backpropagation through time) for MinGRU.
//
// Recomputes forward quantities at each step, then propagates:
//   d_h = d_h_out[t] + d_h_next    (external + future gradient)
//
// Through the update rule h = (1-z)*h_prev + z*h_tilde:
//   d_h_tilde = d_h * z
//   d_z       = d_h * (h_tilde - h_prev)
//   d_h_prev  = d_h * (1 - z)
//
// Gate gradient via sigmoid derivative:
//   d_pre_z   = d_z * z * (1 - z)
//
// Weight gradients d_Wz, d_Wh are accumulated over all timesteps.
// ---------------------------------------------------------------------------
void CpuFallbackBackend::mingru_backward(void* d_x_ptr, void* d_Wz_ptr, void* d_Wh_ptr,
                                           const void* d_h_out_ptr, const void* x_ptr,
                                           const void* h_all_ptr, const void* h_init_ptr,
                                           const void* Wz_ptr, const void* Wh_ptr,
                                           int batch, int seq, int d)
{
    auto* const d_x = static_cast<float*>(d_x_ptr);
    auto* const d_Wz = static_cast<float*>(d_Wz_ptr);
    auto* const d_Wh = static_cast<float*>(d_Wh_ptr);
    const auto* const d_h_out = static_cast<const float*>(d_h_out_ptr);
    const auto* const x = static_cast<const float*>(x_ptr);
    const auto* const h_all = static_cast<const float*>(h_all_ptr);
    const auto* const h_init = static_cast<const float*>(h_init_ptr);
    const auto* const Wz = static_cast<const float*>(Wz_ptr);
    const auto* const Wh = static_cast<const float*>(Wh_ptr);

    // 1. Gradient flowing back from future timesteps
    std::vector<float> d_h_next(batch * d, 0.0f);

    // 2. Per-timestep temporaries
    std::vector<float> d_h_tilde_vec(d);
    std::vector<float> d_pre_z_vec(d);
    std::vector<float> z_vec(d);

    // 3. BPTT: iterate backward from seq-1 to 0
    for (int s = seq - 1; s >= 0; --s) {
        for (int b = 0; b < batch; ++b) {
            const float* const x_t = x + (b * seq + s) * d;

            // 3a. First pass: compute per-dimension quantities, accumulate weight grads
            for (int i = 0; i < d; ++i) {
                const float h_prev = (s > 0) ? h_all[(b * seq + s - 1) * d + i] : h_init[b * d + i];

                // Recompute gate: pre_z = h_prev + Wz @ x_t
                float pre_z = h_prev;
                for (int j = 0; j < d; ++j) {
                    pre_z += Wz[i * d + j] * x_t[j];
                }
                const float z = 1.0f / (1.0f + std::exp(-pre_z));  // sigmoid
                z_vec[i] = z;

                // Recompute candidate: h_tilde = Wh @ (z * x_t)
                float h_tilde = 0.0f;
                for (int j = 0; j < d; ++j) {
                    h_tilde += Wh[i * d + j] * z * x_t[j];
                }

                // Total gradient into this hidden unit
                const float d_h = d_h_out[(b * seq + s) * d + i] + d_h_next[b * d + i];

                // Through h = (1-z)*h_prev + z*h_tilde
                const float d_h_tilde_i = d_h * z;
                const float d_z_update = d_h * (h_tilde - h_prev);
                const float d_h_prev_update = d_h * (1.0f - z);

                // Through h_tilde = sum_j(Wh[i,j] * z * x_j)
                float wx_sum = 0.0f;
                for (int j = 0; j < d; ++j) {
                    wx_sum += Wh[i * d + j] * x_t[j];
                }
                const float d_z_htilde = d_h_tilde_i * wx_sum;

                // Accumulate d_Wh[i,j] += d_h_tilde * z * x_j
                for (int j = 0; j < d; ++j) {
                    d_Wh[i * d + j] += d_h_tilde_i * z * x_t[j];
                }

                // Total d_z and sigmoid backward: d_pre_z = d_z * z * (1-z)
                const float d_z = d_z_update + d_z_htilde;
                const float d_pre_z = d_z * z * (1.0f - z);

                // Accumulate d_Wz[i,j] += d_pre_z * x_j
                for (int j = 0; j < d; ++j) {
                    d_Wz[i * d + j] += d_pre_z * x_t[j];
                }

                // Store for second pass
                d_h_tilde_vec[i] = d_h_tilde_i;
                d_pre_z_vec[i] = d_pre_z;

                // Propagate gradient to previous timestep
                d_h_next[b * d + i] = d_h_prev_update + d_pre_z;
            }

            // 3b. Second pass: accumulate d_x[b,s,j] from all output dimensions i
            float* const d_x_t = d_x + (b * seq + s) * d;
            for (int j = 0; j < d; ++j) {
                float contrib = 0.0f;
                for (int i = 0; i < d; ++i) {
                    // From h_tilde path: d_h_tilde[i] * Wh[i,j] * z_i
                    contrib += d_h_tilde_vec[i] * Wh[i * d + j] * z_vec[i];
                    // From gate path: d_pre_z[i] * Wz[i,j]
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
// Backward pass for slot-based external memory attention.
//
// Recomputes softmax scores, then propagates through:
//   out = sum_s( alpha_s * value_s )
//
// Gradients:
//   d_score[s]  = sum_i( d_out_i * value[s,i] )
//   d_alpha     = softmax_backward(d_score, alpha)
//   d_pre_sm[s] = alpha_s * (d_score_s - dot(d_score, alpha)) / sqrt(d)
//
//   d_x[i]     += sum_s( d_pre_sm_s * key[s,i] )
//   d_key[s,i] += d_pre_sm_s * x_i              (accumulated over rows)
//   d_val[s,i] += alpha_s * d_out_i              (accumulated over rows)
// ---------------------------------------------------------------------------
void CpuFallbackBackend::slot_memory_backward(void* d_x_ptr, void* d_keys_ptr, void* d_values_ptr,
                                                const void* d_out_ptr, const void* x_ptr,
                                                const void* keys_ptr, const void* values_ptr,
                                                int batch, int seq, int d, int n_slots)
{
    auto* const d_x = static_cast<float*>(d_x_ptr);
    auto* const d_keys = static_cast<float*>(d_keys_ptr);
    auto* const d_values = static_cast<float*>(d_values_ptr);
    const auto* const d_out = static_cast<const float*>(d_out_ptr);
    const auto* const x = static_cast<const float*>(x_ptr);
    const auto* const keys = static_cast<const float*>(keys_ptr);
    const auto* const values = static_cast<const float*>(values_ptr);

    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            const int row = b * seq + s;
            const float* const x_row = x + row * d;
            const float* const d_out_row = d_out + row * d;
            float* const d_x_row = d_x + row * d;

            // 1. Recompute softmax scores
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

            // 2. d_score from d_out: d_score[s] = sum_i(d_out_i * value[s,i])
            std::vector<float> d_score(n_slots, 0.0f);
            for (int slot = 0; slot < n_slots; ++slot) {
                for (int i = 0; i < d; ++i) {
                    d_score[slot] += d_out_row[i] * values[slot * d + i];
                }
            }

            // 3. Softmax backward: d_pre_sm = alpha * (d_score - dot(d_score, alpha)) / sqrt(d)
            float dot_ds = 0.0f;
            for (int slot = 0; slot < n_slots; ++slot) {
                dot_ds += d_score[slot] * scores[slot];
            }
            std::vector<float> d_pre_softmax(n_slots);
            for (int slot = 0; slot < n_slots; ++slot) {
                d_pre_softmax[slot] = scores[slot] * (d_score[slot] - dot_ds) * inv_sqrt_d;
            }

            // 4. d_x: gradient w.r.t. query
            for (int i = 0; i < d; ++i) {
                float val = 0.0f;
                for (int slot = 0; slot < n_slots; ++slot) {
                    val += d_pre_softmax[slot] * keys[slot * d + i];
                }
                d_x_row[i] += val;
            }

            // 5. d_keys: gradient w.r.t. slot keys (accumulated over all rows)
            for (int slot = 0; slot < n_slots; ++slot) {
                for (int i = 0; i < d; ++i) {
                    d_keys[slot * d + i] += d_pre_softmax[slot] * x_row[i];
                }
            }

            // 6. d_values: gradient w.r.t. slot values (accumulated over all rows)
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
// Backward pass for SwiGLU feed-forward network.
//
// Given forward:
//   up     = x @ W_up                             [d_ff]
//   gate   = x @ W_gate                           [d_ff]
//   sig_g  = sigmoid(gate)
//   silu_g = gate * sig_g
//   hidden = up * silu_g                           [d_ff]
//   out    = hidden @ W_down                       [d_model]
//
// Backward through W_down:
//   d_hidden = d_out @ W_down^T
//   d_W_down += hidden^T @ d_out
//
// Backward through activation:
//   d_up   = d_hidden * silu_g
//   d_gate = d_hidden * up * sig_g * (1 + gate*(1-sig_g))
//
// Backward through projections:
//   d_x      += d_up @ W_up^T + d_gate @ W_gate^T
//   d_W_up   += x^T @ d_up
//   d_W_gate += x^T @ d_gate
// ---------------------------------------------------------------------------
void CpuFallbackBackend::swiglu_backward(void* d_x_ptr, void* d_W_up_ptr,
                                           void* d_W_gate_ptr, void* d_W_down_ptr,
                                           const void* d_out_ptr, const void* x_ptr,
                                           const void* W_up_ptr, const void* W_gate_ptr,
                                           const void* W_down_ptr,
                                           int batch, int seq, int d_model, int d_ff)
{
    auto* const d_x = static_cast<float*>(d_x_ptr);
    auto* const d_W_up = static_cast<float*>(d_W_up_ptr);
    auto* const d_W_gate = static_cast<float*>(d_W_gate_ptr);
    auto* const d_W_down = static_cast<float*>(d_W_down_ptr);
    const auto* const d_out = static_cast<const float*>(d_out_ptr);
    const auto* const x = static_cast<const float*>(x_ptr);
    const auto* const W_up = static_cast<const float*>(W_up_ptr);
    const auto* const W_gate = static_cast<const float*>(W_gate_ptr);
    const auto* const W_down = static_cast<const float*>(W_down_ptr);

    // Per-position temporaries
    std::vector<float> up(d_ff);
    std::vector<float> gate(d_ff);
    std::vector<float> sigmoid_g(d_ff);
    std::vector<float> silu_g(d_ff);
    std::vector<float> hidden(d_ff);
    std::vector<float> d_hidden(d_ff);
    std::vector<float> d_up(d_ff);
    std::vector<float> d_gate(d_ff);

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            const int row = b * seq + s;
            const float* const inp = x + row * d_model;
            const float* const d_out_row = d_out + row * d_model;
            float* const d_x_row = d_x + row * d_model;

            // 1. Recompute forward: up = x @ W_up, gate = x @ W_gate
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

            // 2. d_hidden = d_out @ W_down^T
            for (int j = 0; j < d_ff; ++j) {
                float val = 0.0f;
                for (int k = 0; k < d_model; ++k) {
                    val += d_out_row[k] * W_down[j * d_model + k];
                }
                d_hidden[j] = val;
            }

            // 3. d_W_down[j, k] += hidden[j] * d_out[row, k]
            for (int j = 0; j < d_ff; ++j) {
                for (int k = 0; k < d_model; ++k) {
                    d_W_down[j * d_model + k] += hidden[j] * d_out_row[k];
                }
            }

            // 4. Activation backward
            for (int j = 0; j < d_ff; ++j) {
                d_up[j] = d_hidden[j] * silu_g[j];
                d_gate[j] = d_hidden[j] * up[j] * sigmoid_g[j] * (1.0f + gate[j] * (1.0f - sigmoid_g[j]));
            }

            // 5. d_x += d_up @ W_up^T + d_gate @ W_gate^T
            for (int k = 0; k < d_model; ++k) {
                float val = 0.0f;
                for (int j = 0; j < d_ff; ++j) {
                    val += d_up[j] * W_up[k * d_ff + j];
                    val += d_gate[j] * W_gate[k * d_ff + j];
                }
                d_x_row[k] += val;
            }

            // 6. d_W_up[k, j] += inp[k] * d_up[j]
            for (int k = 0; k < d_model; ++k) {
                for (int j = 0; j < d_ff; ++j) {
                    d_W_up[k * d_ff + j] += inp[k] * d_up[j];
                }
            }

            // 7. d_W_gate[k, j] += inp[k] * d_gate[j]
            for (int k = 0; k < d_model; ++k) {
                for (int j = 0; j < d_ff; ++j) {
                    d_W_gate[k * d_ff + j] += inp[k] * d_gate[j];
                }
            }
        }
    }
}

// ===========================================================================
//  Inference Kernels (single-token / single-step)
// ===========================================================================

// ---------------------------------------------------------------------------
// mingru_step
// ---------------------------------------------------------------------------
// Single-step MinGRU for autoregressive inference (batch=1, seq=1).
//
//   z       = sigmoid(Wz @ x + h_prev)
//   h_tilde = Wh @ (z * x)
//   h_out   = (1 - z) * h_prev + z * h_tilde
//
// Identical math to mingru_forward but without the time loop.
// ---------------------------------------------------------------------------
void CpuFallbackBackend::mingru_step(void* h_out, const void* x, const void* h_prev,
                                      const void* Wz, const void* Wh, int d)
{
    auto* const ho = static_cast<float*>(h_out);
    const auto* const xp = static_cast<const float*>(x);
    const auto* const hp = static_cast<const float*>(h_prev);
    const auto* const wz = static_cast<const float*>(Wz);
    const auto* const wh = static_cast<const float*>(Wh);

    for (int i = 0; i < d; ++i) {
        // 1. Gate: z = sigmoid(Wz[i,:] @ x + h_prev[i])
        float z_val = 0.0f;
        for (int j = 0; j < d; ++j) {
            z_val += wz[i * d + j] * xp[j];
        }
        z_val += hp[i];
        z_val = 1.0f / (1.0f + std::exp(-z_val));  // sigmoid

        // 2. Candidate: h_tilde = Wh[i,:] @ (z * x)
        float h_tilde = 0.0f;
        for (int j = 0; j < d; ++j) {
            h_tilde += wh[i * d + j] * (z_val * xp[j]);
        }

        // 3. Update: h_out = (1 - z) * h_prev + z * h_tilde
        ho[i] = (1.0f - z_val) * hp[i] + z_val * h_tilde;
    }
}

// ---------------------------------------------------------------------------
// conv_step
// ---------------------------------------------------------------------------
// Single-step causal convolution for autoregressive inference.
//
// Maintains a circular buffer of recent tokens per branch.  At each
// step the buffer is shifted left, the new token is inserted at the
// end, and the convolution is computed over the buffer contents:
//
//   shift buffer left by 1
//   buffer[ks-1] = x
//   out[i] = sum_br sum_{k=0}^{ks-1} w[br][k][i] * buffer[k][i]
// ---------------------------------------------------------------------------
void CpuFallbackBackend::conv_step(void* out, void* buffer, const void* x, const void* weights,
                                    const int* kernel_sizes, int n_branches, int d)
{
    auto* const o = static_cast<float*>(out);
    const auto* const xp = static_cast<const float*>(x);
    const auto* const wp = static_cast<const float*>(weights);
    auto* const buf = static_cast<float*>(buffer);

    // 1. Find maximum kernel size (buffer layout stride)
    int max_kernel = 0;
    for (int br = 0; br < n_branches; ++br) {
        if (kernel_sizes[br] > max_kernel) max_kernel = kernel_sizes[br];
    }

    // 2. Zero output
    std::memset(o, 0, d * sizeof(float));

    // 3. Per-branch: shift buffer, insert token, convolve
    for (int br = 0; br < n_branches; ++br) {
        const int ks = kernel_sizes[br];
        float* const br_buf = buf + br * max_kernel * d;
        const float* const w_branch = wp + br * max_kernel * d;

        // Shift buffer left by one position
        if (ks > 1) {
            std::memmove(br_buf, br_buf + d, (ks - 1) * d * sizeof(float));
        }

        // Insert new token at the end
        std::memcpy(br_buf + (ks - 1) * d, xp, d * sizeof(float));

        // Depthwise convolution over buffer
        for (int k = 0; k < ks; ++k) {
            const float* const b_k = br_buf + k * d;
            const float* const w_k = w_branch + k * d;
            for (int i = 0; i < d; ++i) {
                o[i] += b_k[i] * w_k[i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// slot_query
// ---------------------------------------------------------------------------
// Single-vector slot memory query for autoregressive inference.
//
//   score_s = (x . key_s) / sqrt(d)
//   alpha   = softmax(scores)
//   out     = sum_s( alpha_s * value_s )
//
// Identical to slot_memory_forward but for a single query vector
// (no batch/seq dimensions).
// ---------------------------------------------------------------------------
void CpuFallbackBackend::slot_query(void* out, const void* x, const void* slot_keys,
                                     const void* slot_values, int d, int n_slots)
{
    auto* const o = static_cast<float*>(out);
    const auto* const xp = static_cast<const float*>(x);
    const auto* const sk = static_cast<const float*>(slot_keys);
    const auto* const sv = static_cast<const float*>(slot_values);

    // 1. Scaled dot-product scores
    std::vector<float> scores(n_slots);
    float max_score = -1e30f;
    for (int slot = 0; slot < n_slots; ++slot) {
        float dot = 0.0f;
        for (int i = 0; i < d; ++i) {
            dot += xp[i] * sk[slot * d + i];
        }
        dot /= std::sqrt(static_cast<float>(d));  // 1/sqrt(d) scaling
        scores[slot] = dot;
        if (dot > max_score) max_score = dot;
    }

    // 2. Softmax normalization
    float sum_exp = 0.0f;
    for (int slot = 0; slot < n_slots; ++slot) {
        scores[slot] = std::exp(scores[slot] - max_score);
        sum_exp += scores[slot];
    }
    for (int slot = 0; slot < n_slots; ++slot) {
        scores[slot] /= sum_exp;
    }

    // 3. Weighted sum of values
    std::memset(o, 0, d * sizeof(float));
    for (int slot = 0; slot < n_slots; ++slot) {
        for (int i = 0; i < d; ++i) {
            o[i] += scores[slot] * sv[slot * d + i];
        }
    }
}

} // namespace rnet::gpu

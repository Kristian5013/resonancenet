#ifdef RNET_HAS_METAL

// ---------------------------------------------------------------------------
// Metal backend implementation.
//
// This file MUST be compiled as Objective-C++ (.mm) on Apple platforms.
// The CMakeLists.txt sets the LANGUAGE property to OBJCXX for this file.
//
// Strategy: allocate Metal shared-memory buffers (MTLResourceStorageModeShared)
// which are CPU-accessible via [buffer contents].  All kernel math runs on the
// CPU through those mapped pointers — identical to the CpuFallbackBackend math.
// This gives proper Metal memory management (unified memory, correct page
// alignment, GPU-visible if we later add .metal compute shaders) while keeping
// the implementation portable and easy to validate.
// ---------------------------------------------------------------------------

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_backend.h"
#include "../../core/logging.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

namespace rnet::gpu {

// ── Helpers to bridge void* ↔ ObjC types ────────────────────────────────

static inline id<MTLDevice> as_device(void* p) {
    return (__bridge id<MTLDevice>)p;
}

static inline id<MTLCommandQueue> as_queue(void* p) {
    return (__bridge id<MTLCommandQueue>)p;
}

static inline id<MTLBuffer> as_buffer(void* p) {
    return (__bridge id<MTLBuffer>)p;
}

// ── Constructor / Destructor ────────────────────────────────────────────

MetalBackend::MetalBackend() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            LogError("Metal: failed to create system default device");
            device_name_ = "Metal Device (unavailable)";
            return;
        }

        // Retain the device — we own it for the lifetime of the backend.
        mtl_device_ = (__bridge_retained void*)device;

        device_name_ = std::string([[device name] UTF8String]);
        total_memory_bytes_ = [device recommendedMaxWorkingSetSize];

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            LogError("Metal: failed to create command queue");
            return;
        }
        mtl_queue_ = (__bridge_retained void*)queue;

        LogPrintf("GPU: initialized Metal backend — %s (%.1f GB)",
                  device_name_.c_str(),
                  static_cast<double>(total_memory_bytes_) / (1024.0 * 1024.0 * 1024.0));
    }
}

MetalBackend::~MetalBackend() {
    @autoreleasepool {
        // Release all outstanding buffer allocations.
        for (auto& [ptr, alloc] : allocations_) {
            if (alloc.buffer) {
                CFRelease(alloc.buffer);
            }
        }
        allocations_.clear();
        allocated_bytes_ = 0;

        if (mtl_queue_) {
            CFRelease(mtl_queue_);
            mtl_queue_ = nullptr;
        }
        if (mtl_device_) {
            CFRelease(mtl_device_);
            mtl_device_ = nullptr;
        }
    }
}

// ── Device Info ─────────────────────────────────────────────────────────

std::string MetalBackend::device_name() const { return device_name_; }

size_t MetalBackend::total_memory() const { return total_memory_bytes_; }

size_t MetalBackend::free_memory() const {
    // Metal does not expose a direct "free memory" query.
    // Return total minus what we have allocated as a rough estimate.
    if (allocated_bytes_ >= total_memory_bytes_) return 0;
    return total_memory_bytes_ - allocated_bytes_;
}

GpuBackendType MetalBackend::type() const { return GpuBackendType::METAL; }

// ── Memory Management ───────────────────────────────────────────────────

void* MetalBackend::alloc(size_t bytes) {
    if (!mtl_device_ || bytes == 0) return nullptr;

    @autoreleasepool {
        id<MTLDevice> device = as_device(mtl_device_);

        // StorageModeShared: CPU and GPU share the same memory pages.
        id<MTLBuffer> buffer = [device newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];
        if (!buffer) {
            LogError("Metal: failed to allocate %zu bytes", bytes);
            return nullptr;
        }

        void* mapped = [buffer contents];
        std::memset(mapped, 0, bytes);

        MetalAllocation alloc;
        alloc.buffer = (__bridge_retained void*)buffer;
        alloc.mapped = mapped;
        alloc.size = bytes;
        allocations_[mapped] = alloc;
        allocated_bytes_ += bytes;

        return mapped;
    }
}

void MetalBackend::free(void* ptr) {
    if (!ptr) return;

    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
        LogError("Metal: free() called on unknown pointer %p", ptr);
        return;
    }

    @autoreleasepool {
        allocated_bytes_ -= it->second.size;
        CFRelease(it->second.buffer);
        allocations_.erase(it);
    }
}

void MetalBackend::copy_to_device(void* dst, const void* src, size_t bytes) {
    // Shared memory: CPU pointers are already device-visible, just memcpy.
    if (dst && src && bytes > 0) {
        std::memcpy(dst, src, bytes);
    }
}

void MetalBackend::copy_to_host(void* dst, const void* src, size_t bytes) {
    // Shared memory: device pointers are already CPU-visible, just memcpy.
    if (dst && src && bytes > 0) {
        std::memcpy(dst, src, bytes);
    }
}

void MetalBackend::synchronize() {
    if (!mtl_queue_) return;

    @autoreleasepool {
        id<MTLCommandQueue> queue = as_queue(mtl_queue_);
        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }
}

void MetalBackend::memset_zero(void* ptr, size_t bytes) {
    if (ptr && bytes > 0) {
        std::memset(ptr, 0, bytes);
    }
}

// ── Training Kernels ────────────────────────────────────────────────────
// All math is identical to CpuFallbackBackend, operating on shared-memory
// mapped pointers.  When Metal compute shaders are added later, these
// functions will dispatch .metal kernels instead.

void MetalBackend::embedding_forward(void* out, const void* weight,
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

void MetalBackend::rmsnorm_forward(void* out, const void* x, const void* scale,
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

void MetalBackend::causal_conv_forward(void* out, const void* x, const void* weights,
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

void MetalBackend::mingru_forward(void* h_out, void* state_out,
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

                // Update: h_new = (1-z)*h_c + z*h_tilde
                float h_new = (1.0f - z_val) * h_c[i] + z_val * h_tilde;
                h_o[i] = h_new;
                h_c[i] = h_new;
            }
        }
    }

    std::memcpy(so, h_cur.data(), batch * d * sizeof(float));
}

void MetalBackend::slot_memory_forward(void* out, const void* x,
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

void MetalBackend::swiglu_forward(void* out, const void* x,
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

            // up = x @ W_up,  gate = x @ W_gate
            for (int i = 0; i < d_ff; ++i) {
                float u = 0.0f, g = 0.0f;
                for (int j = 0; j < d_model; ++j) {
                    u += inp[j] * wu[j * d_ff + i];
                    g += inp[j] * wg[j * d_ff + i];
                }
                up[i] = u;
                gate[i] = g;
            }

            // SwiGLU: hidden = up * silu(gate)
            for (int i = 0; i < d_ff; ++i) {
                float silu = gate[i] / (1.0f + std::exp(-gate[i]));
                hidden[i] = up[i] * silu;
            }

            // out = hidden @ W_down
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

void MetalBackend::cross_entropy_loss(float* loss_out, const void* logits,
                                       const int* targets, int batch, int seq, int vocab) {
    auto* lp = static_cast<const float*>(logits);
    float total_loss = 0.0f;
    int count = batch * seq;

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            const float* row = lp + (b * seq + s) * vocab;
            int tgt = targets[b * seq + s];

            // Log-sum-exp for numerical stability
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

void MetalBackend::adamw_step(void* params, const void* grads, void* m, void* v,
                               float lr, float beta1, float beta2, float eps,
                               float weight_decay, int step, int n_params) {
    auto* p = static_cast<float*>(params);
    auto* g = static_cast<const float*>(grads);
    auto* mp = static_cast<float*>(m);
    auto* vp = static_cast<float*>(v);

    float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
    float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));

    for (int i = 0; i < n_params; ++i) {
        // Weight decay
        p[i] -= lr * weight_decay * p[i];

        // Moment updates
        mp[i] = beta1 * mp[i] + (1.0f - beta1) * g[i];
        vp[i] = beta2 * vp[i] + (1.0f - beta2) * g[i] * g[i];

        // Bias correction
        float m_hat = mp[i] / bc1;
        float v_hat = vp[i] / bc2;

        // Parameter update
        p[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

void MetalBackend::gemm(void* C_ptr, const void* A_ptr, const void* B_ptr,
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

// ── Inference Kernels ───────────────────────────────────────────────────

void MetalBackend::mingru_step(void* h_out, const void* x, const void* h_prev,
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

void MetalBackend::conv_step(void* out, void* buffer, const void* x, const void* weights,
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

        // Shift buffer and insert new token
        if (ks > 1) {
            std::memmove(br_buf, br_buf + d, (ks - 1) * d * sizeof(float));
        }
        std::memcpy(br_buf + (ks - 1) * d, xp, d * sizeof(float));

        // Convolve
        for (int k = 0; k < ks; ++k) {
            const float* b_k = br_buf + k * d;
            const float* w_k = w_branch + k * d;
            for (int i = 0; i < d; ++i) {
                o[i] += b_k[i] * w_k[i];
            }
        }
    }
}

void MetalBackend::slot_query(void* out, const void* x, const void* slot_keys,
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

// ── gemm_ex ────────────────────────────────────────────────────────────

void MetalBackend::gemm_ex(void* C_ptr, const void* A_ptr, const void* B_ptr,
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

// ── Backward Kernels ──────────────────────────────────────────────────
// CPU-side implementations operating on Metal shared (mapped) memory.

void MetalBackend::cross_entropy_backward(void* d_logits_ptr, const void* logits_ptr,
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

void MetalBackend::embedding_backward(void* d_weight_ptr, const void* d_out_ptr,
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

void MetalBackend::rmsnorm_backward(void* d_x_ptr, void* d_scale_ptr, const void* d_out_ptr,
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
            int row_idx = b * seq + s;
            const float* x_row = x + row_idx * d;
            const float* d_out_row = d_out + row_idx * d;
            float* d_x_row = d_x + row_idx * d;

            float ss = 0.0f;
            for (int j = 0; j < d; ++j) ss += x_row[j] * x_row[j];
            ss /= static_cast<float>(d);
            float rms = 1.0f / std::sqrt(ss + eps);

            float ds = 0.0f;
            for (int j = 0; j < d; ++j) ds += d_out_row[j] * x_row[j] * scale[j];

            for (int j = 0; j < d; ++j) {
                d_x_row[j] = rms * (d_out_row[j] * scale[j] - x_row[j] * rms * rms * ds / static_cast<float>(d));
            }

            for (int j = 0; j < d; ++j) {
                d_scale[j] += d_out_row[j] * x_row[j] * rms;
            }
        }
    }
}

void MetalBackend::causal_conv_backward(void* d_x_ptr, void* d_weights_ptr, const void* d_out_ptr,
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

void MetalBackend::mingru_backward(void* d_x_ptr, void* d_Wz_ptr, void* d_Wh_ptr,
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

void MetalBackend::slot_memory_backward(void* d_x_ptr, void* d_keys_ptr, void* d_values_ptr,
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
                for (int i = 0; i < d; ++i) dot += x_row[i] * keys[slot * d + i];
                dot *= inv_sqrt_d;
                scores[slot] = dot;
                if (dot > max_score) max_score = dot;
            }
            float sum_exp = 0.0f;
            for (int slot = 0; slot < n_slots; ++slot) {
                scores[slot] = std::exp(scores[slot] - max_score);
                sum_exp += scores[slot];
            }
            for (int slot = 0; slot < n_slots; ++slot) scores[slot] /= sum_exp;

            std::vector<float> d_score(n_slots, 0.0f);
            for (int slot = 0; slot < n_slots; ++slot) {
                for (int i = 0; i < d; ++i) {
                    d_score[slot] += d_out_row[i] * values[slot * d + i];
                }
            }

            float dot_ds = 0.0f;
            for (int slot = 0; slot < n_slots; ++slot) dot_ds += d_score[slot] * scores[slot];

            std::vector<float> d_pre_softmax(n_slots);
            for (int slot = 0; slot < n_slots; ++slot) {
                d_pre_softmax[slot] = scores[slot] * (d_score[slot] - dot_ds) * inv_sqrt_d;
            }

            for (int i = 0; i < d; ++i) {
                float val = 0.0f;
                for (int slot = 0; slot < n_slots; ++slot) val += d_pre_softmax[slot] * keys[slot * d + i];
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

void MetalBackend::swiglu_backward(void* d_x_ptr, void* d_W_up_ptr, void* d_W_gate_ptr,
                                    void* d_W_down_ptr,
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
                for (int k = 0; k < d_model; ++k) val += d_out_row[k] * W_down[j * d_model + k];
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
                    d_W_gate[k * d_ff + j] += inp[k] * d_gate[j];
                }
            }
        }
    }
}

}  // namespace rnet::gpu

#endif  // RNET_HAS_METAL

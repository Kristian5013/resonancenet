#pragma once

/// @file kernels.h
/// Kernel declarations for GPU training/inference operations.
/// Header-only — actual kernel implementations live in backend-specific files.
///
/// All kernel functions are declared as pure virtuals on GpuBackend (see backend.h).
/// This header provides documentation and parameter conventions.

namespace rnet::gpu {

/// ── Kernel Parameter Conventions ──────────────────────────────────────
///
/// All tensor pointers are device memory unless otherwise noted.
/// Shapes follow [batch, seq, dim] ordering (row-major).
///
/// ── Training Kernels ──────────────────────────────────────────────────
///
/// embedding_forward:
///   out    [batch, seq, d_model]  — output embeddings (fp32)
///   weight [vocab, d_model]      — embedding table (fp32)
///   tokens [batch, seq]          — input token IDs (int32)
///
/// rmsnorm_forward:
///   out   [batch, seq, d]  — normalized output (fp32)
///   x     [batch, seq, d]  — input (fp32)
///   scale [d]              — learnable scale (fp32)
///   eps                    — epsilon for numerical stability
///
/// causal_conv_forward:
///   out          [batch, seq, d]              — output (fp32)
///   x            [batch, seq, d]              — input (fp32)
///   weights      [n_branches, max_kernel, d]  — conv weights (fp32)
///   kernel_sizes [n_branches]                 — kernel size per branch (int32)
///
/// mingru_forward:
///   h_out     [batch, seq, d]  — hidden output sequence (fp32)
///   state_out [batch, d]       — final hidden state (fp32)
///   x         [batch, seq, d]  — input (fp32)
///   h_prev    [batch, d]       — initial hidden state (fp32)
///   Wz        [d, d]           — gate weights (fp32)
///   Wh        [d, d]           — candidate weights (fp32)
///
/// slot_memory_forward:
///   out        [batch, seq, d]      — output (fp32)
///   x          [batch, seq, d]      — input queries (fp32)
///   slot_keys  [n_slots, d]         — slot key vectors (fp32)
///   slot_values[n_slots, d]         — slot value vectors (fp32)
///
/// swiglu_forward:
///   out    [batch, seq, d_model]  — output (fp32)
///   x      [batch, seq, d_model]  — input (fp32)
///   W_up   [d_model, d_ff]        — up-projection (fp32)
///   W_gate [d_model, d_ff]        — gate-projection (fp32)
///   W_down [d_ff, d_model]        — down-projection (fp32)
///
/// cross_entropy_loss:
///   loss_out — scalar on host (fp32)
///   logits   [batch, seq, vocab]  — unnormalized logits (fp32)
///   targets  [batch, seq]         — ground-truth token IDs (int32)
///
/// adamw_step:
///   params, grads, m, v — all [n_params] flattened (fp32)
///   step — current optimizer step (1-based)
///
/// gemm:
///   C [M, N], A [M, K], B [K, N] — general matrix multiply (fp32)
///   C = alpha * A @ B + beta * C
///
/// ── Inference Kernels (single-token step) ─────────────────────────────
///
/// mingru_step:
///   h_out  [d]     — next hidden state (fp32)
///   x      [d]     — single-token input (fp32)
///   h_prev [d]     — previous hidden state (fp32)
///   Wz, Wh [d, d]  — weights (fp32)
///
/// conv_step:
///   out     [d]                        — output (fp32)
///   buffer  [n_branches, max_kernel, d]— rolling buffer (fp32, in/out)
///   x       [d]                        — single-token input (fp32)
///   weights [n_branches, max_kernel, d]— conv weights (fp32)
///
/// slot_query:
///   out        [d]          — output (fp32)
///   x          [d]          — query vector (fp32)
///   slot_keys  [n_slots, d] — slot keys (fp32)
///   slot_values[n_slots, d] — slot values (fp32)

}  // namespace rnet::gpu

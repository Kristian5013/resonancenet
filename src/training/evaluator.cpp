// Copyright (c) 2024-2026 The ResonanceNet Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "training/evaluator.h"

#include "gpu/backend.h"
#include "gpu/tensor.h"
#include "training/checkpoint_io.h"
#include "training/model_config.h"

#include <cmath>

namespace rnet::training {

// ---------------------------------------------------------------------------
// Evaluator
// ---------------------------------------------------------------------------

Evaluator::Evaluator(rnet::gpu::GpuBackend& backend)
    : backend_(backend) {}

// ---------------------------------------------------------------------------
// evaluate
// ---------------------------------------------------------------------------
// Deterministic validation-loss evaluation for PoT consensus verification.
//
// Loads a model checkpoint, allocates GPU buffers (FP32 precision to match
// TrainingEngine), and runs n_batches of forward-only inference on the
// validation split.
//
// Architecture per layer:
//   RMSNorm -> CausalConv -> MinGRU -> SlotMemory -> RMSNorm -> SwiGLU -> Residual
//
// Residual connection uses the gemm trick:
//   residual += ff_out  via  gemm(residual, ff, ones, N, 1, 1, alpha=1, beta=1)
//
// Precision: ALL buffers are FP32 (matching TrainingEngine).  Using BF16
// would produce different val_loss values, breaking consensus verification.
//
// NaN handling: batches with non-finite loss are skipped to prevent a
// single corrupted sample from invalidating the entire evaluation.
// ---------------------------------------------------------------------------
Result<float> Evaluator::evaluate(const std::filesystem::path& checkpoint,
                                   DataLoader& val_data,
                                   int n_batches) {
    using rnet::gpu::DType;
    using rnet::gpu::GpuTensor;

    // 1. Read checkpoint header to obtain model config.
    auto hdr_result = read_checkpoint_header(checkpoint);
    if (hdr_result.is_err()) {
        return Result<float>::err("Failed to read checkpoint header: " + hdr_result.error());
    }
    const auto& hdr = hdr_result.value();
    const auto& config = hdr.config;

    // 2. Read full checkpoint with tensor data.
    auto tensors_result = read_checkpoint(checkpoint);
    if (tensors_result.is_err()) {
        return Result<float>::err("Failed to read checkpoint: " + tensors_result.error());
    }
    const auto& tensors = tensors_result.value();

    // 3. Build name -> tensor lookup.
    std::unordered_map<std::string, const TensorEntry*> tensor_map;
    for (const auto& t : tensors) {
        tensor_map[t.name] = &t;
    }

    auto find_tensor = [&](const std::string& name) -> const TensorEntry* {
        auto it = tensor_map.find(name);
        return (it != tensor_map.end()) ? it->second : nullptr;
    };

    // 4. Derive dimension constants from config.
    const int batch_size = 1;  // one sample at a time for determinism
    const int seq_len = static_cast<int>(config.max_seq_len);
    const int d = static_cast<int>(config.d_model);
    const int d_ff = static_cast<int>(config.d_ff);
    const int vocab = static_cast<int>(config.vocab_size);
    const int n_slots = static_cast<int>(config.n_slots);

    // 5. Allocate activation buffers (FP32 to match TrainingEngine precision).
    GpuTensor input_buf(backend_, {batch_size, seq_len, d}, DType::FP32);
    GpuTensor residual_buf(backend_, {batch_size, seq_len, d}, DType::FP32);
    GpuTensor norm_buf(backend_, {batch_size, seq_len, d}, DType::FP32);
    GpuTensor conv_buf(backend_, {batch_size, seq_len, d}, DType::FP32);
    GpuTensor gru_buf(backend_, {batch_size, seq_len, d}, DType::FP32);
    GpuTensor gru_state(backend_, {batch_size, d}, DType::FP32);
    GpuTensor slot_buf(backend_, {batch_size, seq_len, d}, DType::FP32);
    GpuTensor ff_buf(backend_, {batch_size, seq_len, d}, DType::FP32);
    GpuTensor logits_buf(backend_, {batch_size, seq_len, vocab}, DType::FP32);

    // 6. Allocate token buffers on GPU.
    std::vector<int> host_tokens(batch_size * seq_len);
    std::vector<int> host_targets(batch_size * seq_len);
    void* gpu_tokens = backend_.alloc(batch_size * seq_len * sizeof(int));
    void* gpu_targets = backend_.alloc(batch_size * seq_len * sizeof(int));

    // 7. Upload all weight tensors to GPU (FP32 precision).
    std::unordered_map<std::string, std::unique_ptr<GpuTensor>> gpu_weights;
    for (const auto& t : tensors) {
        auto tensor = std::make_unique<GpuTensor>(backend_, t.shape, DType::FP32);
        tensor->copy_from_host(t.data.data());
        gpu_weights[t.name] = std::move(tensor);
    }

    auto get_weight = [&](const std::string& name) -> void* {
        auto it = gpu_weights.find(name);
        return (it != gpu_weights.end()) ? it->second->data() : nullptr;
    };

    // 8. Prepare kernel sizes array on host.
    int kernel_sizes_host[8];
    for (int i = 0; i < 8; ++i) {
        kernel_sizes_host[i] = config.kernel_sizes[i];
    }

    // 9. Allocate ones buffer for residual addition via gemm trick.
    void* ones_buf = backend_.alloc(sizeof(float));
    float one = 1.0f;
    backend_.copy_to_device(ones_buf, &one, sizeof(float));

    // 10. Evaluate n_batches from the validation split.
    float total_loss = 0.0f;
    int valid_batches = 0;

    val_data.reset();

    for (int b = 0; b < n_batches; ++b) {
        auto batch = val_data.next_batch(batch_size, seq_len);

        // 10a. Upload tokens and targets to GPU.
        backend_.copy_to_device(gpu_tokens, batch.tokens.data(),
                                 batch_size * seq_len * sizeof(int));
        backend_.copy_to_device(gpu_targets, batch.targets.data(),
                                 batch_size * seq_len * sizeof(int));

        // 10b. Embedding lookup.
        void* emb_weight = get_weight("embedding.weight");
        if (!emb_weight) {
            backend_.free(gpu_tokens);
            backend_.free(gpu_targets);
            backend_.free(ones_buf);
            return Result<float>::err("Missing embedding.weight tensor");
        }
        backend_.embedding_forward(input_buf.data(), emb_weight,
                                    static_cast<int*>(gpu_tokens),
                                    batch_size, seq_len, d);

        // 10c. Copy to residual stream.
        backend_.copy_to_device(residual_buf.data(), input_buf.data(),
                                 input_buf.size_bytes());

        // 10d. Per-layer forward pass.
        for (uint32_t layer = 0; layer < config.n_layers; ++layer) {
            std::string prefix = "layers." + std::to_string(layer) + ".";

            // RMSNorm
            void* norm_scale = get_weight(prefix + "rmsnorm.scale");
            backend_.rmsnorm_forward(norm_buf.data(), residual_buf.data(),
                                      norm_scale, batch_size, seq_len, d);

            // CausalConv
            void* conv_w = get_weight(prefix + "conv.weight");
            backend_.causal_conv_forward(conv_buf.data(), norm_buf.data(),
                                          conv_w, kernel_sizes_host,
                                          config.n_conv_branches,
                                          batch_size, seq_len, d);

            // MinGRU
            void* wz = get_weight(prefix + "mingru.Wz");
            void* wh = get_weight(prefix + "mingru.Wh");
            backend_.mingru_forward(gru_buf.data(), gru_state.data(),
                                     conv_buf.data(), gru_state.data(),
                                     wz, wh, batch_size, seq_len, d);

            // SlotMemory
            void* sk = get_weight(prefix + "slot.keys");
            void* sv = get_weight(prefix + "slot.values");
            backend_.slot_memory_forward(slot_buf.data(), gru_buf.data(),
                                          sk, sv, batch_size, seq_len,
                                          d, n_slots);

            // RMSNorm 2
            void* norm2_scale = get_weight(prefix + "rmsnorm2.scale");
            backend_.rmsnorm_forward(norm_buf.data(), slot_buf.data(),
                                      norm2_scale, batch_size, seq_len, d);

            // SwiGLU FFN
            void* w_up = get_weight(prefix + "ffn.W_up");
            void* w_gate = get_weight(prefix + "ffn.W_gate");
            void* w_down = get_weight(prefix + "ffn.W_down");
            backend_.swiglu_forward(ff_buf.data(), norm_buf.data(),
                                     w_up, w_gate, w_down,
                                     batch_size, seq_len, d, d_ff);

            // Residual: residual += ff_out (gemm trick with scalar 1.0)
            int total_elems = batch_size * seq_len * d;
            backend_.gemm(residual_buf.data(), ff_buf.data(), ones_buf,
                           total_elems, 1, 1, 1.0f, 1.0f);
        }

        // 10e. Output projection: logits = residual @ output_proj.
        void* out_proj = get_weight("output.weight");
        if (!out_proj) {
            backend_.free(gpu_tokens);
            backend_.free(gpu_targets);
            backend_.free(ones_buf);
            return Result<float>::err("Missing output.weight tensor");
        }
        backend_.gemm(logits_buf.data(), residual_buf.data(), out_proj,
                       batch_size * seq_len, vocab, d);

        // 10f. Cross-entropy loss.
        float batch_loss = 0.0f;
        backend_.cross_entropy_loss(&batch_loss, logits_buf.data(),
                                     static_cast<int*>(gpu_targets),
                                     batch_size, seq_len, vocab);
        backend_.synchronize();

        // 10g. Skip non-finite losses (NaN guard).
        if (!std::isfinite(batch_loss)) continue;

        total_loss += batch_loss;
        ++valid_batches;
    }

    // 11. Free raw GPU allocations.
    backend_.free(gpu_tokens);
    backend_.free(gpu_targets);
    backend_.free(ones_buf);

    // 12. Return average loss.
    if (valid_batches == 0) {
        return Result<float>::err("No valid batches evaluated");
    }

    return Result<float>::ok(total_loss / static_cast<float>(valid_batches));
}

} // namespace rnet::training

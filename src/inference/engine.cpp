// Copyright (c) 2024-2026 The ResonanceNet Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "inference/engine.h"

#include "core/logging.h"

#include <cstring>

namespace rnet::inference {

// ===========================================================================
// Construction / Destruction
// ===========================================================================

// ---------------------------------------------------------------------------
// InferenceEngine (constructor)
// ---------------------------------------------------------------------------
// Binds the engine to a GPU backend.  No allocations occur until
// load_model() is called.  The backend reference must outlive this object.
// ---------------------------------------------------------------------------
InferenceEngine::InferenceEngine(gpu::GpuBackend& backend)
    : backend_(backend) {}

InferenceEngine::~InferenceEngine() = default;

// ===========================================================================
// Model & Tokenizer Loading
// ===========================================================================

// ---------------------------------------------------------------------------
// load_model
// ---------------------------------------------------------------------------
// Deserialises a .rnet checkpoint in two passes:
//   1. Read the header to extract ModelConfig (d_model, n_layers, etc.)
//   2. Read all tensor entries and upload each one to device memory as BF16.
//
// On success every named parameter is stored in the `weights_` map and
// `model_loaded_` is set so that forward passes can proceed.
// ---------------------------------------------------------------------------
Result<void> InferenceEngine::load_model(const std::filesystem::path& checkpoint) {
    // 1. Read checkpoint header for model configuration
    auto header_res = training::read_checkpoint_header(checkpoint);
    if (header_res.is_err()) {
        return Result<void>::err("Failed to read checkpoint header: " + header_res.error());
    }
    config_ = header_res.value().config;

    LogPrintf("Loading model: %llu params, d_model=%u, n_layers=%u",
              static_cast<unsigned long long>(config_.param_count()),
              config_.d_model, config_.n_layers);

    // 2. Read all tensor entries from the checkpoint file
    auto tensors_res = training::read_checkpoint(checkpoint);
    if (tensors_res.is_err()) {
        return Result<void>::err("Failed to read checkpoint: " + tensors_res.error());
    }

    // 3. Upload each tensor to the GPU
    weights_.clear();
    for (const auto& entry : tensors_res.value()) {
        auto upload_res = upload_tensor(entry);
        if (upload_res.is_err()) {
            return Result<void>::err("Failed to upload tensor '" + entry.name +
                                     "': " + upload_res.error());
        }
    }

    // 4. Mark model as loaded
    model_loaded_ = true;
    LogPrintf("Model loaded: %zu tensors on %s",
              weights_.size(), backend_.device_name().c_str());
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// load_tokenizer
// ---------------------------------------------------------------------------
// Loads BPE vocabulary from `vocab_dir` (expects vocab.json + merges.txt).
// Must be called before generate_text() but is not required for raw
// token-ID generation.
// ---------------------------------------------------------------------------
Result<void> InferenceEngine::load_tokenizer(const std::filesystem::path& vocab_dir) {
    // 1. Delegate to the Tokenizer implementation
    auto res = tokenizer_.load(vocab_dir);
    if (res.is_err()) {
        return Result<void>::err("Failed to load tokenizer: " + res.error());
    }

    // 2. Mark tokenizer as available
    tokenizer_loaded_ = true;
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// is_ready
// ---------------------------------------------------------------------------
// Returns true only when both the model weights and the tokenizer have
// been loaded successfully.  Callers should gate inference on this.
// ---------------------------------------------------------------------------
bool InferenceEngine::is_ready() const {
    return model_loaded_ && tokenizer_loaded_;
}

// ===========================================================================
// Weight Management (private helpers)
// ===========================================================================

// ---------------------------------------------------------------------------
// upload_tensor
// ---------------------------------------------------------------------------
// Transfers a single named tensor from host memory to a GpuTensor in BF16.
//
//   1. Validate that the tensor has a positive element count.
//   2. Allocate a GpuTensor with the same shape.
//   3. Copy host data to device.
//   4. Store the tensor in the weights_ map keyed by name.
// ---------------------------------------------------------------------------
Result<void> InferenceEngine::upload_tensor(const training::TensorEntry& entry) {
    // 1. Compute total element count from shape
    std::vector<int64_t> shape = entry.shape;
    int64_t numel = 1;
    for (auto d : shape) numel *= d;

    if (numel <= 0) {
        return Result<void>::err("Tensor has zero elements");
    }

    // 2. Allocate device tensor in BF16 format
    auto tensor = std::make_unique<gpu::GpuTensor>(
        backend_, shape, gpu::DType::BF16);

    // 3. Upload host data to device
    tensor->copy_from_host(entry.data.data());

    // 4. Store in named weight map
    weights_[entry.name] = std::move(tensor);
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// get_weight
// ---------------------------------------------------------------------------
// Looks up a named weight tensor.  Returns nullptr when the name is not
// found, which callers use to detect optional layers (e.g. slot memory
// may be absent in smaller model variants).
// ---------------------------------------------------------------------------
const gpu::GpuTensor* InferenceEngine::get_weight(const std::string& name) const {
    auto it = weights_.find(name);
    if (it == weights_.end()) return nullptr;
    return it->second.get();
}

// ===========================================================================
// Forward Pass — Layer Components
// ===========================================================================

// ---------------------------------------------------------------------------
// forward_embedding
// ---------------------------------------------------------------------------
// Embedding lookup for a single token:
//
//   x = E[token]       (lookup row `token` from [V x d] embedding table)
//
// Allocates a temporary device buffer for the token index and output
// vector, runs the GPU embedding kernel, then copies the result back
// to the host vector `x`.
// ---------------------------------------------------------------------------
Result<void> InferenceEngine::forward_embedding(int token, std::vector<float>& x) {
    const int d = static_cast<int>(config_.d_model);
    x.resize(d);

    // 1. Retrieve embedding weight matrix E of shape [V, d]
    const auto* emb_w = get_weight("embedding.weight");
    if (!emb_w) {
        return Result<void>::err("Missing embedding.weight");
    }

    // 2. Allocate temporary GPU buffers for token index and output
    int token_arr[1] = {token};
    void* d_tokens = backend_.alloc(sizeof(int));
    void* d_out = backend_.alloc(d * sizeof(float));
    backend_.copy_to_device(d_tokens, token_arr, sizeof(int));

    // 3. Execute GPU embedding lookup: x = E[token]
    backend_.embedding_forward(d_out, emb_w->data(), static_cast<const int*>(d_tokens),
                               1, 1, d);

    // 4. Copy result back to host and free temporaries
    backend_.copy_to_host(x.data(), d_out, d * sizeof(float));
    backend_.free(d_tokens);
    backend_.free(d_out);

    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// forward_layer
// ---------------------------------------------------------------------------
// Processes one transformer-style layer in the recurrent architecture.
// Each layer applies five stages with residual connections:
//
//   Stage 1 — RMSNorm:
//       x_norm = x / sqrt(mean(x^2) + eps) * scale
//
//   Stage 2 — Causal convolution step (multi-branch ring buffer):
//       For each branch b with kernel size k_b, the ring buffer stores
//       the last k_b activations and computes a weighted sum.
//
//   Stage 3 — MinGRU recurrent step:
//       z  = sigmoid(Wz @ x_norm)
//       h' = (1 - z) * h + z * (Wh @ x_norm)
//
//   Stage 4 — Slot memory query (optional):
//       Computes soft attention over a fixed set of key-value slots.
//
//   Stage 5 — RMSNorm + SwiGLU FFN:
//       FFN(x) = (swish(x @ W_up) * (x @ W_gate)) @ W_down
//       x = x + FFN(norm(x))
//
// The hidden state h and convolution buffers are carried in `state`
// between calls, giving O(1) memory per step (no KV cache).
// ---------------------------------------------------------------------------
Result<void> InferenceEngine::forward_layer(int layer, std::vector<float>& x,
                                             InferenceState& state) {
    const int d = static_cast<int>(config_.d_model);
    const int d_ff = static_cast<int>(config_.d_ff);
    const int n_slots = static_cast<int>(config_.n_slots);
    const int n_branches = static_cast<int>(config_.n_conv_branches);
    const std::string prefix = "layer." + std::to_string(layer) + ".";

    // ----- Stage 1: RMSNorm -----
    // x_norm = x / sqrt(mean(x^2) + eps) * scale

    // 1. Retrieve per-layer RMSNorm scale parameter
    const auto* norm_w = get_weight(prefix + "rmsnorm.scale");
    if (!norm_w) {
        return Result<void>::err("Missing " + prefix + "rmsnorm.scale");
    }

    // 2. Upload x to device and apply RMSNorm
    void* d_x = backend_.alloc(d * sizeof(float));
    void* d_normed = backend_.alloc(d * sizeof(float));
    backend_.copy_to_device(d_x, x.data(), d * sizeof(float));
    backend_.rmsnorm_forward(d_normed, d_x, norm_w->data(), 1, 1, d);

    // ----- Stage 2: Causal convolution step (ring buffer) -----

    // 3. Load convolution weights (optional — absent in some configs)
    const auto* conv_w = get_weight(prefix + "conv.weights");
    if (conv_w) {
        // 3a. Prepare kernel sizes for each branch
        int kernel_sizes[8] = {};
        for (int b = 0; b < n_branches && b < 8; ++b) {
            kernel_sizes[b] = config_.kernel_sizes[b];
        }

        void* d_kernel_sizes = backend_.alloc(n_branches * sizeof(int));
        backend_.copy_to_device(d_kernel_sizes, kernel_sizes, n_branches * sizeof(int));

        // 3b. Compute total ring-buffer size across all branches
        size_t buf_size = 0;
        for (int b = 0; b < n_branches; ++b) {
            buf_size += kernel_sizes[b] * d;
        }
        void* d_conv_buf = backend_.alloc(buf_size * sizeof(float));

        // 3c. Upload current convolution buffer state to device
        size_t offset = 0;
        for (int b = 0; b < n_branches; ++b) {
            int k = kernel_sizes[b];
            if (k == 0) continue;
            for (int s = 0; s < k; ++s) {
                backend_.copy_to_device(
                    static_cast<char*>(d_conv_buf) + offset,
                    state.conv_buffers[layer][b][s].data(),
                    d * sizeof(float));
                offset += d * sizeof(float);
            }
        }

        // 3d. Run convolution step kernel on device
        void* d_conv_out = backend_.alloc(d * sizeof(float));
        backend_.conv_step(d_conv_out, d_conv_buf, d_normed, conv_w->data(),
                           static_cast<const int*>(d_kernel_sizes), n_branches, d);

        // 3e. Download updated ring buffer back to host state
        offset = 0;
        for (int b = 0; b < n_branches; ++b) {
            int k = kernel_sizes[b];
            if (k == 0) continue;
            for (int s = 0; s < k; ++s) {
                backend_.copy_to_host(
                    state.conv_buffers[layer][b][s].data(),
                    static_cast<const char*>(d_conv_buf) + offset,
                    d * sizeof(float));
                offset += d * sizeof(float);
            }
        }

        // 3f. Use conv output as input to MinGRU
        backend_.free(d_normed);
        d_normed = d_conv_out;

        backend_.free(d_kernel_sizes);
        backend_.free(d_conv_buf);
    }

    // ----- Stage 3: MinGRU recurrent step -----
    // z  = sigmoid(Wz @ x)
    // h' = (1 - z) * h + z * (Wh @ x)

    // 4. Retrieve MinGRU gate and hidden projection weights
    const auto* wz = get_weight(prefix + "mingru.Wz");
    const auto* wh = get_weight(prefix + "mingru.Wh");
    if (!wz || !wh) {
        return Result<void>::err("Missing MinGRU weights for layer " + std::to_string(layer));
    }

    // 5. Upload previous hidden state and compute MinGRU step
    void* d_h_prev = backend_.alloc(d * sizeof(float));
    void* d_h_out = backend_.alloc(d * sizeof(float));
    backend_.copy_to_device(d_h_prev, state.h_states[layer].data(), d * sizeof(float));

    backend_.mingru_step(d_h_out, d_normed, d_h_prev,
                         wz->data(), wh->data(), d);

    // 6. Download updated hidden state back to host
    backend_.copy_to_host(state.h_states[layer].data(), d_h_out, d * sizeof(float));

    // ----- Stage 4: Slot memory query (optional) -----

    // 7. If slot keys/values exist, compute soft attention over fixed slots
    const auto* slot_keys = get_weight(prefix + "slot.keys");
    const auto* slot_values = get_weight(prefix + "slot.values");

    void* d_slot_out = d_h_out;  // reuse buffer
    if (slot_keys && slot_values) {
        backend_.slot_query(d_slot_out, d_h_out, slot_keys->data(),
                            slot_values->data(), d, n_slots);
    }

    // 8. Apply residual connection: x = x + layer_output
    std::vector<float> slot_out(d);
    backend_.copy_to_host(slot_out.data(), d_slot_out, d * sizeof(float));
    for (int i = 0; i < d; ++i) {
        x[i] += slot_out[i];
    }

    // ----- Stage 5: RMSNorm + SwiGLU FFN -----
    // FFN(x) = (swish(x @ W_up) * (x @ W_gate)) @ W_down

    // 9. Retrieve FFN weights (all four must be present)
    const auto* norm2_w = get_weight(prefix + "rmsnorm2.scale");
    const auto* w_up = get_weight(prefix + "ffn.W_up");
    const auto* w_gate = get_weight(prefix + "ffn.W_gate");
    const auto* w_down = get_weight(prefix + "ffn.W_down");

    if (norm2_w && w_up && w_gate && w_down) {
        // 10. Apply second RMSNorm before FFN
        void* d_x2 = backend_.alloc(d * sizeof(float));
        void* d_normed2 = backend_.alloc(d * sizeof(float));
        backend_.copy_to_device(d_x2, x.data(), d * sizeof(float));
        backend_.rmsnorm_forward(d_normed2, d_x2, norm2_w->data(), 1, 1, d);

        // 11. SwiGLU forward: out = (swish(x @ W_up) * (x @ W_gate)) @ W_down
        void* d_ffn_out = backend_.alloc(d * sizeof(float));
        backend_.swiglu_forward(d_ffn_out, d_normed2, w_up->data(),
                                w_gate->data(), w_down->data(),
                                1, 1, d, d_ff);

        // 12. Apply residual connection: x = x + FFN(norm(x))
        std::vector<float> ffn_out(d);
        backend_.copy_to_host(ffn_out.data(), d_ffn_out, d * sizeof(float));

        for (int i = 0; i < d; ++i) {
            x[i] += ffn_out[i];
        }

        backend_.free(d_x2);
        backend_.free(d_normed2);
        backend_.free(d_ffn_out);
    }

    // 13. Free all remaining device allocations
    backend_.free(d_x);
    backend_.free(d_h_prev);
    if (d_slot_out != d_h_out) backend_.free(d_slot_out);
    backend_.free(d_h_out);
    if (d_normed != d_x) backend_.free(d_normed);

    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// forward_output
// ---------------------------------------------------------------------------
// Final output projection: converts the last-layer hidden vector into
// logits over the full vocabulary.
//
//   1. Apply final RMSNorm (if present):
//       x_norm = x / sqrt(mean(x^2) + eps) * scale
//
//   2. Output projection via GEMM:
//       logits = norm(x) @ W_out^T     ([1,d] x [d,V] = [1,V])
//
// The logits vector is resized to vocab_size and filled on return.
// ---------------------------------------------------------------------------
Result<void> InferenceEngine::forward_output(const std::vector<float>& x,
                                              std::vector<float>& logits) {
    const int d = static_cast<int>(config_.d_model);
    const int vocab = static_cast<int>(config_.vocab_size);

    // 1. Retrieve output projection weight matrix [V, d]
    const auto* out_w = get_weight("output.weight");
    if (!out_w) {
        return Result<void>::err("Missing output.weight");
    }

    // 2. Apply final RMSNorm (optional but standard)
    const auto* final_norm = get_weight("final_rmsnorm.scale");

    void* d_x = backend_.alloc(d * sizeof(float));
    backend_.copy_to_device(d_x, x.data(), d * sizeof(float));

    void* d_normed = d_x;
    if (final_norm) {
        d_normed = backend_.alloc(d * sizeof(float));
        backend_.rmsnorm_forward(d_normed, d_x, final_norm->data(), 1, 1, d);
    }

    // 3. Output projection: logits = x @ W_out^T  ([1,d] x [d,V] = [1,V])
    logits.resize(vocab);
    void* d_logits = backend_.alloc(vocab * sizeof(float));
    backend_.gemm(d_logits, d_normed, out_w->data(), 1, vocab, d);
    backend_.copy_to_host(logits.data(), d_logits, vocab * sizeof(float));

    // 4. Free device memory
    backend_.free(d_x);
    if (d_normed != d_x) backend_.free(d_normed);
    backend_.free(d_logits);

    return Result<void>::ok();
}

// ===========================================================================
// Forward Pass — Full Step
// ===========================================================================

// ---------------------------------------------------------------------------
// forward_step
// ---------------------------------------------------------------------------
// Runs a single-token forward pass through the entire model:
//
//   1. Embedding:  x = E[token]
//   2. Layers:     for l in 0..n_layers: x = Layer_l(x, state)
//   3. Output:     logits = norm(x) @ W_out^T
//
// The recurrent state is updated in-place.  Because the architecture
// uses MinGRU (not attention), there is no KV cache — state size is
// O(n_layers * d_model) regardless of sequence length.
// ---------------------------------------------------------------------------
Result<std::vector<float>> InferenceEngine::forward_step(int token, InferenceState& state) {
    if (!model_loaded_) {
        return Result<std::vector<float>>::err("Model not loaded");
    }

    // 1. Embedding lookup: x = E[token]
    std::vector<float> x;
    auto emb_res = forward_embedding(token, x);
    if (emb_res.is_err()) {
        return Result<std::vector<float>>::err(emb_res.error());
    }

    // 2. Process each layer sequentially
    for (uint32_t l = 0; l < config_.n_layers; ++l) {
        auto layer_res = forward_layer(static_cast<int>(l), x, state);
        if (layer_res.is_err()) {
            return Result<std::vector<float>>::err(layer_res.error());
        }
    }

    // 3. Output projection: logits = norm(x) @ W_out^T
    std::vector<float> logits;
    auto out_res = forward_output(x, logits);
    if (out_res.is_err()) {
        return Result<std::vector<float>>::err(out_res.error());
    }

    // 4. Advance token counter
    state.tokens_processed++;
    return Result<std::vector<float>>::ok(std::move(logits));
}

// ===========================================================================
// Text Generation
// ===========================================================================

// ---------------------------------------------------------------------------
// generate
// ---------------------------------------------------------------------------
// Autoregressive generation loop operating on raw token IDs.
//
// The procedure has two phases:
//
//   Phase 1 — Prefill:
//       Feed every prompt token through forward_step to warm up the
//       recurrent state.  Logits are discarded except for the final
//       prompt token (which seeds the first sampling step).
//
//   Phase 2 — Decode:
//       Repeatedly sample a token from logits, invoke the streaming
//       callback, then run forward_step to obtain the next logits.
//       Generation stops on EOS, max_tokens, or a callback returning
//       false.
//
// NOTE: The last prompt token is currently processed twice during the
// prefill-to-decode transition.  This is a known simplification; in
// production the last logits should be cached directly.
// ---------------------------------------------------------------------------
Result<std::vector<int>> InferenceEngine::generate(const std::vector<int>& prompt,
                                                    int max_tokens,
                                                    const SamplerConfig& config,
                                                    TokenCallback callback) {
    if (!model_loaded_) {
        return Result<std::vector<int>>::err("Model not loaded");
    }

    // 1. Initialise fresh recurrent state and sampler
    InferenceState state = InferenceState::create(config_);
    Sampler sampler(config);

    // 2. Prefill — process all prompt tokens to warm the recurrent state
    for (size_t i = 0; i < prompt.size(); ++i) {
        auto res = forward_step(prompt[i], state);
        if (res.is_err()) {
            return Result<std::vector<int>>::err("Prefill failed at token " +
                                                  std::to_string(i) + ": " + res.error());
        }
    }

    // 3. Obtain seed logits for the first decode step
    std::vector<int> generated;
    generated.reserve(max_tokens);

    std::vector<int> all_tokens(prompt.begin(), prompt.end());

    std::vector<float> logits;
    if (!prompt.empty()) {
        // Re-run last prompt token to capture logits for first sample.
        // This is a known simplification (double-processing the last token).
        auto res = forward_step(prompt.back(), state);
        if (res.is_err()) {
            return Result<std::vector<int>>::err(res.error());
        }
        logits = std::move(res.value());
    } else {
        // No prompt provided — start from EOS/BOS token
        auto res = forward_step(training::Tokenizer::EOS_TOKEN, state);
        if (res.is_err()) {
            return Result<std::vector<int>>::err(res.error());
        }
        logits = std::move(res.value());
    }

    // 4. Decode loop — sample and generate up to max_tokens
    for (int t = 0; t < max_tokens; ++t) {
        // 4a. Sample next token from logits
        int next_token = sampler.sample(logits, all_tokens);

        // 4b. Stop on EOS
        if (next_token == training::Tokenizer::EOS_TOKEN) {
            break;
        }

        generated.push_back(next_token);
        all_tokens.push_back(next_token);

        // 4c. Invoke streaming callback (if provided)
        if (callback) {
            std::string token_text;
            if (tokenizer_loaded_) {
                std::vector<int> single = {next_token};
                token_text = tokenizer_.decode(single);
            }
            if (!callback(next_token, token_text)) {
                break;  // user requested stop
            }
        }

        // 4d. Forward pass for next token's logits
        auto res = forward_step(next_token, state);
        if (res.is_err()) {
            return Result<std::vector<int>>::err("Generation failed at step " +
                                                  std::to_string(t) + ": " + res.error());
        }
        logits = std::move(res.value());
    }

    return Result<std::vector<int>>::ok(std::move(generated));
}

// ---------------------------------------------------------------------------
// generate_text
// ---------------------------------------------------------------------------
// Convenience wrapper around generate() that accepts and returns strings
// instead of raw token IDs.
//
//   1. Encode the prompt string into token IDs via BPE.
//   2. Run autoregressive generation.
//   3. Decode the generated token IDs back into a string.
// ---------------------------------------------------------------------------
Result<std::string> InferenceEngine::generate_text(const std::string& prompt,
                                                    int max_tokens,
                                                    const SamplerConfig& config,
                                                    TokenCallback callback) {
    if (!tokenizer_loaded_) {
        return Result<std::string>::err("Tokenizer not loaded");
    }

    // 1. Tokenise the input prompt
    auto tokens = tokenizer_.encode(prompt);

    // 2. Generate token IDs autoregressively
    auto res = generate(tokens, max_tokens, config, callback);
    if (res.is_err()) {
        return Result<std::string>::err(res.error());
    }

    // 3. Decode generated IDs back to text
    std::string output = tokenizer_.decode(res.value());
    return Result<std::string>::ok(std::move(output));
}

} // namespace rnet::inference

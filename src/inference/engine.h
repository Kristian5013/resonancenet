#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/error.h"
#include "core/sync.h"
#include "gpu/backend.h"
#include "gpu/tensor.h"
#include "inference/sampler.h"
#include "inference/state.h"
#include "training/checkpoint_io.h"
#include "training/model_config.h"
#include "training/tokenizer.h"

namespace rnet::inference {

/// Callback invoked for each generated token (for streaming).
/// Return false to stop generation early.
using TokenCallback = std::function<bool(int token, const std::string& text)>;

/// Inference engine: loads a model checkpoint and generates tokens
/// using O(1) recurrent state (no KV cache).
class InferenceEngine {
public:
    explicit InferenceEngine(gpu::GpuBackend& backend);
    ~InferenceEngine();

    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    /// Load model weights from a .rnet checkpoint file.
    Result<void> load_model(const std::filesystem::path& checkpoint);

    /// Load tokenizer from a directory containing vocab.json + merges.txt.
    Result<void> load_tokenizer(const std::filesystem::path& vocab_dir);

    /// Whether a model is loaded and ready for inference.
    bool is_ready() const;

    /// Get the loaded model configuration.
    const training::ModelConfig& model_config() const { return config_; }

    /// Get the tokenizer (must be loaded).
    const training::Tokenizer& tokenizer() const { return tokenizer_; }

    /// Generate tokens from a prompt (token IDs).
    /// @param prompt     Input token IDs
    /// @param max_tokens Maximum tokens to generate
    /// @param config     Sampling configuration
    /// @param callback   Optional per-token callback for streaming
    /// @return Generated token IDs (not including prompt)
    Result<std::vector<int>> generate(const std::vector<int>& prompt,
                                      int max_tokens,
                                      const SamplerConfig& config,
                                      TokenCallback callback = nullptr);

    /// Generate text from a string prompt.
    Result<std::string> generate_text(const std::string& prompt,
                                      int max_tokens,
                                      const SamplerConfig& config,
                                      TokenCallback callback = nullptr);

    /// Single-token forward pass: computes logits and updates state in-place.
    /// @param token  Input token ID
    /// @param state  Recurrent state (modified in-place)
    /// @return Logits of shape [vocab_size]
    Result<std::vector<float>> forward_step(int token, InferenceState& state);

    /// Get the model's parameter count.
    uint64_t param_count() const { return config_.param_count(); }

    /// Get the model's vocabulary size.
    int vocab_size() const { return static_cast<int>(config_.vocab_size); }

private:
    gpu::GpuBackend& backend_;
    training::ModelConfig config_;
    training::Tokenizer tokenizer_;
    bool model_loaded_ = false;
    bool tokenizer_loaded_ = false;

    /// Named weight tensors on GPU
    std::unordered_map<std::string, std::unique_ptr<gpu::GpuTensor>> weights_;

    /// Helper: upload a tensor from checkpoint to GPU.
    Result<void> upload_tensor(const training::TensorEntry& entry);

    /// Helper: get a weight tensor by name (returns nullptr if not found).
    const gpu::GpuTensor* get_weight(const std::string& name) const;

    /// Layer-wise forward pass helpers
    Result<void> forward_embedding(int token, std::vector<float>& x);
    Result<void> forward_layer(int layer, std::vector<float>& x, InferenceState& state);
    Result<void> forward_output(const std::vector<float>& x, std::vector<float>& logits);
};

}  // namespace rnet::inference

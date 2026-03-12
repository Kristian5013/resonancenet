#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>

#include "core/error.h"
#include "training/data_loader.h"
#include "training/lr_schedule.h"
#include "training/model_config.h"

namespace rnet::gpu {
class GpuBackend;
class GpuTensor;
}

namespace rnet::training {

/// Main training API for the Proof-of-Training blockchain.
///
/// Manages model weights on the GPU, runs training steps (forward + backward),
/// and handles checkpoint save/load. All GPU operations go through GpuBackend.
///
/// Forward pass per layer: rmsnorm -> causal_conv -> mingru -> slot_memory ->
///                          rmsnorm -> swiglu -> residual
class TrainingEngine {
public:
    explicit TrainingEngine(rnet::gpu::GpuBackend& backend);
    ~TrainingEngine();

    // Non-copyable, non-movable (owns GPU resources)
    TrainingEngine(const TrainingEngine&) = delete;
    TrainingEngine& operator=(const TrainingEngine&) = delete;
    TrainingEngine(TrainingEngine&&) = delete;
    TrainingEngine& operator=(TrainingEngine&&) = delete;

    /// Initialize model weights for the given config.
    /// Allocates all GPU tensors and initializes weights (random or zeros).
    Result<void> init(const ModelConfig& config);

    /// Load model weights from a checkpoint file.
    Result<void> load_checkpoint(const std::filesystem::path& path);

    /// Save current model weights to a checkpoint file.
    Result<void> save_checkpoint(const std::filesystem::path& path);

    /// Train for N steps on the provided data loader, returning the final loss.
    /// Uses AdamW optimizer with cosine LR schedule.
    Result<float> train_steps(int n_steps, DataLoader& data);

    /// Evaluate on validation data for n_batches, returning average loss.
    Result<float> evaluate(DataLoader& val_data, int n_batches);

    /// Expand the model for a growth event (increases d_model / adds layers).
    /// Existing weights are preserved; new weights are zero-initialized.
    Result<void> expand_model(const ModelConfig& new_config);

    /// Current model configuration.
    const ModelConfig& config() const;

    /// Current training step count.
    uint64_t step() const;

private:
    rnet::gpu::GpuBackend& backend_;
    ModelConfig config_;
    uint64_t step_ = 0;
    bool initialized_ = false;

    /// Named GPU tensors for model weights (BF16).
    std::unordered_map<std::string, std::unique_ptr<rnet::gpu::GpuTensor>> weights_;

    /// AdamW optimizer state: first moment (m) and second moment (v).
    std::unordered_map<std::string, std::unique_ptr<rnet::gpu::GpuTensor>> adam_m_;
    std::unordered_map<std::string, std::unique_ptr<rnet::gpu::GpuTensor>> adam_v_;

    /// Gradient tensors (same shapes as weights).
    std::unordered_map<std::string, std::unique_ptr<rnet::gpu::GpuTensor>> grads_;

    /// Activation buffers (allocated once during init).
    std::unique_ptr<rnet::gpu::GpuTensor> act_input_;
    std::unique_ptr<rnet::gpu::GpuTensor> act_residual_;
    std::unique_ptr<rnet::gpu::GpuTensor> act_norm_;
    std::unique_ptr<rnet::gpu::GpuTensor> act_conv_;
    std::unique_ptr<rnet::gpu::GpuTensor> act_gru_;
    std::unique_ptr<rnet::gpu::GpuTensor> act_gru_state_;
    std::unique_ptr<rnet::gpu::GpuTensor> act_slot_;
    std::unique_ptr<rnet::gpu::GpuTensor> act_ff_;
    std::unique_ptr<rnet::gpu::GpuTensor> act_logits_;

    /// Token buffers on device.
    void* gpu_tokens_ = nullptr;
    void* gpu_targets_ = nullptr;

    /// LR schedule for training.
    std::unique_ptr<LRSchedule> lr_schedule_;

    /// Kernel sizes on host for passing to backend.
    int kernel_sizes_host_[8] = {};

    /// Allocate a named weight tensor and its optimizer state.
    void alloc_weight(const std::string& name, std::vector<int64_t> shape);

    /// Allocate all activation buffers for the current config.
    void alloc_activations(int batch_size, int seq_len);

    /// Free all GPU resources.
    void free_resources();

    /// Run one forward pass, computing loss. Tokens/targets must already be on GPU.
    float forward_pass(int batch_size, int seq_len);

    /// Run one optimizer step on all weights.
    void optimizer_step(float lr);
};

}  // namespace rnet::training

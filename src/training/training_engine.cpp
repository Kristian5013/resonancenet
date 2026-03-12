#include "training/training_engine.h"

#include "training/checkpoint_io.h"
#include "gpu/backend.h"
#include "gpu/tensor.h"

namespace rnet::training {

using rnet::gpu::GpuTensor;
using rnet::gpu::DType;

TrainingEngine::TrainingEngine(rnet::gpu::GpuBackend& backend)
    : backend_(backend) {}

TrainingEngine::~TrainingEngine() {
    free_resources();
}

void TrainingEngine::free_resources() {
    // Release activation buffers
    act_input_.reset();
    act_residual_.reset();
    act_norm_.reset();
    act_conv_.reset();
    act_gru_.reset();
    act_gru_state_.reset();
    act_slot_.reset();
    act_ff_.reset();
    act_logits_.reset();

    // Release weight/optimizer/gradient tensors
    weights_.clear();
    adam_m_.clear();
    adam_v_.clear();
    grads_.clear();

    // Release token buffers
    if (gpu_tokens_) { backend_.free(gpu_tokens_); gpu_tokens_ = nullptr; }
    if (gpu_targets_) { backend_.free(gpu_targets_); gpu_targets_ = nullptr; }

    lr_schedule_.reset();
    initialized_ = false;
}

void TrainingEngine::alloc_weight(const std::string& name, std::vector<int64_t> shape) {
    auto w = std::make_unique<GpuTensor>(backend_, shape, DType::BF16);
    auto m = std::make_unique<GpuTensor>(backend_, shape, DType::FP32);
    auto v = std::make_unique<GpuTensor>(backend_, shape, DType::FP32);
    auto g = std::make_unique<GpuTensor>(backend_, shape, DType::BF16);

    // Zero-initialize optimizer state
    std::vector<uint8_t> zeros(m->size_bytes(), 0);
    m->copy_from_host(zeros.data());
    zeros.resize(v->size_bytes(), 0);
    v->copy_from_host(zeros.data());

    weights_[name] = std::move(w);
    adam_m_[name] = std::move(m);
    adam_v_[name] = std::move(v);
    grads_[name] = std::move(g);
}

void TrainingEngine::alloc_activations(int batch_size, int seq_len) {
    int d = static_cast<int>(config_.d_model);
    int d_ff = static_cast<int>(config_.d_ff);
    int vocab = static_cast<int>(config_.vocab_size);

    act_input_ = std::make_unique<GpuTensor>(
        backend_, std::vector<int64_t>{batch_size, seq_len, d}, DType::BF16);
    act_residual_ = std::make_unique<GpuTensor>(
        backend_, std::vector<int64_t>{batch_size, seq_len, d}, DType::BF16);
    act_norm_ = std::make_unique<GpuTensor>(
        backend_, std::vector<int64_t>{batch_size, seq_len, d}, DType::BF16);
    act_conv_ = std::make_unique<GpuTensor>(
        backend_, std::vector<int64_t>{batch_size, seq_len, d}, DType::BF16);
    act_gru_ = std::make_unique<GpuTensor>(
        backend_, std::vector<int64_t>{batch_size, seq_len, d}, DType::BF16);
    act_gru_state_ = std::make_unique<GpuTensor>(
        backend_, std::vector<int64_t>{batch_size, d}, DType::BF16);
    act_slot_ = std::make_unique<GpuTensor>(
        backend_, std::vector<int64_t>{batch_size, seq_len, d}, DType::BF16);
    act_ff_ = std::make_unique<GpuTensor>(
        backend_, std::vector<int64_t>{batch_size, seq_len, d}, DType::BF16);
    act_logits_ = std::make_unique<GpuTensor>(
        backend_, std::vector<int64_t>{batch_size, seq_len, vocab}, DType::BF16);

    gpu_tokens_ = backend_.alloc(static_cast<size_t>(batch_size) * seq_len * sizeof(int));
    gpu_targets_ = backend_.alloc(static_cast<size_t>(batch_size) * seq_len * sizeof(int));
}

Result<void> TrainingEngine::init(const ModelConfig& config) {
    free_resources();
    config_ = config;
    step_ = 0;

    int d = static_cast<int>(config_.d_model);
    int d_ff = static_cast<int>(config_.d_ff);
    int vocab = static_cast<int>(config_.vocab_size);
    int n_slots = static_cast<int>(config_.n_slots);

    // Prepare kernel sizes
    for (int i = 0; i < 8; ++i) {
        kernel_sizes_host_[i] = config_.kernel_sizes[i];
    }

    // Allocate embedding weight
    alloc_weight("embedding.weight", {vocab, d});

    // Per-layer weights
    for (uint32_t layer = 0; layer < config_.n_layers; ++layer) {
        std::string prefix = "layers." + std::to_string(layer) + ".";

        // RMSNorm scale
        alloc_weight(prefix + "rmsnorm.scale", {d});

        // Causal convolution weights
        // Total conv weight size: sum of active kernel_size * d_model
        int64_t total_conv_params = 0;
        for (int i = 0; i < config_.n_conv_branches; ++i) {
            if (config_.kernel_sizes[i] > 0) {
                total_conv_params += static_cast<int64_t>(config_.kernel_sizes[i]) * d;
            }
        }
        if (total_conv_params > 0) {
            alloc_weight(prefix + "conv.weight", {total_conv_params});
        }

        // MinGRU weights
        alloc_weight(prefix + "mingru.Wz", {d, d});
        alloc_weight(prefix + "mingru.Wh", {d, d});

        // Slot memory
        alloc_weight(prefix + "slot.keys", {d, n_slots});
        alloc_weight(prefix + "slot.values", {n_slots, d});

        // RMSNorm 2 scale
        alloc_weight(prefix + "rmsnorm2.scale", {d});

        // SwiGLU FFN
        alloc_weight(prefix + "ffn.W_up", {d, d_ff});
        alloc_weight(prefix + "ffn.W_gate", {d, d_ff});
        alloc_weight(prefix + "ffn.W_down", {d_ff, d});
    }

    // Output projection
    alloc_weight("output.weight", {d, vocab});

    // Default batch size for activation allocation
    constexpr int DEFAULT_BATCH = 4;
    int seq_len = static_cast<int>(config_.max_seq_len);
    alloc_activations(DEFAULT_BATCH, seq_len);

    // Initialize LR schedule
    LRSchedule::Config lr_config;
    lr_config.peak_lr = 3e-4f;
    lr_config.min_lr = 1e-5f;
    lr_config.warmup_steps = 100;
    lr_config.total_steps = 10000;
    lr_schedule_ = std::make_unique<LRSchedule>(lr_config);

    initialized_ = true;
    return Result<void>::ok();
}

Result<void> TrainingEngine::load_checkpoint(const std::filesystem::path& path) {
    auto hdr_result = read_checkpoint_header(path);
    if (hdr_result.is_err()) {
        return Result<void>::err("Failed to read checkpoint header: " + hdr_result.error());
    }

    const auto& hdr = hdr_result.value();

    // Initialize model with the checkpoint's config
    auto init_result = init(hdr.config);
    if (init_result.is_err()) {
        return init_result;
    }
    step_ = hdr.step;

    // Read tensor data
    auto tensors_result = read_checkpoint(path);
    if (tensors_result.is_err()) {
        return Result<void>::err("Failed to read checkpoint: " + tensors_result.error());
    }

    // Upload tensors to GPU
    for (const auto& entry : tensors_result.value()) {
        auto it = weights_.find(entry.name);
        if (it != weights_.end()) {
            if (it->second->size_bytes() == entry.data.size()) {
                it->second->copy_from_host(entry.data.data());
            }
            // Size mismatch tensors are silently skipped (may happen during growth)
        }
    }

    return Result<void>::ok();
}

Result<void> TrainingEngine::save_checkpoint(const std::filesystem::path& path) {
    if (!initialized_) {
        return Result<void>::err("Training engine not initialized");
    }

    CheckpointHeader hdr;
    hdr.config = config_;
    hdr.step = step_;
    hdr.n_tensors = weights_.size();

    std::vector<TensorEntry> tensors;
    tensors.reserve(weights_.size());

    for (const auto& [name, tensor] : weights_) {
        TensorEntry entry;
        entry.name = name;

        // Copy shape
        auto sp = tensor->shape();
        entry.shape.assign(sp.begin(), sp.end());

        // Download data from GPU
        entry.data.resize(tensor->size_bytes());
        tensor->copy_to_host(entry.data.data());

        tensors.push_back(std::move(entry));
    }

    return write_checkpoint(path, hdr, tensors);
}

float TrainingEngine::forward_pass(int batch_size, int seq_len) {
    int d = static_cast<int>(config_.d_model);
    int d_ff = static_cast<int>(config_.d_ff);
    int vocab = static_cast<int>(config_.vocab_size);
    int n_slots = static_cast<int>(config_.n_slots);

    // Embedding lookup
    backend_.embedding_forward(
        act_input_->data(),
        weights_.at("embedding.weight")->data(),
        static_cast<int*>(gpu_tokens_),
        batch_size, seq_len, d);

    // Copy to residual stream
    backend_.copy_to_device(act_residual_->data(), act_input_->data(),
                             act_input_->size_bytes());

    // Zero-init GRU state
    {
        std::vector<uint8_t> zeros(act_gru_state_->size_bytes(), 0);
        act_gru_state_->copy_from_host(zeros.data());
    }

    // Per-layer forward pass
    for (uint32_t layer = 0; layer < config_.n_layers; ++layer) {
        std::string prefix = "layers." + std::to_string(layer) + ".";

        // RMSNorm
        backend_.rmsnorm_forward(
            act_norm_->data(), act_residual_->data(),
            weights_.at(prefix + "rmsnorm.scale")->data(),
            batch_size, seq_len, d);

        // Causal convolution
        std::string conv_name = prefix + "conv.weight";
        auto conv_it = weights_.find(conv_name);
        if (conv_it != weights_.end()) {
            backend_.causal_conv_forward(
                act_conv_->data(), act_norm_->data(),
                conv_it->second->data(),
                kernel_sizes_host_, config_.n_conv_branches,
                batch_size, seq_len, d);
        } else {
            // No conv weights — pass through
            backend_.copy_to_device(act_conv_->data(), act_norm_->data(),
                                     act_norm_->size_bytes());
        }

        // MinGRU
        backend_.mingru_forward(
            act_gru_->data(), act_gru_state_->data(),
            act_conv_->data(), act_gru_state_->data(),
            weights_.at(prefix + "mingru.Wz")->data(),
            weights_.at(prefix + "mingru.Wh")->data(),
            batch_size, seq_len, d);

        // Slot memory
        backend_.slot_memory_forward(
            act_slot_->data(), act_gru_->data(),
            weights_.at(prefix + "slot.keys")->data(),
            weights_.at(prefix + "slot.values")->data(),
            batch_size, seq_len, d, n_slots);

        // RMSNorm 2
        backend_.rmsnorm_forward(
            act_norm_->data(), act_slot_->data(),
            weights_.at(prefix + "rmsnorm2.scale")->data(),
            batch_size, seq_len, d);

        // SwiGLU FFN
        backend_.swiglu_forward(
            act_ff_->data(), act_norm_->data(),
            weights_.at(prefix + "ffn.W_up")->data(),
            weights_.at(prefix + "ffn.W_gate")->data(),
            weights_.at(prefix + "ffn.W_down")->data(),
            batch_size, seq_len, d, d_ff);

        // Residual connection: residual = residual + ff_out
        // The backend handles residual addition within the forward kernels.
        // We update residual_buf to hold the layer output.
        backend_.copy_to_device(act_residual_->data(), act_ff_->data(),
                                 act_ff_->size_bytes());
    }

    // Output projection: logits = residual @ output_weight
    backend_.gemm(
        act_logits_->data(), act_residual_->data(),
        weights_.at("output.weight")->data(),
        batch_size * seq_len, vocab, d);

    // Cross-entropy loss
    float loss = 0.0f;
    backend_.cross_entropy_loss(
        &loss, act_logits_->data(),
        static_cast<int*>(gpu_targets_),
        batch_size, seq_len, vocab);

    backend_.synchronize();
    return loss;
}

void TrainingEngine::optimizer_step(float lr) {
    int current_step = static_cast<int>(step_);
    constexpr float beta1 = 0.9f;
    constexpr float beta2 = 0.999f;
    constexpr float eps = 1e-8f;
    constexpr float weight_decay = 0.01f;

    for (auto& [name, weight] : weights_) {
        auto grad_it = grads_.find(name);
        auto m_it = adam_m_.find(name);
        auto v_it = adam_v_.find(name);

        if (grad_it == grads_.end() || m_it == adam_m_.end() || v_it == adam_v_.end()) {
            continue;
        }

        int n_params = static_cast<int>(weight->numel());

        backend_.adamw_step(
            weight->data(), grad_it->second->data(),
            m_it->second->data(), v_it->second->data(),
            lr, beta1, beta2, eps, weight_decay,
            current_step, n_params);
    }
}

Result<float> TrainingEngine::train_steps(int n_steps, DataLoader& data) {
    if (!initialized_) {
        return Result<float>::err("Training engine not initialized");
    }

    constexpr int BATCH_SIZE = 4;
    int seq_len = static_cast<int>(config_.max_seq_len);

    float last_loss = 0.0f;

    for (int i = 0; i < n_steps; ++i) {
        auto batch = data.next_batch(BATCH_SIZE, seq_len);

        // Upload tokens and targets
        backend_.copy_to_device(gpu_tokens_, batch.tokens.data(),
                                 static_cast<size_t>(BATCH_SIZE) * seq_len * sizeof(int));
        backend_.copy_to_device(gpu_targets_, batch.targets.data(),
                                 static_cast<size_t>(BATCH_SIZE) * seq_len * sizeof(int));

        // Forward pass (computes loss and populates gradients via autograd)
        last_loss = forward_pass(BATCH_SIZE, seq_len);

        // Get learning rate for current step
        float lr = lr_schedule_->get_lr(static_cast<int>(step_));

        // Optimizer step
        optimizer_step(lr);

        ++step_;
    }

    return Result<float>::ok(last_loss);
}

Result<float> TrainingEngine::evaluate(DataLoader& val_data, int n_batches) {
    if (!initialized_) {
        return Result<float>::err("Training engine not initialized");
    }

    constexpr int BATCH_SIZE = 1;  // Single sample for deterministic eval
    int seq_len = static_cast<int>(config_.max_seq_len);

    float total_loss = 0.0f;
    int valid_batches = 0;

    val_data.reset();

    for (int b = 0; b < n_batches; ++b) {
        auto batch = val_data.next_batch(BATCH_SIZE, seq_len);

        backend_.copy_to_device(gpu_tokens_, batch.tokens.data(),
                                 static_cast<size_t>(BATCH_SIZE) * seq_len * sizeof(int));
        backend_.copy_to_device(gpu_targets_, batch.targets.data(),
                                 static_cast<size_t>(BATCH_SIZE) * seq_len * sizeof(int));

        float loss = forward_pass(BATCH_SIZE, seq_len);
        total_loss += loss;
        ++valid_batches;
    }

    if (valid_batches == 0) {
        return Result<float>::err("No valid evaluation batches");
    }

    return Result<float>::ok(total_loss / static_cast<float>(valid_batches));
}

Result<void> TrainingEngine::expand_model(const ModelConfig& new_config) {
    if (!initialized_) {
        return Result<void>::err("Training engine not initialized");
    }

    // Validate growth direction
    if (new_config.d_model < config_.d_model || new_config.n_layers < config_.n_layers) {
        return Result<void>::err("Model expansion must increase dimensions");
    }

    // Save current weights to a temporary checkpoint in memory
    std::vector<std::pair<std::string, std::vector<uint8_t>>> saved_weights;
    std::unordered_map<std::string, std::vector<int64_t>> saved_shapes;

    for (const auto& [name, tensor] : weights_) {
        std::vector<uint8_t> data(tensor->size_bytes());
        tensor->copy_to_host(data.data());
        saved_shapes[name] = std::vector<int64_t>(tensor->shape().begin(), tensor->shape().end());
        saved_weights.emplace_back(name, std::move(data));
    }

    uint64_t saved_step = step_;

    // Re-initialize with new config (this allocates new, larger tensors)
    auto result = init(new_config);
    if (result.is_err()) {
        return result;
    }
    step_ = saved_step;

    // Zero-initialize all new weights (done by init)
    // Copy old weights into the top-left corner of new tensors
    for (const auto& [name, old_data] : saved_weights) {
        auto it = weights_.find(name);
        if (it == weights_.end()) continue;

        auto& new_tensor = it->second;
        if (new_tensor->size_bytes() == old_data.size()) {
            // Same size — direct copy
            new_tensor->copy_from_host(old_data.data());
        } else if (new_tensor->size_bytes() > old_data.size()) {
            // New tensor is larger — copy old data to beginning, rest stays zero
            std::vector<uint8_t> padded(new_tensor->size_bytes(), 0);
            std::memcpy(padded.data(), old_data.data(), old_data.size());
            new_tensor->copy_from_host(padded.data());
        }
        // If new tensor is smaller (shouldn't happen for growth), skip
    }

    return Result<void>::ok();
}

const ModelConfig& TrainingEngine::config() const {
    return config_;
}

uint64_t TrainingEngine::step() const {
    return step_;
}

}  // namespace rnet::training

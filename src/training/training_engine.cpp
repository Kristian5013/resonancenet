#include "training/training_engine.h"

#include "training/checkpoint_io.h"
#include "gpu/backend.h"
#include "gpu/tensor.h"

#include <cmath>
#include <cstring>
#include <functional>
#include <random>

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

    // Release per-layer saved activations and gradient buffers
    layer_acts_.clear();
    grad_residual_.reset();
    grad_temp_.reset();
    grad_logits_.reset();
    grad_skip_.reset();

    // Release weight/optimizer/gradient tensors
    weights_.clear();
    adam_m_.clear();
    adam_v_.clear();
    grads_.clear();

    // Release token buffers
    if (gpu_tokens_) { backend_.free(gpu_tokens_); gpu_tokens_ = nullptr; }
    if (gpu_targets_) { backend_.free(gpu_targets_); gpu_targets_ = nullptr; }
    if (ones_buf_) { backend_.free(ones_buf_); ones_buf_ = nullptr; }

    lr_schedule_.reset();
    initialized_ = false;
}

void TrainingEngine::alloc_weight(const std::string& name, std::vector<int64_t> shape) {
    auto w = std::make_unique<GpuTensor>(backend_, shape, DType::FP32);
    auto m = std::make_unique<GpuTensor>(backend_, shape, DType::FP32);
    auto v = std::make_unique<GpuTensor>(backend_, shape, DType::FP32);
    auto g = std::make_unique<GpuTensor>(backend_, shape, DType::FP32);

    // Zero-initialize optimizer state and gradients
    backend_.memset_zero(m->data(), m->size_bytes());
    backend_.memset_zero(v->data(), v->size_bytes());
    backend_.memset_zero(g->data(), g->size_bytes());

    // Initialize weights on CPU and upload
    int64_t numel = w->numel();
    std::vector<float> init_data(numel);

    // Seed from tensor name hash for reproducibility without shared state
    std::hash<std::string> hasher;
    std::mt19937 rng(static_cast<uint32_t>(hasher(name)));

    bool is_scale = (name.find("rmsnorm") != std::string::npos && name.find("scale") != std::string::npos);

    if (is_scale) {
        // RMSNorm scale: initialize to 1.0
        std::fill(init_data.begin(), init_data.end(), 1.0f);
    } else if (shape.size() >= 2) {
        // Weight matrices: Xavier normal initialization
        int64_t fan_in = shape[0];
        int64_t fan_out = shape[1];
        float std_dev = std::sqrt(2.0f / static_cast<float>(fan_in + fan_out));
        std::normal_distribution<float> dist(0.0f, std_dev);
        for (int64_t i = 0; i < numel; ++i) {
            init_data[i] = dist(rng);
        }
    } else {
        // 1D weights (conv, etc.): small uniform
        float bound = 0.02f;
        std::uniform_real_distribution<float> dist(-bound, bound);
        for (int64_t i = 0; i < numel; ++i) {
            init_data[i] = dist(rng);
        }
    }
    w->copy_from_host(init_data.data());

    weights_[name] = std::move(w);
    adam_m_[name] = std::move(m);
    adam_v_[name] = std::move(v);
    grads_[name] = std::move(g);
}

void TrainingEngine::alloc_activations(int batch_size, int seq_len) {
    int d = static_cast<int>(config_.d_model);
    int d_ff = static_cast<int>(config_.d_ff);
    int vocab = static_cast<int>(config_.vocab_size);
    int n_layers = static_cast<int>(config_.n_layers);

    // Shared activation buffers (used during forward and backward)
    act_input_ = std::make_unique<GpuTensor>(backend_, std::vector<int64_t>{batch_size, seq_len, d}, DType::FP32);
    act_residual_ = std::make_unique<GpuTensor>(backend_, std::vector<int64_t>{batch_size, seq_len, d}, DType::FP32);
    act_norm_ = std::make_unique<GpuTensor>(backend_, std::vector<int64_t>{batch_size, seq_len, d}, DType::FP32);
    act_conv_ = std::make_unique<GpuTensor>(backend_, std::vector<int64_t>{batch_size, seq_len, d}, DType::FP32);
    act_gru_ = std::make_unique<GpuTensor>(backend_, std::vector<int64_t>{batch_size, seq_len, d}, DType::FP32);
    act_gru_state_ = std::make_unique<GpuTensor>(backend_, std::vector<int64_t>{batch_size, d}, DType::FP32);
    act_slot_ = std::make_unique<GpuTensor>(backend_, std::vector<int64_t>{batch_size, seq_len, d}, DType::FP32);
    act_ff_ = std::make_unique<GpuTensor>(backend_, std::vector<int64_t>{batch_size, seq_len, d}, DType::FP32);
    act_logits_ = std::make_unique<GpuTensor>(backend_, std::vector<int64_t>{batch_size, seq_len, vocab}, DType::FP32);

    // Per-layer saved activations for backward pass
    layer_acts_.resize(n_layers);
    for (int l = 0; l < n_layers; ++l) {
        auto& la = layer_acts_[l];
        la.residual_in = std::make_unique<GpuTensor>(backend_, std::vector<int64_t>{batch_size, seq_len, d}, DType::FP32);
        la.norm1_out = std::make_unique<GpuTensor>(backend_, std::vector<int64_t>{batch_size, seq_len, d}, DType::FP32);
        la.conv_out = std::make_unique<GpuTensor>(backend_, std::vector<int64_t>{batch_size, seq_len, d}, DType::FP32);
        la.gru_h_init = std::make_unique<GpuTensor>(backend_, std::vector<int64_t>{batch_size, d}, DType::FP32);
        la.gru_out = std::make_unique<GpuTensor>(backend_, std::vector<int64_t>{batch_size, seq_len, d}, DType::FP32);
        la.slot_out = std::make_unique<GpuTensor>(backend_, std::vector<int64_t>{batch_size, seq_len, d}, DType::FP32);
        la.norm2_out = std::make_unique<GpuTensor>(backend_, std::vector<int64_t>{batch_size, seq_len, d}, DType::FP32);
    }

    // Gradient buffers
    grad_residual_ = std::make_unique<GpuTensor>(backend_, std::vector<int64_t>{batch_size, seq_len, d}, DType::FP32);
    grad_temp_ = std::make_unique<GpuTensor>(backend_, std::vector<int64_t>{batch_size, seq_len, d}, DType::FP32);
    grad_logits_ = std::make_unique<GpuTensor>(backend_, std::vector<int64_t>{batch_size, seq_len, vocab}, DType::FP32);
    grad_skip_ = std::make_unique<GpuTensor>(backend_, std::vector<int64_t>{batch_size, seq_len, d}, DType::FP32);

    gpu_tokens_ = backend_.alloc(static_cast<size_t>(batch_size) * seq_len * sizeof(int));
    gpu_targets_ = backend_.alloc(static_cast<size_t>(batch_size) * seq_len * sizeof(int));

    // Scalar buffer for gemm-based vector add: contains 1.0f
    ones_buf_ = backend_.alloc(sizeof(float));
    float one = 1.0f;
    backend_.copy_to_device(ones_buf_, &one, sizeof(float));
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

        // Causal convolution weights: padded layout [n_branches, max_kernel, d]
        // CUDA kernels index as weights[br * max_kernel * d + k * d + i]
        int max_kernel = 0;
        for (int i = 0; i < config_.n_conv_branches; ++i) {
            if (config_.kernel_sizes[i] > max_kernel) max_kernel = config_.kernel_sizes[i];
        }
        if (max_kernel > 0 && config_.n_conv_branches > 0) {
            int64_t conv_weight_size = static_cast<int64_t>(config_.n_conv_branches) * max_kernel * d;
            alloc_weight(prefix + "conv.weight", {conv_weight_size});
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

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

float TrainingEngine::forward_pass(int batch_size, int seq_len) {
    int d = static_cast<int>(config_.d_model);
    int d_ff = static_cast<int>(config_.d_ff);
    int vocab = static_cast<int>(config_.vocab_size);
    int n_slots = static_cast<int>(config_.n_slots);

    // Embedding lookup
    backend_.embedding_forward(act_input_->data(), weights_.at("embedding.weight")->data(),
                                static_cast<int*>(gpu_tokens_), batch_size, seq_len, d);
    backend_.copy_to_device(act_residual_->data(), act_input_->data(), act_input_->size_bytes());

    // Zero-init GRU state
    backend_.memset_zero(act_gru_state_->data(), act_gru_state_->size_bytes());

    for (uint32_t layer = 0; layer < config_.n_layers; ++layer) {
        std::string prefix = "layers." + std::to_string(layer) + ".";
        auto& la = layer_acts_[layer];

        // Save residual input for backward
        backend_.copy_to_device(la.residual_in->data(), act_residual_->data(), act_residual_->size_bytes());

        // RMSNorm 1
        backend_.rmsnorm_forward(act_norm_->data(), act_residual_->data(),
                                  weights_.at(prefix + "rmsnorm.scale")->data(),
                                  batch_size, seq_len, d);
        backend_.copy_to_device(la.norm1_out->data(), act_norm_->data(), act_norm_->size_bytes());

        // Causal conv
        std::string conv_name = prefix + "conv.weight";
        auto conv_it = weights_.find(conv_name);
        if (conv_it != weights_.end()) {
            backend_.causal_conv_forward(act_conv_->data(), act_norm_->data(),
                                          conv_it->second->data(), kernel_sizes_host_,
                                          config_.n_conv_branches, batch_size, seq_len, d);
        } else {
            backend_.copy_to_device(act_conv_->data(), act_norm_->data(), act_norm_->size_bytes());
        }
        backend_.copy_to_device(la.conv_out->data(), act_conv_->data(), act_conv_->size_bytes());

        // Save GRU initial state
        backend_.copy_to_device(la.gru_h_init->data(), act_gru_state_->data(), act_gru_state_->size_bytes());

        // MinGRU
        backend_.mingru_forward(act_gru_->data(), act_gru_state_->data(),
                                 act_conv_->data(), act_gru_state_->data(),
                                 weights_.at(prefix + "mingru.Wz")->data(),
                                 weights_.at(prefix + "mingru.Wh")->data(),
                                 batch_size, seq_len, d);
        backend_.copy_to_device(la.gru_out->data(), act_gru_->data(), act_gru_->size_bytes());

        // Slot memory
        backend_.slot_memory_forward(act_slot_->data(), act_gru_->data(),
                                      weights_.at(prefix + "slot.keys")->data(),
                                      weights_.at(prefix + "slot.values")->data(),
                                      batch_size, seq_len, d, n_slots);
        backend_.copy_to_device(la.slot_out->data(), act_slot_->data(), act_slot_->size_bytes());

        // RMSNorm 2
        backend_.rmsnorm_forward(act_norm_->data(), act_slot_->data(),
                                  weights_.at(prefix + "rmsnorm2.scale")->data(),
                                  batch_size, seq_len, d);
        backend_.copy_to_device(la.norm2_out->data(), act_norm_->data(), act_norm_->size_bytes());

        // SwiGLU FFN
        backend_.swiglu_forward(act_ff_->data(), act_norm_->data(),
                                 weights_.at(prefix + "ffn.W_up")->data(),
                                 weights_.at(prefix + "ffn.W_gate")->data(),
                                 weights_.at(prefix + "ffn.W_down")->data(),
                                 batch_size, seq_len, d, d_ff);

        // Residual connection: residual += ff_out
        // Uses gemm trick: residual[N,1] = 1.0 * ff[N,1] @ [1.0] + 1.0 * residual[N,1]
        int total_elems = batch_size * seq_len * d;
        backend_.gemm(act_residual_->data(), act_ff_->data(), ones_buf_,
                       total_elems, 1, 1, 1.0f, 1.0f);
    }

    // Output projection
    backend_.gemm(act_logits_->data(), act_residual_->data(),
                   weights_.at("output.weight")->data(),
                   batch_size * seq_len, vocab, d);

    // Cross-entropy loss
    float loss = 0.0f;
    backend_.cross_entropy_loss(&loss, act_logits_->data(),
                                 static_cast<int*>(gpu_targets_), batch_size, seq_len, vocab);
    backend_.synchronize();
    return loss;
}

// ---------------------------------------------------------------------------
// Backward pass
// ---------------------------------------------------------------------------

void TrainingEngine::backward_pass(int batch_size, int seq_len) {
    int d = static_cast<int>(config_.d_model);
    int d_ff = static_cast<int>(config_.d_ff);
    int vocab = static_cast<int>(config_.vocab_size);
    int n_slots = static_cast<int>(config_.n_slots);
    int BS = batch_size * seq_len;

    // Zero all gradients
    for (auto& [name, grad] : grads_) {
        backend_.memset_zero(grad->data(), grad->size_bytes());
    }

    // 1. Cross-entropy backward -> grad_logits_
    backend_.cross_entropy_backward(grad_logits_->data(), act_logits_->data(),
                                     static_cast<int*>(gpu_targets_), batch_size, seq_len, vocab);

    // 2. Output projection backward:
    //    logits = residual @ output_weight  where residual [BS, d], output_weight [d, vocab]
    //    d_residual = d_logits @ output_weight^T  [BS, d]
    //    d_output_weight += residual^T @ d_logits  [d, vocab]
    backend_.gemm_ex(grad_residual_->data(), grad_logits_->data(),
                      weights_.at("output.weight")->data(),
                      BS, d, vocab, false, true);  // d_logits @ W^T
    backend_.gemm_ex(grads_.at("output.weight")->data(), act_residual_->data(),
                      grad_logits_->data(),
                      d, vocab, BS, true, false, 1.0f, 1.0f);  // residual^T @ d_logits, accumulate

    // 3. Backward through layers in reverse
    for (int layer = static_cast<int>(config_.n_layers) - 1; layer >= 0; --layer) {
        std::string prefix = "layers." + std::to_string(layer) + ".";
        auto& la = layer_acts_[layer];

        // Save incoming gradient for skip connection: d_skip = d_residual
        backend_.copy_to_device(grad_skip_->data(), grad_residual_->data(), grad_residual_->size_bytes());

        // --- SwiGLU backward ---
        // Forward: ff_out = swiglu(norm2_out), input was norm2_out
        backend_.swiglu_backward(
            grad_temp_->data(),  // d_x (d_norm2)
            grads_.at(prefix + "ffn.W_up")->data(),
            grads_.at(prefix + "ffn.W_gate")->data(),
            grads_.at(prefix + "ffn.W_down")->data(),
            grad_residual_->data(),  // d_out (d_ff)
            la.norm2_out->data(),    // x (input to swiglu)
            weights_.at(prefix + "ffn.W_up")->data(),
            weights_.at(prefix + "ffn.W_gate")->data(),
            weights_.at(prefix + "ffn.W_down")->data(),
            batch_size, seq_len, d, d_ff);

        // --- RMSNorm2 backward ---
        // Forward: norm2 = rmsnorm(slot_out), input was slot_out
        backend_.rmsnorm_backward(
            grad_residual_->data(),  // reuse as d_slot_out
            grads_.at(prefix + "rmsnorm2.scale")->data(),
            grad_temp_->data(),    // d_out (d_norm2)
            la.slot_out->data(),   // x (input to rmsnorm2)
            weights_.at(prefix + "rmsnorm2.scale")->data(),
            batch_size, seq_len, d);

        // --- Slot memory backward ---
        backend_.slot_memory_backward(
            grad_temp_->data(),  // d_x (d_gru_out)
            grads_.at(prefix + "slot.keys")->data(),
            grads_.at(prefix + "slot.values")->data(),
            grad_residual_->data(),  // d_out (d_slot_out)
            la.gru_out->data(),      // x (input to slot memory)
            weights_.at(prefix + "slot.keys")->data(),
            weights_.at(prefix + "slot.values")->data(),
            batch_size, seq_len, d, n_slots);

        // --- MinGRU backward ---
        backend_.mingru_backward(
            grad_residual_->data(),  // reuse as d_conv_out (d_x for mingru)
            grads_.at(prefix + "mingru.Wz")->data(),
            grads_.at(prefix + "mingru.Wh")->data(),
            grad_temp_->data(),        // d_h_out (d_gru_out)
            la.conv_out->data(),       // x (input to mingru)
            la.gru_out->data(),        // h_all (all hidden states)
            la.gru_h_init->data(),     // h_init
            weights_.at(prefix + "mingru.Wz")->data(),
            weights_.at(prefix + "mingru.Wh")->data(),
            batch_size, seq_len, d);

        // --- Causal conv backward ---
        std::string conv_name = prefix + "conv.weight";
        auto conv_it = weights_.find(conv_name);
        if (conv_it != weights_.end()) {
            backend_.causal_conv_backward(
                grad_temp_->data(),  // d_x (d_norm1)
                grads_.at(conv_name)->data(),
                grad_residual_->data(),  // d_out (d_conv_out)
                la.norm1_out->data(),    // x (input to conv)
                conv_it->second->data(), // forward weights
                kernel_sizes_host_, config_.n_conv_branches,
                batch_size, seq_len, d);
        } else {
            // No conv -- pass through gradient
            backend_.copy_to_device(grad_temp_->data(), grad_residual_->data(), grad_residual_->size_bytes());
        }

        // --- RMSNorm1 backward ---
        backend_.rmsnorm_backward(
            grad_residual_->data(),  // d_x (d_residual_in)
            grads_.at(prefix + "rmsnorm.scale")->data(),
            grad_temp_->data(),        // d_out (d_norm1)
            la.residual_in->data(),    // x (input to rmsnorm1)
            weights_.at(prefix + "rmsnorm.scale")->data(),
            batch_size, seq_len, d);

        // Add skip-connection gradient: d_residual = d_through_layer + d_skip
        int total_elems = batch_size * seq_len * d;
        backend_.gemm(grad_residual_->data(), grad_skip_->data(), ones_buf_,
                       total_elems, 1, 1, 1.0f, 1.0f);
    }

    // 4. Embedding backward
    backend_.embedding_backward(
        grads_.at("embedding.weight")->data(),
        grad_residual_->data(),
        static_cast<int*>(gpu_tokens_),
        batch_size, seq_len, d, vocab);
}

// ---------------------------------------------------------------------------
// Optimizer
// ---------------------------------------------------------------------------

void TrainingEngine::optimizer_step(float lr) {
    int current_step = static_cast<int>(step_) + 1;  // 1-indexed for bias correction
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

// ---------------------------------------------------------------------------
// Gradient clipping
// ---------------------------------------------------------------------------

void TrainingEngine::clip_gradients(float max_norm) {
    // Compute global gradient L2 norm across all parameters
    float total_sq = 0.0f;
    for (const auto& [name, grad] : grads_) {
        std::vector<float> host_grad(grad->numel());
        grad->copy_to_host(host_grad.data());
        for (float g : host_grad) {
            total_sq += g * g;
        }
    }
    float global_norm = std::sqrt(total_sq);

    if (global_norm > max_norm) {
        float scale = max_norm / global_norm;
        for (auto& [name, grad] : grads_) {
            std::vector<float> host_grad(grad->numel());
            grad->copy_to_host(host_grad.data());
            for (float& g : host_grad) {
                g *= scale;
            }
            grad->copy_from_host(host_grad.data());
        }
    }
}

// ---------------------------------------------------------------------------
// Training loop
// ---------------------------------------------------------------------------

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

        // Forward pass (computes loss and saves activations)
        last_loss = forward_pass(BATCH_SIZE, seq_len);

        // Reject NaN/Inf loss — indicates numerical divergence
        if (!std::isfinite(last_loss)) {
            return Result<float>::err("training diverged: loss is NaN or Inf");
        }

        // Backward pass (computes gradients for all weights)
        backward_pass(BATCH_SIZE, seq_len);

        // Gradient clipping: cap global gradient norm to 1.0
        clip_gradients(1.0f);

        // Get learning rate for current step
        float lr = lr_schedule_->get_lr(static_cast<int>(step_));

        // Optimizer step
        optimizer_step(lr);

        ++step_;
    }

    return Result<float>::ok(last_loss);
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

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
        if (!std::isfinite(loss)) continue;  // skip corrupted batches
        total_loss += loss;
        ++valid_batches;
    }

    if (valid_batches == 0) {
        return Result<float>::err("No valid evaluation batches");
    }

    return Result<float>::ok(total_loss / static_cast<float>(valid_batches));
}

// ---------------------------------------------------------------------------
// Model expansion (grow-then-train)
// ---------------------------------------------------------------------------

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
            auto old_shape_it = saved_shapes.find(name);
            auto new_shape = new_tensor->shape();

            // Check if this is a 2D+ tensor that changed column dimension
            if (old_shape_it != saved_shapes.end() &&
                old_shape_it->second.size() >= 2 && new_shape.size() >= 2 &&
                old_shape_it->second[1] != new_shape[1]) {
                // Row-by-row copy respecting stride change
                // old layout: rows of old_cols floats
                // new layout: rows of new_cols floats (zero-padded)
                int64_t old_rows = old_shape_it->second[0];
                int64_t old_cols = old_shape_it->second[1];
                int64_t new_cols = new_shape[1];
                size_t old_row_bytes = static_cast<size_t>(old_cols) * sizeof(float);
                size_t new_row_bytes = static_cast<size_t>(new_cols) * sizeof(float);

                std::vector<uint8_t> padded(new_tensor->size_bytes(), 0);
                for (int64_t r = 0; r < old_rows; ++r) {
                    std::memcpy(padded.data() + r * new_row_bytes,
                               old_data.data() + r * old_row_bytes,
                               old_row_bytes);
                }
                new_tensor->copy_from_host(padded.data());
            } else {
                // 1D tensor or same column count: flat copy is correct
                std::vector<uint8_t> padded(new_tensor->size_bytes(), 0);
                std::memcpy(padded.data(), old_data.data(), old_data.size());
                new_tensor->copy_from_host(padded.data());
            }
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

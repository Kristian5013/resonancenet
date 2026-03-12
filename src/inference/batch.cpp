#include "inference/batch.h"

#include "core/logging.h"

namespace rnet::inference {

BatchScheduler::BatchScheduler(InferenceEngine& engine, const BatchConfig& config)
    : engine_(engine), config_(config) {}

BatchScheduler::~BatchScheduler() {
    stop();
}

void BatchScheduler::start() {
    if (running_.exchange(true)) return;  // Already running

    // Start worker threads (one per batch slot)
    int n_workers = std::max(1, config_.max_batch_size);
    workers_.reserve(n_workers);
    for (int i = 0; i < n_workers; ++i) {
        workers_.emplace_back([this]() { worker_loop(); });
    }

    LogPrintf("BatchScheduler started with %d workers", n_workers);
}

void BatchScheduler::stop() {
    if (!running_.exchange(false)) return;  // Already stopped

    // Wake up all workers
    queue_cv_.notify_all();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();

    LogPrintf("BatchScheduler stopped");
}

Result<RequestId> BatchScheduler::submit(InferenceRequest request) {
    core::UniqueLock lock(mutex_);

    if (!running_.load(std::memory_order_relaxed)) {
        return Result<RequestId>::err("Scheduler is not running");
    }

    if (static_cast<int>(pending_queue_.size()) >= config_.max_queue_size) {
        return Result<RequestId>::err("Queue is full");
    }

    // Clamp max_tokens
    if (request.max_tokens > config_.max_tokens_per_request) {
        request.max_tokens = config_.max_tokens_per_request;
    }

    RequestId id = next_id_.fetch_add(1);
    request.id = id;
    request.status = RequestStatus::PENDING;
    request.submit_time = std::chrono::steady_clock::now();

    auto shared_req = std::make_shared<InferenceRequest>(std::move(request));
    requests_[id] = shared_req;
    pending_queue_.push(id);

    lock.unlock();
    queue_cv_.notify_one();

    return Result<RequestId>::ok(id);
}

Result<void> BatchScheduler::cancel(RequestId id) {
    core::UniqueLock lock(mutex_);
    auto it = requests_.find(id);
    if (it == requests_.end()) {
        return Result<void>::err("Request not found");
    }

    auto& req = it->second;
    if (req->status == RequestStatus::PENDING || req->status == RequestStatus::PROCESSING) {
        req->status = RequestStatus::CANCELLED;
        result_cv_.notify_all();
        return Result<void>::ok();
    }

    return Result<void>::err("Request already completed or failed");
}

Result<RequestStatus> BatchScheduler::get_status(RequestId id) const {
    core::UniqueLock lock(mutex_);
    auto it = requests_.find(id);
    if (it == requests_.end()) {
        return Result<RequestStatus>::err("Request not found");
    }
    return Result<RequestStatus>::ok(it->second->status);
}

Result<InferenceRequest> BatchScheduler::wait_for_result(RequestId id,
                                                          std::chrono::milliseconds timeout) {
    core::UniqueLock lock(mutex_);
    auto it = requests_.find(id);
    if (it == requests_.end()) {
        return Result<InferenceRequest>::err("Request not found");
    }

    auto& req = it->second;

    auto is_done = [&req]() {
        return req->status == RequestStatus::COMPLETED ||
               req->status == RequestStatus::FAILED ||
               req->status == RequestStatus::CANCELLED;
    };

    if (!is_done()) {
        if (timeout.count() > 0) {
            result_cv_.wait_for(lock, timeout, is_done);
        } else {
            result_cv_.wait(lock, is_done);
        }
    }

    if (!is_done()) {
        return Result<InferenceRequest>::err("Timed out waiting for result");
    }

    InferenceRequest result = *req;

    // Clean up completed request
    requests_.erase(it);

    if (result.status == RequestStatus::FAILED) {
        return Result<InferenceRequest>::err(result.error_message);
    }

    return Result<InferenceRequest>::ok(std::move(result));
}

int BatchScheduler::queue_size() const {
    core::UniqueLock lock(mutex_);
    return static_cast<int>(pending_queue_.size());
}

int BatchScheduler::active_count() const {
    return active_count_.load(std::memory_order_relaxed);
}

void BatchScheduler::worker_loop() {
    while (running_.load(std::memory_order_relaxed)) {
        std::shared_ptr<InferenceRequest> request;

        {
            core::UniqueLock lock(mutex_);
            queue_cv_.wait_for(lock, std::chrono::milliseconds(config_.batch_timeout_ms),
                               [this]() {
                                   return !pending_queue_.empty() ||
                                          !running_.load(std::memory_order_relaxed);
                               });

            if (!running_.load(std::memory_order_relaxed)) break;
            if (pending_queue_.empty()) continue;

            RequestId id = pending_queue_.front();
            pending_queue_.pop();

            auto it = requests_.find(id);
            if (it == requests_.end()) continue;

            request = it->second;
            if (request->status == RequestStatus::CANCELLED) continue;

            request->status = RequestStatus::PROCESSING;
            request->start_time = std::chrono::steady_clock::now();
        }

        active_count_.fetch_add(1);
        process_request(request);
        active_count_.fetch_sub(1);

        result_cv_.notify_all();
    }
}

void BatchScheduler::process_request(std::shared_ptr<InferenceRequest> request) {
    auto gen_res = engine_.generate(
        request->prompt_tokens,
        request->max_tokens,
        request->sampler_config,
        [&request](int token, const std::string& text) -> bool {
            // Check for cancellation
            if (request->status == RequestStatus::CANCELLED) {
                return false;
            }
            // Forward to user callback
            if (request->stream_callback) {
                return request->stream_callback(token, text);
            }
            return true;
        });

    {
        core::UniqueLock lock(mutex_);
        request->end_time = std::chrono::steady_clock::now();

        if (request->status == RequestStatus::CANCELLED) {
            return;  // Already cancelled
        }

        if (gen_res.is_err()) {
            request->status = RequestStatus::FAILED;
            request->error_message = gen_res.error();
        } else {
            request->status = RequestStatus::COMPLETED;
            request->output_tokens = std::move(gen_res.value());
        }
    }
}

}  // namespace rnet::inference

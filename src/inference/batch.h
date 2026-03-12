#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "core/error.h"
#include "core/sync.h"
#include "inference/engine.h"
#include "inference/sampler.h"

namespace rnet::inference {

/// Unique request identifier.
using RequestId = uint64_t;

/// Status of a batched inference request.
enum class RequestStatus : uint8_t {
    PENDING,      ///< Queued, waiting to be processed
    PROCESSING,   ///< Currently being generated
    COMPLETED,    ///< Generation finished
    FAILED,       ///< Generation failed with error
    CANCELLED,    ///< Cancelled by caller
};

/// A single inference request in the batch queue.
struct InferenceRequest {
    RequestId id = 0;
    std::vector<int> prompt_tokens;
    int max_tokens = 256;
    SamplerConfig sampler_config;
    TokenCallback stream_callback;

    // Output (filled when complete)
    std::vector<int> output_tokens;
    std::string error_message;
    RequestStatus status = RequestStatus::PENDING;

    // Timing
    std::chrono::steady_clock::time_point submit_time;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
};

/// Configuration for the batch scheduler.
struct BatchConfig {
    int max_batch_size = 8;           ///< Maximum concurrent requests
    int max_queue_size = 256;         ///< Maximum pending requests before rejection
    int batch_timeout_ms = 50;        ///< Wait time to fill batch before processing
    int max_tokens_per_request = 2048; ///< Hard limit on output tokens
};

/// Batch scheduler: queues inference requests and processes them in batches.
/// Each request runs sequentially (MinGRU is inherently sequential per request),
/// but multiple requests can be interleaved to hide latency.
class BatchScheduler {
public:
    BatchScheduler(InferenceEngine& engine, const BatchConfig& config = {});
    ~BatchScheduler();

    BatchScheduler(const BatchScheduler&) = delete;
    BatchScheduler& operator=(const BatchScheduler&) = delete;

    /// Start the background processing thread(s).
    void start();

    /// Stop processing and drain the queue.
    void stop();

    /// Submit a request. Returns a request ID for tracking.
    Result<RequestId> submit(InferenceRequest request);

    /// Cancel a pending or in-progress request.
    Result<void> cancel(RequestId id);

    /// Check the status of a request.
    Result<RequestStatus> get_status(RequestId id) const;

    /// Wait for a request to complete and retrieve the result.
    /// Blocks until the request finishes, fails, or is cancelled.
    Result<InferenceRequest> wait_for_result(RequestId id,
                                              std::chrono::milliseconds timeout = std::chrono::milliseconds(0));

    /// Current number of pending requests in the queue.
    int queue_size() const;

    /// Current number of actively processing requests.
    int active_count() const;

    /// Whether the scheduler is running.
    bool is_running() const { return running_.load(std::memory_order_relaxed); }

private:
    InferenceEngine& engine_;
    BatchConfig config_;

    mutable core::Mutex mutex_;
    core::CondVar queue_cv_;
    core::CondVar result_cv_;

    std::queue<RequestId> pending_queue_;
    std::unordered_map<RequestId, std::shared_ptr<InferenceRequest>> requests_;

    std::atomic<bool> running_{false};
    std::atomic<RequestId> next_id_{1};
    std::atomic<int> active_count_{0};

    std::vector<std::thread> workers_;

    /// Worker thread function.
    void worker_loop();

    /// Process a single request.
    void process_request(std::shared_ptr<InferenceRequest> request);
};

}  // namespace rnet::inference

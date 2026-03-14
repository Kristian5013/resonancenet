// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "inference/batch.h"

#include "core/logging.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <utility>

namespace rnet::inference {

// ===========================================================================
//  Construction
// ===========================================================================

BatchScheduler::BatchScheduler(InferenceEngine& engine, const BatchConfig& config)
    : engine_(engine), config_(config) {}

BatchScheduler::~BatchScheduler() {
    stop();
}

// ===========================================================================
//  Lifecycle
// ===========================================================================

// ---------------------------------------------------------------------------
// start
// ---------------------------------------------------------------------------
// Design: Launch N worker threads (one per batch slot), thread-pool pattern.
//   Each worker loops on the shared queue, competing for requests via the
//   condition variable.  max_batch_size controls concurrency — one thread per
//   slot ensures the GPU is never starved when multiple requests arrive.
// ---------------------------------------------------------------------------
void BatchScheduler::start() {
    // 1. Atomically flip running flag; bail if already running
    if (running_.exchange(true)) return;

    // 2. Spawn one worker thread per batch slot
    int n_workers = std::max(1, config_.max_batch_size);
    workers_.reserve(n_workers);
    for (int i = 0; i < n_workers; ++i) {
        workers_.emplace_back([this]() { worker_loop(); });
    }

    LogPrintf("BatchScheduler started with %d workers", n_workers);
}

// ---------------------------------------------------------------------------
// stop
// ---------------------------------------------------------------------------
// Design: Signal shutdown via atomic flag, wake all workers via CV, join
//   threads.  The atomic exchange guarantees stop() is idempotent — only the
//   first caller performs the teardown, subsequent calls are no-ops.
// ---------------------------------------------------------------------------
void BatchScheduler::stop() {
    // 1. Atomically clear running flag; bail if already stopped
    if (!running_.exchange(false)) return;

    // 2. Wake up all workers so they observe the flag change
    queue_cv_.notify_all();

    // 3. Join every worker thread
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();

    LogPrintf("BatchScheduler stopped");
}

// ===========================================================================
//  Request Submission
// ===========================================================================

// ---------------------------------------------------------------------------
// submit
// ---------------------------------------------------------------------------
// Design: Thread-safe queue insertion with capacity check, max_tokens
//   clamping.  The lock is held only long enough to validate state, enqueue,
//   and register the request — then released before signalling the CV so
//   workers can wake without contention on the mutex.
// ---------------------------------------------------------------------------
Result<RequestId> BatchScheduler::submit(InferenceRequest request) {
    core::UniqueLock lock(mutex_);

    // 1. Reject if scheduler is not running
    if (!running_.load(std::memory_order_relaxed)) {
        return Result<RequestId>::err("Scheduler is not running");
    }

    // 2. Reject if queue is at capacity
    if (static_cast<int>(pending_queue_.size()) >= config_.max_queue_size) {
        return Result<RequestId>::err("Queue is full");
    }

    // 3. Clamp max_tokens to the per-request hard limit
    if (request.max_tokens > config_.max_tokens_per_request) {
        request.max_tokens = config_.max_tokens_per_request;
    }

    // 4. Assign monotonic ID and stamp metadata
    RequestId id = next_id_.fetch_add(1);
    request.id = id;
    request.status = RequestStatus::PENDING;
    request.submit_time = std::chrono::steady_clock::now();

    // 5. Store shared ownership and enqueue the ID
    auto shared_req = std::make_shared<InferenceRequest>(std::move(request));
    requests_[id] = shared_req;
    pending_queue_.push(id);

    // 6. Release lock, then signal one waiting worker
    lock.unlock();
    queue_cv_.notify_one();

    return Result<RequestId>::ok(id);
}

// ===========================================================================
//  Request Management
// ===========================================================================

// ---------------------------------------------------------------------------
// cancel
// ---------------------------------------------------------------------------
// Design: State transition: PENDING/PROCESSING -> CANCELLED.  Only requests
//   still in-flight can be cancelled; completed or failed requests are
//   immutable.  The result CV is broadcast so that any thread blocked in
//   wait_for_result sees the cancellation immediately.
// ---------------------------------------------------------------------------
Result<void> BatchScheduler::cancel(RequestId id) {
    core::UniqueLock lock(mutex_);

    // 1. Look up the request by ID
    auto it = requests_.find(id);
    if (it == requests_.end()) {
        return Result<void>::err("Request not found");
    }

    // 2. Transition to CANCELLED if still in-flight
    auto& req = it->second;
    if (req->status == RequestStatus::PENDING || req->status == RequestStatus::PROCESSING) {
        req->status = RequestStatus::CANCELLED;
        result_cv_.notify_all();
        return Result<void>::ok();
    }

    return Result<void>::err("Request already completed or failed");
}

// ---------------------------------------------------------------------------
// get_status
// ---------------------------------------------------------------------------
Result<RequestStatus> BatchScheduler::get_status(RequestId id) const {
    core::UniqueLock lock(mutex_);

    // 1. Look up the request by ID
    auto it = requests_.find(id);
    if (it == requests_.end()) {
        return Result<RequestStatus>::err("Request not found");
    }

    return Result<RequestStatus>::ok(it->second->status);
}

// ---------------------------------------------------------------------------
// wait_for_result
// ---------------------------------------------------------------------------
// Design: Blocking wait with condition variable, optional timeout, cleanup on
//   completion.  When timeout is zero the call blocks indefinitely; otherwise
//   it returns an error after the deadline.  The request entry is erased from
//   the map upon retrieval so callers cannot double-read results.
// ---------------------------------------------------------------------------
Result<InferenceRequest> BatchScheduler::wait_for_result(RequestId id,
                                                          std::chrono::milliseconds timeout) {
    core::UniqueLock lock(mutex_);

    // 1. Look up the request by ID
    auto it = requests_.find(id);
    if (it == requests_.end()) {
        return Result<InferenceRequest>::err("Request not found");
    }

    auto& req = it->second;

    // 2. Build completion predicate
    auto is_done = [&req]() {
        return req->status == RequestStatus::COMPLETED ||
               req->status == RequestStatus::FAILED ||
               req->status == RequestStatus::CANCELLED;
    };

    // 3. Block until done or timeout expires
    if (!is_done()) {
        if (timeout.count() > 0) {
            result_cv_.wait_for(lock, timeout, is_done);
        } else {
            result_cv_.wait(lock, is_done);
        }
    }

    // 4. Check whether we actually finished or timed out
    if (!is_done()) {
        return Result<InferenceRequest>::err("Timed out waiting for result");
    }

    // 5. Copy result and erase the request entry
    InferenceRequest result = *req;
    requests_.erase(it);

    // 6. Return error result for failed requests, success otherwise
    if (result.status == RequestStatus::FAILED) {
        return Result<InferenceRequest>::err(result.error_message);
    }

    return Result<InferenceRequest>::ok(std::move(result));
}

// ---------------------------------------------------------------------------
// queue_size
// ---------------------------------------------------------------------------
int BatchScheduler::queue_size() const {
    core::UniqueLock lock(mutex_);
    return static_cast<int>(pending_queue_.size());
}

// ---------------------------------------------------------------------------
// active_count
// ---------------------------------------------------------------------------
int BatchScheduler::active_count() const {
    return active_count_.load(std::memory_order_relaxed);
}

// ===========================================================================
//  Worker Loop
// ===========================================================================

// ---------------------------------------------------------------------------
// worker_loop
// ---------------------------------------------------------------------------
// Design: Consumer pattern: wait on CV -> dequeue -> process -> notify result.
//   Each worker blocks on the queue CV with a bounded timeout so it can
//   periodically re-check the shutdown flag.  On wake, the worker dequeues a
//   single request, transitions it to PROCESSING, then drops the lock before
//   performing the (potentially expensive) inference call.
// ---------------------------------------------------------------------------
void BatchScheduler::worker_loop() {
    while (running_.load(std::memory_order_relaxed)) {
        std::shared_ptr<InferenceRequest> request;

        {
            core::UniqueLock lock(mutex_);

            // 1. Wait for work or shutdown signal
            queue_cv_.wait_for(lock, std::chrono::milliseconds(config_.batch_timeout_ms),
                               [this]() {
                                   return !pending_queue_.empty() ||
                                          !running_.load(std::memory_order_relaxed);
                               });

            // 2. Re-check shutdown after wake
            if (!running_.load(std::memory_order_relaxed)) break;

            // 3. Spurious wake or timeout with no work — retry
            if (pending_queue_.empty()) continue;

            // 4. Dequeue the front request ID
            RequestId id = pending_queue_.front();
            pending_queue_.pop();

            // 5. Look up the shared request; skip if already gone
            auto it = requests_.find(id);
            if (it == requests_.end()) continue;

            request = it->second;

            // 6. Skip cancelled requests
            if (request->status == RequestStatus::CANCELLED) continue;

            // 7. Transition to PROCESSING and stamp start time
            request->status = RequestStatus::PROCESSING;
            request->start_time = std::chrono::steady_clock::now();
        }

        // 8. Process outside the lock
        active_count_.fetch_add(1);
        process_request(request);
        active_count_.fetch_sub(1);

        // 9. Notify any threads waiting on results
        result_cv_.notify_all();
    }
}

// ===========================================================================
//  Request Processing
// ===========================================================================

// ---------------------------------------------------------------------------
// process_request
// ---------------------------------------------------------------------------
// Design: Delegate to InferenceEngine::generate with cancellation check
//   callback.  The streaming callback serves double duty: it forwards tokens
//   to the caller's optional stream_callback AND checks the cancellation flag
//   on every token, returning false to abort generation early if the request
//   was cancelled from another thread.
// ---------------------------------------------------------------------------
void BatchScheduler::process_request(std::shared_ptr<InferenceRequest> request) {
    // 1. Invoke the engine with a streaming/cancellation callback
    auto gen_res = engine_.generate(
        request->prompt_tokens,
        request->max_tokens,
        request->sampler_config,
        [&request](int token, const std::string& text) -> bool {
            // 2. Check for cancellation on every token
            if (request->status == RequestStatus::CANCELLED) {
                return false;
            }
            // 3. Forward to user callback if provided
            if (request->stream_callback) {
                return request->stream_callback(token, text);
            }
            return true;
        });

    {
        core::UniqueLock lock(mutex_);

        // 4. Stamp completion time
        request->end_time = std::chrono::steady_clock::now();

        // 5. Bail if cancelled while we were generating
        if (request->status == RequestStatus::CANCELLED) {
            return;
        }

        // 6. Record outcome
        if (gen_res.is_err()) {
            request->status = RequestStatus::FAILED;
            request->error_message = gen_res.error();
        } else {
            request->status = RequestStatus::COMPLETED;
            request->output_tokens = std::move(gen_res.value());
        }
    }
}

} // namespace rnet::inference

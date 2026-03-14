// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "core/thread.h"

#include "core/logging.h"

#include <deque>
#include <exception>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

namespace rnet::core {

// ===========================================================================
//  Thread naming
// ===========================================================================

static thread_local std::string t_thread_name = "unknown";

// ---------------------------------------------------------------------------
// set_thread_name / get_thread_name
//
// Sets both the application-level thread-local name and the OS-level
// thread description (visible in debuggers and profilers).
// Windows: SetThreadDescription (Win10 1607+).
// Linux:   pthread_setname_np (max 15 chars + null).
// macOS:   pthread_setname_np (current thread variant).
// ---------------------------------------------------------------------------
void set_thread_name(const std::string& name) {
    t_thread_name = name;

#ifdef _WIN32
    // 1. Convert UTF-8 name to wide string for Win32 API.
    int len = MultiByteToWideChar(CP_UTF8, 0, name.c_str(),
                                   -1, nullptr, 0);
    if (len > 0) {
        std::vector<wchar_t> wide(static_cast<size_t>(len));
        MultiByteToWideChar(CP_UTF8, 0, name.c_str(), -1,
                            wide.data(), len);
        SetThreadDescription(GetCurrentThread(), wide.data());
    }
#elif defined(__linux__)
    // 1. Truncate to 15 chars (pthread_setname_np limit).
    std::string truncated = name.substr(0, 15);
    pthread_setname_np(pthread_self(), truncated.c_str());
#elif defined(__APPLE__)
    pthread_setname_np(name.c_str());
#endif
}

std::string get_thread_name() {
    return t_thread_name;
}

// ---------------------------------------------------------------------------
// trace_thread
//
// Wraps a thread body with name-setting, logging, and exception handling.
// All application threads should be launched through this function or
// through ThreadGroup::create_thread (which calls this internally).
// ---------------------------------------------------------------------------
void trace_thread(const std::string& name,
                  std::function<void()> fn) {
    set_thread_name(name);
    try {
        LogPrintf("Thread %s started", name.c_str());
        fn();
        LogPrintf("Thread %s exited", name.c_str());
    } catch (const std::exception& e) {
        LogError("Thread %s exception: %s", name.c_str(), e.what());
    } catch (...) {
        LogError("Thread %s: unknown exception", name.c_str());
    }
}

// ===========================================================================
//  ShutdownFlag
// ===========================================================================

// ---------------------------------------------------------------------------
// ShutdownFlag — process-wide atomic boolean for cooperative shutdown.
//
// All long-running loops check shutdown_requested() and exit cleanly
// when it becomes true.
// ---------------------------------------------------------------------------
ShutdownFlag& ShutdownFlag::instance() {
    static ShutdownFlag flag;
    return flag;
}

void ShutdownFlag::request_shutdown() {
    shutdown_.store(true, std::memory_order_release);
}

bool ShutdownFlag::is_shutdown_requested() const {
    return shutdown_.load(std::memory_order_acquire);
}

void ShutdownFlag::reset() {
    shutdown_.store(false, std::memory_order_release);
}

bool shutdown_requested() {
    return ShutdownFlag::instance().is_shutdown_requested();
}

void request_shutdown() {
    ShutdownFlag::instance().request_shutdown();
}

// ===========================================================================
//  ThreadGroup
// ===========================================================================

// ---------------------------------------------------------------------------
// ThreadGroup — named-thread collection with join-all-on-destruct.
//
// Each thread is launched via trace_thread for uniform logging and
// exception safety.
// ---------------------------------------------------------------------------
ThreadGroup::~ThreadGroup() {
    join_all();
}

void ThreadGroup::create_thread(const std::string& name,
                                 std::function<void()> fn) {
    threads_.push_back({name, std::thread([name, fn = std::move(fn)]() {
        trace_thread(name, fn);
    })});
}

void ThreadGroup::join_all() {
    for (auto& ti : threads_) {
        if (ti.thread.joinable()) {
            ti.thread.join();
        }
    }
    threads_.clear();
}

size_t ThreadGroup::size() const {
    return threads_.size();
}

bool ThreadGroup::all_done() const {
    for (const auto& ti : threads_) {
        if (ti.thread.joinable()) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// get_num_cores
//
// Returns hardware concurrency, falling back to 1 if the OS reports 0.
// ---------------------------------------------------------------------------
unsigned int get_num_cores() {
    unsigned int cores = std::thread::hardware_concurrency();
    return cores > 0 ? cores : 1;
}

// ===========================================================================
//  WorkQueue
// ===========================================================================

// ---------------------------------------------------------------------------
// WorkQueue — bounded, blocking FIFO for work items.
//
// Producers call enqueue(); consumer threads call run() which blocks
// until work is available or shutdown is signalled.  max_depth of 0
// means unlimited depth.
// ---------------------------------------------------------------------------
WorkQueue::WorkQueue(size_t max_depth)
    : max_depth_(max_depth) {}

WorkQueue::~WorkQueue() {
    shutdown();
}

bool WorkQueue::enqueue(std::function<void()> fn) {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (shutdown_) return false;
        if (max_depth_ > 0 && queue_.size() >= max_depth_) {
            return false;
        }
        queue_.push_back(std::move(fn));
    }
    cv_.notify_one();
    return true;
}

void WorkQueue::run() {
    while (true) {
        std::function<void()> fn;
        {
            // 1. Wait for work or shutdown signal.
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() {
                return shutdown_ || !queue_.empty();
            });
            if (shutdown_ && queue_.empty()) return;
            // 2. Dequeue one item.
            fn = std::move(queue_.front());
            queue_.pop_front();
        }
        // 3. Execute outside the lock.
        try {
            fn();
        } catch (const std::exception& e) {
            LogError("WorkQueue exception: %s", e.what());
        } catch (...) {
            LogError("WorkQueue: unknown exception");
        }
    }
}

void WorkQueue::shutdown() {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        shutdown_ = true;
    }
    cv_.notify_all();
}

size_t WorkQueue::pending() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return queue_.size();
}

bool WorkQueue::is_shutdown() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return shutdown_;
}

// ===========================================================================
//  ThreadPool
// ===========================================================================

// ---------------------------------------------------------------------------
// ThreadPool — fixed-size pool backed by a shared WorkQueue.
//
// Defaults to hardware_concurrency threads if num_threads is 0.
// Destruction calls stop() which drains remaining work and joins all
// worker threads.
// ---------------------------------------------------------------------------
ThreadPool::ThreadPool(size_t num_threads)
    : queue_(0) {
    if (num_threads == 0) {
        num_threads = get_num_cores();
    }
    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this, i]() {
            set_thread_name("pool-" + std::to_string(i));
            queue_.run();
        });
    }
}

ThreadPool::~ThreadPool() {
    stop();
}

bool ThreadPool::submit(std::function<void()> fn) {
    return queue_.enqueue(std::move(fn));
}

void ThreadPool::stop() {
    if (stopped_) return;
    stopped_ = true;
    queue_.shutdown();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
    workers_.clear();
}

size_t ThreadPool::num_threads() const {
    return workers_.size();
}

} // namespace rnet::core

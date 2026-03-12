#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace rnet::core {

/// Set the name of the current thread (for debugging/logging).
void set_thread_name(const std::string& name);

/// Get the name of the current thread.
std::string get_thread_name();

/// Run a function in a named thread with exception logging.
/// The function runs synchronously in the current thread.
void trace_thread(const std::string& name,
                  std::function<void()> fn);

/// Global shutdown flag
class ShutdownFlag {
public:
    static ShutdownFlag& instance();

    void request_shutdown();
    bool is_shutdown_requested() const;
    void reset();

private:
    ShutdownFlag() = default;
    std::atomic<bool> shutdown_{false};
};

/// Check if shutdown was requested
bool shutdown_requested();

/// Request global shutdown
void request_shutdown();

/// Thread group: manage a collection of threads that can be
/// joined together.
class ThreadGroup {
public:
    ThreadGroup() = default;
    ~ThreadGroup();

    ThreadGroup(const ThreadGroup&) = delete;
    ThreadGroup& operator=(const ThreadGroup&) = delete;
    ThreadGroup(ThreadGroup&&) = default;
    ThreadGroup& operator=(ThreadGroup&&) = default;

    /// Create and start a new named thread
    void create_thread(const std::string& name,
                       std::function<void()> fn);

    /// Join all threads
    void join_all();

    /// Number of active threads
    size_t size() const;

    /// Check if all threads have completed
    bool all_done() const;

private:
    struct ThreadInfo {
        std::string name;
        std::thread thread;
    };
    std::vector<ThreadInfo> threads_;
};

/// Get the number of hardware threads available
unsigned int get_num_cores();

/// WorkQueue: a simple work-stealing queue for thread pools.
/// Consumers call run() to process items until shutdown.
class WorkQueue {
public:
    explicit WorkQueue(size_t max_depth = 0);
    ~WorkQueue();

    WorkQueue(const WorkQueue&) = delete;
    WorkQueue& operator=(const WorkQueue&) = delete;

    /// Enqueue a work item. Returns false if shutting down.
    bool enqueue(std::function<void()> fn);

    /// Run the work queue loop (blocks until shutdown).
    /// Call from worker threads.
    void run();

    /// Request shutdown; wakes all waiting workers.
    void shutdown();

    /// Get number of items waiting
    size_t pending() const;

    /// Check if shutdown was requested
    bool is_shutdown() const;

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::function<void()>> queue_;
    size_t max_depth_;
    bool shutdown_ = false;
};

/// Simple thread pool built on WorkQueue
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = 0);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /// Submit a work item
    bool submit(std::function<void()> fn);

    /// Shutdown and join all threads
    void stop();

    /// Number of worker threads
    size_t num_threads() const;

private:
    WorkQueue queue_;
    std::vector<std::thread> workers_;
    bool stopped_ = false;
};

}  // namespace rnet::core

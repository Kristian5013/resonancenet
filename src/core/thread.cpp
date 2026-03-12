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

static thread_local std::string t_thread_name = "unknown";

void set_thread_name(const std::string& name) {
    t_thread_name = name;

#ifdef _WIN32
    // Windows: SetThreadDescription (Windows 10 1607+)
    // Convert to wide string
    int len = MultiByteToWideChar(CP_UTF8, 0, name.c_str(),
                                   -1, nullptr, 0);
    if (len > 0) {
        std::vector<wchar_t> wide(static_cast<size_t>(len));
        MultiByteToWideChar(CP_UTF8, 0, name.c_str(), -1,
                            wide.data(), len);
        SetThreadDescription(GetCurrentThread(), wide.data());
    }
#elif defined(__linux__)
    // Linux: pthread_setname_np (max 16 chars including null)
    std::string truncated = name.substr(0, 15);
    pthread_setname_np(pthread_self(), truncated.c_str());
#elif defined(__APPLE__)
    pthread_setname_np(name.c_str());
#endif
}

std::string get_thread_name() {
    return t_thread_name;
}

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

// ─── ShutdownFlag ────────────────────────────────────────────────────

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

// ─── ThreadGroup ─────────────────────────────────────────────────────

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

unsigned int get_num_cores() {
    unsigned int cores = std::thread::hardware_concurrency();
    return cores > 0 ? cores : 1;
}

// ─── WorkQueue ───────────────────────────────────────────────────────

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
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() {
                return shutdown_ || !queue_.empty();
            });
            if (shutdown_ && queue_.empty()) return;
            fn = std::move(queue_.front());
            queue_.pop_front();
        }
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

// ─── ThreadPool ──────────────────────────────────────────────────────

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

}  // namespace rnet::core

#pragma once

#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <chrono>
#include <thread>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <cassert>
#include <functional>

namespace rnet::core {

/// Primary mutex type — NON-RECURSIVE by design.
using Mutex = std::mutex;

/// Recursive mutex for exceptional cases only.
using RecursiveMutex = std::recursive_mutex;

/// Shared (reader-writer) mutex
using SharedMutex = std::shared_mutex;

/// Condition variable
using CondVar = std::condition_variable;

/// RAII lock types
using UniqueLock = std::unique_lock<Mutex>;
using SharedLock = std::shared_lock<SharedMutex>;
using RecursiveLock = std::unique_lock<RecursiveMutex>;

// Concatenation helpers for LOCK macro
#define RNET_PASTE2(a, b) a ## b
#define RNET_PASTE(a, b) RNET_PASTE2(a, b)

/// LOCK(mutex) — creates a unique_lock with auto-generated name.
#define LOCK(cs) \
    std::unique_lock<decltype(cs)> RNET_PASTE(lock_, __LINE__)(cs)

/// LOCK2(cs1, cs2) — lock two mutexes without deadlock.
#define LOCK2(cs1, cs2) \
    std::scoped_lock RNET_PASTE(lock_, __LINE__)(cs1, cs2)

/// TRY_LOCK(mutex, name) — attempts to lock; `name` is bool success flag.
#define TRY_LOCK(cs, name) \
    std::unique_lock<decltype(cs)> name(cs, std::try_to_lock)

/// WAIT(cv, lock, pred) — wait on condition variable with predicate.
#define WAIT(cv, lock, pred) \
    (cv).wait(lock, [&]() { return pred; })

#ifdef RNET_DEBUG_LOCKORDER

/// Debug lock-order tracking to detect potential deadlocks.
/// Each thread records the order in which it acquires locks.
/// If thread A locks (M1, M2) and thread B locks (M2, M1), flag it.
class LockOrderTracker {
public:
    static LockOrderTracker& instance();

    void track_lock(const void* mutex_addr, const char* name,
                    const char* file, int line);
    void track_unlock(const void* mutex_addr);
    void check_order(const void* mutex_addr, const char* name,
                     const char* file, int line);

private:
    LockOrderTracker() = default;

    struct LockInfo {
        const void* addr = nullptr;
        std::string name;
        std::string file;
        int line = 0;
    };

    Mutex tracker_mutex_;

    // Per-thread held locks (ordered by acquisition)
    std::map<std::thread::id, std::vector<LockInfo>> held_locks_;

    // Known lock orderings: if (A, B) is in the set, A must be locked
    // before B.
    std::set<std::pair<const void*, const void*>> observed_order_;
};

#define DEBUG_LOCK(cs, name) \
    LockOrderTracker::instance().check_order(&(cs), name, __FILE__, __LINE__)
#define DEBUG_UNLOCK(cs) \
    LockOrderTracker::instance().track_unlock(&(cs))

#else

#define DEBUG_LOCK(cs, name) ((void)0)
#define DEBUG_UNLOCK(cs) ((void)0)

#endif  // RNET_DEBUG_LOCKORDER

/// Semaphore (counting) built on mutex + condvar
class Semaphore {
public:
    explicit Semaphore(int initial_count = 0)
        : count_(initial_count) {}

    void post() {
        {
            std::unique_lock<Mutex> lock(mutex_);
            ++count_;
        }
        cv_.notify_one();
    }

    void wait() {
        std::unique_lock<Mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return count_ > 0; });
        --count_;
    }

    bool try_wait() {
        std::unique_lock<Mutex> lock(mutex_);
        if (count_ > 0) {
            --count_;
            return true;
        }
        return false;
    }

private:
    Mutex mutex_;
    CondVar cv_;
    int count_;
};

/// CountdownLatch: blocks until a counter reaches zero.
class CountdownLatch {
public:
    explicit CountdownLatch(int count) : count_(count) {}

    void count_down() {
        {
            std::unique_lock<Mutex> lock(mutex_);
            if (count_ > 0) --count_;
        }
        if (count_ == 0) cv_.notify_all();
    }

    void wait() {
        std::unique_lock<Mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return count_ <= 0; });
    }

    template<typename Rep, typename Period>
    bool wait_for(const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<Mutex> lock(mutex_);
        return cv_.wait_for(lock, timeout,
                            [this]() { return count_ <= 0; });
    }

    int count() const {
        std::unique_lock<Mutex> lock(mutex_);
        return count_;
    }

private:
    mutable Mutex mutex_;
    CondVar cv_;
    int count_;
};

/// OnceFlag: execute a function exactly once, thread-safely.
class OnceFlag {
public:
    template<typename Fn>
    void call_once(Fn&& fn) {
        std::call_once(flag_, std::forward<Fn>(fn));
    }

private:
    std::once_flag flag_;
};

/// SpinLock: busy-wait lock for very short critical sections.
/// Only use when lock hold time is sub-microsecond.
class SpinLock {
public:
    void lock() {
        while (flag_.test_and_set(std::memory_order_acquire)) {
            // Spin — add pause hint for x86
#ifdef _MSC_VER
            _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
#endif
        }
    }

    void unlock() {
        flag_.clear(std::memory_order_release);
    }

    bool try_lock() {
        return !flag_.test_and_set(std::memory_order_acquire);
    }

private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

#ifdef _MSC_VER
#include <intrin.h>
#endif

/// AtomicCounter: convenient atomic integer with named operations.
class AtomicCounter {
public:
    explicit AtomicCounter(int64_t initial = 0)
        : value_(initial) {}

    int64_t increment() {
        return value_.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    int64_t decrement() {
        return value_.fetch_sub(1, std::memory_order_relaxed) - 1;
    }

    int64_t add(int64_t delta) {
        return value_.fetch_add(delta, std::memory_order_relaxed)
               + delta;
    }

    int64_t get() const {
        return value_.load(std::memory_order_relaxed);
    }

    void set(int64_t val) {
        value_.store(val, std::memory_order_relaxed);
    }

    void reset() { set(0); }

private:
    std::atomic<int64_t> value_;
};

}  // namespace rnet::core

// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "core/sync.h"

namespace rnet::core {

// ===========================================================================
//  Lock-order tracking (debug builds only)
// ===========================================================================

#ifdef RNET_DEBUG_LOCKORDER

// ---------------------------------------------------------------------------
// LockOrderTracker
//
// Records the order in which mutexes are acquired per-thread.  If
// thread A locks (M1 then M2) and thread B later locks (M2 then M1),
// check_order fires an assertion — signalling a potential deadlock.
//
// Only compiled when RNET_DEBUG_LOCKORDER is defined; zero overhead in
// release builds.
// ---------------------------------------------------------------------------
LockOrderTracker& LockOrderTracker::instance() {
    static LockOrderTracker tracker;
    return tracker;
}

void LockOrderTracker::track_lock(const void* mutex_addr,
                                   const char* name,
                                   const char* file, int line) {
    std::unique_lock<Mutex> guard(tracker_mutex_);
    auto tid = std::this_thread::get_id();
    auto& locks = held_locks_[tid];
    locks.push_back({mutex_addr, name ? name : "", file ? file : "", line});
}

void LockOrderTracker::track_unlock(const void* mutex_addr) {
    std::unique_lock<Mutex> guard(tracker_mutex_);
    auto tid = std::this_thread::get_id();
    auto it = held_locks_.find(tid);
    if (it == held_locks_.end()) return;

    auto& locks = it->second;
    for (auto li = locks.begin(); li != locks.end(); ++li) {
        if (li->addr == mutex_addr) {
            locks.erase(li);
            break;
        }
    }
    if (locks.empty()) {
        held_locks_.erase(it);
    }
}

void LockOrderTracker::check_order(const void* mutex_addr,
                                    const char* name,
                                    const char* file, int line) {
    std::unique_lock<Mutex> guard(tracker_mutex_);
    auto tid = std::this_thread::get_id();
    auto& locks = held_locks_[tid];

    // 1. For each lock already held, record the forward ordering and
    //    check for the reverse ordering having been observed previously.
    for (const auto& held : locks) {
        auto pair_fwd = std::make_pair(held.addr, mutex_addr);
        auto pair_rev = std::make_pair(mutex_addr, held.addr);

        if (observed_order_.count(pair_rev)) {
            // 2. Potential deadlock detected.
            assert(false &&
                "Lock order inversion detected! Potential deadlock.");
        }
        observed_order_.insert(pair_fwd);
    }

    track_lock(mutex_addr, name, file, line);
}

#endif // RNET_DEBUG_LOCKORDER

// ===========================================================================
//  Debug helpers (always available)
// ===========================================================================

// ---------------------------------------------------------------------------
// Lock/unlock counters for diagnostics.
//
// Incremented by debug-instrumented lock wrappers.  Useful for
// profiling lock contention in non-release builds.
// ---------------------------------------------------------------------------
static std::mutex g_global_mutex;
static std::atomic<int64_t> g_lock_count{0};
static std::atomic<int64_t> g_unlock_count{0};

namespace debug {

int64_t total_lock_count() {
    return g_lock_count.load(std::memory_order_relaxed);
}

int64_t total_unlock_count() {
    return g_unlock_count.load(std::memory_order_relaxed);
}

void increment_lock_count() {
    g_lock_count.fetch_add(1, std::memory_order_relaxed);
}

void increment_unlock_count() {
    g_unlock_count.fetch_add(1, std::memory_order_relaxed);
}

void reset_lock_stats() {
    g_lock_count.store(0, std::memory_order_relaxed);
    g_unlock_count.store(0, std::memory_order_relaxed);
}

} // namespace debug

} // namespace rnet::core

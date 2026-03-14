// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

// Own header.
#include "node/shutdown.h"

// Standard library.
#include <atomic>
#include <condition_variable>
#include <mutex>

namespace rnet::node {

namespace {

std::atomic<bool> g_shutdown_requested{false};
std::mutex        g_shutdown_mutex;
std::condition_variable g_shutdown_cv;

} // namespace

// ---------------------------------------------------------------------------
// request_shutdown
// ---------------------------------------------------------------------------
// Sets the shutdown flag and wakes any thread blocked in
// wait_for_shutdown().
// ---------------------------------------------------------------------------
void request_shutdown()
{
    g_shutdown_requested.store(true, std::memory_order_release);
    std::lock_guard<std::mutex> lock(g_shutdown_mutex);
    g_shutdown_cv.notify_all();
}

// ---------------------------------------------------------------------------
// shutdown_requested
// ---------------------------------------------------------------------------
bool shutdown_requested()
{
    return g_shutdown_requested.load(std::memory_order_acquire);
}

// ---------------------------------------------------------------------------
// wait_for_shutdown
// ---------------------------------------------------------------------------
// Blocks the calling thread until request_shutdown() is called.
// ---------------------------------------------------------------------------
void wait_for_shutdown()
{
    std::unique_lock<std::mutex> lock(g_shutdown_mutex);
    g_shutdown_cv.wait(lock, [] {
        return g_shutdown_requested.load(std::memory_order_acquire);
    });
}

// ---------------------------------------------------------------------------
// reset_shutdown
// ---------------------------------------------------------------------------
// Clears the shutdown flag (used in tests / regtest restart).
// ---------------------------------------------------------------------------
void reset_shutdown()
{
    g_shutdown_requested.store(false, std::memory_order_release);
}

} // namespace rnet::node

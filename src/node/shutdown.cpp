#include "node/shutdown.h"

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace rnet::node {

namespace {

std::atomic<bool> g_shutdown_requested{false};
std::mutex        g_shutdown_mutex;
std::condition_variable g_shutdown_cv;

} // anonymous namespace

void request_shutdown()
{
    g_shutdown_requested.store(true, std::memory_order_release);
    // Wake anyone blocked in wait_for_shutdown().
    std::lock_guard<std::mutex> lock(g_shutdown_mutex);
    g_shutdown_cv.notify_all();
}

bool shutdown_requested()
{
    return g_shutdown_requested.load(std::memory_order_acquire);
}

void wait_for_shutdown()
{
    std::unique_lock<std::mutex> lock(g_shutdown_mutex);
    g_shutdown_cv.wait(lock, [] {
        return g_shutdown_requested.load(std::memory_order_acquire);
    });
}

void reset_shutdown()
{
    g_shutdown_requested.store(false, std::memory_order_release);
}

} // namespace rnet::node

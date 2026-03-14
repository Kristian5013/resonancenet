// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "core/time.h"

#include <ctime>
#include <iomanip>
#include <sstream>

namespace rnet::core {

// ===========================================================================
//  Free-function time accessors
// ===========================================================================

// ---------------------------------------------------------------------------
// get_time / get_time_millis / get_time_micros
//
// Wall-clock accessors that honour MockableClock for deterministic tests.
// ---------------------------------------------------------------------------
int64_t get_time() {
    return MockableClock::instance().now();
}

int64_t get_time_millis() {
    return MockableClock::instance().now_millis();
}

int64_t get_time_micros() {
    return MockableClock::instance().now_micros();
}

// ---------------------------------------------------------------------------
// get_steady_millis / get_steady_micros
//
// Monotonic clock — not affected by wall-clock adjustments or mocking.
// Suitable for elapsed-time measurement and throttle logic.
// ---------------------------------------------------------------------------
int64_t get_steady_millis() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               now.time_since_epoch())
        .count();
}

int64_t get_steady_micros() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(
               now.time_since_epoch())
        .count();
}

// ---------------------------------------------------------------------------
// format_iso_time / format_time
//
// format_iso_time: UTC ISO-8601 (e.g. "2025-01-15T08:30:00Z").
// format_time:     local-time display (e.g. "2025-01-15 09:30:00").
// ---------------------------------------------------------------------------
std::string format_iso_time(int64_t time_sec) {
    std::time_t t = static_cast<std::time_t>(time_sec);
    std::tm tm_buf{};
#ifdef _WIN32
    gmtime_s(&tm_buf, &t);
#else
    gmtime_r(&t, &tm_buf);
#endif
    char buf[64];
    std::snprintf(buf, sizeof(buf),
                  "%04d-%02d-%02dT%02d:%02d:%02dZ",
                  tm_buf.tm_year + 1900, tm_buf.tm_mon + 1,
                  tm_buf.tm_mday, tm_buf.tm_hour,
                  tm_buf.tm_min, tm_buf.tm_sec);
    return std::string(buf);
}

std::string format_time(int64_t time_sec) {
    std::time_t t = static_cast<std::time_t>(time_sec);
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    char buf[64];
    std::snprintf(buf, sizeof(buf),
                  "%04d-%02d-%02d %02d:%02d:%02d",
                  tm_buf.tm_year + 1900, tm_buf.tm_mon + 1,
                  tm_buf.tm_mday, tm_buf.tm_hour,
                  tm_buf.tm_min, tm_buf.tm_sec);
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// parse_iso_time
//
// Parses an ISO-8601 string into a Unix timestamp.  Returns -1 on failure.
// ---------------------------------------------------------------------------
int64_t parse_iso_time(const std::string& str) {
    std::tm tm_buf{};
    std::istringstream ss(str);
    ss >> std::get_time(&tm_buf, "%Y-%m-%dT%H:%M:%S");
    if (ss.fail()) return -1;

#ifdef _WIN32
    return static_cast<int64_t>(_mkgmtime(&tm_buf));
#else
    return static_cast<int64_t>(timegm(&tm_buf));
#endif
}

// ===========================================================================
//  MockableClock
// ===========================================================================

// ---------------------------------------------------------------------------
// MockableClock — singleton wall-clock with optional fixed-time override.
//
// When mock_time_ > 0 every call returns the frozen value (scaled to the
// requested resolution).  This keeps integration tests deterministic
// without touching the system clock.
// ---------------------------------------------------------------------------
MockableClock& MockableClock::instance() {
    static MockableClock clock;
    return clock;
}

void MockableClock::set_mock_time(int64_t time_sec) {
    mock_time_.store(time_sec, std::memory_order_relaxed);
}

int64_t MockableClock::now() const {
    int64_t mock = mock_time_.load(std::memory_order_relaxed);
    if (mock > 0) return mock;
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

int64_t MockableClock::now_millis() const {
    int64_t mock = mock_time_.load(std::memory_order_relaxed);
    if (mock > 0) return mock * 1000;
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

int64_t MockableClock::now_micros() const {
    int64_t mock = mock_time_.load(std::memory_order_relaxed);
    if (mock > 0) return mock * 1000000;
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

bool MockableClock::is_mocked() const {
    return mock_time_.load(std::memory_order_relaxed) > 0;
}

// ===========================================================================
//  Timer
// ===========================================================================

// ---------------------------------------------------------------------------
// Timer — lightweight monotonic stopwatch.
//
// Constructed at "now"; call reset() to restart.  Provides elapsed time
// in milliseconds, microseconds, seconds, or a human-friendly string.
// ---------------------------------------------------------------------------
Timer::Timer() : start_(std::chrono::steady_clock::now()) {}

void Timer::reset() {
    start_ = std::chrono::steady_clock::now();
}

int64_t Timer::elapsed_ms() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               now - start_)
        .count();
}

int64_t Timer::elapsed_us() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(
               now - start_)
        .count();
}

double Timer::elapsed_sec() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start_).count();
}

std::string Timer::elapsed_str() const {
    auto ms = elapsed_ms();
    if (ms < 1000) {
        return std::to_string(ms) + "ms";
    }
    if (ms < 60000) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.2fs",
                      static_cast<double>(ms) / 1000.0);
        return std::string(buf);
    }
    auto secs = ms / 1000;
    auto mins = secs / 60;
    secs %= 60;
    return std::to_string(mins) + "m" + std::to_string(secs) + "s";
}

// ===========================================================================
//  Throttle
// ===========================================================================

// ---------------------------------------------------------------------------
// Throttle — rate-gates a repeating action to at most once per interval.
//
// check() returns true at most once every interval_ms_ milliseconds.
// Uses steady clock to avoid wall-clock drift.
// ---------------------------------------------------------------------------
Throttle::Throttle(int64_t interval_ms)
    : interval_ms_(interval_ms) {}

bool Throttle::check() {
    auto now = get_steady_millis();
    if (now - last_time_ >= interval_ms_) {
        last_time_ = now;
        return true;
    }
    return false;
}

void Throttle::reset() {
    last_time_ = 0;
}

// ===========================================================================
//  Deadline
// ===========================================================================

// ---------------------------------------------------------------------------
// Deadline — absolute point in monotonic time for timeout checks.
//
// Three factory methods:
//   from_now(ms) — expires after the given duration.
//   expired()    — already expired at construction.
//   never()      — never expires (infinite timeout).
// ---------------------------------------------------------------------------
Deadline Deadline::from_now(int64_t ms) {
    Deadline d;
    d.deadline_ms_ = get_steady_millis() + ms;
    d.never_ = false;
    return d;
}

Deadline Deadline::expired() {
    Deadline d;
    d.deadline_ms_ = 0;
    d.never_ = false;
    return d;
}

Deadline Deadline::never() {
    Deadline d;
    d.never_ = true;
    return d;
}

bool Deadline::is_expired() const {
    if (never_) return false;
    return get_steady_millis() >= deadline_ms_;
}

int64_t Deadline::remaining_ms() const {
    if (never_) return std::numeric_limits<int64_t>::max();
    int64_t diff = deadline_ms_ - get_steady_millis();
    return diff > 0 ? diff : 0;
}

// ===========================================================================
//  RateLimiter
// ===========================================================================

// ---------------------------------------------------------------------------
// RateLimiter — sliding-window event counter.
//
// Allows up to max_events within any contiguous window_ms period.
// Expired timestamps are pruned lazily on each allow() call.
// ---------------------------------------------------------------------------
RateLimiter::RateLimiter(int max_events, int64_t window_ms)
    : max_events_(max_events), window_ms_(window_ms) {
    timestamps_.reserve(static_cast<size_t>(max_events));
}

bool RateLimiter::allow() {
    auto now = get_steady_millis();
    auto cutoff = now - window_ms_;

    // 1. Remove expired timestamps.
    while (!timestamps_.empty() && timestamps_.front() < cutoff) {
        timestamps_.erase(timestamps_.begin());
    }

    // 2. Check capacity.
    if (static_cast<int>(timestamps_.size()) >= max_events_) {
        return false;
    }

    // 3. Record this event.
    timestamps_.push_back(now);
    return true;
}

void RateLimiter::reset() {
    timestamps_.clear();
}

// ===========================================================================
//  PeriodicTimer
// ===========================================================================

// ---------------------------------------------------------------------------
// PeriodicTimer — fires a callback at most once per interval.
//
// Call check() from a polling loop; the callback executes inline when
// the interval has elapsed since the last firing.
// ---------------------------------------------------------------------------
PeriodicTimer::PeriodicTimer(int64_t interval_ms,
                              std::function<void()> callback)
    : interval_ms_(interval_ms), callback_(std::move(callback)) {}

void PeriodicTimer::check() {
    auto now = get_steady_millis();
    if (now - last_fire_ >= interval_ms_) {
        last_fire_ = now;
        if (callback_) callback_();
    }
}

void PeriodicTimer::reset() {
    last_fire_ = 0;
}

void PeriodicTimer::set_interval(int64_t ms) {
    interval_ms_ = ms;
}

} // namespace rnet::core

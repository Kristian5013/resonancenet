#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace rnet::core {

/// Get current time in seconds since epoch
int64_t get_time();

/// Get current time in milliseconds since epoch
int64_t get_time_millis();

/// Get current time in microseconds since epoch
int64_t get_time_micros();

/// Get monotonic time (for measuring durations)
int64_t get_steady_millis();
int64_t get_steady_micros();

/// Format a Unix timestamp to ISO 8601 string
std::string format_iso_time(int64_t time_sec);

/// Format a Unix timestamp to a human-readable string
std::string format_time(int64_t time_sec);

/// Parse ISO 8601 date-time string to Unix timestamp
/// Returns -1 on failure.
int64_t parse_iso_time(const std::string& str);

/// Mockable clock for testing
class MockableClock {
public:
    static MockableClock& instance();

    /// Set mock time. 0 = use real time.
    void set_mock_time(int64_t time_sec);

    /// Get current time (mock or real)
    int64_t now() const;
    int64_t now_millis() const;
    int64_t now_micros() const;

    /// Check if mocking is active
    bool is_mocked() const;

private:
    MockableClock() = default;
    std::atomic<int64_t> mock_time_{0};
};

/// Timer utility for benchmarking
class Timer {
public:
    Timer();

    /// Reset the timer
    void reset();

    /// Get elapsed time in various units
    int64_t elapsed_ms() const;
    int64_t elapsed_us() const;
    double elapsed_sec() const;

    /// Get elapsed time as formatted string
    std::string elapsed_str() const;

private:
    std::chrono::steady_clock::time_point start_;
};

/// Throttle: limits how often an action can occur
class Throttle {
public:
    explicit Throttle(int64_t interval_ms);

    /// Returns true if enough time has passed since last call
    bool check();

    /// Reset the throttle timer
    void reset();

private:
    int64_t interval_ms_;
    int64_t last_time_ = 0;
};

/// Deadline: a point in time that can be checked against.
class Deadline {
public:
    /// Create a deadline N milliseconds from now
    static Deadline from_now(int64_t ms);

    /// Create an already-expired deadline
    static Deadline expired();

    /// Create a deadline that never expires
    static Deadline never();

    /// Check if the deadline has passed
    bool is_expired() const;

    /// Milliseconds remaining (0 if expired)
    int64_t remaining_ms() const;

private:
    Deadline() = default;
    int64_t deadline_ms_ = 0;
    bool never_ = false;
};

/// Rate limiter: allows N events per time window
class RateLimiter {
public:
    RateLimiter(int max_events, int64_t window_ms);

    /// Returns true if the event is allowed (under the rate limit)
    bool allow();

    /// Reset the rate limiter
    void reset();

private:
    int max_events_;
    int64_t window_ms_;
    std::vector<int64_t> timestamps_;
};

/// PeriodicTimer: calls a callback at regular intervals
class PeriodicTimer {
public:
    PeriodicTimer(int64_t interval_ms,
                  std::function<void()> callback);

    /// Check if it's time to fire; if so, call the callback
    void check();

    /// Reset the timer
    void reset();

    /// Change the interval
    void set_interval(int64_t ms);

private:
    int64_t interval_ms_;
    int64_t last_fire_ = 0;
    std::function<void()> callback_;
};

}  // namespace rnet::core

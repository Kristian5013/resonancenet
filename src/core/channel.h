#pragma once

#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>
#include <chrono>

namespace rnet::core {

/// Thread-safe MPSC (multi-producer, single-consumer) channel.
/// Uses mutex + condition_variable internally.
template<typename T>
class Channel {
public:
    explicit Channel(size_t max_size = 0)
        : max_size_(max_size) {}

    ~Channel() { close(); }

    Channel(const Channel&) = delete;
    Channel& operator=(const Channel&) = delete;

    /// Push an item into the channel.
    /// Returns false if channel is closed.
    /// If max_size > 0, blocks until space is available.
    bool push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (closed_) return false;

        if (max_size_ > 0) {
            not_full_.wait(lock, [this]() {
                return closed_ || queue_.size() < max_size_;
            });
            if (closed_) return false;
        }

        queue_.push(std::move(item));
        not_empty_.notify_one();
        return true;
    }

    /// Try to push without blocking.
    /// Returns false if channel is closed or full (when bounded).
    bool try_push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (closed_) return false;
        if (max_size_ > 0 && queue_.size() >= max_size_) return false;

        queue_.push(std::move(item));
        not_empty_.notify_one();
        return true;
    }

    /// Pop an item from the channel (blocking).
    /// Returns nullopt if channel is closed and empty.
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this]() {
            return closed_ || !queue_.empty();
        });

        if (queue_.empty()) return std::nullopt;

        T item = std::move(queue_.front());
        queue_.pop();

        if (max_size_ > 0) {
            not_full_.notify_one();
        }

        return item;
    }

    /// Try to pop without blocking.
    std::optional<T> try_pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty()) return std::nullopt;

        T item = std::move(queue_.front());
        queue_.pop();

        if (max_size_ > 0) {
            not_full_.notify_one();
        }

        return item;
    }

    /// Pop with timeout.
    /// Returns nullopt if timeout expires or channel is closed and empty.
    template<typename Rep, typename Period>
    std::optional<T> pop_for(
        const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        bool got = not_empty_.wait_for(lock, timeout, [this]() {
            return closed_ || !queue_.empty();
        });

        if (!got || queue_.empty()) return std::nullopt;

        T item = std::move(queue_.front());
        queue_.pop();

        if (max_size_ > 0) {
            not_full_.notify_one();
        }

        return item;
    }

    /// Close the channel. Wakes all waiting threads.
    /// After closing, push() returns false. pop() drains remaining items.
    void close() {
        std::unique_lock<std::mutex> lock(mutex_);
        closed_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    /// Check if channel is closed
    bool is_closed() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return closed_;
    }

    /// Check if channel is empty
    bool empty() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    /// Current number of items in the channel
    size_t size() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.size();
    }

    /// Drain all items from the channel
    std::vector<T> drain() {
        std::unique_lock<std::mutex> lock(mutex_);
        std::vector<T> items;
        items.reserve(queue_.size());
        while (!queue_.empty()) {
            items.push_back(std::move(queue_.front()));
            queue_.pop();
        }
        if (max_size_ > 0) {
            not_full_.notify_all();
        }
        return items;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::queue<T> queue_;
    size_t max_size_;
    bool closed_ = false;
};

}  // namespace rnet::core

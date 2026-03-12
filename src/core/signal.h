#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <mutex>
#include <utility>
#include <vector>

namespace rnet::core {

/// Connection handle — returned by Signal::connect, used to disconnect.
class Connection {
public:
    Connection() = default;
    explicit Connection(uint64_t id) : id_(id) {}

    uint64_t id() const { return id_; }
    bool is_valid() const { return id_ != 0; }

    bool operator==(const Connection& other) const {
        return id_ == other.id_;
    }

private:
    uint64_t id_ = 0;
};

/// ScopedConnection — automatically disconnects on destruction.
/// Must be used with a disconnect callback.
class ScopedConnection {
public:
    ScopedConnection() = default;

    ScopedConnection(Connection conn,
                     std::function<void(Connection)> disconnector)
        : conn_(conn), disconnector_(std::move(disconnector)) {}

    ~ScopedConnection() {
        disconnect();
    }

    ScopedConnection(const ScopedConnection&) = delete;
    ScopedConnection& operator=(const ScopedConnection&) = delete;

    ScopedConnection(ScopedConnection&& other) noexcept
        : conn_(other.conn_),
          disconnector_(std::move(other.disconnector_)) {
        other.conn_ = Connection();
    }

    ScopedConnection& operator=(ScopedConnection&& other) noexcept {
        if (this != &other) {
            disconnect();
            conn_ = other.conn_;
            disconnector_ = std::move(other.disconnector_);
            other.conn_ = Connection();
        }
        return *this;
    }

    void disconnect() {
        if (conn_.is_valid() && disconnector_) {
            disconnector_(conn_);
            conn_ = Connection();
        }
    }

    Connection connection() const { return conn_; }

private:
    Connection conn_;
    std::function<void(Connection)> disconnector_;
};

/// Signal<Args...> — simple signal/slot mechanism.
/// Thread-safe. Multiple slots can be connected.
template<typename... Args>
class Signal {
public:
    using Slot = std::function<void(Args...)>;

    Signal() = default;
    ~Signal() = default;

    Signal(const Signal&) = delete;
    Signal& operator=(const Signal&) = delete;
    Signal(Signal&&) = default;
    Signal& operator=(Signal&&) = default;

    /// Connect a slot. Returns a Connection handle for disconnecting.
    Connection connect(Slot slot) {
        std::lock_guard<std::mutex> lock(mutex_);
        uint64_t id = ++next_id_;
        slots_.push_back({id, std::move(slot)});
        return Connection(id);
    }

    /// Connect a slot and return a ScopedConnection that auto-disconnects.
    ScopedConnection connect_scoped(Slot slot) {
        auto conn = connect(std::move(slot));
        return ScopedConnection(conn, [this](Connection c) {
            disconnect(c);
        });
    }

    /// Disconnect a slot by its Connection handle.
    void disconnect(Connection conn) {
        std::lock_guard<std::mutex> lock(mutex_);
        slots_.erase(
            std::remove_if(slots_.begin(), slots_.end(),
                           [&](const SlotEntry& entry) {
                               return entry.id == conn.id();
                           }),
            slots_.end());
    }

    /// Disconnect all slots.
    void disconnect_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        slots_.clear();
    }

    /// Emit the signal, calling all connected slots.
    void emit(Args... args) const {
        // Copy slots under lock, then call without lock to avoid
        // deadlock if a slot modifies the signal.
        std::vector<SlotEntry> slots_copy;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            slots_copy = slots_;
        }
        for (const auto& entry : slots_copy) {
            if (entry.slot) {
                entry.slot(args...);
            }
        }
    }

    /// Shorthand for emit
    void operator()(Args... args) const {
        emit(std::forward<Args>(args)...);
    }

    /// Number of connected slots
    size_t num_slots() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return slots_.size();
    }

    /// Check if any slots are connected
    bool has_connections() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return !slots_.empty();
    }

private:
    struct SlotEntry {
        uint64_t id;
        Slot slot;
    };

    mutable std::mutex mutex_;
    std::vector<SlotEntry> slots_;
    uint64_t next_id_ = 0;
};

/// Signal with a return value that collects results from all slots.
template<typename R, typename... Args>
class CollectingSignal {
public:
    using Slot = std::function<R(Args...)>;

    Connection connect(Slot slot) {
        std::lock_guard<std::mutex> lock(mutex_);
        uint64_t id = ++next_id_;
        slots_.push_back({id, std::move(slot)});
        return Connection(id);
    }

    void disconnect(Connection conn) {
        std::lock_guard<std::mutex> lock(mutex_);
        slots_.erase(
            std::remove_if(slots_.begin(), slots_.end(),
                           [&](const SlotEntry& entry) {
                               return entry.id == conn.id();
                           }),
            slots_.end());
    }

    void disconnect_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        slots_.clear();
    }

    std::vector<R> emit(Args... args) const {
        std::vector<SlotEntry> slots_copy;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            slots_copy = slots_;
        }
        std::vector<R> results;
        results.reserve(slots_copy.size());
        for (const auto& entry : slots_copy) {
            if (entry.slot) {
                results.push_back(entry.slot(args...));
            }
        }
        return results;
    }

private:
    struct SlotEntry {
        uint64_t id;
        Slot slot;
    };

    mutable std::mutex mutex_;
    std::vector<SlotEntry> slots_;
    uint64_t next_id_ = 0;
};

}  // namespace rnet::core

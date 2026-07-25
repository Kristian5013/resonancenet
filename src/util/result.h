// Explicit, fail-closed error handling.
//
// Consensus code does not throw and does not return bare bools that a caller can
// forget to check. Every fallible operation returns Result<T>, which cannot be
// dereferenced without having been checked, and carries a human-readable reason.
//
// This exists because of a concrete failure mode found in the Lattica audit: a
// checkpoint loader that ignored fread() return values loaded a truncated file as
// zeroes and reported success. Nothing here can do that.
#pragma once

#include <cassert>
#include <optional>
#include <string>
#include <utility>

namespace rnet {

class Error {
public:
    Error() = default;
    explicit Error(std::string what) : what_(std::move(what)) {}

    const std::string& message() const { return what_; }

private:
    std::string what_;
};

template <typename T>
class [[nodiscard]] Result {
public:
    Result(T value) : value_(std::move(value)) {}                 // NOLINT(google-explicit-constructor)
    Result(Error err) : error_(std::move(err)) {}                 // NOLINT(google-explicit-constructor)

    bool ok() const { return value_.has_value(); }
    explicit operator bool() const { return ok(); }

    const T& value() const {
        assert(ok() && "Result::value() on an error result");
        return *value_;
    }
    T& value() {
        assert(ok() && "Result::value() on an error result");
        return *value_;
    }
    T value_or(T fallback) const { return ok() ? *value_ : std::move(fallback); }

    const std::string& error() const { return error_.message(); }

private:
    std::optional<T> value_;
    Error error_;
};

// Result<void> equivalent: an operation that either succeeded or explains why not.
class [[nodiscard]] Status {
public:
    Status() = default;
    Status(Error err) : error_(std::move(err)), failed_(true) {}  // NOLINT(google-explicit-constructor)

    static Status Ok() { return Status(); }

    bool ok() const { return !failed_; }
    explicit operator bool() const { return ok(); }
    const std::string& error() const { return error_.message(); }

private:
    Error error_;
    bool failed_{false};
};

inline Error Err(std::string what) { return Error(std::move(what)); }

}  // namespace rnet

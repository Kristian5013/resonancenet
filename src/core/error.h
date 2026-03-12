#pragma once

#include <string>
#include <string_view>
#include <optional>
#include <functional>
#include <type_traits>
#include <utility>
#include <stdexcept>

namespace rnet::core {

/// Result<T> — monadic error handling. Never throw exceptions.
template<typename T>
class Result {
public:
    static Result ok(T value) {
        Result r;
        r.value_ = std::move(value);
        return r;
    }

    static Result err(std::string message) {
        Result r;
        r.error_ = std::move(message);
        return r;
    }

    bool is_ok() const noexcept { return value_.has_value(); }
    bool is_err() const noexcept { return error_.has_value(); }

    const T& value() const& {
        if (!value_) throw std::runtime_error("Result::value() on error: " + error_.value_or("unknown"));
        return *value_;
    }
    T& value() & {
        if (!value_) throw std::runtime_error("Result::value() on error: " + error_.value_or("unknown"));
        return *value_;
    }
    T&& value() && {
        if (!value_) throw std::runtime_error("Result::value() on error: " + error_.value_or("unknown"));
        return std::move(*value_);
    }

    const std::string& error() const& { return *error_; }
    std::string& error() & { return *error_; }

    const T& value_or(const T& fallback) const& {
        return is_ok() ? *value_ : fallback;
    }

    template<typename Fn>
    auto map(Fn&& fn) const -> Result<decltype(fn(std::declval<const T&>()))> {
        using U = decltype(fn(std::declval<const T&>()));
        if (is_ok()) {
            return Result<U>::ok(fn(*value_));
        }
        return Result<U>::err(*error_);
    }

    template<typename Fn>
    auto and_then(Fn&& fn) const -> decltype(fn(std::declval<const T&>())) {
        using RetType = decltype(fn(std::declval<const T&>()));
        if (is_ok()) {
            return fn(*value_);
        }
        return RetType::err(*error_);
    }

    template<typename Fn>
    Result or_else(Fn&& fn) const {
        if (is_ok()) {
            return *this;
        }
        return fn(*error_);
    }

    explicit operator bool() const noexcept { return is_ok(); }

private:
    Result() = default;
    std::optional<T> value_;
    std::optional<std::string> error_;
};

/// Specialization for Result<void>
template<>
class Result<void> {
public:
    static Result ok() {
        Result r;
        r.ok_ = true;
        return r;
    }

    static Result err(std::string message) {
        Result r;
        r.ok_ = false;
        r.error_ = std::move(message);
        return r;
    }

    bool is_ok() const noexcept { return ok_; }
    bool is_err() const noexcept { return !ok_; }

    const std::string& error() const& { return *error_; }
    std::string& error() & { return *error_; }

    template<typename Fn>
    Result and_then(Fn&& fn) const {
        if (is_ok()) {
            return fn();
        }
        return *this;
    }

    template<typename Fn>
    Result or_else(Fn&& fn) const {
        if (is_ok()) {
            return *this;
        }
        return fn(*error_);
    }

    explicit operator bool() const noexcept { return ok_; }

private:
    Result() = default;
    bool ok_ = false;
    std::optional<std::string> error_;
};

/// Error codes for common failures
enum class ErrorCode : int {
    OK = 0,
    INVALID_ARGUMENT = 1,
    OUT_OF_RANGE = 2,
    NOT_FOUND = 3,
    ALREADY_EXISTS = 4,
    PERMISSION_DENIED = 5,
    IO_FAILURE = 6,
    PARSE_FAILURE = 7,
    NETWORK_FAILURE = 8,
    CRYPTO_FAILURE = 9,
    CONSENSUS_FAILURE = 10,
    WALLET_FAILURE = 11,
    DATABASE_FAILURE = 12,
    TIMEOUT = 13,
    CANCELLED = 14,
    INTERNAL = 15,
};

/// Convert ErrorCode to string
inline std::string_view error_code_name(ErrorCode code) {
    switch (code) {
        case ErrorCode::OK:                return "OK";
        case ErrorCode::INVALID_ARGUMENT:  return "INVALID_ARGUMENT";
        case ErrorCode::OUT_OF_RANGE:      return "OUT_OF_RANGE";
        case ErrorCode::NOT_FOUND:         return "NOT_FOUND";
        case ErrorCode::ALREADY_EXISTS:    return "ALREADY_EXISTS";
        case ErrorCode::PERMISSION_DENIED: return "PERMISSION_DENIED";
        case ErrorCode::IO_FAILURE:        return "IO_FAILURE";
        case ErrorCode::PARSE_FAILURE:     return "PARSE_FAILURE";
        case ErrorCode::NETWORK_FAILURE:   return "NETWORK_FAILURE";
        case ErrorCode::CRYPTO_FAILURE:    return "CRYPTO_FAILURE";
        case ErrorCode::CONSENSUS_FAILURE: return "CONSENSUS_FAILURE";
        case ErrorCode::WALLET_FAILURE:    return "WALLET_FAILURE";
        case ErrorCode::DATABASE_FAILURE:  return "DATABASE_FAILURE";
        case ErrorCode::TIMEOUT:           return "TIMEOUT";
        case ErrorCode::CANCELLED:         return "CANCELLED";
        case ErrorCode::INTERNAL:          return "INTERNAL";
        default:                           return "UNKNOWN";
    }
}

/// Rich error with code + message
struct Error {
    ErrorCode code = ErrorCode::INTERNAL;
    std::string message;

    Error() = default;
    Error(ErrorCode c, std::string msg)
        : code(c), message(std::move(msg)) {}

    std::string to_string() const {
        return std::string(error_code_name(code)) + ": " + message;
    }
};

/// TypedResult<T> — Result with structured Error (code + message)
template<typename T>
class TypedResult {
public:
    static TypedResult ok(T value) {
        TypedResult r;
        r.value_ = std::move(value);
        return r;
    }

    static TypedResult err(ErrorCode code, std::string message) {
        TypedResult r;
        r.error_ = Error{code, std::move(message)};
        return r;
    }

    static TypedResult err(Error e) {
        TypedResult r;
        r.error_ = std::move(e);
        return r;
    }

    bool is_ok() const noexcept { return value_.has_value(); }
    bool is_err() const noexcept { return error_.has_value(); }

    const T& value() const& { return *value_; }
    T& value() & { return *value_; }
    T&& value() && { return std::move(*value_); }

    const Error& error() const& { return *error_; }
    ErrorCode error_code() const { return error_->code; }
    const std::string& error_message() const { return error_->message; }

    explicit operator bool() const noexcept { return is_ok(); }

private:
    TypedResult() = default;
    std::optional<T> value_;
    std::optional<Error> error_;
};

/// Specialization for TypedResult<void>
template<>
class TypedResult<void> {
public:
    static TypedResult ok() {
        TypedResult r;
        r.ok_ = true;
        return r;
    }

    static TypedResult err(ErrorCode code, std::string message) {
        TypedResult r;
        r.ok_ = false;
        r.error_ = Error{code, std::move(message)};
        return r;
    }

    bool is_ok() const noexcept { return ok_; }
    bool is_err() const noexcept { return !ok_; }

    const Error& error() const& { return *error_; }
    ErrorCode error_code() const { return error_->code; }

    explicit operator bool() const noexcept { return ok_; }

private:
    TypedResult() = default;
    bool ok_ = false;
    std::optional<Error> error_;
};

// ─── Error construction helpers ─────────────────────────────────────

std::string format_error(ErrorCode code, const std::string& msg);
std::string format_error_chain(const std::vector<Error>& errors);

Error make_error(ErrorCode code, const std::string& msg);
Error make_invalid_arg(const std::string& msg);
Error make_not_found(const std::string& msg);
Error make_io_error(const std::string& msg);
Error make_parse_error(const std::string& msg);
Error make_timeout(const std::string& msg);
Error make_internal(const std::string& msg);

}  // namespace rnet::core

namespace rnet {
    using core::Result;
    using core::TypedResult;
    using core::ErrorCode;
    using core::Error;
}  // namespace rnet

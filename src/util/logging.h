// Leveled logging. Deliberately tiny and dependency-free: a node that cannot log
// must still be able to start, so logging never allocates on the failure path and
// never throws.
#pragma once

#include <format>
#include <string>
#include <string_view>

namespace rnet::util {

enum class LogLevel : int {
    Error = 0,
    Warn = 1,
    Info = 2,
    Debug = 3,
    Trace = 4,
};

void SetLogLevel(LogLevel level);
LogLevel GetLogLevel();
bool ParseLogLevel(std::string_view name, LogLevel& out);

// Writes "<ISO-8601 UTC> <LEVEL> <message>" to stderr.
void LogRaw(LogLevel level, std::string_view message);

template <typename... Args>
void Log(LogLevel level, std::format_string<Args...> fmt, Args&&... args) {
    if (level > GetLogLevel()) return;
    LogRaw(level, std::format(fmt, std::forward<Args>(args)...));
}

#define RNET_LOG_ERROR(...) ::rnet::util::Log(::rnet::util::LogLevel::Error, __VA_ARGS__)
#define RNET_LOG_WARN(...) ::rnet::util::Log(::rnet::util::LogLevel::Warn, __VA_ARGS__)
#define RNET_LOG_INFO(...) ::rnet::util::Log(::rnet::util::LogLevel::Info, __VA_ARGS__)
#define RNET_LOG_DEBUG(...) ::rnet::util::Log(::rnet::util::LogLevel::Debug, __VA_ARGS__)
#define RNET_LOG_TRACE(...) ::rnet::util::Log(::rnet::util::LogLevel::Trace, __VA_ARGS__)

}  // namespace rnet::util

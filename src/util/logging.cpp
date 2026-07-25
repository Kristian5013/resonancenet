#include "util/logging.h"

#include <atomic>
#include <chrono>
#include <cstdio>

namespace rnet::util {

namespace {
std::atomic<LogLevel> g_level{LogLevel::Info};

const char* LevelName(LogLevel level) {
    switch (level) {
        case LogLevel::Error: return "ERROR";
        case LogLevel::Warn:  return "WARN ";
        case LogLevel::Info:  return "INFO ";
        case LogLevel::Debug: return "DEBUG";
        case LogLevel::Trace: return "TRACE";
    }
    return "?????";
}
}  // namespace

void SetLogLevel(LogLevel level) { g_level.store(level, std::memory_order_relaxed); }
LogLevel GetLogLevel() { return g_level.load(std::memory_order_relaxed); }

bool ParseLogLevel(std::string_view name, LogLevel& out) {
    if (name == "error") { out = LogLevel::Error; return true; }
    if (name == "warn")  { out = LogLevel::Warn;  return true; }
    if (name == "info")  { out = LogLevel::Info;  return true; }
    if (name == "debug") { out = LogLevel::Debug; return true; }
    if (name == "trace") { out = LogLevel::Trace; return true; }
    return false;
}

void LogRaw(LogLevel level, std::string_view message) {
    const auto now = std::chrono::system_clock::now();
    const std::string ts = std::format("{:%FT%TZ}", std::chrono::floor<std::chrono::seconds>(now));
    std::fprintf(stderr, "%s %s %.*s\n", ts.c_str(), LevelName(level),
                 static_cast<int>(message.size()), message.data());
}

}  // namespace rnet::util

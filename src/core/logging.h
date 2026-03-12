#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <string_view>
#include <vector>
#include <mutex>

namespace rnet::core {

/// Log categories — bitfield for efficient filtering
enum class LogCategory : uint32_t {
    NONE       = 0,
    NET        = (1 << 0),
    MEMPOOL    = (1 << 1),
    VALIDATION = (1 << 2),
    MINING     = (1 << 3),
    TRAINING   = (1 << 4),
    RPC        = (1 << 5),
    WALLET     = (1 << 6),
    LIGHTNING  = (1 << 7),
    DB         = (1 << 8),
    LOCK       = (1 << 9),
    RAND       = (1 << 10),
    PRUNE      = (1 << 11),
    HTTP       = (1 << 12),
    BENCH      = (1 << 13),
    QT_GUI     = (1 << 14),
    ALL        = 0xFFFFFFFF
};

inline LogCategory operator|(LogCategory a, LogCategory b) {
    return static_cast<LogCategory>(
        static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

inline LogCategory operator&(LogCategory a, LogCategory b) {
    return static_cast<LogCategory>(
        static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}

inline LogCategory& operator|=(LogCategory& a, LogCategory b) {
    a = a | b;
    return a;
}

/// Log severity levels
enum class LogLevel : int {
    TRACE = 0,
    DBG   = 1,
    INFO  = 2,
    WARN  = 3,
    LOG_ERROR = 4,  // Avoid Windows ERROR macro
    FATAL = 5
};

/// Convert category/level to string
std::string_view log_category_name(LogCategory cat);
std::string_view log_level_name(LogLevel level);

/// Parse category name to enum
LogCategory log_category_from_name(std::string_view name);

/// Logger singleton — thread-safe
class Logger {
public:
    static Logger& instance();

    void enable_category(LogCategory cat);
    void disable_category(LogCategory cat);
    bool is_enabled(LogCategory cat) const;

    void set_level(LogLevel level);
    LogLevel get_level() const;

    void set_print_to_console(bool enable);
    void set_print_to_file(bool enable);

    bool open_log_file(const std::string& path);
    void close_log_file();

    /// Main log function
    void log_write(LogCategory cat, LogLevel level,
                   const char* file, int line,
                   const std::string& message);

    /// Formatted log
    template<typename... Args>
    void log_printf(LogCategory cat, LogLevel level,
                    const char* file, int line,
                    const char* fmt, Args&&... args);

    /// Callback for GUI or other consumers
    using LogCallback = std::function<void(
        LogCategory, LogLevel, const std::string&)>;
    void set_callback(LogCallback cb);

    /// Shrink log file if it exceeds max_bytes
    void shrink_log_file(size_t max_bytes);

    /// Flush log file
    void flush();

    /// Get all buffered messages (for startup before file is opened)
    std::vector<std::string> get_buffered_messages() const;

private:
    Logger();
    ~Logger();

    mutable std::mutex mutex_;
    std::atomic<uint32_t> enabled_categories_{
        static_cast<uint32_t>(LogCategory::ALL)};
    std::atomic<int> min_level_{static_cast<int>(LogLevel::INFO)};
    bool print_to_console_ = true;
    bool print_to_file_ = false;
    FILE* log_file_ = nullptr;
    std::string log_file_path_;
    LogCallback callback_;
    std::vector<std::string> buffer_;  // Pre-file-open buffer
    bool started_logging_ = false;
};

/// Format a log message with snprintf
std::string format_log_message(const char* fmt, ...);

// Implementation of template log_printf
template<typename... Args>
void Logger::log_printf(LogCategory cat, LogLevel level,
                        const char* file, int line,
                        const char* fmt, Args&&... args) {
    if (!is_enabled(cat)) return;
    if (static_cast<int>(level) < min_level_.load(
            std::memory_order_relaxed)) return;
    char buf[4096];
    std::snprintf(buf, sizeof(buf), fmt,
                  std::forward<Args>(args)...);
    log_write(cat, level, file, line, std::string(buf));
}

/// Log categories as strings for iteration
struct LogCategoryEntry {
    LogCategory cat;
    const char* name;
};

inline constexpr LogCategoryEntry ALL_LOG_CATEGORIES[] = {
    {LogCategory::NET,        "net"},
    {LogCategory::MEMPOOL,    "mempool"},
    {LogCategory::VALIDATION, "validation"},
    {LogCategory::MINING,     "mining"},
    {LogCategory::TRAINING,   "training"},
    {LogCategory::RPC,        "rpc"},
    {LogCategory::WALLET,     "wallet"},
    {LogCategory::LIGHTNING,  "lightning"},
    {LogCategory::DB,         "db"},
    {LogCategory::LOCK,       "lock"},
    {LogCategory::RAND,       "rand"},
    {LogCategory::PRUNE,      "prune"},
    {LogCategory::HTTP,       "http"},
    {LogCategory::BENCH,      "bench"},
    {LogCategory::QT_GUI,     "qt"},
};

/// Get a list of all available log category names
std::vector<std::string> get_log_category_names();

/// Parse a comma-separated list of categories
LogCategory parse_log_categories(std::string_view str);

}  // namespace rnet::core

/// Convenience macros
#define LogPrint(category, ...) \
    do { \
        auto& logger_ = ::rnet::core::Logger::instance(); \
        if (logger_.is_enabled(::rnet::core::LogCategory::category)) { \
            logger_.log_printf( \
                ::rnet::core::LogCategory::category, \
                ::rnet::core::LogLevel::INFO, \
                __FILE__, __LINE__, __VA_ARGS__); \
        } \
    } while(0)

#define LogPrintf(...) \
    do { \
        ::rnet::core::Logger::instance().log_printf( \
            ::rnet::core::LogCategory::ALL, \
            ::rnet::core::LogLevel::INFO, \
            __FILE__, __LINE__, __VA_ARGS__); \
    } while(0)

#define LogError(...) \
    do { \
        ::rnet::core::Logger::instance().log_printf( \
            ::rnet::core::LogCategory::ALL, \
            ::rnet::core::LogLevel::LOG_ERROR, \
            __FILE__, __LINE__, __VA_ARGS__); \
    } while(0)

#define LogDebug(category, ...) \
    do { \
        auto& logger_ = ::rnet::core::Logger::instance(); \
        if (logger_.is_enabled(::rnet::core::LogCategory::category)) { \
            logger_.log_printf( \
                ::rnet::core::LogCategory::category, \
                ::rnet::core::LogLevel::DBG, \
                __FILE__, __LINE__, __VA_ARGS__); \
        } \
    } while(0)

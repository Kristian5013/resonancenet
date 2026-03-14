// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "core/logging.h"

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <thread>

namespace rnet::core {

// ===========================================================================
//  Category / level name mappings
// ===========================================================================

// ---------------------------------------------------------------------------
// log_category_name
//   Returns the short display name for a log category enum value.
// ---------------------------------------------------------------------------
std::string_view log_category_name(LogCategory cat) {
    switch (cat) {
        case LogCategory::NONE:       return "none";
        case LogCategory::NET:        return "net";
        case LogCategory::MEMPOOL:    return "mempool";
        case LogCategory::VALIDATION: return "validation";
        case LogCategory::MINING:     return "mining";
        case LogCategory::TRAINING:   return "training";
        case LogCategory::RPC:        return "rpc";
        case LogCategory::WALLET:     return "wallet";
        case LogCategory::LIGHTNING:  return "lightning";
        case LogCategory::DB:         return "db";
        case LogCategory::LOCK:       return "lock";
        case LogCategory::RAND:       return "rand";
        case LogCategory::PRUNE:      return "prune";
        case LogCategory::HTTP:       return "http";
        case LogCategory::BENCH:      return "bench";
        case LogCategory::QT_GUI:     return "qt";
        case LogCategory::ALL:        return "all";
        default:                      return "unknown";
    }
}

// ---------------------------------------------------------------------------
// log_level_name
//   Returns the uppercase label for a log level.
// ---------------------------------------------------------------------------
std::string_view log_level_name(LogLevel level) {
    switch (level) {
        case LogLevel::TRACE:     return "TRACE";
        case LogLevel::DBG:       return "DEBUG";
        case LogLevel::INFO:      return "INFO";
        case LogLevel::WARN:      return "WARN";
        case LogLevel::LOG_ERROR: return "ERROR";
        case LogLevel::FATAL:     return "FATAL";
        default:                  return "?????";
    }
}

// ---------------------------------------------------------------------------
// log_category_from_name
//   Reverse lookup: string name -> LogCategory enum.  Returns NONE for
//   unrecognised names.
// ---------------------------------------------------------------------------
LogCategory log_category_from_name(std::string_view name) {
    if (name == "net")        return LogCategory::NET;
    if (name == "mempool")    return LogCategory::MEMPOOL;
    if (name == "validation") return LogCategory::VALIDATION;
    if (name == "mining")     return LogCategory::MINING;
    if (name == "training")   return LogCategory::TRAINING;
    if (name == "rpc")        return LogCategory::RPC;
    if (name == "wallet")     return LogCategory::WALLET;
    if (name == "lightning")  return LogCategory::LIGHTNING;
    if (name == "db")         return LogCategory::DB;
    if (name == "lock")       return LogCategory::LOCK;
    if (name == "rand")       return LogCategory::RAND;
    if (name == "prune")      return LogCategory::PRUNE;
    if (name == "http")       return LogCategory::HTTP;
    if (name == "bench")      return LogCategory::BENCH;
    if (name == "qt")         return LogCategory::QT_GUI;
    if (name == "all")        return LogCategory::ALL;
    if (name == "none")       return LogCategory::NONE;
    return LogCategory::NONE;
}

// ===========================================================================
//  Logger lifetime
// ===========================================================================

// ---------------------------------------------------------------------------
// Constructor / destructor
//   Default-constructed.  Destructor ensures the log file is flushed and
//   closed.
// ---------------------------------------------------------------------------
Logger::Logger() = default;

Logger::~Logger() {
    close_log_file();
}

// ---------------------------------------------------------------------------
// instance
//   Meyer's singleton -- thread-safe static initialisation.
// ---------------------------------------------------------------------------
Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

// ===========================================================================
//  Category and level control
// ===========================================================================

// ---------------------------------------------------------------------------
// enable_category / disable_category
//   Atomically set or clear category bits in the bitmask.
// ---------------------------------------------------------------------------
void Logger::enable_category(LogCategory cat) {
    enabled_categories_.fetch_or(
        static_cast<uint32_t>(cat), std::memory_order_relaxed);
}

void Logger::disable_category(LogCategory cat) {
    enabled_categories_.fetch_and(
        ~static_cast<uint32_t>(cat), std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// is_enabled
//   Returns true if the given category bit is set.
// ---------------------------------------------------------------------------
bool Logger::is_enabled(LogCategory cat) const {
    uint32_t mask = enabled_categories_.load(std::memory_order_relaxed);
    return (mask & static_cast<uint32_t>(cat)) != 0;
}

// ---------------------------------------------------------------------------
// set_level / get_level
//   Controls the minimum severity threshold.
// ---------------------------------------------------------------------------
void Logger::set_level(LogLevel level) {
    min_level_.store(static_cast<int>(level), std::memory_order_relaxed);
}

LogLevel Logger::get_level() const {
    return static_cast<LogLevel>(
        min_level_.load(std::memory_order_relaxed));
}

// ===========================================================================
//  Output destination control
// ===========================================================================

// ---------------------------------------------------------------------------
// set_print_to_console / set_print_to_file
//   Toggle whether log lines are written to stderr and/or the log file.
// ---------------------------------------------------------------------------
void Logger::set_print_to_console(bool enable) {
    std::lock_guard<std::mutex> lock(mutex_);
    print_to_console_ = enable;
}

void Logger::set_print_to_file(bool enable) {
    std::lock_guard<std::mutex> lock(mutex_);
    print_to_file_ = enable;
}

// ===========================================================================
//  Log file management
// ===========================================================================

// ---------------------------------------------------------------------------
// open_log_file
//   Opens (or re-opens) the log file in append mode.  Any messages
//   buffered before the file was available are flushed immediately.
// ---------------------------------------------------------------------------
bool Logger::open_log_file(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    // 1. Close existing file if any
    if (log_file_) {
        std::fclose(log_file_);
        log_file_ = nullptr;
    }

    // 2. Open new file
    log_file_ = std::fopen(path.c_str(), "a");
    if (!log_file_) {
        return false;
    }
    log_file_path_ = path;
    print_to_file_ = true;
    started_logging_ = true;

    // 3. Flush buffered messages
    for (const auto& msg : buffer_) {
        std::fputs(msg.c_str(), log_file_);
    }
    buffer_.clear();

    return true;
}

// ---------------------------------------------------------------------------
// close_log_file
//   Flushes and closes the current log file handle.
// ---------------------------------------------------------------------------
void Logger::close_log_file() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (log_file_) {
        std::fclose(log_file_);
        log_file_ = nullptr;
    }
}

// ===========================================================================
//  Timestamp formatting
// ===========================================================================

// ---------------------------------------------------------------------------
// make_timestamp
//   Produces a "YYYY-MM-DD HH:MM:SS.mmm" timestamp string using the
//   system clock with millisecond precision.
// ---------------------------------------------------------------------------
static std::string make_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &time_t_now);
#else
    localtime_r(&time_t_now, &tm_buf);
#endif

    char time_str[64];
    std::snprintf(time_str, sizeof(time_str),
                  "%04d-%02d-%02d %02d:%02d:%02d.%03d",
                  tm_buf.tm_year + 1900, tm_buf.tm_mon + 1,
                  tm_buf.tm_mday, tm_buf.tm_hour, tm_buf.tm_min,
                  tm_buf.tm_sec, static_cast<int>(ms.count()));
    return std::string(time_str);
}

// ===========================================================================
//  Core logging
// ===========================================================================

// ---------------------------------------------------------------------------
// log_write
//   Central log-dispatch method.  Checks category and level filters, formats
//   the line with timestamp/category/level prefix, then routes to console,
//   file, buffer, and/or callback as configured.
// ---------------------------------------------------------------------------
void Logger::log_write(LogCategory cat, LogLevel level,
                        const char* /*file*/, int /*line*/,
                        const std::string& message) {
    // 1. Category filter
    if (!is_enabled(cat)) return;

    // 2. Level filter
    if (static_cast<int>(level) < min_level_.load(
            std::memory_order_relaxed)) return;

    // 3. Format the log line
    std::string timestamp = make_timestamp();
    auto cat_name = log_category_name(cat);
    auto lvl_name = log_level_name(level);

    char formatted[4096];
    std::snprintf(formatted, sizeof(formatted),
                  "[%s] [%.*s] [%.*s] %s\n",
                  timestamp.c_str(),
                  static_cast<int>(lvl_name.size()), lvl_name.data(),
                  static_cast<int>(cat_name.size()), cat_name.data(),
                  message.c_str());

    std::string line(formatted);

    std::lock_guard<std::mutex> lock(mutex_);

    // 4. Console output
    if (print_to_console_) {
        std::fputs(line.c_str(), stderr);
    }

    // 5. File output or pre-file buffer
    if (print_to_file_ && log_file_) {
        std::fputs(line.c_str(), log_file_);
        std::fflush(log_file_);
    } else if (!started_logging_) {
        buffer_.push_back(line);
        // 6. Cap buffer size
        if (buffer_.size() > 10000) {
            buffer_.erase(buffer_.begin(),
                          buffer_.begin() + 5000);
        }
    }

    // 7. Callback notification
    if (callback_) {
        callback_(cat, level, message);
    }
}

// ---------------------------------------------------------------------------
// set_callback
//   Registers a function to be called for every emitted log line (after
//   formatting).  Pass nullptr to remove.
// ---------------------------------------------------------------------------
void Logger::set_callback(LogCallback cb) {
    std::lock_guard<std::mutex> lock(mutex_);
    callback_ = std::move(cb);
}

// ---------------------------------------------------------------------------
// shrink_log_file
//   Truncates the log file to approximately max_bytes/2 by keeping only
//   the tail portion, starting at the first complete line boundary.
// ---------------------------------------------------------------------------
void Logger::shrink_log_file(size_t max_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (log_file_path_.empty()) return;

    // 1. Check current size
    std::error_code ec;
    auto file_size = std::filesystem::file_size(log_file_path_, ec);
    if (ec || file_size <= max_bytes) return;

    // 2. Close the file before rewriting
    if (log_file_) {
        std::fclose(log_file_);
        log_file_ = nullptr;
    }

    // 3. Read the tail portion
    size_t keep = max_bytes / 2;
    std::vector<char> tail(keep);

    FILE* f = std::fopen(log_file_path_.c_str(), "rb");
    if (f) {
        std::fseek(f, static_cast<long>(file_size - keep), SEEK_SET);
        size_t read = std::fread(tail.data(), 1, keep, f);
        std::fclose(f);

        // 4. Find first newline to start at a clean line
        size_t start = 0;
        for (size_t i = 0; i < read; ++i) {
            if (tail[i] == '\n') {
                start = i + 1;
                break;
            }
        }

        // 5. Rewrite the file with just the tail
        f = std::fopen(log_file_path_.c_str(), "w");
        if (f) {
            std::fwrite(tail.data() + start, 1, read - start, f);
            std::fclose(f);
        }
    }

    // 6. Re-open in append mode
    log_file_ = std::fopen(log_file_path_.c_str(), "a");
}

// ---------------------------------------------------------------------------
// flush
//   Forces any buffered file output to disk.
// ---------------------------------------------------------------------------
void Logger::flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (log_file_) {
        std::fflush(log_file_);
    }
}

// ---------------------------------------------------------------------------
// get_buffered_messages
//   Returns a copy of the pre-file message buffer (useful for the GUI to
//   display early startup messages).
// ---------------------------------------------------------------------------
std::vector<std::string> Logger::get_buffered_messages() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffer_;
}

// ===========================================================================
//  Free functions
// ===========================================================================

// ---------------------------------------------------------------------------
// get_log_category_names
//   Returns a list of all known category names (from the global table).
// ---------------------------------------------------------------------------
std::vector<std::string> get_log_category_names() {
    std::vector<std::string> names;
    for (const auto& entry : ALL_LOG_CATEGORIES) {
        names.emplace_back(entry.name);
    }
    return names;
}

// ---------------------------------------------------------------------------
// parse_log_categories
//   Parses a comma-separated category string into a combined bitmask.
//   Special tokens: "1"/"all" -> ALL, "0"/"none" -> NONE.
// ---------------------------------------------------------------------------
LogCategory parse_log_categories(std::string_view str) {
    LogCategory result = LogCategory::NONE;
    size_t start = 0;
    while (start < str.size()) {
        // 1. Extract next comma-delimited token
        auto pos = str.find(',', start);
        std::string_view token;
        if (pos == std::string_view::npos) {
            token = str.substr(start);
            start = str.size();
        } else {
            token = str.substr(start, pos - start);
            start = pos + 1;
        }
        // 2. Trim whitespace from the token
        auto ts = token.find_first_not_of(" \t");
        auto te = token.find_last_not_of(" \t");
        if (ts != std::string_view::npos) {
            token = token.substr(ts, te - ts + 1);
        }
        // 3. Handle special values
        if (token == "1" || token == "all") {
            return LogCategory::ALL;
        }
        if (token == "0" || token == "none") {
            return LogCategory::NONE;
        }
        // 4. Merge the category bit
        result |= log_category_from_name(token);
    }
    return result;
}

// ---------------------------------------------------------------------------
// format_log_message
//   printf-style formatting helper that returns a std::string.
// ---------------------------------------------------------------------------
std::string format_log_message(const char* fmt, ...) {
    char buf[4096];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    return std::string(buf);
}

} // namespace rnet::core

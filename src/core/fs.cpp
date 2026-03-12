#include "core/fs.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#else
#include <sys/file.h>
#include <unistd.h>
#endif

namespace rnet::core {

static fs::path g_data_dir;
static std::mutex g_data_dir_mutex;

#ifdef _WIN32
static int g_lock_fd = -1;
static HANDLE g_lock_handle = INVALID_HANDLE_VALUE;
#else
static int g_lock_fd = -1;
#endif

fs::path get_default_data_dir() {
#ifdef _WIN32
    char app_data[MAX_PATH] = {0};
    if (SHGetFolderPathA(nullptr, CSIDL_APPDATA, nullptr, 0,
                         app_data) == S_OK) {
        return fs::path(app_data) / "ResonanceNet";
    }
    // Fallback
    const char* home = std::getenv("USERPROFILE");
    if (home) {
        return fs::path(home) / "AppData" / "Roaming" / "ResonanceNet";
    }
    return fs::path("C:\\ResonanceNet");
#elif defined(__APPLE__)
    const char* home = std::getenv("HOME");
    if (home) {
        return fs::path(home) / "Library" / "Application Support"
               / "ResonanceNet";
    }
    return fs::path("/tmp/ResonanceNet");
#else
    const char* home = std::getenv("HOME");
    if (home) {
        return fs::path(home) / ".resonancenet";
    }
    return fs::path("/tmp/.resonancenet");
#endif
}

fs::path get_data_dir() {
    std::lock_guard<std::mutex> lock(g_data_dir_mutex);
    if (g_data_dir.empty()) {
        return get_default_data_dir();
    }
    return g_data_dir;
}

void set_data_dir(const fs::path& path) {
    std::lock_guard<std::mutex> lock(g_data_dir_mutex);
    g_data_dir = path;
}

Result<void> ensure_directory(const fs::path& path) {
    std::error_code ec;
    if (fs::exists(path, ec)) {
        if (!fs::is_directory(path, ec)) {
            return Result<void>::err(
                "Path exists but is not a directory: " +
                path.string());
        }
        return Result<void>::ok();
    }
    if (!fs::create_directories(path, ec)) {
        return Result<void>::err(
            "Failed to create directory: " + path.string() +
            " (" + ec.message() + ")");
    }
    return Result<void>::ok();
}

Result<void> lock_directory(const fs::path& dir,
                            const std::string& lockfile) {
    auto lock_path = dir / lockfile;

#ifdef _WIN32
    g_lock_handle = CreateFileA(
        lock_path.string().c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0,  // No sharing
        nullptr,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);

    if (g_lock_handle == INVALID_HANDLE_VALUE) {
        return Result<void>::err(
            "Cannot obtain lock on " + lock_path.string() +
            ". Another instance may be running.");
    }
    return Result<void>::ok();
#else
    g_lock_fd = open(lock_path.c_str(),
                     O_RDWR | O_CREAT, 0644);
    if (g_lock_fd < 0) {
        return Result<void>::err(
            "Cannot create lock file: " + lock_path.string());
    }
    if (flock(g_lock_fd, LOCK_EX | LOCK_NB) != 0) {
        close(g_lock_fd);
        g_lock_fd = -1;
        return Result<void>::err(
            "Cannot obtain lock on " + lock_path.string() +
            ". Another instance may be running.");
    }
    return Result<void>::ok();
#endif
}

void unlock_directory(const fs::path& dir,
                      const std::string& lockfile) {
#ifdef _WIN32
    if (g_lock_handle != INVALID_HANDLE_VALUE) {
        CloseHandle(g_lock_handle);
        g_lock_handle = INVALID_HANDLE_VALUE;
    }
    auto lock_path = dir / lockfile;
    std::error_code ec;
    fs::remove(lock_path, ec);
#else
    if (g_lock_fd >= 0) {
        flock(g_lock_fd, LOCK_UN);
        close(g_lock_fd);
        g_lock_fd = -1;
    }
    auto lock_path = dir / lockfile;
    std::error_code ec;
    fs::remove(lock_path, ec);
#endif
}

fs::path abs_path_for_config_val(const fs::path& path) {
    if (path.is_absolute()) return path;
    return get_data_dir() / path;
}

fs::path get_blocks_dir() {
    return get_data_dir() / "blocks";
}

fs::path get_chainstate_dir() {
    return get_data_dir() / "chainstate";
}

fs::path get_wallet_dir() {
    return get_data_dir() / "wallets";
}

fs::path get_log_dir() {
    return get_data_dir();
}

Result<void> rename_file(const fs::path& src, const fs::path& dst) {
    std::error_code ec;
    fs::rename(src, dst, ec);
    if (ec) {
        return Result<void>::err(
            "Failed to rename " + src.string() + " to " +
            dst.string() + ": " + ec.message());
    }
    return Result<void>::ok();
}

Result<void> copy_file(const fs::path& src, const fs::path& dst) {
    std::error_code ec;
    fs::copy_file(src, dst,
                  fs::copy_options::overwrite_existing, ec);
    if (ec) {
        return Result<void>::err(
            "Failed to copy " + src.string() + " to " +
            dst.string() + ": " + ec.message());
    }
    return Result<void>::ok();
}

bool file_exists(const fs::path& path) {
    std::error_code ec;
    return fs::exists(path, ec);
}

uint64_t file_size(const fs::path& path) {
    std::error_code ec;
    auto sz = fs::file_size(path, ec);
    if (ec) return 0;
    return sz;
}

Result<std::string> read_file(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return Result<std::string>::err(
            "Cannot open file: " + path.string());
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return Result<std::string>::ok(ss.str());
}

Result<void> write_file_atomic(const fs::path& path,
                               const std::string& data) {
    auto temp_path = path;
    temp_path += ".tmp";

    {
        std::ofstream file(temp_path,
                           std::ios::binary | std::ios::trunc);
        if (!file.is_open()) {
            return Result<void>::err(
                "Cannot open temp file: " + temp_path.string());
        }
        file.write(data.data(),
                   static_cast<std::streamsize>(data.size()));
        if (!file.good()) {
            return Result<void>::err(
                "Failed to write temp file: " + temp_path.string());
        }
    }

    return rename_file(temp_path, path);
}

fs::path get_temp_dir() {
    return fs::temp_directory_path();
}

std::vector<fs::path> list_files(const fs::path& dir,
                                  const std::string& extension) {
    std::vector<fs::path> result;
    std::error_code ec;
    if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) {
        return result;
    }

    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file(ec)) continue;
        if (!extension.empty()) {
            auto ext = entry.path().extension().string();
            if (ext != extension) continue;
        }
        result.push_back(entry.path());
    }

    std::sort(result.begin(), result.end());
    return result;
}

uint64_t directory_size(const fs::path& dir) {
    uint64_t total = 0;
    std::error_code ec;
    if (!fs::exists(dir, ec)) return 0;

    for (const auto& entry :
         fs::recursive_directory_iterator(dir, ec)) {
        if (entry.is_regular_file(ec)) {
            total += entry.file_size(ec);
        }
    }
    return total;
}

Result<void> remove_directory(const fs::path& dir) {
    std::error_code ec;
    if (!fs::exists(dir, ec)) {
        return Result<void>::ok();
    }
    auto count = fs::remove_all(dir, ec);
    if (ec) {
        return Result<void>::err(
            "Failed to remove directory: " + dir.string() +
            " (" + ec.message() + ")");
    }
    (void)count;
    return Result<void>::ok();
}

fs::path get_temp_file_path(const std::string& prefix) {
    auto temp = fs::temp_directory_path();
    // Generate a pseudo-unique filename using current time
    auto now = std::chrono::steady_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count();
    std::string name = prefix + std::to_string(ns);
    return temp / name;
}

bool is_path_inside(const fs::path& path, const fs::path& parent) {
    std::error_code ec;
    auto abs_path = fs::weakly_canonical(path, ec);
    auto abs_parent = fs::weakly_canonical(parent, ec);

    auto path_str = abs_path.string();
    auto parent_str = abs_parent.string();

    if (path_str.size() < parent_str.size()) return false;
    return path_str.substr(0, parent_str.size()) == parent_str;
}

}  // namespace rnet::core

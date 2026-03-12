#pragma once

#include <filesystem>
#include <mutex>
#include <string>

#include "core/error.h"

namespace rnet::core {

namespace fs = std::filesystem;

/// Get the default data directory for ResonanceNet
/// Linux:   ~/.resonancenet/
/// macOS:   ~/Library/Application Support/ResonanceNet/
/// Windows: %APPDATA%\ResonanceNet
fs::path get_default_data_dir();

/// Get the current data directory (may be overridden by -datadir)
/// Must call set_data_dir() first, or it returns the default.
fs::path get_data_dir();

/// Set the data directory (call during startup from -datadir arg)
void set_data_dir(const fs::path& path);

/// Ensure a directory exists, creating it if necessary
Result<void> ensure_directory(const fs::path& path);

/// Create a lock file to prevent multiple instances
Result<void> lock_directory(const fs::path& dir,
                            const std::string& lockfile = ".lock");

/// Release the directory lock
void unlock_directory(const fs::path& dir,
                      const std::string& lockfile = ".lock");

/// Convert a config-relative path to absolute
fs::path abs_path_for_config_val(const fs::path& path);

/// Get path for specific subdirectories
fs::path get_blocks_dir();
fs::path get_chainstate_dir();
fs::path get_wallet_dir();
fs::path get_log_dir();

/// File utilities
Result<void> rename_file(const fs::path& src, const fs::path& dst);
Result<void> copy_file(const fs::path& src, const fs::path& dst);
bool file_exists(const fs::path& path);
uint64_t file_size(const fs::path& path);

/// Read entire file into string
Result<std::string> read_file(const fs::path& path);

/// Write string to file atomically (write to temp, then rename)
Result<void> write_file_atomic(const fs::path& path,
                               const std::string& data);

/// Get temp directory path
fs::path get_temp_dir();

/// List files in a directory matching an optional extension filter
std::vector<fs::path> list_files(const fs::path& dir,
                                 const std::string& extension = "");

/// Get total size of all files in a directory (recursive)
uint64_t directory_size(const fs::path& dir);

/// Remove a directory and all contents
Result<void> remove_directory(const fs::path& dir);

/// Create a unique temporary file path
fs::path get_temp_file_path(const std::string& prefix = "rnet_");

/// Check if a path is inside another path (for security checks)
bool is_path_inside(const fs::path& path, const fs::path& parent);

}  // namespace rnet::core

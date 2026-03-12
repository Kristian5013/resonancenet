#pragma once

#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "core/error.h"

namespace rnet::core {

/// ArgsManager — command-line argument and config file parsing.
/// Supports:
///   -key=value
///   -key (boolean true)
///   -nokey (boolean false)
///   config file: key=value lines
class ArgsManager {
public:
    ArgsManager() = default;
    ~ArgsManager() = default;

    ArgsManager(const ArgsManager&) = delete;
    ArgsManager& operator=(const ArgsManager&) = delete;

    /// Register an argument with description and default
    void add_arg(const std::string& name, const std::string& help,
                 bool has_value = true);

    /// Parse command-line arguments
    Result<void> parse_args(int argc, const char* const argv[]);

    /// Parse from a vector of strings
    Result<void> parse_args(const std::vector<std::string>& args);

    /// Read a config file (key=value format, # for comments)
    Result<void> read_config_file(const std::string& path);

    /// Get argument value
    std::optional<std::string> get_arg(const std::string& name) const;

    /// Get argument as integer
    std::optional<int64_t> get_int_arg(const std::string& name) const;

    /// Get argument as bool (-key = true, -nokey = false)
    bool get_bool_arg(const std::string& name,
                      bool default_val = false) const;

    /// Get argument as double
    std::optional<double> get_double_arg(
        const std::string& name) const;

    /// Get all values for a multi-value argument
    std::vector<std::string> get_args(const std::string& name) const;

    /// Check if argument was provided
    bool is_set(const std::string& name) const;

    /// Get the data directory
    std::string get_data_dir() const;

    /// Set a default value
    void set_default(const std::string& name,
                     const std::string& value);

    /// Force-set a value (for testing)
    void force_set(const std::string& name, const std::string& value);

    /// Clear all parsed values
    void clear();

    /// Get usage/help string
    std::string get_help_message() const;

    /// Get the network name (main, test, regtest)
    std::string get_network() const;

    /// Get list of non-option arguments (positional)
    const std::vector<std::string>& get_positional_args() const;

    /// Dump all settings to a string (for debugging)
    std::string dump_settings() const;

    /// Generate a default config file string
    std::string generate_default_config() const;

    /// Get all settings as a flat map (merged: defaults < config < args)
    std::map<std::string, std::string> get_all_settings() const;

private:
    mutable std::mutex mutex_;

    // Registered arguments: name -> (help text, has_value)
    struct ArgDef {
        std::string help;
        bool has_value = true;
    };
    std::map<std::string, ArgDef> arg_defs_;

    // Parsed values: name -> list of values (for multi-value support)
    std::map<std::string, std::vector<std::string>> args_;

    // Config file values (lower priority than command-line)
    std::map<std::string, std::vector<std::string>> config_args_;

    // Default values
    std::map<std::string, std::string> defaults_;

    // Boolean flags set via -noXXX
    std::set<std::string> negated_args_;

    // Positional arguments
    std::vector<std::string> positional_args_;

    // Helper to normalize arg name (strip leading dashes)
    static std::string normalize_name(std::string_view name);

    // Internal get that checks args_ then config_args_ then defaults_
    std::optional<std::string> get_internal(
        const std::string& name) const;
};

}  // namespace rnet::core

#include "core/config.h"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <sstream>

namespace rnet::core {

std::string ArgsManager::normalize_name(std::string_view name) {
    // Strip leading dashes
    while (!name.empty() && name.front() == '-') {
        name.remove_prefix(1);
    }
    return std::string(name);
}

void ArgsManager::add_arg(const std::string& name,
                           const std::string& help,
                           bool has_value) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto norm = normalize_name(name);
    arg_defs_[norm] = {help, has_value};
}

Result<void> ArgsManager::parse_args(int argc,
                                      const char* const argv[]) {
    std::vector<std::string> args;
    args.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }
    return parse_args(args);
}

Result<void> ArgsManager::parse_args(
    const std::vector<std::string>& args) {
    std::lock_guard<std::mutex> lock(mutex_);
    args_.clear();
    negated_args_.clear();
    positional_args_.clear();

    // Skip argv[0] (program name)
    for (size_t i = 1; i < args.size(); ++i) {
        const auto& arg = args[i];

        if (arg.empty() || arg[0] != '-') {
            positional_args_.push_back(arg);
            continue;
        }

        // Split on '='
        auto eq_pos = arg.find('=');
        std::string name_part;
        std::string value_part;

        if (eq_pos != std::string::npos) {
            name_part = arg.substr(0, eq_pos);
            value_part = arg.substr(eq_pos + 1);
        } else {
            name_part = arg;
            value_part = "1";  // Boolean flag
        }

        auto norm = normalize_name(name_part);

        // Handle -noXXX negation
        if (norm.size() > 2 && norm.substr(0, 2) == "no") {
            std::string base = norm.substr(2);
            negated_args_.insert(base);
            args_[base].push_back("0");
            continue;
        }

        args_[norm].push_back(value_part);
    }

    return Result<void>::ok();
}

Result<void> ArgsManager::read_config_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return Result<void>::err("Cannot open config file: " + path);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    std::string line;
    int line_num = 0;

    std::string current_section;  // For [section] support

    while (std::getline(file, line)) {
        ++line_num;

        // Trim whitespace
        auto start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        line = line.substr(start);

        // Skip comments
        if (line[0] == '#' || line[0] == ';') continue;

        // Remove inline comments (but not inside quotes)
        bool in_quotes = false;
        for (size_t i = 0; i < line.size(); ++i) {
            if (line[i] == '"') in_quotes = !in_quotes;
            if (!in_quotes && line[i] == '#') {
                line = line.substr(0, i);
                break;
            }
        }

        // Trim trailing whitespace
        auto end = line.find_last_not_of(" \t\r\n");
        if (end != std::string::npos) {
            line = line.substr(0, end + 1);
        }

        if (line.empty()) continue;

        // Handle [section] headers
        if (line.front() == '[' && line.back() == ']') {
            current_section = line.substr(1, line.size() - 2);
            // Trim section name
            auto sec_start = current_section.find_first_not_of(" \t");
            auto sec_end = current_section.find_last_not_of(" \t");
            if (sec_start != std::string::npos) {
                current_section = current_section.substr(
                    sec_start, sec_end - sec_start + 1);
            }
            continue;
        }

        // Parse key=value
        auto eq_pos = line.find('=');
        std::string key_str;
        std::string val_str;

        if (eq_pos == std::string::npos) {
            key_str = line;
            val_str = "1";
        } else {
            key_str = line.substr(0, eq_pos);
            val_str = line.substr(eq_pos + 1);
        }

        // Trim key and value
        auto key_end = key_str.find_last_not_of(" \t");
        if (key_end != std::string::npos) {
            key_str = key_str.substr(0, key_end + 1);
        }
        auto val_start = val_str.find_first_not_of(" \t");
        if (val_start != std::string::npos) {
            val_str = val_str.substr(val_start);
        }
        auto val_end = val_str.find_last_not_of(" \t");
        if (val_end != std::string::npos) {
            val_str = val_str.substr(0, val_end + 1);
        }

        // Strip quotes from value
        if (val_str.size() >= 2 &&
            val_str.front() == '"' && val_str.back() == '"') {
            val_str = val_str.substr(1, val_str.size() - 2);
        }

        // Prefix with section if present (e.g., [test] port=1234
        // becomes test.port=1234)
        auto name = normalize_name(key_str);
        if (!current_section.empty()) {
            name = current_section + "." + name;
        }

        config_args_[name].push_back(val_str);
    }

    return Result<void>::ok();
}

std::optional<std::string> ArgsManager::get_internal(
    const std::string& name) const {
    auto norm = normalize_name(name);

    // Command-line takes priority
    auto it = args_.find(norm);
    if (it != args_.end() && !it->second.empty()) {
        return it->second.back();
    }

    // Then config file
    auto cit = config_args_.find(norm);
    if (cit != config_args_.end() && !cit->second.empty()) {
        return cit->second.back();
    }

    // Then defaults
    auto dit = defaults_.find(norm);
    if (dit != defaults_.end()) {
        return dit->second;
    }

    return std::nullopt;
}

std::optional<std::string> ArgsManager::get_arg(
    const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return get_internal(name);
}

std::optional<int64_t> ArgsManager::get_int_arg(
    const std::string& name) const {
    auto val = get_arg(name);
    if (!val) return std::nullopt;

    int64_t result = 0;
    auto [ptr, ec] = std::from_chars(
        val->data(), val->data() + val->size(), result);
    if (ec != std::errc()) return std::nullopt;
    return result;
}

bool ArgsManager::get_bool_arg(const std::string& name,
                                bool default_val) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto norm = normalize_name(name);

    if (negated_args_.count(norm)) return false;

    auto val = get_internal(norm);
    if (!val) return default_val;

    if (*val == "0" || *val == "false" || *val == "no") return false;
    return true;
}

std::optional<double> ArgsManager::get_double_arg(
    const std::string& name) const {
    auto val = get_arg(name);
    if (!val) return std::nullopt;

    try {
        return std::stod(*val);
    } catch (...) {
        return std::nullopt;
    }
}

std::vector<std::string> ArgsManager::get_args(
    const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto norm = normalize_name(name);
    std::vector<std::string> result;

    auto it = args_.find(norm);
    if (it != args_.end()) {
        result.insert(result.end(),
                      it->second.begin(), it->second.end());
    }

    auto cit = config_args_.find(norm);
    if (cit != config_args_.end()) {
        result.insert(result.end(),
                      cit->second.begin(), cit->second.end());
    }

    return result;
}

bool ArgsManager::is_set(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto norm = normalize_name(name);
    return args_.count(norm) > 0 || config_args_.count(norm) > 0;
}

std::string ArgsManager::get_data_dir() const {
    auto dir = get_arg("datadir");
    if (dir) return *dir;
    return "";
}

void ArgsManager::set_default(const std::string& name,
                               const std::string& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    defaults_[normalize_name(name)] = value;
}

void ArgsManager::force_set(const std::string& name,
                             const std::string& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    args_[normalize_name(name)] = {value};
}

void ArgsManager::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    args_.clear();
    config_args_.clear();
    negated_args_.clear();
    positional_args_.clear();
}

std::string ArgsManager::get_help_message() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;
    oss << "Options:\n";

    for (const auto& [name, def] : arg_defs_) {
        oss << "  -" << name;
        if (def.has_value) {
            oss << "=<value>";
        }
        oss << "\n";
        if (!def.help.empty()) {
            oss << "      " << def.help << "\n";
        }
    }
    return oss.str();
}

std::string ArgsManager::get_network() const {
    if (get_bool_arg("regtest")) return "regtest";
    if (get_bool_arg("testnet")) return "testnet";
    return "main";
}

const std::vector<std::string>&
ArgsManager::get_positional_args() const {
    return positional_args_;
}

// ─── Utility: dump all settings ──────────────────────────────────────

std::string ArgsManager::dump_settings() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;

    oss << "# Command-line arguments:\n";
    for (const auto& [name, values] : args_) {
        for (const auto& val : values) {
            oss << name << "=" << val << "\n";
        }
    }

    oss << "\n# Config file arguments:\n";
    for (const auto& [name, values] : config_args_) {
        for (const auto& val : values) {
            oss << name << "=" << val << "\n";
        }
    }

    oss << "\n# Default values:\n";
    for (const auto& [name, val] : defaults_) {
        oss << name << "=" << val << "\n";
    }

    return oss.str();
}

std::string ArgsManager::generate_default_config() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;

    oss << "# ResonanceNet configuration file\n";
    oss << "# Lines beginning with # are comments.\n";
    oss << "# See rnetd -help for all options.\n\n";

    for (const auto& [name, def] : arg_defs_) {
        oss << "# " << def.help << "\n";
        auto dit = defaults_.find(name);
        if (dit != defaults_.end()) {
            oss << "#" << name << "=" << dit->second << "\n";
        } else if (def.has_value) {
            oss << "#" << name << "=\n";
        } else {
            oss << "#" << name << "=1\n";
        }
        oss << "\n";
    }

    oss << "# Network sections:\n";
    oss << "# [test]\n";
    oss << "# [regtest]\n";

    return oss.str();
}

std::map<std::string, std::string>
ArgsManager::get_all_settings() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<std::string, std::string> result;

    // Defaults first (lowest priority)
    for (const auto& [name, val] : defaults_) {
        result[name] = val;
    }

    // Config file (medium priority)
    for (const auto& [name, values] : config_args_) {
        if (!values.empty()) {
            result[name] = values.back();
        }
    }

    // Command-line (highest priority)
    for (const auto& [name, values] : args_) {
        if (!values.empty()) {
            result[name] = values.back();
        }
    }

    return result;
}

}  // namespace rnet::core

#include "rpc/util.h"

#include <algorithm>
#include <sstream>

#include "core/logging.h"

namespace rnet::rpc {

// ── RPCTable ────────────────────────────────────────────────────────

void RPCTable::register_command(RPCCommand cmd) {
    std::string name = cmd.name;
    index_[name] = commands_.size();
    commands_.push_back(std::move(cmd));
}

const RPCCommand* RPCTable::find(const std::string& name) const {
    auto it = index_.find(name);
    if (it == index_.end()) return nullptr;
    return &commands_[it->second];
}

std::string RPCTable::help(const std::string& command_name) const {
    if (!command_name.empty()) {
        const auto* cmd = find(command_name);
        if (!cmd) return "help: unknown command: " + command_name;
        return cmd->name + "\n" + cmd->help;
    }

    // Group by category
    std::map<std::string, std::vector<const RPCCommand*>> by_cat;
    for (const auto& cmd : commands_) {
        by_cat[cmd.category].push_back(&cmd);
    }

    std::ostringstream out;
    for (const auto& [cat, cmds] : by_cat) {
        out << "== " << cat << " ==\n";
        for (const auto* cmd : cmds) {
            out << cmd->name;
            // Show first line of help
            if (!cmd->help.empty()) {
                auto nl = cmd->help.find('\n');
                out << " - " << cmd->help.substr(0, nl);
            }
            out << "\n";
        }
        out << "\n";
    }
    return out.str();
}

// ── Param helpers ───────────────────────────────────────────────────

static const JsonValue NULL_PARAM;

const JsonValue& get_param(const RPCRequest& req, size_t index) {
    if (req.params.is_array() && index < req.params.size()) {
        return req.params[index];
    }
    return NULL_PARAM;
}

const JsonValue& get_param_optional(const RPCRequest& req, size_t index) {
    return get_param(req, index);
}

std::string require_string_param(const RPCRequest& req, size_t index,
                                 const std::string& name) {
    const auto& p = get_param(req, index);
    if (!p.is_string()) {
        // We signal error by returning empty — callers should check
        return {};
    }
    return p.as_string();
}

int64_t require_int_param(const RPCRequest& req, size_t index,
                          const std::string& name) {
    const auto& p = get_param(req, index);
    if (p.is_int()) return p.as_int();
    if (p.is_double()) return static_cast<int64_t>(p.as_double());
    return 0;
}

bool require_bool_param(const RPCRequest& req, size_t index,
                        const std::string& name) {
    const auto& p = get_param(req, index);
    if (p.is_bool()) return p.as_bool();
    return false;
}

bool is_valid_hex(const std::string& s) {
    if (s.size() % 2 != 0) return false;
    for (char c : s) {
        if (!((c >= '0' && c <= '9') ||
              (c >= 'a' && c <= 'f') ||
              (c >= 'A' && c <= 'F'))) {
            return false;
        }
    }
    return true;
}

static int hex_char_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
    std::vector<uint8_t> result;
    if (hex.size() % 2 != 0) return result;
    result.reserve(hex.size() / 2);
    for (size_t i = 0; i + 1 < hex.size(); i += 2) {
        int hi = hex_char_val(hex[i]);
        int lo = hex_char_val(hex[i + 1]);
        if (hi < 0 || lo < 0) { result.clear(); return result; }
        result.push_back(static_cast<uint8_t>((hi << 4) | lo));
    }
    return result;
}

std::string bytes_to_hex(const uint8_t* data, size_t len) {
    static constexpr char hex_chars[] = "0123456789abcdef";
    std::string result;
    result.reserve(len * 2);
    for (size_t i = 0; i < len; ++i) {
        result.push_back(hex_chars[(data[i] >> 4) & 0x0F]);
        result.push_back(hex_chars[data[i] & 0x0F]);
    }
    return result;
}

JsonValue make_rpc_error(int code, const std::string& message) {
    JsonValue err = JsonValue::object();
    err.set("code", JsonValue(static_cast<int64_t>(code)));
    err.set("message", JsonValue(message));
    return err;
}

JsonValue rpc_error_response(int code, const std::string& message) {
    // Return the error object — caller wraps it into RPCResponse
    return make_rpc_error(code, message);
}

}  // namespace rnet::rpc

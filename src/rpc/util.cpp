// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "rpc/util.h"

#include "core/logging.h"

#include <algorithm>
#include <sstream>

namespace rnet::rpc {

// ===========================================================================
//  RPCTable -- command registry
// ===========================================================================

// ---------------------------------------------------------------------------
// register_command -- insert a command into the table, indexed by name.
// ---------------------------------------------------------------------------
void RPCTable::register_command(RPCCommand cmd) {
    // 1. Store the name before the move.
    std::string name = cmd.name;
    // 2. Map the name to the command's index in the vector.
    index_[name] = commands_.size();
    commands_.push_back(std::move(cmd));
}

// ---------------------------------------------------------------------------
// find -- look up a command by name; returns nullptr if not registered.
// ---------------------------------------------------------------------------
const RPCCommand* RPCTable::find(const std::string& name) const {
    auto it = index_.find(name);
    if (it == index_.end()) return nullptr;
    return &commands_[it->second];
}

// ---------------------------------------------------------------------------
// help -- generate help text for one command or all commands grouped by
// category.  When command_name is empty, lists every registered command
// with the first line of its help string.
// ---------------------------------------------------------------------------
std::string RPCTable::help(const std::string& command_name) const {
    // 1. Single-command help.
    if (!command_name.empty()) {
        const auto* cmd = find(command_name);
        if (!cmd) return "help: unknown command: " + command_name;
        return cmd->name + "\n" + cmd->help;
    }

    // 2. Group all commands by category.
    std::map<std::string, std::vector<const RPCCommand*>> by_cat;
    for (const auto& cmd : commands_) {
        by_cat[cmd.category].push_back(&cmd);
    }

    // 3. Format the output.
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

// ===========================================================================
//  Parameter helpers
// ===========================================================================

static const JsonValue NULL_PARAM;

// ---------------------------------------------------------------------------
// get_param -- return the positional parameter at `index`, or a null
// JsonValue sentinel when out of range.
// ---------------------------------------------------------------------------
const JsonValue& get_param(const RPCRequest& req, size_t index) {
    if (req.params.is_array() && index < req.params.size()) {
        return req.params[index];
    }
    return NULL_PARAM;
}

// ---------------------------------------------------------------------------
// get_param_optional -- identical to get_param (callers use the name for
// documentation clarity).
// ---------------------------------------------------------------------------
const JsonValue& get_param_optional(const RPCRequest& req, size_t index) {
    return get_param(req, index);
}

// ---------------------------------------------------------------------------
// require_string_param -- extract a string parameter or return empty on
// type mismatch (callers must check the result).
// ---------------------------------------------------------------------------
std::string require_string_param(const RPCRequest& req, size_t index,
                                 const std::string& name) {
    const auto& p = get_param(req, index);
    if (!p.is_string()) {
        return {};
    }
    return p.as_string();
}

// ---------------------------------------------------------------------------
// require_int_param -- extract an integer parameter, accepting doubles via
// truncation.  Returns 0 on type mismatch.
// ---------------------------------------------------------------------------
int64_t require_int_param(const RPCRequest& req, size_t index,
                          const std::string& name) {
    const auto& p = get_param(req, index);
    if (p.is_int()) return p.as_int();
    if (p.is_double()) return static_cast<int64_t>(p.as_double());
    return 0;
}

// ---------------------------------------------------------------------------
// require_bool_param -- extract a boolean parameter.  Returns false on
// type mismatch.
// ---------------------------------------------------------------------------
bool require_bool_param(const RPCRequest& req, size_t index,
                        const std::string& name) {
    const auto& p = get_param(req, index);
    if (p.is_bool()) return p.as_bool();
    return false;
}

// ===========================================================================
//  Hex utilities
// ===========================================================================

// ---------------------------------------------------------------------------
// is_valid_hex -- return true when `s` is an even-length string of
// hexadecimal characters [0-9a-fA-F].
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// hex_char_val -- convert a single hex character to its 0-15 value.
// ---------------------------------------------------------------------------
static int hex_char_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// ---------------------------------------------------------------------------
// hex_to_bytes -- decode a hex string into a byte vector.  Returns an
// empty vector on invalid input.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// bytes_to_hex -- encode a byte buffer as lowercase hex.
// ---------------------------------------------------------------------------
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

// ===========================================================================
//  RPC error helpers
// ===========================================================================

// ---------------------------------------------------------------------------
// make_rpc_error -- build a JSON-RPC error object with code and message.
// ---------------------------------------------------------------------------
JsonValue make_rpc_error(int code, const std::string& message) {
    JsonValue err = JsonValue::object();
    err.set("code", JsonValue(static_cast<int64_t>(code)));
    err.set("message", JsonValue(message));
    return err;
}

// ---------------------------------------------------------------------------
// rpc_error_response -- convenience alias for make_rpc_error.
// ---------------------------------------------------------------------------
JsonValue rpc_error_response(int code, const std::string& message) {
    return make_rpc_error(code, message);
}

} // namespace rnet::rpc

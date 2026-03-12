#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "rpc/request.h"

// Forward declarations
namespace rnet::node { struct NodeContext; }

namespace rnet::rpc {

// ── RPC Error Codes ─────────────────────────────────────────────────

/// Standard JSON-RPC 2.0 error codes
static constexpr int RPC_INVALID_REQUEST  = -32600;
static constexpr int RPC_METHOD_NOT_FOUND = -32601;
static constexpr int RPC_INVALID_PARAMS   = -32602;
static constexpr int RPC_INTERNAL_ERROR   = -32603;
static constexpr int RPC_PARSE_ERROR      = -32700;

/// Bitcoin-style custom error codes
static constexpr int RPC_MISC_ERROR                  = -1;
static constexpr int RPC_TYPE_ERROR                  = -3;
static constexpr int RPC_INVALID_ADDRESS_OR_KEY      = -5;
static constexpr int RPC_OUT_OF_MEMORY               = -7;
static constexpr int RPC_INVALID_PARAMETER           = -8;
static constexpr int RPC_DATABASE_ERROR              = -20;
static constexpr int RPC_DESERIALIZATION_ERROR       = -22;
static constexpr int RPC_VERIFY_ERROR                = -25;
static constexpr int RPC_VERIFY_REJECTED             = -26;
static constexpr int RPC_VERIFY_ALREADY_IN_CHAIN     = -27;
static constexpr int RPC_IN_WARMUP                   = -28;
static constexpr int RPC_METHOD_DEPRECATED           = -32;

/// Wallet errors
static constexpr int RPC_WALLET_ERROR                = -4;
static constexpr int RPC_WALLET_INSUFFICIENT_FUNDS   = -6;
static constexpr int RPC_WALLET_KEYPOOL_RAN_OUT      = -12;
static constexpr int RPC_WALLET_UNLOCK_NEEDED        = -13;
static constexpr int RPC_WALLET_PASSPHRASE_INCORRECT = -14;
static constexpr int RPC_WALLET_NOT_FOUND            = -18;
static constexpr int RPC_WALLET_NOT_SPECIFIED        = -19;

/// P2P client errors
static constexpr int RPC_CLIENT_NOT_CONNECTED        = -9;
static constexpr int RPC_CLIENT_IN_INITIAL_DOWNLOAD  = -10;
static constexpr int RPC_CLIENT_NODE_ALREADY_ADDED   = -23;
static constexpr int RPC_CLIENT_NODE_NOT_ADDED       = -24;
static constexpr int RPC_CLIENT_NODE_NOT_CONNECTED   = -29;
static constexpr int RPC_CLIENT_INVALID_IP_OR_SUBNET = -30;
static constexpr int RPC_CLIENT_P2P_DISABLED         = -31;

// ── RPC Command ─────────────────────────────────────────────────────

/// Handler function signature
using RPCHandler = std::function<JsonValue(const RPCRequest& req,
                                           node::NodeContext& ctx)>;

/// An RPC command entry
struct RPCCommand {
    std::string name;
    RPCHandler  handler;
    std::string help;
    std::string category;
};

// ── RPC Table ───────────────────────────────────────────────────────

/// Registry of all RPC commands. Thread-safe for reads after init.
class RPCTable {
public:
    RPCTable() = default;

    /// Register a command. Call during init only.
    void register_command(RPCCommand cmd);

    /// Look up a command by name. Returns nullptr if not found.
    const RPCCommand* find(const std::string& name) const;

    /// Get all registered commands
    const std::vector<RPCCommand>& commands() const { return commands_; }

    /// Get help text for a specific command, or all commands if name is empty
    std::string help(const std::string& command_name = "") const;

private:
    std::vector<RPCCommand> commands_;
    std::unordered_map<std::string, size_t> index_;
};

// ── Param helpers ───────────────────────────────────────────────────

/// Get positional param or throw RPC error
const JsonValue& get_param(const RPCRequest& req, size_t index);

/// Get optional positional param (returns null if absent)
const JsonValue& get_param_optional(const RPCRequest& req, size_t index);

/// Require a string param at position
std::string require_string_param(const RPCRequest& req, size_t index,
                                 const std::string& name);

/// Require an integer param at position
int64_t require_int_param(const RPCRequest& req, size_t index,
                          const std::string& name);

/// Require a bool param at position
bool require_bool_param(const RPCRequest& req, size_t index,
                        const std::string& name);

/// Validate hex string (even length, all hex chars)
bool is_valid_hex(const std::string& s);

/// Convert hex string to bytes
std::vector<uint8_t> hex_to_bytes(const std::string& hex);

/// Convert bytes to hex string
std::string bytes_to_hex(const uint8_t* data, size_t len);

/// Build a JSON-RPC error response JsonValue
JsonValue make_rpc_error(int code, const std::string& message);

/// Throw helper — returns a JsonValue that represents an error
/// (caller should return this from handler to signal error)
JsonValue rpc_error_response(int code, const std::string& message);

}  // namespace rnet::rpc

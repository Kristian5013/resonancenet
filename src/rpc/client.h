#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include "core/error.h"
#include "rpc/request.h"

namespace rnet::rpc {

/// RPCClient — connects to rnetd's JSON-RPC server and sends requests.
/// Used by rnet-cli and other tools that need to communicate with the daemon.
class RPCClient {
public:
    RPCClient();
    ~RPCClient();

    /// Set server connection parameters
    void set_host(const std::string& host) { host_ = host; }
    void set_port(uint16_t port) { port_ = port; }

    /// Set authentication via rpcuser/rpcpassword
    void set_credentials(const std::string& user, const std::string& password);

    /// Set authentication via cookie file
    bool load_cookie(const std::filesystem::path& cookie_path);

    /// Set the data directory to look for the cookie file automatically
    void set_data_dir(const std::filesystem::path& dir) { data_dir_ = dir; }

    /// Auto-detect authentication: try cookie file in data_dir first,
    /// fall back to credentials.
    bool auto_auth();

    /// Send a JSON-RPC request and get the response.
    /// Connects, sends, reads response, disconnects per-call.
    core::Result<RPCResponse> call(const std::string& method,
                                   const JsonValue& params = JsonValue::array());

    /// Send a raw JSON-RPC request
    core::Result<RPCResponse> send(const RPCRequest& req);

    /// Get the last HTTP status code received
    int last_http_status() const { return last_http_status_; }

    /// Get the last error message
    const std::string& last_error() const { return last_error_; }

private:
    /// Connect to the server, send request, read response
    core::Result<std::string> http_post(const std::string& body);

    /// Base64 encode for auth header
    static std::string base64_encode(const std::string& input);

    std::string host_ = "127.0.0.1";
    uint16_t port_ = 9554;

    std::string auth_header_;  // "Basic <base64>"
    std::filesystem::path data_dir_;

    int last_http_status_ = 0;
    std::string last_error_;
};

}  // namespace rnet::rpc

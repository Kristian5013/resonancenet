#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "core/sync.h"
#include "rpc/request.h"
#include "rpc/util.h"

// Forward declarations
namespace rnet::node { struct NodeContext; }

namespace rnet::rpc {

/// HTTP request parsed from the socket
struct HttpRequest {
    std::string method;       // "GET", "POST"
    std::string uri;          // "/"
    std::string body;
    std::string authorization; // "Basic ..." header value
    std::string content_type;
    int64_t content_length = 0;
};

/// Cookie-based auth file for RPC
static constexpr const char* COOKIE_FILE_NAME = ".cookie";

/// RPCServer — HTTP + JSON-RPC server for rnetd.
/// Binds to a port, accepts connections, parses JSON-RPC, dispatches
/// to registered handlers. Uses platform sockets (no libevent).
class RPCServer {
public:
    RPCServer();
    ~RPCServer();

    // Non-copyable
    RPCServer(const RPCServer&) = delete;
    RPCServer& operator=(const RPCServer&) = delete;

    /// Set the node context (provides access to all subsystems)
    void set_context(node::NodeContext* ctx) { ctx_ = ctx; }

    /// Set RPC credentials (rpcuser / rpcpassword)
    void set_credentials(const std::string& user,
                         const std::string& password);

    /// Set the data directory (for cookie file)
    void set_data_dir(const std::filesystem::path& dir) { data_dir_ = dir; }

    /// Get the RPC command table (for registration)
    RPCTable& table() { return table_; }
    const RPCTable& table() const { return table_; }

    /// Start the server on the given port. Non-blocking: spawns a
    /// background thread that accepts connections.
    bool start(uint16_t port = 9554, const std::string& bind_addr = "127.0.0.1");

    /// Stop the server and close all connections
    void stop();

    /// Check if the server is running
    bool is_running() const { return running_.load(); }

    /// Get the port we are listening on
    uint16_t port() const { return port_; }

    /// Get the startup time (Unix seconds)
    int64_t startup_time() const { return startup_time_; }

private:
    /// Main accept loop (runs in server_thread_)
    void accept_loop();

    /// Handle a single client connection
    void handle_connection(int64_t client_sock);

    /// Read a full HTTP request from the socket
    bool read_http_request(int64_t sock, HttpRequest& req);

    /// Send an HTTP response
    void send_http_response(int64_t sock, int status_code,
                            const std::string& status_text,
                            const std::string& body);

    /// Authenticate the request
    bool authenticate(const HttpRequest& req);

    /// Process a JSON-RPC request and return the response
    RPCResponse process_request(const RPCRequest& rpc_req);

    /// Generate a random cookie and write to the cookie file
    bool generate_cookie();

    /// Delete the cookie file on shutdown
    void delete_cookie();

    /// Base64 encode
    static std::string base64_encode(const std::string& input);

    /// Base64 decode
    static std::string base64_decode(const std::string& input);

    node::NodeContext* ctx_ = nullptr;
    RPCTable table_;

    std::string rpc_user_;
    std::string rpc_password_;
    std::string cookie_user_;      // "__cookie__"
    std::string cookie_password_;  // random hex string
    std::filesystem::path data_dir_;

    uint16_t port_ = 9554;
    int64_t listen_sock_ = -1;
    std::atomic<bool> running_{false};
    std::thread server_thread_;
    int64_t startup_time_ = 0;

    core::Mutex mutex_;
};

}  // namespace rnet::rpc

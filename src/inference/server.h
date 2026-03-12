#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "core/error.h"
#include "inference/batch.h"
#include "inference/engine.h"

namespace rnet::inference {

/// Server configuration.
struct ServerConfig {
    std::string host = "127.0.0.1";
    uint16_t port = 8080;
    int max_connections = 64;
    std::string api_key;               ///< Optional API key (empty = no auth)
    std::string model_name = "rnet-1"; ///< Model name reported in /v1/models
};

/// Parsed HTTP request.
struct HttpRequest {
    std::string method;     ///< GET, POST, etc.
    std::string path;       ///< /v1/completions, etc.
    std::string body;       ///< Request body (JSON)
    std::string api_key;    ///< From Authorization header
    std::string content_type;
    bool stream = false;    ///< SSE streaming requested
};

/// HTTP response to send.
struct HttpResponse {
    int status_code = 200;
    std::string content_type = "application/json";
    std::string body;
    bool is_sse = false;    ///< Server-Sent Events mode
};

/// Callback for SSE (Server-Sent Events) streaming.
/// Called with each chunk; return false to stop.
using SseCallback = std::function<bool(const std::string& event_data)>;

/// OpenAI-compatible HTTP inference server.
///
/// Endpoints:
///   POST /v1/completions       — Text completion
///   POST /v1/chat/completions  — Chat completion
///   GET  /v1/models            — List available models
///   GET  /health               — Health check
///
/// Wire format follows the OpenAI API specification.
class InferenceServer {
public:
    InferenceServer(InferenceEngine& engine,
                    BatchScheduler& scheduler,
                    const ServerConfig& config = {});
    ~InferenceServer();

    InferenceServer(const InferenceServer&) = delete;
    InferenceServer& operator=(const InferenceServer&) = delete;

    /// Start the HTTP server (binds and listens).
    Result<void> start();

    /// Stop the server and close all connections.
    void stop();

    /// Whether the server is running.
    bool is_running() const { return running_.load(std::memory_order_relaxed); }

    /// Get the bound port (useful when port 0 is specified for auto-assign).
    uint16_t bound_port() const { return bound_port_; }

    /// Process a single HTTP request (for testing without network).
    HttpResponse handle_request(const HttpRequest& request);

private:
    InferenceEngine& engine_;
    BatchScheduler& scheduler_;
    ServerConfig config_;

    std::atomic<bool> running_{false};
    uint16_t bound_port_ = 0;
    std::thread listener_thread_;

    // Platform socket handle (opaque)
    intptr_t listen_socket_ = -1;

    /// Listener loop: accept connections and dispatch.
    void listener_loop();

    /// Handle a single client connection.
    void handle_connection(intptr_t client_socket);

    /// Parse raw HTTP request bytes into HttpRequest struct.
    static Result<HttpRequest> parse_http_request(const std::string& raw);

    /// Serialize HttpResponse to raw HTTP bytes.
    static std::string serialize_http_response(const HttpResponse& response);

    // --- Route handlers ---

    /// POST /v1/completions
    HttpResponse handle_completions(const HttpRequest& request);

    /// POST /v1/chat/completions
    HttpResponse handle_chat_completions(const HttpRequest& request);

    /// GET /v1/models
    HttpResponse handle_models(const HttpRequest& request);

    /// GET /health
    HttpResponse handle_health(const HttpRequest& request);

    /// Validate API key if configured.
    bool check_auth(const HttpRequest& request, HttpResponse& error_response);

    // --- JSON helpers ---

    /// Build an OpenAI-compatible error response.
    static HttpResponse make_error_response(int status_code,
                                             const std::string& error_type,
                                             const std::string& message);

    /// Build a completion response JSON.
    std::string build_completion_json(const std::string& id,
                                      const std::string& model,
                                      const std::string& text,
                                      int prompt_tokens,
                                      int completion_tokens,
                                      const std::string& finish_reason);

    /// Build a chat completion response JSON.
    std::string build_chat_completion_json(const std::string& id,
                                            const std::string& model,
                                            const std::string& role,
                                            const std::string& content,
                                            int prompt_tokens,
                                            int completion_tokens,
                                            const std::string& finish_reason);

    /// Generate a unique request ID.
    static std::string generate_request_id();

    /// Simple JSON value extraction (no dependency on external JSON library).
    static std::string json_get_string(const std::string& json, const std::string& key);
    static int json_get_int(const std::string& json, const std::string& key, int default_val);
    static float json_get_float(const std::string& json, const std::string& key, float default_val);
    static bool json_get_bool(const std::string& json, const std::string& key, bool default_val);
};

}  // namespace rnet::inference

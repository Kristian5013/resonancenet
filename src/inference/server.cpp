// Copyright (c) 2024-2026 The ResonanceNet Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "inference/server.h"

#include "core/logging.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstring>
#include <sstream>

// ===========================================================================
// Platform Abstraction
// ===========================================================================

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "Ws2_32.lib")

namespace {
struct WinsockInit {
    WinsockInit() {
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
    }
    ~WinsockInit() { WSACleanup(); }
};
static WinsockInit g_winsock_init;
using socket_t = SOCKET;
constexpr socket_t INVALID_SOCK = INVALID_SOCKET;
inline int close_socket(socket_t s) { return closesocket(s); }
}  // namespace

#else
#  include <arpa/inet.h>
#  include <netinet/in.h>
#  include <sys/socket.h>
#  include <unistd.h>

namespace {
using socket_t = int;
constexpr socket_t INVALID_SOCK = -1;
inline int close_socket(socket_t s) { return close(s); }
}  // namespace
#endif

namespace rnet::inference {

// ===========================================================================
// Lifecycle
// ===========================================================================

// ---------------------------------------------------------------------------
// InferenceServer (constructor)
// ---------------------------------------------------------------------------
// Captures references to the shared InferenceEngine and BatchScheduler that
// back this server, plus a copy of the immutable ServerConfig (host, port,
// API key, model name, max connections).  No sockets are opened here.
// ---------------------------------------------------------------------------
InferenceServer::InferenceServer(InferenceEngine& engine,
                                  BatchScheduler& scheduler,
                                  const ServerConfig& config)
    : engine_(engine), scheduler_(scheduler), config_(config) {}

// ---------------------------------------------------------------------------
// ~InferenceServer
// ---------------------------------------------------------------------------
// Ensures the listener socket is closed and the accept-thread is joined
// before the object is destroyed, preventing dangling-socket leaks.
// ---------------------------------------------------------------------------
InferenceServer::~InferenceServer() {
    stop();
}

// ---------------------------------------------------------------------------
// start
// ---------------------------------------------------------------------------
// Opens a TCP listener socket, binds it to the configured host:port, and
// spawns a background thread that loops on accept().  On success the actual
// bound port is stored in bound_port_ (useful when port 0 is requested).
// The running_ flag is set atomically so that concurrent start() calls are
// rejected with an error.
// ---------------------------------------------------------------------------
Result<void> InferenceServer::start() {
    // 1. Atomically claim the running slot; reject double-start.
    if (running_.exchange(true)) {
        return Result<void>::err("Server already running");
    }

    // 2. Create a TCP stream socket.
    socket_t sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCK) {
        running_.store(false);
        return Result<void>::err("Failed to create socket");
    }

    // 3. Allow address reuse so restarts don't hit TIME_WAIT.
    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char*>(&opt), sizeof(opt));

    // 4. Bind to the configured host and port.
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(config_.port);
    inet_pton(AF_INET, config_.host.c_str(), &addr.sin_addr);

    if (bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        close_socket(sock);
        running_.store(false);
        return Result<void>::err("Failed to bind to " + config_.host + ":" +
                                  std::to_string(config_.port));
    }

    // 5. Mark socket as passive listener.
    if (listen(sock, config_.max_connections) != 0) {
        close_socket(sock);
        running_.store(false);
        return Result<void>::err("Failed to listen on socket");
    }

    // 6. Query the actual bound port (relevant when port 0 was requested).
    sockaddr_in bound_addr{};
    socklen_t addr_len = sizeof(bound_addr);
    getsockname(sock, reinterpret_cast<sockaddr*>(&bound_addr), &addr_len);
    bound_port_ = ntohs(bound_addr.sin_port);

    // 7. Stash the socket handle and spawn the accept thread.
    listen_socket_ = static_cast<intptr_t>(sock);

    listener_thread_ = std::thread([this]() { listener_loop(); });

    LogPrintf("Inference server started on %s:%u",
              config_.host.c_str(), bound_port_);
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// stop
// ---------------------------------------------------------------------------
// Tears down the server gracefully: clears the running_ flag, closes the
// listener socket (which unblocks a pending accept()), and joins the
// listener thread.  Safe to call multiple times.
// ---------------------------------------------------------------------------
void InferenceServer::stop() {
    // 1. Flip running_ to false; bail if already stopped.
    if (!running_.exchange(false)) return;

    // 2. Close the listening socket to unblock accept().
    if (listen_socket_ != -1) {
        close_socket(static_cast<socket_t>(listen_socket_));
        listen_socket_ = -1;
    }

    // 3. Join the listener thread.
    if (listener_thread_.joinable()) {
        listener_thread_.join();
    }

    LogPrintf("Inference server stopped");
}

// ===========================================================================
// Network Loop
// ===========================================================================

// ---------------------------------------------------------------------------
// listener_loop
// ---------------------------------------------------------------------------
// Blocking accept-loop that runs on a dedicated thread.  Each accepted
// connection is dispatched to handle_connection() on a detached thread.
// In production this would use a bounded thread-pool; the detached-thread
// approach is intentionally simple for the initial implementation.
// ---------------------------------------------------------------------------
void InferenceServer::listener_loop() {
    while (running_.load(std::memory_order_relaxed)) {
        // 1. Block until a client connects or the socket is closed.
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        socket_t client = accept(static_cast<socket_t>(listen_socket_),
                                  reinterpret_cast<sockaddr*>(&client_addr),
                                  &client_len);

        if (client == INVALID_SOCK) {
            // 2. accept() failed -- either shutting down or transient error.
            continue;
        }

        // 3. Dispatch to a detached thread for request handling.
        std::thread([this, client]() {
            handle_connection(static_cast<intptr_t>(client));
        }).detach();
    }
}

// ---------------------------------------------------------------------------
// handle_connection
// ---------------------------------------------------------------------------
// Owns the full lifecycle of a single HTTP connection: recv the raw bytes,
// parse the HTTP request, route to the appropriate handler, serialise the
// response, send it back, and close the socket.  The receive buffer is
// capped at 65'536 bytes (64 KiB) which is sufficient for typical
// inference API payloads.
// ---------------------------------------------------------------------------
void InferenceServer::handle_connection(intptr_t client_socket) {
    socket_t sock = static_cast<socket_t>(client_socket);

    // 1. Read the HTTP request (up to 64 KiB).
    std::string raw;
    raw.resize(65'536);
    int n = recv(sock, raw.data(), static_cast<int>(raw.size()), 0);
    if (n <= 0) {
        close_socket(sock);
        return;
    }
    raw.resize(n);

    // 2. Parse the raw bytes into an HttpRequest struct.
    auto req_res = parse_http_request(raw);
    if (req_res.is_err()) {
        auto resp = make_error_response(400, "invalid_request", req_res.error());
        auto resp_str = serialize_http_response(resp);
        send(sock, resp_str.c_str(), static_cast<int>(resp_str.size()), 0);
        close_socket(sock);
        return;
    }

    // 3. Route the request and send the response.
    HttpResponse response = handle_request(req_res.value());
    auto resp_str = serialize_http_response(response);
    send(sock, resp_str.c_str(), static_cast<int>(resp_str.size()), 0);
    close_socket(sock);
}

// ===========================================================================
// Request Routing
// ===========================================================================

// ---------------------------------------------------------------------------
// handle_request
// ---------------------------------------------------------------------------
// Central dispatch table for the OpenAI-compatible REST API.  Routes are
// matched in order:
//   POST /v1/completions       -> handle_completions
//   POST /v1/chat/completions  -> handle_chat_completions
//   GET  /v1/models            -> handle_models
//   GET  /health               -> handle_health
// All routes pass through check_auth() first when an API key is configured.
// ---------------------------------------------------------------------------
HttpResponse InferenceServer::handle_request(const HttpRequest& request) {
    // 1. Verify API-key authentication if configured.
    HttpResponse auth_error;
    if (!check_auth(request, auth_error)) {
        return auth_error;
    }

    // 2. Match method + path against the routing table.
    if (request.method == "POST" && request.path == "/v1/completions") {
        return handle_completions(request);
    }
    if (request.method == "POST" && request.path == "/v1/chat/completions") {
        return handle_chat_completions(request);
    }
    if (request.method == "GET" && request.path == "/v1/models") {
        return handle_models(request);
    }
    if (request.method == "GET" && request.path == "/health") {
        return handle_health(request);
    }

    // 3. No matching route found.
    return make_error_response(404, "not_found", "Unknown endpoint: " + request.path);
}

// ---------------------------------------------------------------------------
// check_auth
// ---------------------------------------------------------------------------
// Validates the Bearer token from the Authorization header against the
// configured api_key.  When no key is configured, all requests are allowed.
// ---------------------------------------------------------------------------
bool InferenceServer::check_auth(const HttpRequest& request, HttpResponse& error_response) {
    // 1. If no API key is configured, skip authentication.
    if (config_.api_key.empty()) return true;

    // 2. Reject if the token is missing or does not match.
    if (request.api_key.empty() || request.api_key != config_.api_key) {
        error_response = make_error_response(401, "unauthorized", "Invalid API key");
        return false;
    }
    return true;
}

// ===========================================================================
// API Handlers
// ===========================================================================

// ---------------------------------------------------------------------------
// handle_completions
// ---------------------------------------------------------------------------
// POST /v1/completions -- OpenAI-compatible text completion endpoint.
// Extracts prompt, sampling parameters from the JSON body, runs inference
// through InferenceEngine::generate_text(), and returns the result in the
// standard OpenAI completion response format with usage statistics.
// ---------------------------------------------------------------------------
HttpResponse InferenceServer::handle_completions(const HttpRequest& request) {
    // 1. Guard: model must be loaded.
    if (!engine_.is_ready()) {
        return make_error_response(503, "model_not_loaded", "Model is not loaded");
    }

    // 2. Extract prompt and sampling parameters from the request body.
    std::string prompt = json_get_string(request.body, "prompt");
    int max_tokens = json_get_int(request.body, "max_tokens", 256);
    float temperature = json_get_float(request.body, "temperature", 1.0f);
    float top_p = json_get_float(request.body, "top_p", 0.9f);
    int top_k = json_get_int(request.body, "top_k", 50);
    float rep_penalty = json_get_float(request.body, "repetition_penalty", 1.0f);

    // 3. Populate the sampler configuration.
    SamplerConfig sampler;
    sampler.temperature = temperature;
    sampler.top_p = top_p;
    sampler.top_k = top_k;
    sampler.repetition_penalty = rep_penalty;

    // 4. Run text generation.
    auto result = engine_.generate_text(prompt, max_tokens, sampler);
    if (result.is_err()) {
        return make_error_response(500, "generation_error", result.error());
    }

    // 5. Count tokens for usage statistics.
    auto tokens_in = engine_.tokenizer().encode(prompt);
    auto tokens_out = engine_.tokenizer().encode(result.value());

    // 6. Build the OpenAI-format JSON response.
    std::string req_id = generate_request_id();
    std::string json = build_completion_json(
        req_id, config_.model_name, result.value(),
        static_cast<int>(tokens_in.size()),
        static_cast<int>(tokens_out.size()),
        "stop");

    HttpResponse resp;
    resp.status_code = 200;
    resp.body = json;
    return resp;
}

// ---------------------------------------------------------------------------
// handle_chat_completions
// ---------------------------------------------------------------------------
// POST /v1/chat/completions -- OpenAI-compatible chat completion endpoint.
// Performs a simplified extraction of the last user message content from the
// messages array (a full implementation would parse the complete array and
// apply a chat template).  Returns the result in the standard OpenAI chat
// completion response format with usage statistics.
// ---------------------------------------------------------------------------
HttpResponse InferenceServer::handle_chat_completions(const HttpRequest& request) {
    // 1. Guard: model must be loaded.
    if (!engine_.is_ready()) {
        return make_error_response(503, "model_not_loaded", "Model is not loaded");
    }

    // 2. Extract the last user message as the prompt (simplified).
    std::string content = json_get_string(request.body, "content");
    if (content.empty()) {
        // 3. Fallback: find last occurrence of "content" in the messages array.
        size_t pos = request.body.rfind("\"content\"");
        if (pos != std::string::npos) {
            content = json_get_string(request.body.substr(pos), "content");
        }
    }

    if (content.empty()) {
        return make_error_response(400, "invalid_request", "No content found in request");
    }

    // 4. Extract sampling parameters.
    int max_tokens = json_get_int(request.body, "max_tokens", 256);
    float temperature = json_get_float(request.body, "temperature", 1.0f);
    float top_p = json_get_float(request.body, "top_p", 0.9f);

    SamplerConfig sampler;
    sampler.temperature = temperature;
    sampler.top_p = top_p;

    // 5. Run text generation.
    auto result = engine_.generate_text(content, max_tokens, sampler);
    if (result.is_err()) {
        return make_error_response(500, "generation_error", result.error());
    }

    // 6. Count tokens for usage statistics.
    auto tokens_in = engine_.tokenizer().encode(content);
    auto tokens_out = engine_.tokenizer().encode(result.value());

    // 7. Build the OpenAI chat-format JSON response.
    std::string req_id = generate_request_id();
    std::string json = build_chat_completion_json(
        req_id, config_.model_name, "assistant", result.value(),
        static_cast<int>(tokens_in.size()),
        static_cast<int>(tokens_out.size()),
        "stop");

    HttpResponse resp;
    resp.status_code = 200;
    resp.body = json;
    return resp;
}

// ---------------------------------------------------------------------------
// handle_models
// ---------------------------------------------------------------------------
// GET /v1/models -- Returns the list of available models in the OpenAI
// List Models response format.  Currently a single model is reported,
// identified by config_.model_name.
// ---------------------------------------------------------------------------
HttpResponse InferenceServer::handle_models(const HttpRequest& /*request*/) {
    std::string json = R"({"object":"list","data":[{"id":")" +
                       config_.model_name +
                       R"(","object":"model","created":0,"owned_by":"rnet"}]})";

    HttpResponse resp;
    resp.status_code = 200;
    resp.body = json;
    return resp;
}

// ---------------------------------------------------------------------------
// handle_health
// ---------------------------------------------------------------------------
// GET /health -- Returns a JSON health-check payload indicating whether the
// model is loaded and ready for inference.  Returns HTTP 200 when ready,
// HTTP 503 when still loading.
// ---------------------------------------------------------------------------
HttpResponse InferenceServer::handle_health(const HttpRequest& /*request*/) {
    bool ready = engine_.is_ready();
    std::string json = R"({"status":")" +
                       std::string(ready ? "ok" : "loading") +
                       R"(","model_loaded":)" +
                       (ready ? "true" : "false") +
                       R"(})";

    HttpResponse resp;
    resp.status_code = ready ? 200 : 503;
    resp.body = json;
    return resp;
}

// ===========================================================================
// HTTP Parsing
// ===========================================================================

// ---------------------------------------------------------------------------
// parse_http_request
// ---------------------------------------------------------------------------
// Parses a raw HTTP/1.1 request byte-string into an HttpRequest struct.
// Algorithm:
//   1. Split on the first \r\n to get the request line.
//   2. Tokenise the request line on spaces: METHOD SP PATH SP VERSION.
//   3. Strip query parameters from the path.
//   4. Iterate subsequent header lines (terminated by \r\n\r\n), extracting
//      Content-Type and Authorization (Bearer token).
//   5. Everything after the blank line is the body.
//   6. Check for "stream":true in the body for SSE streaming requests.
// ---------------------------------------------------------------------------
Result<HttpRequest> InferenceServer::parse_http_request(const std::string& raw) {
    HttpRequest req;

    // 1. Find end of the request line.
    auto line_end = raw.find("\r\n");
    if (line_end == std::string::npos) {
        return Result<HttpRequest>::err("Malformed HTTP request");
    }

    // 2. Parse request line: "METHOD /path HTTP/1.1".
    std::string request_line = raw.substr(0, line_end);
    auto sp1 = request_line.find(' ');
    auto sp2 = request_line.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) {
        return Result<HttpRequest>::err("Malformed request line");
    }

    req.method = request_line.substr(0, sp1);
    req.path = request_line.substr(sp1 + 1, sp2 - sp1 - 1);

    // 3. Strip query parameters from path.
    auto qpos = req.path.find('?');
    if (qpos != std::string::npos) {
        req.path = req.path.substr(0, qpos);
    }

    // 4. Parse headers until the blank line (\r\n\r\n).
    size_t pos = line_end + 2;
    while (pos < raw.size()) {
        auto next_end = raw.find("\r\n", pos);
        if (next_end == std::string::npos || next_end == pos) {
            pos = (next_end == pos) ? next_end + 2 : raw.size();
            break;
        }

        std::string header_line = raw.substr(pos, next_end - pos);
        auto colon = header_line.find(':');
        if (colon != std::string::npos) {
            std::string key = header_line.substr(0, colon);
            std::string value = header_line.substr(colon + 1);
            // Trim leading whitespace from value
            while (!value.empty() && value[0] == ' ') value = value.substr(1);

            // Lowercase key for case-insensitive comparison
            std::string key_lower = key;
            std::transform(key_lower.begin(), key_lower.end(), key_lower.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

            if (key_lower == "content-type") {
                req.content_type = value;
            } else if (key_lower == "authorization") {
                // Extract token from "Bearer <key>"
                if (value.size() > 7 && value.substr(0, 7) == "Bearer ") {
                    req.api_key = value.substr(7);
                }
            }
        }
        pos = next_end + 2;
    }

    // 5. Body is everything after the blank line.
    if (pos < raw.size()) {
        req.body = raw.substr(pos);
    }

    // 6. Check if streaming is requested.
    req.stream = json_get_bool(req.body, "stream", false);

    return Result<HttpRequest>::ok(std::move(req));
}

// ---------------------------------------------------------------------------
// serialize_http_response
// ---------------------------------------------------------------------------
// Serialises an HttpResponse into a complete HTTP/1.1 response byte-string
// including status line, headers (Content-Type, Content-Length, CORS, and
// Connection: close), and body.
// ---------------------------------------------------------------------------
std::string InferenceServer::serialize_http_response(const HttpResponse& response) {
    std::ostringstream ss;

    // 1. Write status line.
    ss << "HTTP/1.1 " << response.status_code << " ";

    switch (response.status_code) {
        case 200: ss << "OK"; break;
        case 400: ss << "Bad Request"; break;
        case 401: ss << "Unauthorized"; break;
        case 404: ss << "Not Found"; break;
        case 500: ss << "Internal Server Error"; break;
        case 503: ss << "Service Unavailable"; break;
        default: ss << "Unknown"; break;
    }

    // 2. Write response headers.
    ss << "\r\n";
    ss << "Content-Type: " << response.content_type << "\r\n";
    ss << "Content-Length: " << response.body.size() << "\r\n";
    ss << "Access-Control-Allow-Origin: *\r\n";
    ss << "Connection: close\r\n";
    ss << "\r\n";

    // 3. Append body.
    ss << response.body;

    return ss.str();
}

// ===========================================================================
// JSON Helpers
// ===========================================================================

// ---------------------------------------------------------------------------
// json_get_string
// ---------------------------------------------------------------------------
// Minimal JSON string-value extractor (no external dependency).  Searches
// for the key in the raw JSON, skips the colon and whitespace, then reads
// the quoted value while un-escaping \\, \", \n, \t, and \r sequences per
// RFC 8259 section 7.
// ---------------------------------------------------------------------------
std::string InferenceServer::json_get_string(const std::string& json,
                                              const std::string& key) {
    // 1. Search for the quoted key.
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";

    // 2. Skip past the key, colon, and surrounding whitespace.
    pos += search.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == ':' || json[pos] == '\t' || json[pos] == '\n' || json[pos] == '\r')) ++pos;

    // 3. Expect opening double-quote.
    if (pos >= json.size() || json[pos] != '"') return "";
    ++pos;

    // 4. Read characters, un-escaping per RFC 8259.
    std::string result;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            ++pos;
            switch (json[pos]) {
                case '"':  result += '"'; break;
                case '\\': result += '\\'; break;
                case 'n':  result += '\n'; break;
                case 't':  result += '\t'; break;
                case 'r':  result += '\r'; break;
                default:   result += json[pos]; break;
            }
        } else {
            result += json[pos];
        }
        ++pos;
    }
    return result;
}

// ---------------------------------------------------------------------------
// json_get_int
// ---------------------------------------------------------------------------
// Extracts an integer value for the given key using std::from_chars.
// Returns default_val if the key is missing or the value is not a valid
// integer literal.
// ---------------------------------------------------------------------------
int InferenceServer::json_get_int(const std::string& json, const std::string& key,
                                   int default_val) {
    // 1. Locate the key.
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return default_val;

    // 2. Skip past key, colon, whitespace.
    pos += search.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == ':' || json[pos] == '\t' || json[pos] == '\n' || json[pos] == '\r')) ++pos;

    // 3. Parse the integer with from_chars.
    int result = default_val;
    auto [ptr, ec] = std::from_chars(json.data() + pos,
                                      json.data() + json.size(),
                                      result);
    if (ec != std::errc{}) return default_val;
    return result;
}

// ---------------------------------------------------------------------------
// json_get_float
// ---------------------------------------------------------------------------
// Extracts a floating-point value for the given key using strtof (since
// std::from_chars for float is not universally available across all
// compilers).  Returns default_val if the key is missing or the value
// cannot be parsed.
// ---------------------------------------------------------------------------
float InferenceServer::json_get_float(const std::string& json, const std::string& key,
                                       float default_val) {
    // 1. Locate the key.
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return default_val;

    // 2. Skip past key, colon, whitespace.
    pos += search.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == ':' || json[pos] == '\t' || json[pos] == '\n' || json[pos] == '\r')) ++pos;

    // 3. Parse via strtof for maximum compiler compatibility.
    const char* start = json.data() + pos;
    char* end = nullptr;
    float result = std::strtof(start, &end);
    if (end == start) return default_val;
    return result;
}

// ---------------------------------------------------------------------------
// json_get_bool
// ---------------------------------------------------------------------------
// Extracts a boolean value for the given key by matching the literal strings
// "true" or "false".  Returns default_val if the key is missing or the
// value is neither literal.
// ---------------------------------------------------------------------------
bool InferenceServer::json_get_bool(const std::string& json, const std::string& key,
                                     bool default_val) {
    // 1. Locate the key.
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return default_val;

    // 2. Skip past key, colon, whitespace.
    pos += search.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == ':' || json[pos] == '\t' || json[pos] == '\n' || json[pos] == '\r')) ++pos;

    // 3. Match "true" or "false" literals.
    if (pos + 4 <= json.size() && json.substr(pos, 4) == "true") return true;
    if (pos + 5 <= json.size() && json.substr(pos, 5) == "false") return false;
    return default_val;
}

// ===========================================================================
// JSON Builders
// ===========================================================================

// ---------------------------------------------------------------------------
// make_error_response
// ---------------------------------------------------------------------------
// Constructs an HttpResponse with the given HTTP status code and a JSON
// error body matching the OpenAI error envelope: {"error":{"type":...,"message":...}}.
// ---------------------------------------------------------------------------
HttpResponse InferenceServer::make_error_response(int status_code,
                                                    const std::string& error_type,
                                                    const std::string& message) {
    HttpResponse resp;
    resp.status_code = status_code;
    resp.body = R"({"error":{"type":")" + error_type +
                R"(","message":")" + message + R"("}})";
    return resp;
}

// ---------------------------------------------------------------------------
// escape_json_string
// ---------------------------------------------------------------------------
// Escapes a raw string for safe embedding inside a JSON double-quoted value
// per RFC 8259 section 7.  Handles the mandatory escapes (backslash,
// double-quote, and control characters U+0000..U+001F) plus the common
// whitespace shortcuts (\n, \r, \t).  Control characters below U+0020 that
// lack a shortcut are emitted as \uXXXX.
// ---------------------------------------------------------------------------
static std::string escape_json_string(const std::string& s) {
    std::string result;
    result.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"':  result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(c));
                    result += buf;
                } else {
                    result += c;
                }
                break;
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// build_completion_json
// ---------------------------------------------------------------------------
// Builds a JSON response body matching the OpenAI text-completion format:
//   { id, object:"text_completion", created, model, choices:[{text,...}],
//     usage:{prompt_tokens, completion_tokens, total_tokens} }
// The "created" timestamp is the current Unix epoch in seconds.
// ---------------------------------------------------------------------------
std::string InferenceServer::build_completion_json(const std::string& id,
                                                    const std::string& model,
                                                    const std::string& text,
                                                    int prompt_tokens,
                                                    int completion_tokens,
                                                    const std::string& finish_reason) {
    // 1. Capture current Unix epoch timestamp (seconds).
    auto now = std::chrono::system_clock::now();
    auto epoch = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count();

    // 2. Assemble the JSON envelope.
    std::ostringstream ss;
    ss << R"({"id":")" << id
       << R"(","object":"text_completion","created":)" << epoch
       << R"(,"model":")" << model
       << R"(","choices":[{"text":")" << escape_json_string(text)
       << R"(","index":0,"finish_reason":")" << finish_reason
       << R"("}],"usage":{"prompt_tokens":)" << prompt_tokens
       << R"(,"completion_tokens":)" << completion_tokens
       << R"(,"total_tokens":)" << (prompt_tokens + completion_tokens)
       << R"(}})";
    return ss.str();
}

// ---------------------------------------------------------------------------
// build_chat_completion_json
// ---------------------------------------------------------------------------
// Builds a JSON response body matching the OpenAI chat-completion format:
//   { id, object:"chat.completion", created, model,
//     choices:[{index, message:{role, content}, finish_reason}],
//     usage:{prompt_tokens, completion_tokens, total_tokens} }
// The "created" timestamp is the current Unix epoch in seconds.
// ---------------------------------------------------------------------------
std::string InferenceServer::build_chat_completion_json(const std::string& id,
                                                         const std::string& model,
                                                         const std::string& role,
                                                         const std::string& content,
                                                         int prompt_tokens,
                                                         int completion_tokens,
                                                         const std::string& finish_reason) {
    // 1. Capture current Unix epoch timestamp (seconds).
    auto now = std::chrono::system_clock::now();
    auto epoch = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count();

    // 2. Assemble the JSON envelope.
    std::ostringstream ss;
    ss << R"({"id":")" << id
       << R"(","object":"chat.completion","created":)" << epoch
       << R"(,"model":")" << model
       << R"(","choices":[{"index":0,"message":{"role":")" << role
       << R"(","content":")" << escape_json_string(content)
       << R"("},"finish_reason":")" << finish_reason
       << R"("}],"usage":{"prompt_tokens":)" << prompt_tokens
       << R"(,"completion_tokens":)" << completion_tokens
       << R"(,"total_tokens":)" << (prompt_tokens + completion_tokens)
       << R"(}})";
    return ss.str();
}

// ---------------------------------------------------------------------------
// generate_request_id
// ---------------------------------------------------------------------------
// Produces a unique request identifier of the form "cmpl-<hex-timestamp>-<seq>"
// using a monotonic clock and an atomic counter.  The combination guarantees
// uniqueness within a single process without external coordination.
// ---------------------------------------------------------------------------
std::string InferenceServer::generate_request_id() {
    static std::atomic<uint64_t> counter{0};
    // 1. Sample monotonic clock for the timestamp component.
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    // 2. Atomically increment the sequence counter.
    uint64_t id = counter.fetch_add(1);
    // 3. Format as "cmpl-<hex_timestamp>-<seq>".
    std::ostringstream ss;
    ss << "cmpl-" << std::hex << now << "-" << id;
    return ss.str();
}

} // namespace rnet::inference

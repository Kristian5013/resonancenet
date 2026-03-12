#include "inference/server.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstring>
#include <sstream>

#include "core/logging.h"

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

// ────────────────────────────────────────────────────────────────────
// Construction / destruction
// ────────────────────────────────────────────────────────────────────

InferenceServer::InferenceServer(InferenceEngine& engine,
                                  BatchScheduler& scheduler,
                                  const ServerConfig& config)
    : engine_(engine), scheduler_(scheduler), config_(config) {}

InferenceServer::~InferenceServer() {
    stop();
}

// ────────────────────────────────────────────────────────────────────
// Start / stop
// ────────────────────────────────────────────────────────────────────

Result<void> InferenceServer::start() {
    if (running_.exchange(true)) {
        return Result<void>::err("Server already running");
    }

    socket_t sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCK) {
        running_.store(false);
        return Result<void>::err("Failed to create socket");
    }

    // Allow address reuse
    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char*>(&opt), sizeof(opt));

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

    if (listen(sock, config_.max_connections) != 0) {
        close_socket(sock);
        running_.store(false);
        return Result<void>::err("Failed to listen on socket");
    }

    // Get actual bound port
    sockaddr_in bound_addr{};
    socklen_t addr_len = sizeof(bound_addr);
    getsockname(sock, reinterpret_cast<sockaddr*>(&bound_addr), &addr_len);
    bound_port_ = ntohs(bound_addr.sin_port);

    listen_socket_ = static_cast<intptr_t>(sock);

    listener_thread_ = std::thread([this]() { listener_loop(); });

    LogPrintf("Inference server started on %s:%u",
              config_.host.c_str(), bound_port_);
    return Result<void>::ok();
}

void InferenceServer::stop() {
    if (!running_.exchange(false)) return;

    // Close listening socket to unblock accept()
    if (listen_socket_ != -1) {
        close_socket(static_cast<socket_t>(listen_socket_));
        listen_socket_ = -1;
    }

    if (listener_thread_.joinable()) {
        listener_thread_.join();
    }

    LogPrintf("Inference server stopped");
}

// ────────────────────────────────────────────────────────────────────
// Network loop
// ────────────────────────────────────────────────────────────────────

void InferenceServer::listener_loop() {
    while (running_.load(std::memory_order_relaxed)) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        socket_t client = accept(static_cast<socket_t>(listen_socket_),
                                  reinterpret_cast<sockaddr*>(&client_addr),
                                  &client_len);

        if (client == INVALID_SOCK) {
            // accept() failed — either we are shutting down or error
            continue;
        }

        // Handle in a detached thread (simple approach; production would use a thread pool)
        std::thread([this, client]() {
            handle_connection(static_cast<intptr_t>(client));
        }).detach();
    }
}

void InferenceServer::handle_connection(intptr_t client_socket) {
    socket_t sock = static_cast<socket_t>(client_socket);

    // Read the HTTP request (up to 64KB)
    std::string raw;
    raw.resize(65536);
    int n = recv(sock, raw.data(), static_cast<int>(raw.size()), 0);
    if (n <= 0) {
        close_socket(sock);
        return;
    }
    raw.resize(n);

    auto req_res = parse_http_request(raw);
    if (req_res.is_err()) {
        auto resp = make_error_response(400, "invalid_request", req_res.error());
        auto resp_str = serialize_http_response(resp);
        send(sock, resp_str.c_str(), static_cast<int>(resp_str.size()), 0);
        close_socket(sock);
        return;
    }

    HttpResponse response = handle_request(req_res.value());
    auto resp_str = serialize_http_response(response);
    send(sock, resp_str.c_str(), static_cast<int>(resp_str.size()), 0);
    close_socket(sock);
}

// ────────────────────────────────────────────────────────────────────
// Request routing
// ────────────────────────────────────────────────────────────────────

HttpResponse InferenceServer::handle_request(const HttpRequest& request) {
    // Auth check
    HttpResponse auth_error;
    if (!check_auth(request, auth_error)) {
        return auth_error;
    }

    // Route
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

    return make_error_response(404, "not_found", "Unknown endpoint: " + request.path);
}

bool InferenceServer::check_auth(const HttpRequest& request, HttpResponse& error_response) {
    if (config_.api_key.empty()) return true;  // No auth required

    if (request.api_key.empty() || request.api_key != config_.api_key) {
        error_response = make_error_response(401, "unauthorized", "Invalid API key");
        return false;
    }
    return true;
}

// ────────────────────────────────────────────────────────────────────
// POST /v1/completions
// ────────────────────────────────────────────────────────────────────

HttpResponse InferenceServer::handle_completions(const HttpRequest& request) {
    if (!engine_.is_ready()) {
        return make_error_response(503, "model_not_loaded", "Model is not loaded");
    }

    std::string prompt = json_get_string(request.body, "prompt");
    int max_tokens = json_get_int(request.body, "max_tokens", 256);
    float temperature = json_get_float(request.body, "temperature", 1.0f);
    float top_p = json_get_float(request.body, "top_p", 0.9f);
    int top_k = json_get_int(request.body, "top_k", 50);
    float rep_penalty = json_get_float(request.body, "repetition_penalty", 1.0f);

    SamplerConfig sampler;
    sampler.temperature = temperature;
    sampler.top_p = top_p;
    sampler.top_k = top_k;
    sampler.repetition_penalty = rep_penalty;

    auto result = engine_.generate_text(prompt, max_tokens, sampler);
    if (result.is_err()) {
        return make_error_response(500, "generation_error", result.error());
    }

    auto tokens_in = engine_.tokenizer().encode(prompt);
    // Approximate output token count
    auto tokens_out = engine_.tokenizer().encode(result.value());

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

// ────────────────────────────────────────────────────────────────────
// POST /v1/chat/completions
// ────────────────────────────────────────────────────────────────────

HttpResponse InferenceServer::handle_chat_completions(const HttpRequest& request) {
    if (!engine_.is_ready()) {
        return make_error_response(503, "model_not_loaded", "Model is not loaded");
    }

    // Extract the last user message as the prompt (simplified chat handling).
    // A full implementation would parse the messages array and apply a chat template.
    // For now, we look for the last "content" field.
    std::string content = json_get_string(request.body, "content");
    if (content.empty()) {
        // Try to find content in a messages array (very basic extraction)
        // Look for last occurrence of "content":"..."
        size_t pos = request.body.rfind("\"content\"");
        if (pos != std::string::npos) {
            content = json_get_string(request.body.substr(pos), "content");
        }
    }

    if (content.empty()) {
        return make_error_response(400, "invalid_request", "No content found in request");
    }

    int max_tokens = json_get_int(request.body, "max_tokens", 256);
    float temperature = json_get_float(request.body, "temperature", 1.0f);
    float top_p = json_get_float(request.body, "top_p", 0.9f);

    SamplerConfig sampler;
    sampler.temperature = temperature;
    sampler.top_p = top_p;

    auto result = engine_.generate_text(content, max_tokens, sampler);
    if (result.is_err()) {
        return make_error_response(500, "generation_error", result.error());
    }

    auto tokens_in = engine_.tokenizer().encode(content);
    auto tokens_out = engine_.tokenizer().encode(result.value());

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

// ────────────────────────────────────────────────────────────────────
// GET /v1/models
// ────────────────────────────────────────────────────────────────────

HttpResponse InferenceServer::handle_models(const HttpRequest& /*request*/) {
    std::string json = R"({"object":"list","data":[{"id":")" +
                       config_.model_name +
                       R"(","object":"model","created":0,"owned_by":"rnet"}]})";

    HttpResponse resp;
    resp.status_code = 200;
    resp.body = json;
    return resp;
}

// ────────────────────────────────────────────────────────────────────
// GET /health
// ────────────────────────────────────────────────────────────────────

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

// ────────────────────────────────────────────────────────────────────
// HTTP parsing / serialization
// ────────────────────────────────────────────────────────────────────

Result<HttpRequest> InferenceServer::parse_http_request(const std::string& raw) {
    HttpRequest req;

    // Find end of first line
    auto line_end = raw.find("\r\n");
    if (line_end == std::string::npos) {
        return Result<HttpRequest>::err("Malformed HTTP request");
    }

    // Parse request line: "METHOD /path HTTP/1.1"
    std::string request_line = raw.substr(0, line_end);
    auto sp1 = request_line.find(' ');
    auto sp2 = request_line.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) {
        return Result<HttpRequest>::err("Malformed request line");
    }

    req.method = request_line.substr(0, sp1);
    req.path = request_line.substr(sp1 + 1, sp2 - sp1 - 1);

    // Strip query parameters from path
    auto qpos = req.path.find('?');
    if (qpos != std::string::npos) {
        req.path = req.path.substr(0, qpos);
    }

    // Parse headers
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

            // Lowercase key for comparison
            std::string key_lower = key;
            std::transform(key_lower.begin(), key_lower.end(), key_lower.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

            if (key_lower == "content-type") {
                req.content_type = value;
            } else if (key_lower == "authorization") {
                // "Bearer <key>"
                if (value.size() > 7 && value.substr(0, 7) == "Bearer ") {
                    req.api_key = value.substr(7);
                }
            }
        }
        pos = next_end + 2;
    }

    // Body is everything after the blank line
    if (pos < raw.size()) {
        req.body = raw.substr(pos);
    }

    // Check if streaming is requested
    req.stream = json_get_bool(req.body, "stream", false);

    return Result<HttpRequest>::ok(std::move(req));
}

std::string InferenceServer::serialize_http_response(const HttpResponse& response) {
    std::ostringstream ss;
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

    ss << "\r\n";
    ss << "Content-Type: " << response.content_type << "\r\n";
    ss << "Content-Length: " << response.body.size() << "\r\n";
    ss << "Access-Control-Allow-Origin: *\r\n";
    ss << "Connection: close\r\n";
    ss << "\r\n";
    ss << response.body;

    return ss.str();
}

// ────────────────────────────────────────────────────────────────────
// JSON helpers (minimal, no external dependency)
// ────────────────────────────────────────────────────────────────────

std::string InferenceServer::json_get_string(const std::string& json,
                                              const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";

    pos += search.size();
    // Skip whitespace and colon
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == ':' || json[pos] == '\t' || json[pos] == '\n' || json[pos] == '\r')) ++pos;

    if (pos >= json.size() || json[pos] != '"') return "";
    ++pos;  // Skip opening quote

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

int InferenceServer::json_get_int(const std::string& json, const std::string& key,
                                   int default_val) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return default_val;

    pos += search.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == ':' || json[pos] == '\t' || json[pos] == '\n' || json[pos] == '\r')) ++pos;

    int result = default_val;
    auto [ptr, ec] = std::from_chars(json.data() + pos,
                                      json.data() + json.size(),
                                      result);
    if (ec != std::errc{}) return default_val;
    return result;
}

float InferenceServer::json_get_float(const std::string& json, const std::string& key,
                                       float default_val) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return default_val;

    pos += search.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == ':' || json[pos] == '\t' || json[pos] == '\n' || json[pos] == '\r')) ++pos;

    // Use strtof since from_chars for float is not universally available on all compilers
    const char* start = json.data() + pos;
    char* end = nullptr;
    float result = std::strtof(start, &end);
    if (end == start) return default_val;
    return result;
}

bool InferenceServer::json_get_bool(const std::string& json, const std::string& key,
                                     bool default_val) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return default_val;

    pos += search.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == ':' || json[pos] == '\t' || json[pos] == '\n' || json[pos] == '\r')) ++pos;

    if (pos + 4 <= json.size() && json.substr(pos, 4) == "true") return true;
    if (pos + 5 <= json.size() && json.substr(pos, 5) == "false") return false;
    return default_val;
}

// ────────────────────────────────────────────────────────────────────
// JSON builders
// ────────────────────────────────────────────────────────────────────

HttpResponse InferenceServer::make_error_response(int status_code,
                                                    const std::string& error_type,
                                                    const std::string& message) {
    HttpResponse resp;
    resp.status_code = status_code;
    resp.body = R"({"error":{"type":")" + error_type +
                R"(","message":")" + message + R"("}})";
    return resp;
}

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

std::string InferenceServer::build_completion_json(const std::string& id,
                                                    const std::string& model,
                                                    const std::string& text,
                                                    int prompt_tokens,
                                                    int completion_tokens,
                                                    const std::string& finish_reason) {
    auto now = std::chrono::system_clock::now();
    auto epoch = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count();

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

std::string InferenceServer::build_chat_completion_json(const std::string& id,
                                                         const std::string& model,
                                                         const std::string& role,
                                                         const std::string& content,
                                                         int prompt_tokens,
                                                         int completion_tokens,
                                                         const std::string& finish_reason) {
    auto now = std::chrono::system_clock::now();
    auto epoch = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count();

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

std::string InferenceServer::generate_request_id() {
    static std::atomic<uint64_t> counter{0};
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    uint64_t id = counter.fetch_add(1);
    std::ostringstream ss;
    ss << "cmpl-" << std::hex << now << "-" << id;
    return ss.str();
}

}  // namespace rnet::inference

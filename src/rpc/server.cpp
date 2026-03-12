#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using socket_t = SOCKET;
static constexpr socket_t INVALID_SOCK = INVALID_SOCKET;
#define CLOSE_SOCKET closesocket
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
using socket_t = int;
static constexpr socket_t INVALID_SOCK = -1;
#define CLOSE_SOCKET close
#endif

#include "rpc/server.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>

#include "core/logging.h"
#include "core/random.h"
#include "core/time.h"
#include "node/context.h"

namespace rnet::rpc {

// ── Winsock init helper ─────────────────────────────────────────────

#ifdef _WIN32
namespace {
struct WinsockInit {
    WinsockInit() {
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
    }
    ~WinsockInit() { WSACleanup(); }
};
static WinsockInit g_winsock_init;
}  // anonymous namespace
#endif

// ── RPCServer lifecycle ─────────────────────────────────────────────

RPCServer::RPCServer() = default;

RPCServer::~RPCServer() {
    stop();
}

void RPCServer::set_credentials(const std::string& user,
                                const std::string& password) {
    rpc_user_ = user;
    rpc_password_ = password;
}

bool RPCServer::start(uint16_t port, const std::string& bind_addr) {
    if (running_.load()) return false;

    port_ = port;

    // Create listening socket
    socket_t sock = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCK) {
        LogError("RPC: failed to create listen socket");
        return false;
    }

    // Allow address reuse
    int optval = 1;
    ::setsockopt(sock, SOL_SOCKET, SO_REUSEADDR,
                 reinterpret_cast<const char*>(&optval), sizeof(optval));

    // Bind
    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    ::inet_pton(AF_INET, bind_addr.c_str(), &addr.sin_addr);

    if (::bind(sock, reinterpret_cast<struct sockaddr*>(&addr),
               sizeof(addr)) != 0) {
        LogError("RPC: failed to bind to %s:%d", bind_addr.c_str(), port);
        CLOSE_SOCKET(sock);
        return false;
    }

    if (::listen(sock, 16) != 0) {
        LogError("RPC: failed to listen on port %d", port);
        CLOSE_SOCKET(sock);
        return false;
    }

    listen_sock_ = static_cast<int64_t>(sock);
    startup_time_ = core::get_time();

    // Generate auth cookie
    generate_cookie();

    running_.store(true);
    server_thread_ = std::thread([this]() { accept_loop(); });

    LogPrint(RPC, "RPC server listening on %s:%d", bind_addr.c_str(), port);
    return true;
}

void RPCServer::stop() {
    if (!running_.load()) return;

    running_.store(false);

    // Close the listen socket to unblock accept()
    if (listen_sock_ >= 0) {
        CLOSE_SOCKET(static_cast<socket_t>(listen_sock_));
        listen_sock_ = -1;
    }

    if (server_thread_.joinable()) {
        server_thread_.join();
    }

    delete_cookie();
    LogPrint(RPC, "RPC server stopped");
}

// ── Accept loop ─────────────────────────────────────────────────────

void RPCServer::accept_loop() {
    while (running_.load()) {
        struct sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);

        socket_t client = ::accept(
            static_cast<socket_t>(listen_sock_),
            reinterpret_cast<struct sockaddr*>(&client_addr),
            &client_len);

        if (client == INVALID_SOCK) {
            if (!running_.load()) break;  // shutdown
            continue;
        }

        // Set a receive timeout so we don't block forever
#ifdef _WIN32
        DWORD timeout_ms = 30000;
        ::setsockopt(client, SOL_SOCKET, SO_RCVTIMEO,
                     reinterpret_cast<const char*>(&timeout_ms),
                     sizeof(timeout_ms));
#else
        struct timeval tv;
        tv.tv_sec = 30;
        tv.tv_usec = 0;
        ::setsockopt(client, SOL_SOCKET, SO_RCVTIMEO,
                     reinterpret_cast<const char*>(&tv), sizeof(tv));
#endif

        // Handle synchronously (single-threaded for simplicity)
        handle_connection(static_cast<int64_t>(client));
        CLOSE_SOCKET(client);
    }
}

// ── Connection handling ─────────────────────────────────────────────

void RPCServer::handle_connection(int64_t client_sock) {
    HttpRequest http_req;
    if (!read_http_request(client_sock, http_req)) {
        send_http_response(client_sock, 400, "Bad Request",
                           "{\"error\":\"malformed HTTP request\"}");
        return;
    }

    // Only accept POST for JSON-RPC
    if (http_req.method != "POST") {
        send_http_response(client_sock, 405, "Method Not Allowed",
                           "{\"error\":\"use POST for JSON-RPC\"}");
        return;
    }

    // Authenticate
    if (!authenticate(http_req)) {
        send_http_response(client_sock, 401, "Unauthorized",
                           "{\"error\":\"authentication failed\"}");
        return;
    }

    // Parse JSON-RPC request
    JsonValue json;
    if (!parse_json(http_req.body, json)) {
        auto resp = RPCResponse::make_error(RPC_PARSE_ERROR,
                                            "JSON parse error", JsonValue());
        send_http_response(client_sock, 200, "OK", resp.to_json());
        return;
    }

    // Check for batch requests (array of requests)
    if (json.is_array()) {
        JsonValue results = JsonValue::array();
        for (size_t i = 0; i < json.size(); ++i) {
            RPCRequest rpc_req;
            if (!RPCRequest::from_json(json[i], rpc_req)) {
                results.push_back(parse_json(
                    RPCResponse::make_error(RPC_INVALID_REQUEST,
                                            "invalid request",
                                            JsonValue()).to_json()));
            } else {
                auto resp = process_request(rpc_req);
                JsonValue resp_json;
                parse_json(resp.to_json(), resp_json);
                results.push_back(std::move(resp_json));
            }
        }
        send_http_response(client_sock, 200, "OK", results.to_string());
        return;
    }

    RPCRequest rpc_req;
    if (!RPCRequest::from_json(json, rpc_req)) {
        auto resp = RPCResponse::make_error(RPC_INVALID_REQUEST,
                                            "invalid JSON-RPC request",
                                            JsonValue());
        send_http_response(client_sock, 200, "OK", resp.to_json());
        return;
    }

    auto resp = process_request(rpc_req);
    send_http_response(client_sock, 200, "OK", resp.to_json());
}

RPCResponse RPCServer::process_request(const RPCRequest& rpc_req) {
    LogPrint(RPC, "RPC call: %s", rpc_req.method.c_str());

    const RPCCommand* cmd = table_.find(rpc_req.method);
    if (!cmd) {
        return RPCResponse::make_error(
            RPC_METHOD_NOT_FOUND,
            "Method not found: " + rpc_req.method,
            rpc_req.id);
    }

    if (!ctx_) {
        return RPCResponse::make_error(
            RPC_INTERNAL_ERROR,
            "Server not fully initialized",
            rpc_req.id);
    }

    try {
        JsonValue result = cmd->handler(rpc_req, *ctx_);

        // Check if the handler returned an error object
        if (result.is_object() && result.has_key("code") &&
            result.has_key("message")) {
            RPCResponse resp;
            resp.result = JsonValue();
            resp.error = std::move(result);
            resp.id = rpc_req.id;
            return resp;
        }

        return RPCResponse::success(std::move(result), rpc_req.id);
    } catch (const std::exception& e) {
        LogError("RPC: exception in %s: %s",
                 rpc_req.method.c_str(), e.what());
        return RPCResponse::make_error(
            RPC_INTERNAL_ERROR,
            std::string("internal error: ") + e.what(),
            rpc_req.id);
    }
}

// ── HTTP parsing ────────────────────────────────────────────────────

bool RPCServer::read_http_request(int64_t sock, HttpRequest& req) {
    // Read data into buffer (up to 4MB)
    static constexpr size_t MAX_REQUEST_SIZE = 4 * 1024 * 1024;
    std::string buf;
    buf.reserve(8192);

    char chunk[4096];
    bool headers_done = false;

    while (buf.size() < MAX_REQUEST_SIZE) {
        int n = ::recv(static_cast<socket_t>(sock), chunk, sizeof(chunk), 0);
        if (n <= 0) break;
        buf.append(chunk, static_cast<size_t>(n));

        // Check if we have the full headers
        auto header_end = buf.find("\r\n\r\n");
        if (header_end != std::string::npos) {
            headers_done = true;

            // Parse content-length from headers
            std::string headers_str = buf.substr(0, header_end);

            // Find Content-Length
            req.content_length = 0;
            auto cl_pos = headers_str.find("Content-Length:");
            if (cl_pos == std::string::npos) {
                cl_pos = headers_str.find("content-length:");
            }
            if (cl_pos != std::string::npos) {
                auto val_start = cl_pos + 15;  // length of "Content-Length:"
                while (val_start < headers_str.size() &&
                       headers_str[val_start] == ' ') {
                    ++val_start;
                }
                auto val_end = headers_str.find("\r\n", val_start);
                if (val_end == std::string::npos) val_end = headers_str.size();
                std::string cl_str = headers_str.substr(
                    val_start, val_end - val_start);
                try {
                    req.content_length = std::stoll(cl_str);
                } catch (...) {
                    req.content_length = 0;
                }
            }

            // Check if we have the full body
            size_t body_start = header_end + 4;
            size_t body_received = buf.size() - body_start;
            if (static_cast<int64_t>(body_received) >=
                req.content_length) {
                break;
            }
        }
    }

    if (!headers_done) return false;

    // Parse request line
    auto first_line_end = buf.find("\r\n");
    if (first_line_end == std::string::npos) return false;

    std::string first_line = buf.substr(0, first_line_end);

    // "POST / HTTP/1.1"
    auto sp1 = first_line.find(' ');
    if (sp1 == std::string::npos) return false;
    req.method = first_line.substr(0, sp1);

    auto sp2 = first_line.find(' ', sp1 + 1);
    if (sp2 == std::string::npos) return false;
    req.uri = first_line.substr(sp1 + 1, sp2 - sp1 - 1);

    // Parse headers
    auto header_end = buf.find("\r\n\r\n");
    std::string header_section = buf.substr(first_line_end + 2,
                                            header_end - first_line_end - 2);

    // Extract Authorization header
    size_t pos = 0;
    while (pos < header_section.size()) {
        auto line_end = header_section.find("\r\n", pos);
        if (line_end == std::string::npos) line_end = header_section.size();
        std::string line = header_section.substr(pos, line_end - pos);

        auto colon = line.find(':');
        if (colon != std::string::npos) {
            std::string key = line.substr(0, colon);
            std::string val = line.substr(colon + 1);
            // Trim leading spaces
            while (!val.empty() && val[0] == ' ') val.erase(0, 1);

            // Case-insensitive comparison
            std::string key_lower = key;
            for (auto& c : key_lower) c = static_cast<char>(std::tolower(c));

            if (key_lower == "authorization") {
                req.authorization = val;
            } else if (key_lower == "content-type") {
                req.content_type = val;
            }
        }

        pos = line_end + 2;
    }

    // Extract body
    size_t body_start = header_end + 4;
    if (body_start < buf.size()) {
        req.body = buf.substr(body_start);
    }

    return true;
}

void RPCServer::send_http_response(int64_t sock, int status_code,
                                   const std::string& status_text,
                                   const std::string& body) {
    std::ostringstream response;
    response << "HTTP/1.1 " << status_code << " " << status_text << "\r\n";
    response << "Content-Type: application/json\r\n";
    response << "Content-Length: " << body.size() << "\r\n";
    response << "Connection: close\r\n";
    response << "\r\n";
    response << body;

    std::string resp_str = response.str();
    const char* data = resp_str.c_str();
    size_t remaining = resp_str.size();

    while (remaining > 0) {
        int sent = ::send(static_cast<socket_t>(sock), data,
                          static_cast<int>(remaining), 0);
        if (sent <= 0) break;
        data += sent;
        remaining -= static_cast<size_t>(sent);
    }
}

// ── Authentication ──────────────────────────────────────────────────

bool RPCServer::authenticate(const HttpRequest& req) {
    if (req.authorization.empty()) {
        // No auth header — check if auth is required
        if (rpc_user_.empty() && cookie_password_.empty()) {
            return true;  // No auth configured
        }
        return false;
    }

    // Expect "Basic <base64(user:password)>"
    if (req.authorization.substr(0, 6) != "Basic ") {
        return false;
    }

    std::string decoded = base64_decode(req.authorization.substr(6));
    auto colon = decoded.find(':');
    if (colon == std::string::npos) return false;

    std::string user = decoded.substr(0, colon);
    std::string pass = decoded.substr(colon + 1);

    // Check rpcuser/rpcpassword
    if (!rpc_user_.empty()) {
        if (user == rpc_user_ && pass == rpc_password_) {
            return true;
        }
    }

    // Check cookie auth
    if (!cookie_password_.empty()) {
        if (user == cookie_user_ && pass == cookie_password_) {
            return true;
        }
    }

    return false;
}

bool RPCServer::generate_cookie() {
    cookie_user_ = "__cookie__";
    cookie_password_ = core::random_hex_string(32);  // 32 random bytes -> 64 hex chars

    if (data_dir_.empty()) return false;

    auto cookie_path = data_dir_ / COOKIE_FILE_NAME;
    std::ofstream out(cookie_path, std::ios::trunc);
    if (!out) {
        LogError("RPC: failed to write cookie file: %s",
                 cookie_path.string().c_str());
        return false;
    }

    out << cookie_user_ << ":" << cookie_password_ << std::endl;
    out.close();

    LogPrint(RPC, "RPC cookie written to %s", cookie_path.string().c_str());
    return true;
}

void RPCServer::delete_cookie() {
    if (data_dir_.empty()) return;
    auto cookie_path = data_dir_ / COOKIE_FILE_NAME;
    std::error_code ec;
    std::filesystem::remove(cookie_path, ec);
}

// ── Base64 ──────────────────────────────────────────────────────────

static constexpr char BASE64_CHARS[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string RPCServer::base64_encode(const std::string& input) {
    std::string out;
    size_t i = 0;
    uint8_t arr3[3];
    uint8_t arr4[4];
    size_t in_len = input.size();
    const auto* bytes = reinterpret_cast<const uint8_t*>(input.data());

    while (in_len--) {
        arr3[i++] = *(bytes++);
        if (i == 3) {
            arr4[0] = (arr3[0] & 0xfc) >> 2;
            arr4[1] = ((arr3[0] & 0x03) << 4) + ((arr3[1] & 0xf0) >> 4);
            arr4[2] = ((arr3[1] & 0x0f) << 2) + ((arr3[2] & 0xc0) >> 6);
            arr4[3] = arr3[2] & 0x3f;
            for (i = 0; i < 4; i++) out += BASE64_CHARS[arr4[i]];
            i = 0;
        }
    }

    if (i) {
        for (size_t j = i; j < 3; j++) arr3[j] = 0;
        arr4[0] = (arr3[0] & 0xfc) >> 2;
        arr4[1] = ((arr3[0] & 0x03) << 4) + ((arr3[1] & 0xf0) >> 4);
        arr4[2] = ((arr3[1] & 0x0f) << 2) + ((arr3[2] & 0xc0) >> 6);
        arr4[3] = arr3[2] & 0x3f;
        for (size_t j = 0; j < i + 1; j++) out += BASE64_CHARS[arr4[j]];
        while (i++ < 3) out += '=';
    }
    return out;
}

std::string RPCServer::base64_decode(const std::string& input) {
    auto find_char = [](char c) -> int {
        const char* p = std::strchr(BASE64_CHARS, c);
        if (p) return static_cast<int>(p - BASE64_CHARS);
        return -1;
    };

    std::string out;
    int i = 0;
    uint8_t arr4[4];
    uint8_t arr3[3];

    for (char c : input) {
        if (c == '=' || c == '\n' || c == '\r') break;
        int val = find_char(c);
        if (val < 0) continue;
        arr4[i++] = static_cast<uint8_t>(val);
        if (i == 4) {
            arr3[0] = (arr4[0] << 2) + ((arr4[1] & 0x30) >> 4);
            arr3[1] = ((arr4[1] & 0xf) << 4) + ((arr4[2] & 0x3c) >> 2);
            arr3[2] = ((arr4[2] & 0x3) << 6) + arr4[3];
            for (i = 0; i < 3; i++) out += static_cast<char>(arr3[i]);
            i = 0;
        }
    }

    if (i) {
        for (int j = i; j < 4; j++) arr4[j] = 0;
        arr3[0] = (arr4[0] << 2) + ((arr4[1] & 0x30) >> 4);
        arr3[1] = ((arr4[1] & 0xf) << 4) + ((arr4[2] & 0x3c) >> 2);
        arr3[2] = ((arr4[2] & 0x3) << 6) + arr4[3];
        for (int j = 0; j < i - 1; j++) out += static_cast<char>(arr3[j]);
    }
    return out;
}

}  // namespace rnet::rpc

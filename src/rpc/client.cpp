// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

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
#include <netdb.h>
#include <unistd.h>
using socket_t = int;
static constexpr socket_t INVALID_SOCK = -1;
#define CLOSE_SOCKET close
#endif

#include "rpc/client.h"

#include "core/logging.h"

#include <cstring>
#include <fstream>
#include <sstream>

namespace rnet::rpc {

// ===========================================================================
//  Winsock initialisation (client-side)
// ===========================================================================

#ifdef _WIN32
namespace {

// ---------------------------------------------------------------------------
// RAII Winsock bootstrap -- mirrors the server-side initialiser so both
// sides of the RPC boundary can work independently.
// ---------------------------------------------------------------------------
struct WinsockInitClient {
    WinsockInitClient() {
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
    }
    ~WinsockInitClient() { WSACleanup(); }
};
static WinsockInitClient g_winsock_init_client;

}  // anonymous namespace
#endif

// ===========================================================================
//  RPCClient -- lifecycle
// ===========================================================================

RPCClient::RPCClient() = default;
RPCClient::~RPCClient() = default;

// ---------------------------------------------------------------------------
// set_credentials -- store HTTP Basic auth from user/password pair.
// ---------------------------------------------------------------------------
void RPCClient::set_credentials(const std::string& user,
                                const std::string& password) {
    // 1. Encode "user:password" as Base64 for the Authorization header.
    auth_header_ = "Basic " + base64_encode(user + ":" + password);
}

// ---------------------------------------------------------------------------
// load_cookie -- read the .cookie file written by rnetd at startup.
// Format: "__cookie__:hexstring"
// ---------------------------------------------------------------------------
bool RPCClient::load_cookie(const std::filesystem::path& cookie_path) {
    // 1. Open the cookie file.
    std::ifstream in(cookie_path);
    if (!in) return false;

    // 2. Read the single-line cookie.
    std::string line;
    if (!std::getline(in, line)) return false;

    // 3. Split on the colon separator.
    auto colon = line.find(':');
    if (colon == std::string::npos) return false;

    std::string user = line.substr(0, colon);
    std::string pass = line.substr(colon + 1);

    // 4. Trim trailing whitespace.
    while (!pass.empty() && (pass.back() == '\n' || pass.back() == '\r' ||
                             pass.back() == ' ')) {
        pass.pop_back();
    }

    // 5. Store as Basic auth header.
    auth_header_ = "Basic " + base64_encode(user + ":" + pass);
    return true;
}

// ---------------------------------------------------------------------------
// auto_auth -- try cookie-based auth, fall back to existing credentials.
// ---------------------------------------------------------------------------
bool RPCClient::auto_auth() {
    // 1. Attempt cookie file in the configured data directory.
    if (!data_dir_.empty()) {
        auto cookie_path = data_dir_ / ".cookie";
        if (load_cookie(cookie_path)) return true;
    }
    // 2. Fall back to previously set credentials.
    return !auth_header_.empty();
}

// ===========================================================================
//  RPCClient -- RPC call / send
// ===========================================================================

// ---------------------------------------------------------------------------
// call -- convenience wrapper: builds an RPCRequest and delegates to send().
// ---------------------------------------------------------------------------
core::Result<RPCResponse> RPCClient::call(const std::string& method,
                                          const JsonValue& params) {
    RPCRequest req;
    req.method = method;
    req.params = params;
    req.id = JsonValue(static_cast<int64_t>(1));
    return send(req);
}

// ---------------------------------------------------------------------------
// send -- serialise the RPCRequest to JSON-RPC 2.0, POST it, parse reply.
// ---------------------------------------------------------------------------
core::Result<RPCResponse> RPCClient::send(const RPCRequest& req) {
    // 1. Build the JSON-RPC request body.
    JsonValue body = JsonValue::object();
    body.set("jsonrpc", JsonValue("2.0"));
    body.set("method", JsonValue(req.method));
    body.set("params", req.params);
    body.set("id", req.id);

    std::string json_body = body.to_string();

    // 2. Perform the HTTP POST.
    auto result = http_post(json_body);
    if (result.is_err()) {
        return core::Result<RPCResponse>::err(result.error());
    }

    // 3. Parse the response JSON.
    JsonValue resp_json;
    if (!parse_json(result.value(), resp_json)) {
        return core::Result<RPCResponse>::err("failed to parse response JSON");
    }

    // 4. Extract result / error / id fields.
    RPCResponse response;
    response.result = resp_json["result"];
    response.error = resp_json["error"];
    response.id = resp_json["id"];

    return core::Result<RPCResponse>::ok(std::move(response));
}

// ===========================================================================
//  RPCClient -- HTTP transport
// ===========================================================================

// ---------------------------------------------------------------------------
// http_post -- raw HTTP/1.1 POST over a blocking TCP socket.
//
// Design: intentionally minimal (no HTTP library dependency).  The server
// always responds with Connection: close, so we simply read until EOF.
// ---------------------------------------------------------------------------
core::Result<std::string> RPCClient::http_post(const std::string& body) {
    // 1. Resolve the target address.
    struct addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    std::string port_str = std::to_string(port_);
    struct addrinfo* result = nullptr;
    int rc = ::getaddrinfo(host_.c_str(), port_str.c_str(), &hints, &result);
    if (rc != 0 || !result) {
        last_error_ = "failed to resolve host: " + host_;
        return core::Result<std::string>::err(last_error_);
    }

    // 2. Create the TCP socket.
    socket_t sock = ::socket(result->ai_family, result->ai_socktype,
                             result->ai_protocol);
    if (sock == INVALID_SOCK) {
        ::freeaddrinfo(result);
        last_error_ = "failed to create socket";
        return core::Result<std::string>::err(last_error_);
    }

    // 3. Set send/receive timeouts (30 s).
#ifdef _WIN32
    DWORD timeout_ms = 30000;
    ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO,
                 reinterpret_cast<const char*>(&timeout_ms),
                 sizeof(timeout_ms));
    ::setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO,
                 reinterpret_cast<const char*>(&timeout_ms),
                 sizeof(timeout_ms));
#else
    struct timeval tv;
    tv.tv_sec = 30;
    tv.tv_usec = 0;
    ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO,
                 reinterpret_cast<const char*>(&tv), sizeof(tv));
    ::setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO,
                 reinterpret_cast<const char*>(&tv), sizeof(tv));
#endif

    // 4. Connect to the server.
    if (::connect(sock, result->ai_addr,
                  static_cast<int>(result->ai_addrlen)) != 0) {
        CLOSE_SOCKET(sock);
        ::freeaddrinfo(result);
        last_error_ = "failed to connect to " + host_ + ":" + port_str;
        return core::Result<std::string>::err(last_error_);
    }
    ::freeaddrinfo(result);

    // 5. Build the HTTP request.
    std::ostringstream http_req;
    http_req << "POST / HTTP/1.1\r\n";
    http_req << "Host: " << host_ << "\r\n";
    http_req << "Content-Type: application/json\r\n";
    http_req << "Content-Length: " << body.size() << "\r\n";
    if (!auth_header_.empty()) {
        http_req << "Authorization: " << auth_header_ << "\r\n";
    }
    http_req << "Connection: close\r\n";
    http_req << "\r\n";
    http_req << body;

    std::string request = http_req.str();

    // 6. Send the request bytes.
    const char* send_ptr = request.c_str();
    size_t send_remaining = request.size();
    while (send_remaining > 0) {
        int sent = ::send(sock, send_ptr, static_cast<int>(send_remaining), 0);
        if (sent <= 0) {
            CLOSE_SOCKET(sock);
            last_error_ = "send failed";
            return core::Result<std::string>::err(last_error_);
        }
        send_ptr += sent;
        send_remaining -= static_cast<size_t>(sent);
    }

    // 7. Read the full response (server sends Connection: close).
    std::string response;
    response.reserve(8192);
    char buf[4096];
    while (true) {
        int n = ::recv(sock, buf, sizeof(buf), 0);
        if (n <= 0) break;
        response.append(buf, static_cast<size_t>(n));
    }
    CLOSE_SOCKET(sock);

    // 8. Locate the header/body boundary.
    auto header_end = response.find("\r\n\r\n");
    if (header_end == std::string::npos) {
        last_error_ = "malformed HTTP response";
        return core::Result<std::string>::err(last_error_);
    }

    // 9. Parse the HTTP status code from the first line.
    auto first_line_end = response.find("\r\n");
    std::string status_line = response.substr(0, first_line_end);
    // "HTTP/1.1 200 OK"
    auto sp1 = status_line.find(' ');
    if (sp1 != std::string::npos) {
        auto sp2 = status_line.find(' ', sp1 + 1);
        std::string code_str = status_line.substr(sp1 + 1,
                                                  sp2 - sp1 - 1);
        try {
            last_http_status_ = std::stoi(code_str);
        } catch (...) {
            last_http_status_ = 0;
        }
    }

    // 10. Extract the response body.
    std::string resp_body = response.substr(header_end + 4);

    // 11. Reject authentication failures early.
    if (last_http_status_ == 401) {
        last_error_ = "authentication failed (HTTP 401)";
        return core::Result<std::string>::err(last_error_);
    }

    return core::Result<std::string>::ok(std::move(resp_body));
}

// ===========================================================================
//  Base64 encoder (used for HTTP Basic auth)
// ===========================================================================

static constexpr char BASE64_TABLE[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// ---------------------------------------------------------------------------
// base64_encode -- RFC 4648 standard Base64 encoding.
// ---------------------------------------------------------------------------
std::string RPCClient::base64_encode(const std::string& input) {
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
            for (i = 0; i < 4; i++) out += BASE64_TABLE[arr4[i]];
            i = 0;
        }
    }

    if (i) {
        for (size_t j = i; j < 3; j++) arr3[j] = 0;
        arr4[0] = (arr3[0] & 0xfc) >> 2;
        arr4[1] = ((arr3[0] & 0x03) << 4) + ((arr3[1] & 0xf0) >> 4);
        arr4[2] = ((arr3[1] & 0x0f) << 2) + ((arr3[2] & 0xc0) >> 6);
        arr4[3] = arr3[2] & 0x3f;
        for (size_t j = 0; j < i + 1; j++) out += BASE64_TABLE[arr4[j]];
        while (i++ < 3) out += '=';
    }
    return out;
}

} // namespace rnet::rpc

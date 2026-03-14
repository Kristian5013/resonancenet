// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

// Project headers.
#include "core/config.h"
#include "core/error.h"
#include "core/hex.h"
#include "core/logging.h"

// Standard library.
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Platform-specific socket headers.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using socket_t = SOCKET;
static constexpr socket_t INVALID_SOCK = INVALID_SOCKET;
static void close_socket(socket_t s) { closesocket(s); }
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using socket_t = int;
static constexpr socket_t INVALID_SOCK = -1;
static void close_socket(socket_t s) { close(s); }
#endif

// ===========================================================================
//  Minimal HTTP client
// ===========================================================================

struct HttpResponse {
    int status_code = 0;
    std::string body;
};

// ---------------------------------------------------------------------------
// connect_to_host
// ---------------------------------------------------------------------------
// Opens a TCP connection to the given host:port using getaddrinfo for
// protocol-agnostic resolution.
// ---------------------------------------------------------------------------
static socket_t connect_to_host(const std::string& host, uint16_t port)
{
    struct addrinfo hints{}, *result = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    std::string port_str = std::to_string(port);
    int rc = getaddrinfo(host.c_str(), port_str.c_str(), &hints, &result);
    if (rc != 0 || !result) {
        return INVALID_SOCK;
    }

    socket_t sock = INVALID_SOCK;
    for (auto* rp = result; rp != nullptr; rp = rp->ai_next) {
        sock = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sock == INVALID_SOCK) continue;
        if (connect(sock, rp->ai_addr, static_cast<int>(rp->ai_addrlen)) == 0) {
            break;
        }
        close_socket(sock);
        sock = INVALID_SOCK;
    }
    freeaddrinfo(result);
    return sock;
}

// ---------------------------------------------------------------------------
// send_all
// ---------------------------------------------------------------------------
static bool send_all(socket_t sock, const std::string& data)
{
    size_t total = 0;
    while (total < data.size()) {
        int sent = send(sock, data.data() + total,
                        static_cast<int>(data.size() - total), 0);
        if (sent <= 0) return false;
        total += static_cast<size_t>(sent);
    }
    return true;
}

// ---------------------------------------------------------------------------
// recv_all
// ---------------------------------------------------------------------------
static std::string recv_all(socket_t sock)
{
    std::string result;
    std::array<char, 8192> buf{};
    for (;;) {
        int n = recv(sock, buf.data(), static_cast<int>(buf.size()), 0);
        if (n <= 0) break;
        result.append(buf.data(), static_cast<size_t>(n));
    }
    return result;
}

// ---------------------------------------------------------------------------
// base64_encode
// ---------------------------------------------------------------------------
static std::string base64_encode(const std::string& input)
{
    static constexpr char table[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string output;
    output.reserve(((input.size() + 2) / 3) * 4);
    for (size_t i = 0; i < input.size(); i += 3) {
        uint32_t n = static_cast<uint8_t>(input[i]) << 16;
        if (i + 1 < input.size()) n |= static_cast<uint8_t>(input[i + 1]) << 8;
        if (i + 2 < input.size()) n |= static_cast<uint8_t>(input[i + 2]);
        output.push_back(table[(n >> 18) & 0x3F]);
        output.push_back(table[(n >> 12) & 0x3F]);
        output.push_back((i + 1 < input.size()) ? table[(n >> 6) & 0x3F] : '=');
        output.push_back((i + 2 < input.size()) ? table[n & 0x3F] : '=');
    }
    return output;
}

// ---------------------------------------------------------------------------
// http_post
// ---------------------------------------------------------------------------
// Sends an HTTP POST request and returns the parsed response.
// ---------------------------------------------------------------------------
static HttpResponse http_post(const std::string& host, uint16_t port,
                               const std::string& path,
                               const std::string& body,
                               const std::string& auth_header)
{
    HttpResponse resp;
    socket_t sock = connect_to_host(host, port);
    if (sock == INVALID_SOCK) {
        resp.status_code = -1;
        resp.body = "Connection refused";
        return resp;
    }

    // 1. Build and send the request.
    std::ostringstream req;
    req << "POST " << path << " HTTP/1.0\r\n"
        << "Host: " << host << ":" << port << "\r\n"
        << "Content-Type: application/json\r\n"
        << "Content-Length: " << body.size() << "\r\n";
    if (!auth_header.empty()) {
        req << "Authorization: Basic " << auth_header << "\r\n";
    }
    req << "\r\n" << body;

    if (!send_all(sock, req.str())) {
        close_socket(sock);
        resp.status_code = -2;
        resp.body = "Send failed";
        return resp;
    }

    // 2. Shutdown write side so server knows we are done.
#ifdef _WIN32
    shutdown(sock, SD_SEND);
#else
    shutdown(sock, SHUT_WR);
#endif

    // 3. Receive and parse the response.
    std::string raw = recv_all(sock);
    close_socket(sock);

    auto header_end = raw.find("\r\n\r\n");
    if (header_end == std::string::npos) {
        resp.status_code = -3;
        resp.body = "Malformed HTTP response";
        return resp;
    }

    // Parse status line: "HTTP/1.x NNN ...".
    auto first_line_end = raw.find("\r\n");
    std::string status_line = raw.substr(0, first_line_end);
    auto space1 = status_line.find(' ');
    if (space1 != std::string::npos) {
        resp.status_code = std::atoi(status_line.c_str() + space1 + 1);
    }
    resp.body = raw.substr(header_end + 4);
    return resp;
}

// ===========================================================================
//  JSON-RPC helpers (minimal, no external dep)
// ===========================================================================

// ---------------------------------------------------------------------------
// build_json_rpc
// ---------------------------------------------------------------------------
// Constructs a JSON-RPC 1.0 request body with auto-detection of parameter
// types (number, boolean, null, or string).
// ---------------------------------------------------------------------------
static std::string build_json_rpc(const std::string& method,
                                   const std::vector<std::string>& params,
                                   int id = 1)
{
    std::ostringstream ss;
    ss << "{\"jsonrpc\":\"1.0\",\"id\":" << id
       << ",\"method\":\"" << method << "\",\"params\":[";
    for (size_t i = 0; i < params.size(); ++i) {
        if (i > 0) ss << ",";
        const auto& p = params[i];
        if (p == "true" || p == "false" || p == "null") {
            ss << p;
        } else {
            // 1. Detect if param is a number.
            bool is_number = !p.empty();
            bool has_dot = false;
            for (size_t j = 0; j < p.size(); ++j) {
                char c = p[j];
                if (c == '-' && j == 0) continue;
                if (c == '.' && !has_dot) { has_dot = true; continue; }
                if (c < '0' || c > '9') { is_number = false; break; }
            }
            if (is_number && !p.empty()) {
                ss << p;
            } else {
                // 2. Escape as JSON string.
                ss << "\"";
                for (char c : p) {
                    if (c == '\"') ss << "\\\"";
                    else if (c == '\\') ss << "\\\\";
                    else if (c == '\n') ss << "\\n";
                    else if (c == '\r') ss << "\\r";
                    else if (c == '\t') ss << "\\t";
                    else ss << c;
                }
                ss << "\"";
            }
        }
    }
    ss << "]}";
    return ss.str();
}

// ---------------------------------------------------------------------------
// extract_json_field
// ---------------------------------------------------------------------------
// Very basic JSON field extractor -- handles strings, objects, arrays,
// null, numbers, and booleans.
// ---------------------------------------------------------------------------
static std::string extract_json_field(const std::string& json,
                                       const std::string& field)
{
    std::string key = "\"" + field + "\"";
    auto pos = json.find(key);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + key.size());
    if (pos == std::string::npos) return "";
    pos++;
    while (pos < json.size() && json[pos] == ' ') pos++;
    if (pos >= json.size()) return "";

    if (json[pos] == '"') {
        // String value.
        pos++;
        std::string result;
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                pos++;
                if (json[pos] == 'n') result += '\n';
                else if (json[pos] == 't') result += '\t';
                else result += json[pos];
            } else {
                result += json[pos];
            }
            pos++;
        }
        return result;
    } else if (json[pos] == '{' || json[pos] == '[') {
        // Object or array -- find matching brace.
        int depth = 1;
        char open = json[pos], close_c = (open == '{') ? '}' : ']';
        size_t start = pos;
        pos++;
        while (pos < json.size() && depth > 0) {
            if (json[pos] == open) depth++;
            else if (json[pos] == close_c) depth--;
            pos++;
        }
        return json.substr(start, pos - start);
    } else if (json.substr(pos, 4) == "null") {
        return "null";
    } else {
        // Number or boolean.
        size_t start = pos;
        while (pos < json.size() && json[pos] != ',' && json[pos] != '}') pos++;
        std::string val = json.substr(start, pos - start);
        while (!val.empty() && (val.back() == ' ' || val.back() == '\r' || val.back() == '\n'))
            val.pop_back();
        return val;
    }
}

// ===========================================================================
//  Chat REPL mode
// ===========================================================================

// ---------------------------------------------------------------------------
// run_chat_mode
// ---------------------------------------------------------------------------
// Interactive AI chat loop: sends user messages via the "chat" RPC method
// and prints responses until the user types "exit" or "quit".
// ---------------------------------------------------------------------------
static int run_chat_mode(const std::string& host, uint16_t port,
                          const std::string& auth)
{
    fprintf(stderr, "ResonanceNet AI Chat (connected to %s:%u)\n", host.c_str(), port);
    fprintf(stderr, "Type your message and press Enter. Type 'exit' to quit.\n\n");

    std::string line;
    while (true) {
        fprintf(stderr, "You> ");
        if (!std::getline(std::cin, line)) break;
        if (line == "exit" || line == "quit") break;
        if (line.empty()) continue;

        std::string rpc_body = build_json_rpc("chat", {line});
        auto resp = http_post(host, port, "/", rpc_body, auth);

        if (resp.status_code < 0) {
            fprintf(stderr, "Error: %s\n", resp.body.c_str());
            continue;
        }

        std::string error_val = extract_json_field(resp.body, "error");
        if (!error_val.empty() && error_val != "null") {
            std::string msg = extract_json_field(error_val, "message");
            fprintf(stderr, "RPC error: %s\n", msg.empty() ? error_val.c_str() : msg.c_str());
            continue;
        }

        std::string result = extract_json_field(resp.body, "result");
        fprintf(stdout, "AI> %s\n\n", result.c_str());
    }
    return 0;
}

// ===========================================================================
//  Argument parsing
// ===========================================================================

struct CliConfig {
    std::string host = "127.0.0.1";
    uint16_t port = 9554;
    std::string rpcuser;
    std::string rpcpassword;
    std::string cookie_file;
    std::string method;
    std::vector<std::string> params;
    bool chat_mode = false;
};

// ---------------------------------------------------------------------------
// print_usage
// ---------------------------------------------------------------------------
static void print_usage()
{
    fprintf(stderr,
        "Usage: rnet-cli [options] <command> [params...]\n"
        "\n"
        "Options:\n"
        "  -rpcconnect=<host>    Host for RPC connection (default: 127.0.0.1)\n"
        "  -rpcport=<port>       Port for RPC connection (default: 9554)\n"
        "  -rpcuser=<user>       RPC username\n"
        "  -rpcpassword=<pw>     RPC password\n"
        "  -rpccookiefile=<path> Path to .cookie file\n"
        "  -help                 Show this help message\n"
        "\n"
        "Special modes:\n"
        "  rnet-cli chat         Interactive AI chat REPL\n"
        "\n"
        "Common commands:\n"
        "  getblockchaininfo     Get blockchain state\n"
        "  getblockcount         Get current block height\n"
        "  getbestblockhash      Get hash of best block\n"
        "  getpeerinfo           Get connected peer info\n"
        "  getmininginfo         Get mining state\n"
        "  getnewaddress         Generate a new address\n"
        "  getbalance             Get wallet balance\n"
        "  sendtoaddress <addr> <amount>  Send coins\n"
        "  stop                  Shut down rnetd\n"
    );
}

// ---------------------------------------------------------------------------
// parse_cli_args
// ---------------------------------------------------------------------------
static CliConfig parse_cli_args(int argc, char* argv[])
{
    CliConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("-rpcconnect=") == 0) {
            cfg.host = arg.substr(12);
        } else if (arg.find("-rpcport=") == 0) {
            cfg.port = static_cast<uint16_t>(std::atoi(arg.c_str() + 9));
        } else if (arg.find("-rpcuser=") == 0) {
            cfg.rpcuser = arg.substr(9);
        } else if (arg.find("-rpcpassword=") == 0) {
            cfg.rpcpassword = arg.substr(13);
        } else if (arg.find("-rpccookiefile=") == 0) {
            cfg.cookie_file = arg.substr(15);
        } else if (arg == "-help" || arg == "--help" || arg == "-h") {
            print_usage();
            std::exit(0);
        } else if (arg[0] != '-') {
            if (cfg.method.empty()) {
                cfg.method = arg;
                if (cfg.method == "chat") {
                    cfg.chat_mode = true;
                }
            } else {
                cfg.params.push_back(arg);
            }
        }
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// read_cookie_file
// ---------------------------------------------------------------------------
static std::string read_cookie_file(const std::string& path)
{
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return "";
    std::array<char, 512> buf{};
    size_t n = fread(buf.data(), 1, buf.size() - 1, f);
    fclose(f);
    buf[n] = '\0';
    return std::string(buf.data());
}

// ===========================================================================
//  Main
// ===========================================================================

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
// Entry point for rnet-cli.  Parses arguments, builds auth credentials,
// then dispatches to either chat REPL mode or a single JSON-RPC call.
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
#ifdef _WIN32
    WSADATA wsa_data;
    WSAStartup(MAKEWORD(2, 2), &wsa_data);
#endif

    CliConfig cfg = parse_cli_args(argc, argv);

    if (cfg.method.empty() && !cfg.chat_mode) {
        print_usage();
        return 1;
    }

    // 1. Build auth header.
    std::string auth;
    if (!cfg.rpcuser.empty()) {
        auth = base64_encode(cfg.rpcuser + ":" + cfg.rpcpassword);
    } else if (!cfg.cookie_file.empty()) {
        std::string cookie = read_cookie_file(cfg.cookie_file);
        if (!cookie.empty()) {
            auth = base64_encode(cookie);
        }
    }

    // 2. Chat mode.
    if (cfg.chat_mode) {
        int rc = run_chat_mode(cfg.host, cfg.port, auth);
#ifdef _WIN32
        WSACleanup();
#endif
        return rc;
    }

    // 3. Single RPC call.
    std::string rpc_body = build_json_rpc(cfg.method, cfg.params);
    auto resp = http_post(cfg.host, cfg.port, "/", rpc_body, auth);

    if (resp.status_code < 0) {
        fprintf(stderr, "error: could not connect to server %s:%u (%s)\n",
                cfg.host.c_str(), cfg.port, resp.body.c_str());
#ifdef _WIN32
        WSACleanup();
#endif
        return 1;
    }

    if (resp.status_code == 401) {
        fprintf(stderr, "error: authorization failed (check -rpcuser/-rpcpassword)\n");
#ifdef _WIN32
        WSACleanup();
#endif
        return 1;
    }

    // 4. Parse result.
    std::string error_val = extract_json_field(resp.body, "error");
    if (!error_val.empty() && error_val != "null") {
        std::string code = extract_json_field(error_val, "code");
        std::string msg = extract_json_field(error_val, "message");
        fprintf(stderr, "error code: %s\nerror message:\n%s\n",
                code.c_str(), msg.c_str());
#ifdef _WIN32
        WSACleanup();
#endif
        return 1;
    }

    std::string result = extract_json_field(resp.body, "result");
    if (result != "null") {
        fprintf(stdout, "%s\n", result.c_str());
    }

#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}

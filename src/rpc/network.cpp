#include "rpc/network.h"

#include <cstdio>

#include "core/logging.h"
#include "net/conn_manager.h"
#include "net/connection.h"
#include "node/context.h"

namespace rnet::rpc {

// ── getnetworkinfo ──────────────────────────────────────────────────

static JsonValue rpc_getnetworkinfo(const RPCRequest& req,
                                    node::NodeContext& ctx) {
    JsonValue result = JsonValue::object();

    result.set("version", JsonValue(static_cast<int64_t>(40000)));  // v4.0.0
    result.set("subversion", JsonValue(std::string("/ResonanceNet:4.0.0/")));
    result.set("protocolversion", JsonValue(static_cast<int64_t>(70016)));

    if (ctx.connman) {
        result.set("connections",
                    JsonValue(static_cast<int64_t>(ctx.connman->connection_count())));
        result.set("connections_in", JsonValue(static_cast<int64_t>(0)));
        result.set("connections_out",
                    JsonValue(static_cast<int64_t>(ctx.connman->connection_count())));
        result.set("networkactive", JsonValue(ctx.connman->is_running()));
    } else {
        result.set("connections", JsonValue(static_cast<int64_t>(0)));
        result.set("connections_in", JsonValue(static_cast<int64_t>(0)));
        result.set("connections_out", JsonValue(static_cast<int64_t>(0)));
        result.set("networkactive", JsonValue(false));
    }

    result.set("relayfee", JsonValue(0.00001));
    result.set("incrementalfee", JsonValue(0.00001));
    result.set("localrelay", JsonValue(true));
    result.set("timeoffset", JsonValue(static_cast<int64_t>(0)));

    // Networks
    JsonValue networks = JsonValue::array();
    {
        JsonValue net = JsonValue::object();
        net.set("name", JsonValue(std::string("ipv4")));
        net.set("limited", JsonValue(false));
        net.set("reachable", JsonValue(true));
        networks.push_back(std::move(net));
    }
    {
        JsonValue net = JsonValue::object();
        net.set("name", JsonValue(std::string("ipv6")));
        net.set("limited", JsonValue(true));
        net.set("reachable", JsonValue(false));
        networks.push_back(std::move(net));
    }
    result.set("networks", std::move(networks));

    result.set("warnings", JsonValue(std::string("")));

    return result;
}

// ── getpeerinfo ─────────────────────────────────────────────────────

static JsonValue rpc_getpeerinfo(const RPCRequest& req,
                                 node::NodeContext& ctx) {
    JsonValue peers = JsonValue::array();

    if (!ctx.connman) return peers;

    auto connections = ctx.connman->get_connections();
    for (const auto& conn : connections) {
        if (!conn) continue;

        JsonValue peer = JsonValue::object();
        peer.set("id", JsonValue(static_cast<int64_t>(conn->id())));
        peer.set("addr", JsonValue(conn->addr().to_string()));
        peer.set("addrlocal", JsonValue(std::string("127.0.0.1")));
        peer.set("network", JsonValue(std::string("ipv4")));
        peer.set("services", JsonValue(
            static_cast<int64_t>(conn->services())));
        peer.set("relaytxes", JsonValue(true));
        peer.set("lastsend",
                 JsonValue(static_cast<int64_t>(conn->last_send_time())));
        peer.set("lastrecv",
                 JsonValue(static_cast<int64_t>(conn->last_recv_time())));
        peer.set("bytessent",
                 JsonValue(static_cast<int64_t>(conn->bytes_sent())));
        peer.set("bytesrecv",
                 JsonValue(static_cast<int64_t>(conn->bytes_recv())));
        peer.set("conntime",
                 JsonValue(static_cast<int64_t>(conn->connect_time())));
        peer.set("version",
                 JsonValue(static_cast<int64_t>(conn->version())));
        peer.set("subver", JsonValue(conn->user_agent()));
        peer.set("inbound", JsonValue(conn->is_inbound()));
        peer.set("startingheight",
                 JsonValue(static_cast<int64_t>(conn->start_height())));
        peer.set("synced_headers",
                 JsonValue(static_cast<int64_t>(conn->best_known_height())));
        peer.set("synced_blocks",
                 JsonValue(static_cast<int64_t>(conn->best_known_height())));
        peer.set("pingtime",
                 JsonValue(static_cast<double>(conn->ping_time_ms()) / 1000.0));

        peers.push_back(std::move(peer));
    }

    return peers;
}

// ── getconnectioncount ──────────────────────────────────────────────

static JsonValue rpc_getconnectioncount(const RPCRequest& req,
                                        node::NodeContext& ctx) {
    if (!ctx.connman) return JsonValue(static_cast<int64_t>(0));
    return JsonValue(static_cast<int64_t>(ctx.connman->connection_count()));
}

// ── addnode ─────────────────────────────────────────────────────────

static JsonValue rpc_addnode(const RPCRequest& req,
                             node::NodeContext& ctx) {
    const auto& node_param = get_param(req, 0);
    const auto& cmd_param = get_param(req, 1);

    if (!node_param.is_string() || !cmd_param.is_string()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "node (string) and command (string) required");
    }

    if (!ctx.connman) {
        return make_rpc_error(RPC_CLIENT_P2P_DISABLED, "P2P is disabled");
    }

    std::string addr_str = node_param.as_string();
    std::string command = cmd_param.as_string();

    // Parse "host:port" string into CNetAddr
    auto parse_addr = [](const std::string& s) -> net::CNetAddr {
        net::CNetAddr addr;
        std::string host = s;
        uint16_t port = 9555;  // default P2P port
        auto colon = s.rfind(':');
        if (colon != std::string::npos) {
            host = s.substr(0, colon);
            try {
                port = static_cast<uint16_t>(std::stoi(s.substr(colon + 1)));
            } catch (...) {}
        }
        // Parse dotted-quad IPv4
        unsigned a = 0, b = 0, c = 0, d = 0;
        if (std::sscanf(host.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
            addr.set_ipv4(static_cast<uint8_t>(a), static_cast<uint8_t>(b),
                          static_cast<uint8_t>(c), static_cast<uint8_t>(d));
        }
        addr.port = port;
        return addr;
    };

    if (command == "add") {
        auto addr = parse_addr(addr_str);
        auto result = ctx.connman->connect_to(addr);
        if (result.is_err()) {
            return make_rpc_error(RPC_CLIENT_NODE_ALREADY_ADDED,
                                  result.error());
        }

        LogPrint(RPC, "addnode: connected to %s", addr_str.c_str());
    } else if (command == "remove") {
        // Would need to find and disconnect by address
        LogPrint(RPC, "addnode: remove not yet implemented for %s",
                 addr_str.c_str());
    } else if (command == "onetry") {
        auto addr = parse_addr(addr_str);
        ctx.connman->connect_to(addr);  // Fire and forget
    } else {
        return make_rpc_error(RPC_INVALID_PARAMETER,
                              "command must be add, remove, or onetry");
    }

    return JsonValue();  // null = success
}

// ── disconnectnode ──────────────────────────────────────────────────

static JsonValue rpc_disconnectnode(const RPCRequest& req,
                                    node::NodeContext& ctx) {
    if (!ctx.connman) {
        return make_rpc_error(RPC_CLIENT_P2P_DISABLED, "P2P is disabled");
    }

    const auto& addr_param = get_param(req, 0);
    const auto& id_param = get_param_optional(req, 1);

    if (id_param.is_int()) {
        uint64_t conn_id = static_cast<uint64_t>(id_param.as_int());
        ctx.connman->disconnect(conn_id);
        return JsonValue();
    }

    if (addr_param.is_string()) {
        // Find by address and disconnect
        auto connections = ctx.connman->get_connections();
        for (const auto& conn : connections) {
            if (conn && conn->addr().to_string() == addr_param.as_string()) {
                ctx.connman->disconnect(conn->id());
                return JsonValue();
            }
        }
        return make_rpc_error(RPC_CLIENT_NODE_NOT_CONNECTED,
                              "Node not found: " + addr_param.as_string());
    }

    return make_rpc_error(RPC_INVALID_PARAMS,
                          "address (string) or nodeid (int) required");
}

// ── Registration ────────────────────────────────────────────────────

void register_network_rpcs(RPCTable& table) {
    table.register_command({
        "getnetworkinfo",
        rpc_getnetworkinfo,
        "Returns network-related information (connections, version, etc.).",
        "Network"
    });

    table.register_command({
        "getpeerinfo",
        rpc_getpeerinfo,
        "Returns information about each connected peer.",
        "Network"
    });

    table.register_command({
        "getconnectioncount",
        rpc_getconnectioncount,
        "Returns the number of connections to other nodes.",
        "Network"
    });

    table.register_command({
        "addnode",
        rpc_addnode,
        "Attempts to add or remove a node from the addnode list.\n"
        "Arguments: node (string), command (add|remove|onetry)",
        "Network"
    });

    table.register_command({
        "disconnectnode",
        rpc_disconnectnode,
        "Disconnect from a peer.\n"
        "Arguments: address (string) or nodeid (int)",
        "Network"
    });
}

}  // namespace rnet::rpc

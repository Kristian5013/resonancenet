#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

// Forward declarations only — full headers in context.cpp
namespace rnet::chain { class CChainState; }
namespace rnet::mempool { class CTxMemPool; }
namespace rnet::net {
    class ConnManager;
    class BlockSync;
    class AddrManager;
    class MsgHandler;
}
namespace rnet::consensus { struct ConsensusParams; }
namespace rnet::core { class ArgsManager; }
namespace rnet::rpc { class RPCServer; }

namespace rnet::node {

/// NodeContext — dependency-injection container that owns all major subsystems.
/// Created once at startup, passed by reference to components that need
/// cross-module access. Non-copyable, non-movable (owns everything).
struct NodeContext {
    NodeContext();
    ~NodeContext();

    // Non-copyable, non-movable
    NodeContext(const NodeContext&) = delete;
    NodeContext& operator=(const NodeContext&) = delete;
    NodeContext(NodeContext&&) = delete;
    NodeContext& operator=(NodeContext&&) = delete;

    // ── Subsystem pointers (set during init) ────────────────────────

    std::unique_ptr<chain::CChainState>      chainstate;
    std::unique_ptr<mempool::CTxMemPool>      mempool;
    std::unique_ptr<net::ConnManager>          connman;
    std::unique_ptr<net::BlockSync>           block_sync;
    std::unique_ptr<net::MsgHandler>          msg_handler;
    std::unique_ptr<net::AddrManager>         addrman;
    std::unique_ptr<rpc::RPCServer>           rpc_server;

    // ── Args manager ────────────────────────────────────────────────

    std::unique_ptr<core::ArgsManager>        args;

    // ── Configuration (populated from args during init) ─────────────

    std::filesystem::path data_dir;
    std::string network = "mainnet";       // "mainnet", "testnet", "regtest"
    uint16_t listen_port  = 9555;
    uint16_t rpc_port     = 9554;
    bool     listen       = true;
    bool     rpc_enabled  = true;
    int      max_connections = 125;

    std::vector<std::string> connect_nodes; // -connect= peers (exclusive)
    std::vector<std::string> add_nodes;     // -addnode= peers

    // ── Runtime state ───────────────────────────────────────────────

    std::atomic<bool> shutdown_requested{false};

    /// Request orderly shutdown
    void request_shutdown() { shutdown_requested.store(true, std::memory_order_release); }

    /// Check whether shutdown has been requested
    bool is_shutdown_requested() const { return shutdown_requested.load(std::memory_order_acquire); }
};

} // namespace rnet::node

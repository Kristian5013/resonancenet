#include "node/init.h"

#include "node/context.h"
#include "node/shutdown.h"

// Subsystem headers (full includes)
#include "chain/chainstate.h"
#include "chain/coins.h"
#include "chain/storage.h"
#include "chain/utxo_db.h"
#include "common/chainparams.h"
#include "consensus/params.h"
#include "core/config.h"
#include "core/fs.h"
#include "core/logging.h"
#include "mempool/pool.h"
#include "net/addr_man.h"
#include "net/conn_manager.h"
#include "net/msg_handler.h"
#include "net/protocol.h"
#include "net/sync.h"
#include "rpc/server.h"
#include "rpc/blockchain.h"
#include "rpc/control.h"
#include "rpc/lightning_rpc.h"
#include "rpc/mining.h"
#include "rpc/misc.h"
#include "rpc/network.h"
#include "rpc/rawtransaction.h"
#include "rpc/training_rpc.h"
#include "rpc/wallet_rpc.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <csignal>
#endif

namespace rnet::node {

// ────────────────────────────────────────────────────────────────────
// Signal handler — sets global + per-context shutdown flags
// ────────────────────────────────────────────────────────────────────

namespace {

NodeContext* g_node_ctx = nullptr;

#ifdef _WIN32
BOOL WINAPI console_handler(DWORD ctrl_type) {
    if (ctrl_type == CTRL_C_EVENT || ctrl_type == CTRL_BREAK_EVENT) {
        request_shutdown();
        if (g_node_ctx) g_node_ctx->request_shutdown();
        return TRUE;
    }
    return FALSE;
}
#else
void posix_signal_handler(int /*sig*/) {
    request_shutdown();
    if (g_node_ctx) g_node_ctx->request_shutdown();
}
#endif

void install_signal_handlers() {
#ifdef _WIN32
    SetConsoleCtrlHandler(console_handler, TRUE);
#else
    struct sigaction sa{};
    sa.sa_handler = posix_signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT,  &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
#endif
}

} // anonymous namespace

// ────────────────────────────────────────────────────────────────────
// register_args
// ────────────────────────────────────────────────────────────────────

void register_args(NodeContext& ctx)
{
    if (!ctx.args) {
        ctx.args = std::make_unique<core::ArgsManager>();
    }
    auto& a = *ctx.args;

    a.add_arg("help",          "Print this help message and exit",    false);
    a.add_arg("version",       "Print version and exit",             false);
    a.add_arg("datadir",       "Specify data directory",             true);
    a.add_arg("conf",          "Config file (default: resonancenet.conf)", true);

    a.add_arg("testnet",       "Use the test network",               false);
    a.add_arg("regtest",       "Use the regression test network",    false);

    a.add_arg("listen",        "Accept incoming connections (default: 1)", true);
    a.add_arg("port",          "Listen on <port> (default: 9555)",   true);
    a.add_arg("maxconnections","Maximum number of connections",      true);
    a.add_arg("connect",       "Connect only to the specified node", true);
    a.add_arg("addnode",       "Add a node to connect to",          true);

    a.add_arg("rpcport",       "Listen for JSON-RPC on <port>",     true);
    a.add_arg("norpc",         "Disable JSON-RPC server",           false);

    a.add_arg("debug",         "Enable debug logging (category)",   true);
    a.add_arg("printtoconsole","Print log to stdout",               false);

    a.add_arg("gen",           "Generate blocks (mining)",          false);
    a.add_arg("daemon",        "Run in the background",             false);
    a.add_arg("pid",           "PID file (default: rnetd.pid)",     true);
}

// ────────────────────────────────────────────────────────────────────
// parse_args
// ────────────────────────────────────────────────────────────────────

Result<void> parse_args(NodeContext& ctx, int argc, const char* const argv[])
{
    register_args(ctx);

    auto res = ctx.args->parse_args(argc, argv);
    if (res.is_err()) {
        return Result<void>::err("Failed to parse arguments: " + res.error());
    }

    auto& a = *ctx.args;

    if (a.get_bool_arg("regtest")) {
        ctx.network = "regtest";
    } else if (a.get_bool_arg("testnet")) {
        ctx.network = "testnet";
    } else {
        ctx.network = "mainnet";
    }

    if (a.is_set("datadir")) {
        ctx.data_dir = std::filesystem::path(a.get_arg("datadir").value());
    } else {
        ctx.data_dir = core::get_default_data_dir();
    }

    if (auto v = a.get_int_arg("port")) {
        ctx.listen_port = static_cast<uint16_t>(*v);
    }
    if (auto v = a.get_int_arg("rpcport")) {
        ctx.rpc_port = static_cast<uint16_t>(*v);
    }

    ctx.listen = a.get_bool_arg("listen", true);
    ctx.rpc_enabled = !a.get_bool_arg("norpc");

    if (auto v = a.get_int_arg("maxconnections")) {
        ctx.max_connections = static_cast<int>(*v);
    }

    ctx.connect_nodes = a.get_args("connect");
    ctx.add_nodes     = a.get_args("addnode");

    return Result<void>::ok();
}

// ────────────────────────────────────────────────────────────────────
// init_data_dir
// ────────────────────────────────────────────────────────────────────

Result<void> init_data_dir(NodeContext& ctx)
{
    std::filesystem::path dir = ctx.data_dir;
    if (ctx.network == "testnet") {
        dir /= "testnet";
    } else if (ctx.network == "regtest") {
        dir /= "regtest";
    }
    ctx.data_dir = dir;

    auto res = core::ensure_directory(dir);
    if (res.is_err()) {
        return Result<void>::err("Cannot create data directory: " + res.error());
    }

    auto lock_res = core::lock_directory(dir);
    if (lock_res.is_err()) {
        return Result<void>::err(
            "Cannot acquire lock. Is another rnetd instance running? " +
            lock_res.error());
    }

    core::set_data_dir(dir);

    auto log_path = dir / "debug.log";
    auto& logger = core::Logger::instance();
    logger.open_log_file(log_path.string());
    logger.set_print_to_file(true);

    if (ctx.args && ctx.args->get_bool_arg("printtoconsole")) {
        logger.set_print_to_console(true);
    }

    if (ctx.args && ctx.args->is_set("debug")) {
        auto cats = ctx.args->get_args("debug");
        for (const auto& c : cats) {
            if (c == "all" || c == "1") {
                logger.enable_category(core::LogCategory::ALL);
            } else {
                logger.enable_category(core::log_category_from_name(c));
            }
        }
        logger.set_level(core::LogLevel::DBG);
    }

    LogPrintf("ResonanceNet v0.1.0 starting (%s)", ctx.network.c_str());
    LogPrintf("Data directory: %s", dir.string().c_str());

    return Result<void>::ok();
}

// ────────────────────────────────────────────────────────────────────
// init_chain
// ────────────────────────────────────────────────────────────────────

Result<void> init_chain(NodeContext& ctx)
{
    LogPrintf("Initialising chain state...");

    common::NetworkType net_type = common::parse_network_type(ctx.network);
    common::ChainParams::select(net_type);
    const auto& chain_params = common::ChainParams::get();
    const auto& consensus    = chain_params.consensus();

    if (!ctx.args || !ctx.args->is_set("port")) {
        ctx.listen_port = consensus.default_port;
    }
    if (!ctx.args || !ctx.args->is_set("rpcport")) {
        ctx.rpc_port = consensus.rpc_port;
    }

    auto blocks_dir = core::get_blocks_dir();
    auto storage_res = core::ensure_directory(blocks_dir);
    if (storage_res.is_err()) {
        return Result<void>::err("Cannot create blocks dir: " + storage_res.error());
    }

    auto storage = std::make_unique<chain::BlockStorage>(blocks_dir);

    auto chainstate_dir = core::get_chainstate_dir();
    auto cs_dir_res = core::ensure_directory(chainstate_dir);
    if (cs_dir_res.is_err()) {
        return Result<void>::err("Cannot create chainstate dir: " + cs_dir_res.error());
    }

    auto coins_view = std::make_unique<chain::CCoinsViewDB>(chainstate_dir);

    ctx.chainstate = std::make_unique<chain::CChainState>(
        consensus,
        std::move(coins_view),
        std::move(storage));

    auto genesis_res = ctx.chainstate->load_genesis();
    if (genesis_res.is_err()) {
        return Result<void>::err("Failed to load genesis: " + genesis_res.error());
    }

    LogPrintf("Chain initialised: height=%d", ctx.chainstate->height());

    return Result<void>::ok();
}

// ────────────────────────────────────────────────────────────────────
// init_mempool
// ────────────────────────────────────────────────────────────────────

Result<void> init_mempool(NodeContext& ctx)
{
    LogPrintf("Creating transaction mempool...");
    ctx.mempool = std::make_unique<mempool::CTxMemPool>();
    return Result<void>::ok();
}

// ────────────────────────────────────────────────────────────────────
// init_network
// ────────────────────────────────────────────────────────────────────

Result<void> init_network(NodeContext& ctx)
{
    LogPrintf("Initialising P2P network...");

    // Set the active network magic so that all P2P framing uses the
    // correct magic bytes for this network (mainnet/testnet/regtest).
    const auto& chain_params = common::ChainParams::get();
    net::set_network_magic(chain_params.consensus().magic);
    LogPrintf("Network magic set for %s", ctx.network.c_str());

    // Address manager
    ctx.addrman = std::make_unique<net::AddrManager>();
    {
        auto peers_path = ctx.data_dir / "peers.dat";
        if (core::file_exists(peers_path)) {
            auto load_res = ctx.addrman->load(peers_path.string());
            if (load_res.is_err()) {
                LogPrintf("Warning: could not load peers.dat: %s",
                          load_res.error().c_str());
            } else {
                LogPrintf("Loaded %zu known addresses", ctx.addrman->size());
            }
        }

        if (ctx.addrman->size() < 10) {
            auto seeds = net::AddrManager::get_default_seeds(ctx.network);
            LogPrintf("Loading seed nodes for %s (%zu seeds)",
                      ctx.network.c_str(), seeds.size());
            ctx.addrman->add(seeds, "hardcoded-seed");
        }
    }

    // Connection manager
    ctx.connman = std::make_unique<net::ConnManager>();
    ctx.connman->set_user_agent("/ResonanceNet:2.0.0/");
    ctx.connman->set_local_services(net::NODE_NETWORK);
    if (ctx.chainstate) {
        ctx.connman->set_best_height(ctx.chainstate->height());
    }

    // Block sync manager
    if (ctx.chainstate) {
        ctx.block_sync = std::make_unique<net::BlockSync>(
            *ctx.chainstate, *ctx.connman);
    }

    // Message handler — dispatches P2P messages to chain/mempool
    ctx.msg_handler = std::make_unique<net::MsgHandler>(
        *ctx.connman, ctx.addrman.get());

    // Wire up: do we have this inv item?
    ctx.msg_handler->set_have_item(
        [&ctx](const net::CInv& inv) -> bool {
            if (inv.type == net::InvType::INV_BLOCK ||
                inv.type == net::InvType::INV_WITNESS_BLOCK) {
                return ctx.chainstate &&
                       ctx.chainstate->lookup_block_index(inv.hash) != nullptr;
            }
            if (inv.type == net::InvType::INV_TX ||
                inv.type == net::InvType::INV_WITNESS_TX) {
                return ctx.mempool && ctx.mempool->exists(inv.hash);
            }
            return false;
        });

    // Wire up: get serialized block data by hash
    ctx.msg_handler->set_get_block_data(
        [&ctx](const rnet::uint256& hash) -> std::vector<uint8_t> {
            if (!ctx.chainstate) return {};
            auto* idx = ctx.chainstate->lookup_block_index(hash);
            if (!idx || idx->file_number < 0) return {};
            chain::DiskBlockPos dpos;
            dpos.file_number = idx->file_number;
            dpos.pos = idx->data_pos;
            auto res = ctx.chainstate->storage().read_block(dpos);
            if (res.is_err()) return {};
            core::DataStream ss;
            res.value().serialize(ss);
            return {ss.data(), ss.data() + ss.size()};
        });

    // Wire up: get serialized tx data by hash
    ctx.msg_handler->set_get_tx_data(
        [&ctx](const rnet::uint256& hash) -> std::vector<uint8_t> {
            if (!ctx.mempool) return {};
            auto tx = ctx.mempool->get(hash);
            if (!tx) return {};
            core::DataStream ss;
            tx->serialize(ss);
            return {ss.data(), ss.data() + ss.size()};
        });

    // Wire up: getblocks response — find fork point from locator, return hashes
    ctx.msg_handler->set_get_block_hashes(
        [&ctx](const std::vector<rnet::uint256>& locator,
               const rnet::uint256& stop_hash,
               int max_count) -> std::vector<rnet::uint256> {
            if (!ctx.chainstate) return {};
            // Find fork point: walk locator, find first hash on active chain
            int fork_height = 0;
            for (const auto& loc_hash : locator) {
                if (ctx.chainstate->is_on_active_chain(loc_hash)) {
                    auto* idx = ctx.chainstate->lookup_block_index(loc_hash);
                    if (idx) fork_height = idx->height;
                    break;
                }
            }
            // Return block hashes from fork_height+1 up to max_count
            std::vector<rnet::uint256> result;
            int tip_height = ctx.chainstate->height();
            for (int h = fork_height + 1;
                 h <= tip_height && static_cast<int>(result.size()) < max_count;
                 ++h) {
                auto* idx = ctx.chainstate->get_block_by_height(h);
                if (!idx) break;
                result.push_back(idx->block_hash);
                if (idx->block_hash == stop_hash) break;
            }
            return result;
        });

    // Wire up: getheaders response — return serialized headers
    ctx.msg_handler->set_get_headers(
        [&ctx](const std::vector<rnet::uint256>& locator,
               const rnet::uint256& stop_hash,
               int max_count) -> std::vector<std::vector<uint8_t>> {
            if (!ctx.chainstate) return {};
            int fork_height = 0;
            for (const auto& loc_hash : locator) {
                if (ctx.chainstate->is_on_active_chain(loc_hash)) {
                    auto* idx = ctx.chainstate->lookup_block_index(loc_hash);
                    if (idx) fork_height = idx->height;
                    break;
                }
            }
            std::vector<std::vector<uint8_t>> result;
            int tip_height = ctx.chainstate->height();
            for (int h = fork_height + 1;
                 h <= tip_height && static_cast<int>(result.size()) < max_count;
                 ++h) {
                auto* idx = ctx.chainstate->get_block_by_height(h);
                if (!idx) break;
                core::DataStream ss;
                idx->header.serialize(ss);
                result.emplace_back(ss.data(), ss.data() + ss.size());
                if (idx->block_hash == stop_hash) break;
            }
            return result;
        });

    // Wire up: received block → accept into chain
    ctx.msg_handler->set_on_new_block(
        [&ctx](uint64_t peer_id,
               const std::vector<uint8_t>& block_data) -> bool {
            if (!ctx.chainstate) return false;
            core::DataStream ss(
                std::span<const uint8_t>(block_data.data(), block_data.size()));
            primitives::CBlock block;
            try { block.unserialize(ss); } catch (...) { return false; }
            auto res = ctx.chainstate->accept_block(block);
            if (res.is_err()) {
                LogPrint(NET, "Block from peer %llu rejected: %s",
                         static_cast<unsigned long long>(peer_id),
                         res.error().c_str());
                return false;
            }
            LogPrintf("Accepted block height=%d from peer %llu",
                     res.value()->height,
                     static_cast<unsigned long long>(peer_id));

            // Feed into BlockSync so it can track progress & request more
            if (ctx.block_sync) {
                net::CPeer peer;
                peer.id = peer_id;
                ctx.block_sync->on_block(peer, block);
            }
            return true;
        });

    // Wire up: received tx → accept into mempool and relay
    ctx.msg_handler->set_on_new_tx(
        [&ctx](uint64_t peer_id,
               const std::vector<uint8_t>& tx_data) -> bool {
            if (!ctx.mempool) return false;
            core::DataStream ss(
                std::span<const uint8_t>(tx_data.data(), tx_data.size()));
            auto tx_ptr = std::make_shared<primitives::CTransaction>();
            try {
                const_cast<primitives::CTransaction&>(*tx_ptr).unserialize(ss);
            } catch (...) { return false; }
            // Check if we already have it
            if (ctx.mempool->exists(tx_ptr->txid())) return true;
            int height = ctx.chainstate ? ctx.chainstate->height() : 0;
            float val_loss = 0.0f;
            if (ctx.chainstate && ctx.chainstate->tip()) {
                val_loss = ctx.chainstate->tip()->val_loss;
            }
            auto res = ctx.mempool->add_tx(tx_ptr, 0, height, val_loss);
            if (res.is_err()) return false;
            // Relay to other peers (forward the raw data, avoid re-serialize)
            ctx.connman->broadcast_except(peer_id, net::msg::TX,
                std::span<const uint8_t>(tx_data.data(), tx_data.size()));
            return true;
        });

    // Wire up: get mempool txids
    ctx.msg_handler->set_get_mempool_txids(
        [&ctx]() -> std::vector<rnet::uint256> {
            if (!ctx.mempool) return {};
            return ctx.mempool->get_txids();
        });

    // Wire up: received headers → feed into BlockSync
    ctx.msg_handler->set_on_headers(
        [&ctx](uint64_t peer_id,
               const std::vector<uint8_t>& headers_data) -> bool {
            if (!ctx.block_sync || !ctx.chainstate) return false;

            core::DataStream ss(
                std::span<const uint8_t>(headers_data.data(), headers_data.size()));

            uint64_t count = core::unserialize_compact_size(ss);
            std::vector<primitives::CBlockHeader> headers;
            headers.reserve(static_cast<size_t>(count));

            for (uint64_t i = 0; i < count; ++i) {
                primitives::CBlockHeader hdr;
                try {
                    hdr.unserialize(ss);
                    // Skip tx_count (always 0 in headers message)
                    core::unserialize_compact_size(ss);
                } catch (...) {
                    LogPrintf("Failed to parse header %llu from peer %llu",
                             static_cast<unsigned long long>(i),
                             static_cast<unsigned long long>(peer_id));
                    return false;
                }
                headers.push_back(std::move(hdr));
            }

            LogPrintf("Received %zu headers from peer %llu",
                     headers.size(),
                     static_cast<unsigned long long>(peer_id));

            // Create a temporary CPeer for BlockSync interface
            net::CPeer peer;
            peer.id = peer_id;
            ctx.block_sync->on_headers(peer, headers);
            return true;
        });

    // Register all P2P message handlers
    ctx.msg_handler->register_handlers();

    // Block broadcast: when chainstate connects a new block, announce to peers
    if (ctx.chainstate) {
        ctx.chainstate->on_block_connected.connect(
            [&ctx](const chain::CBlockIndex* pindex) {
                if (!ctx.connman || !pindex) return;

                // Update ConnManager best height so version messages are correct
                ctx.connman->set_best_height(pindex->height);

                // Send inv for the new block to all peers
                net::CInv inv(net::InvType::INV_BLOCK, pindex->block_hash);
                core::DataStream ss;
                core::serialize_compact_size(ss, 1);
                inv.serialize(ss);
                ctx.connman->broadcast(net::msg::INV, ss.span());
                LogPrintf("Broadcast block inv height=%d to peers",
                         pindex->height);
            });
    }

    // IBD: when a new peer connects with a higher height, start syncing
    if (ctx.block_sync && ctx.chainstate) {
        ctx.connman->on_connected.connect(
            [&ctx](net::CConnection& conn) {
                int32_t peer_height = conn.start_height();
                int32_t our_height = ctx.chainstate->height();

                LogPrintf("Peer %llu connected: their height=%d, our height=%d",
                         static_cast<unsigned long long>(conn.id()),
                         peer_height, our_height);

                if (peer_height > our_height) {
                    // This peer has blocks we need — start IBD
                    net::CPeer peer;
                    peer.id = conn.id();
                    peer.start_height = peer_height;

                    ctx.block_sync->start();
                    ctx.block_sync->on_new_peer(peer);

                    LogPrintf("Starting IBD from peer %llu (need blocks %d..%d)",
                             static_cast<unsigned long long>(conn.id()),
                             our_height + 1, peer_height);
                }
            });
    }

    // Start listening if configured
    if (ctx.listen) {
        auto start_res = ctx.connman->start(ctx.listen_port);
        if (start_res.is_err()) {
            LogPrintf("Warning: could not start P2P listener: %s",
                      start_res.error().c_str());
        }
    }

    // Connect to -connect= and -addnode= peers
    auto connect_to_peers = [&](const std::vector<std::string>& peers) {
        for (const auto& peer_str : peers) {
            // Parse "host:port"
            std::string host = peer_str;
            uint16_t port = ctx.listen_port;

            auto colon = peer_str.rfind(':');
            if (colon != std::string::npos) {
                host = peer_str.substr(0, colon);
                try {
                    port = static_cast<uint16_t>(std::stoi(peer_str.substr(colon + 1)));
                } catch (...) {
                    LogPrintf("Warning: invalid port in peer address: %s", peer_str.c_str());
                    continue;
                }
            }

            // Parse IPv4
            net::CNetAddr addr;
            unsigned int a, b, c, d;
            if (std::sscanf(host.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
                addr.set_ipv4(static_cast<uint8_t>(a), static_cast<uint8_t>(b),
                              static_cast<uint8_t>(c), static_cast<uint8_t>(d));
                addr.port = port;

                LogPrintf("Connecting to peer %s:%u ...", host.c_str(), port);
                auto res = ctx.connman->connect_to(addr);
                if (res.is_ok()) {
                    LogPrintf("Connected to %s:%u (id=%llu)",
                              host.c_str(), port, res.value());
                } else {
                    LogPrintf("Failed to connect to %s:%u: %s",
                              host.c_str(), port, res.error().c_str());
                }
            } else {
                LogPrintf("Warning: cannot parse peer address: %s", peer_str.c_str());
            }
        }
    };

    connect_to_peers(ctx.connect_nodes);
    connect_to_peers(ctx.add_nodes);

    LogPrintf("Network initialised");
    return Result<void>::ok();
}

// ────────────────────────────────────────────────────────────────────
// init_rpc
// ────────────────────────────────────────────────────────────────────

Result<void> init_rpc(NodeContext& ctx)
{
    if (!ctx.rpc_enabled) {
        LogPrintf("JSON-RPC server disabled (-norpc)");
        return Result<void>::ok();
    }

    LogPrintf("Starting JSON-RPC server on port %u...", ctx.rpc_port);

    ctx.rpc_server = std::make_unique<rpc::RPCServer>();
    ctx.rpc_server->set_context(&ctx);
    ctx.rpc_server->set_data_dir(ctx.data_dir);

    // Register all RPC command handlers
    auto& table = ctx.rpc_server->table();
    rpc::register_blockchain_rpcs(table);
    rpc::register_control_rpcs(table);
    rpc::register_lightning_rpcs(table);
    rpc::register_mining_rpcs(table);
    rpc::register_misc_rpcs(table);
    rpc::register_network_rpcs(table);
    rpc::register_rawtransaction_rpcs(table);
    rpc::register_training_rpcs(table);
    rpc::register_wallet_rpcs(table);

    if (!ctx.rpc_server->start(ctx.rpc_port)) {
        return Result<void>::err("Failed to start RPC server on port " +
                                  std::to_string(ctx.rpc_port));
    }

    LogPrintf("JSON-RPC server started on port %u", ctx.rpc_port);
    return Result<void>::ok();
}

// ────────────────────────────────────────────────────────────────────
// app_init
// ────────────────────────────────────────────────────────────────────

Result<void> app_init(NodeContext& ctx, int argc, const char* const argv[])
{
    g_node_ctx = &ctx;
    reset_shutdown();
    install_signal_handlers();

    auto args_res = parse_args(ctx, argc, argv);
    if (args_res.is_err()) return args_res;

    if (ctx.args->is_set("help") || ctx.args->is_set("version")) {
        return Result<void>::ok();
    }

    auto dir_res = init_data_dir(ctx);
    if (dir_res.is_err()) return dir_res;

    auto chain_res = init_chain(ctx);
    if (chain_res.is_err()) return chain_res;

    auto mem_res = init_mempool(ctx);
    if (mem_res.is_err()) return mem_res;

    auto net_res = init_network(ctx);
    if (net_res.is_err()) return net_res;

    auto rpc_res = init_rpc(ctx);
    if (rpc_res.is_err()) return rpc_res;

    LogPrintf("Initialisation complete");
    return Result<void>::ok();
}

// ────────────────────────────────────────────────────────────────────
// wait_for_shutdown
// ────────────────────────────────────────────────────────────────────

void wait_for_shutdown(NodeContext& /*ctx*/)
{
    LogPrintf("Node running. Press Ctrl+C to stop.");
    rnet::node::wait_for_shutdown();
}

// ────────────────────────────────────────────────────────────────────
// app_shutdown
// ────────────────────────────────────────────────────────────────────

void app_shutdown(NodeContext& ctx)
{
    LogPrintf("Shutting down...");

    if (ctx.rpc_server) {
        ctx.rpc_server->stop();
        ctx.rpc_server.reset();
    }

    if (ctx.block_sync) {
        ctx.block_sync->stop();
        ctx.block_sync.reset();
    }

    if (ctx.connman) {
        ctx.connman->stop();
    }

    // Save peers database
    if (ctx.addrman) {
        auto peers_path = ctx.data_dir / "peers.dat";
        auto save_res = ctx.addrman->save(peers_path.string());
        if (save_res.is_err()) {
            LogPrintf("Warning: could not save peers.dat: %s",
                      save_res.error().c_str());
        }
    }

    // Release subsystems in reverse order
    ctx.msg_handler.reset();
    ctx.addrman.reset();
    ctx.connman.reset();
    ctx.mempool.reset();
    ctx.chainstate.reset();

    core::unlock_directory(ctx.data_dir);
    core::Logger::instance().close_log_file();

    g_node_ctx = nullptr;

    LogPrintf("Shutdown complete.");
}

} // namespace rnet::node

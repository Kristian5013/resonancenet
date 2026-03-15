// Copyright (c) 2024-2026 The ResonanceNet Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "node/init.h"

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
#include "net/checkpoint_store.h"
#include "net/conn_manager.h"
#include "net/msg_handler.h"
#include "net/protocol.h"
#include "net/sync.h"
#include "node/context.h"
#include "node/shutdown.h"
#include "rpc/blockchain.h"
#include "rpc/control.h"
#include "rpc/lightning_rpc.h"
#include "rpc/mining.h"
#include "rpc/misc.h"
#include "rpc/network.h"
#include "rpc/rawtransaction.h"
#include "rpc/server.h"
#include "rpc/training_rpc.h"
#include "rpc/wallet_rpc.h"
#include "wallet/wallet.h"

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

// ===========================================================================
// Signal Handling
// ===========================================================================

namespace {

NodeContext* g_node_ctx = nullptr;

// ---------------------------------------------------------------------------
// console_handler / posix_signal_handler
// ---------------------------------------------------------------------------
// OS-specific signal handlers that set both the global shutdown flag and the
// per-context shutdown flag.  On Windows we handle CTRL_C and CTRL_BREAK
// console events; on POSIX we intercept SIGINT and SIGTERM.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// install_signal_handlers
// ---------------------------------------------------------------------------
// Registers the platform-appropriate signal/event handlers so that the node
// can perform a graceful shutdown when the operator presses Ctrl+C or sends
// SIGTERM.
// ---------------------------------------------------------------------------
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

// ===========================================================================
// Argument Registration
// ===========================================================================

// ---------------------------------------------------------------------------
// register_args
// ---------------------------------------------------------------------------
// Registers every CLI/config-file argument that rnetd accepts.  Follows
// Bitcoin Core convention: boolean flags take no value, string/int args
// require a value.  Called early before any parsing so --help can enumerate
// all known options.
// ---------------------------------------------------------------------------
void register_args(NodeContext& ctx)
{
    // 1. Ensure the ArgsManager exists
    if (!ctx.args) {
        ctx.args = std::make_unique<core::ArgsManager>();
    }
    auto& a = *ctx.args;

    // 2. General options
    a.add_arg("help",          "Print this help message and exit",    false);
    a.add_arg("version",       "Print version and exit",             false);
    a.add_arg("datadir",       "Specify data directory",             true);
    a.add_arg("conf",          "Config file (default: resonancenet.conf)", true);

    // 3. Network selection
    a.add_arg("testnet",       "Use the test network",               false);
    a.add_arg("regtest",       "Use the regression test network",    false);

    // 4. P2P networking
    a.add_arg("listen",        "Accept incoming connections (default: 1)", true);
    a.add_arg("port",          "Listen on <port> (default: 9555)",   true);
    a.add_arg("maxconnections","Maximum number of connections",      true);
    a.add_arg("connect",       "Connect only to the specified node", true);
    a.add_arg("addnode",       "Add a node to connect to",          true);

    // 5. JSON-RPC
    a.add_arg("rpcport",       "Listen for JSON-RPC on <port>",     true);
    a.add_arg("norpc",         "Disable JSON-RPC server",           false);

    // 6. Logging / debugging
    a.add_arg("debug",         "Enable debug logging (category)",   true);
    a.add_arg("printtoconsole","Print log to stdout",               false);

    // 7. Mining and daemon
    a.add_arg("gen",           "Generate blocks (mining)",          false);
    a.add_arg("daemon",        "Run in the background",             false);
    a.add_arg("pid",           "PID file (default: rnetd.pid)",     true);
}

// ===========================================================================
// Argument Parsing
// ===========================================================================

// ---------------------------------------------------------------------------
// parse_args
// ---------------------------------------------------------------------------
// Parses command-line arguments and selects the active network
// (mainnet / testnet / regtest).  Populates the NodeContext with data-dir
// path, port overrides, peer lists, and boolean flags used by later init
// stages.  Returns an error Result if the raw argument parse fails.
// ---------------------------------------------------------------------------
Result<void> parse_args(NodeContext& ctx, int argc, const char* const argv[])
{
    // 1. Register all known arguments, then parse
    register_args(ctx);

    auto res = ctx.args->parse_args(argc, argv);
    if (res.is_err()) {
        return Result<void>::err("Failed to parse arguments: " + res.error());
    }

    auto& a = *ctx.args;

    // 2. Select network
    if (a.get_bool_arg("regtest")) {
        ctx.network = "regtest";
    } else if (a.get_bool_arg("testnet")) {
        ctx.network = "testnet";
    } else {
        ctx.network = "mainnet";
    }

    // 3. Resolve data directory
    if (a.is_set("datadir")) {
        ctx.data_dir = std::filesystem::path(a.get_arg("datadir").value());
    } else {
        ctx.data_dir = core::get_default_data_dir();
    }

    // 4. Port overrides
    if (auto v = a.get_int_arg("port")) {
        ctx.listen_port = static_cast<uint16_t>(*v);
    }
    if (auto v = a.get_int_arg("rpcport")) {
        ctx.rpc_port = static_cast<uint16_t>(*v);
    }

    // 5. Boolean flags
    ctx.listen = a.get_bool_arg("listen", true);
    ctx.rpc_enabled = !a.get_bool_arg("norpc");

    // 6. Connection limits
    if (auto v = a.get_int_arg("maxconnections")) {
        ctx.max_connections = static_cast<int>(*v);
    }

    // 7. Explicit peer lists (-connect= and -addnode=)
    ctx.connect_nodes = a.get_args("connect");
    ctx.add_nodes     = a.get_args("addnode");

    return Result<void>::ok();
}

// ===========================================================================
// Data Directory
// ===========================================================================

// ---------------------------------------------------------------------------
// init_data_dir
// ---------------------------------------------------------------------------
// Creates the data directory (appending a network-specific subdirectory for
// testnet/regtest), acquires a filesystem lock to prevent concurrent rnetd
// instances, and initialises the logging subsystem with the appropriate
// log-level and output targets.
// ---------------------------------------------------------------------------
Result<void> init_data_dir(NodeContext& ctx)
{
    // 1. Append network-specific subdirectory
    std::filesystem::path dir = ctx.data_dir;
    if (ctx.network == "testnet") {
        dir /= "testnet";
    } else if (ctx.network == "regtest") {
        dir /= "regtest";
    }
    ctx.data_dir = dir;

    // 2. Create the directory tree
    auto res = core::ensure_directory(dir);
    if (res.is_err()) {
        return Result<void>::err("Cannot create data directory: " + res.error());
    }

    // 3. Acquire filesystem lock (prevents two rnetd on same datadir)
    auto lock_res = core::lock_directory(dir);
    if (lock_res.is_err()) {
        return Result<void>::err(
            "Cannot acquire lock. Is another rnetd instance running? " +
            lock_res.error());
    }

    // 4. Set the global data-dir so other subsystems can locate it
    core::set_data_dir(dir);

    // 5. Open the log file
    auto log_path = dir / "debug.log";
    auto& logger = core::Logger::instance();
    logger.open_log_file(log_path.string());
    logger.set_print_to_file(true);

    // 6. Console logging (if requested)
    if (ctx.args && ctx.args->get_bool_arg("printtoconsole")) {
        logger.set_print_to_console(true);
    }

    // 7. Debug categories
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

// ===========================================================================
// Chain State
// ===========================================================================

// ---------------------------------------------------------------------------
// init_chain
// ---------------------------------------------------------------------------
// Brings up the chain subsystem in a strict order: select consensus params,
// create the blocks/ and chainstate/ directories, open the LevelDB-backed
// coins view, construct the CChainState, and load (or create) the genesis
// block.  The chain must be fully initialised before the mempool or network
// layers start.
// ---------------------------------------------------------------------------
Result<void> init_chain(NodeContext& ctx)
{
    LogPrintf("Initialising chain state...");

    // 1. Select consensus parameters for the active network
    common::NetworkType net_type = common::parse_network_type(ctx.network);
    common::ChainParams::select(net_type);
    const auto& chain_params = common::ChainParams::get();
    const auto& consensus    = chain_params.consensus();

    // 2. Apply default ports from consensus if not overridden by CLI
    if (!ctx.args || !ctx.args->is_set("port")) {
        ctx.listen_port = consensus.default_port;   // 9'555 mainnet
    }
    if (!ctx.args || !ctx.args->is_set("rpcport")) {
        ctx.rpc_port = consensus.rpc_port;          // 9'556 mainnet
    }

    // 3. Create blocks directory and open block storage
    auto blocks_dir = core::get_blocks_dir();
    auto storage_res = core::ensure_directory(blocks_dir);
    if (storage_res.is_err()) {
        return Result<void>::err("Cannot create blocks dir: " + storage_res.error());
    }

    auto storage = std::make_unique<chain::BlockStorage>(blocks_dir);

    // 4. Create chainstate directory and open LevelDB coins view
    auto chainstate_dir = core::get_chainstate_dir();
    auto cs_dir_res = core::ensure_directory(chainstate_dir);
    if (cs_dir_res.is_err()) {
        return Result<void>::err("Cannot create chainstate dir: " + cs_dir_res.error());
    }

    auto coins_view = std::make_unique<chain::CCoinsViewDB>(chainstate_dir);

    // 5. Construct the chainstate object
    ctx.chainstate = std::make_unique<chain::CChainState>(
        consensus,
        std::move(coins_view),
        std::move(storage));

    // 6. Load or create the genesis block
    auto genesis_res = ctx.chainstate->load_genesis();
    if (genesis_res.is_err()) {
        return Result<void>::err("Failed to load genesis: " + genesis_res.error());
    }

    // 7. Reload block index from disk (restores chain state across restarts)
    auto load_res = ctx.chainstate->load_block_index();
    if (load_res.is_err()) {
        return Result<void>::err(
            "Failed to load block index: " + load_res.error());
    }

    LogPrintf("Chain initialised: height=%d", ctx.chainstate->height());

    return Result<void>::ok();
}

// ===========================================================================
// Mempool
// ===========================================================================

// ---------------------------------------------------------------------------
// init_mempool
// ---------------------------------------------------------------------------
// Creates the in-memory transaction pool that holds unconfirmed transactions
// until they are included in a block.  The mempool enforces fee-rate
// ordering and UTXO-expiry aware admission via the val_loss threshold.
// ---------------------------------------------------------------------------
Result<void> init_mempool(NodeContext& ctx)
{
    LogPrintf("Creating transaction mempool...");

    // 1. Allocate the mempool (default size limits from consensus)
    ctx.mempool = std::make_unique<mempool::CTxMemPool>();

    return Result<void>::ok();
}

// ===========================================================================
// P2P Network
// ===========================================================================

// ---------------------------------------------------------------------------
// init_network
// ---------------------------------------------------------------------------
// Initialises the full P2P networking stack.  This is the most complex init
// function because it wires together five subsystems:
//   1. Network magic selection (mainnet / testnet / regtest framing)
//   2. AddrManager with seed-node bootstrap
//   3. ConnManager — listener, outbound connection slots
//   4. BlockSync for Initial Block Download (IBD)
//   5. MsgHandler — routes inv/block/tx/headers messages to chain & mempool
// After all callbacks are registered it starts listening and connects to
// any explicitly-requested peers (-connect=, -addnode=).
// ---------------------------------------------------------------------------
Result<void> init_network(NodeContext& ctx)
{
    LogPrintf("Initialising P2P network...");

    // 1. Set the active network magic for P2P framing
    const auto& chain_params = common::ChainParams::get();
    net::set_network_magic(chain_params.consensus().magic);
    LogPrintf("Network magic set for %s", ctx.network.c_str());

    // 2. Address manager — load peers.dat or seed from hardcoded list
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

    // 3. Checkpoint store — file-based storage for model checkpoints
    ctx.checkpoint_store = std::make_unique<net::CheckpointStore>(ctx.data_dir);
    LogPrintf("Checkpoint store: %s", ctx.checkpoint_store->root().string().c_str());

    // 4. Connection manager — user-agent, services, best height
    ctx.connman = std::make_unique<net::ConnManager>();
    ctx.connman->set_user_agent("/ResonanceNet:2.0.0/");
    ctx.connman->set_local_services(net::NODE_NETWORK | net::NODE_CHECKPOINT);
    ctx.connman->set_addrman(ctx.addrman.get());
    if (ctx.chainstate) {
        ctx.connman->set_best_height(ctx.chainstate->height());
    }

    // 4. Block sync manager for IBD
    if (ctx.chainstate) {
        ctx.block_sync = std::make_unique<net::BlockSync>(
            *ctx.chainstate, *ctx.connman);
    }

    // 5. Message handler — dispatches P2P messages to chain/mempool
    ctx.msg_handler = std::make_unique<net::MsgHandler>(
        *ctx.connman, ctx.addrman.get());

    // 5a. Wire up checkpoint store for P2P checkpoint transfer
    ctx.msg_handler->set_checkpoint_store(ctx.checkpoint_store.get());

    // 6. Wire up: do we have this inv item?
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

    // 7. Wire up: get serialized block data by hash
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

    // 8. Wire up: get serialized tx data by hash
    ctx.msg_handler->set_get_tx_data(
        [&ctx](const rnet::uint256& hash) -> std::vector<uint8_t> {
            if (!ctx.mempool) return {};
            auto tx = ctx.mempool->get(hash);
            if (!tx) return {};
            core::DataStream ss;
            tx->serialize(ss);
            return {ss.data(), ss.data() + ss.size()};
        });

    // 9. Wire up: getblocks response — find fork point from locator, return hashes
    ctx.msg_handler->set_get_block_hashes(
        [&ctx](const std::vector<rnet::uint256>& locator,
               const rnet::uint256& stop_hash,
               int max_count) -> std::vector<rnet::uint256> {
            if (!ctx.chainstate) return {};
            // 9a. Walk locator to find first hash on active chain (fork point)
            int fork_height = 0;
            for (const auto& loc_hash : locator) {
                if (ctx.chainstate->is_on_active_chain(loc_hash)) {
                    auto* idx = ctx.chainstate->lookup_block_index(loc_hash);
                    if (idx) fork_height = idx->height;
                    break;
                }
            }
            // 9b. Return block hashes from fork_height+1 up to max_count
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

    // 10. Wire up: getheaders response — return serialized headers
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

    // 11. Wire up: received block -> accept into chain
    ctx.msg_handler->set_on_new_block(
        [&ctx](uint64_t peer_id,
               const std::vector<uint8_t>& block_data) -> bool {
            if (!ctx.chainstate) return false;
            core::DataStream ss(
                std::span<const uint8_t>(block_data.data(), block_data.size()));
            primitives::CBlock block;
            try { block.unserialize(ss); } catch (...) { return false; }

            // Skip if block is already known (prevents relay loops).
            auto block_hash = block.hash();
            if (ctx.chainstate->lookup_block_index(block_hash)) {
                return true;  // already have it, no relay needed
            }

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

            // 11a. Relay the accepted block to all other connected peers.
            //       Forward the raw serialised bytes we already have,
            //       skipping the peer that sent us this block.
            ctx.connman->broadcast_except(peer_id, net::msg::BLOCK,
                std::span<const uint8_t>(block_data.data(), block_data.size()));
            LogPrintf("Relayed block height=%d to peers (from peer %llu)",
                     res.value()->height,
                     static_cast<unsigned long long>(peer_id));

            // 11b. Feed into BlockSync so it can track progress & request more
            if (ctx.block_sync) {
                net::CPeer peer;
                peer.id = peer_id;
                ctx.block_sync->on_block(peer, block);
            }
            return true;
        });

    // 12. Wire up: received tx -> accept into mempool and relay
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
            // 12a. Check if we already have it
            if (ctx.mempool->exists(tx_ptr->txid())) return true;
            int height = ctx.chainstate ? ctx.chainstate->height() : 0;
            float val_loss = 0.0f;
            if (ctx.chainstate && ctx.chainstate->tip()) {
                val_loss = ctx.chainstate->tip()->val_loss;
            }
            auto res = ctx.mempool->add_tx(tx_ptr, 0, height, val_loss);
            if (res.is_err()) return false;
            // 12b. Relay to other peers (forward raw data, avoid re-serialize)
            ctx.connman->broadcast_except(peer_id, net::msg::TX,
                std::span<const uint8_t>(tx_data.data(), tx_data.size()));
            return true;
        });

    // 13. Wire up: get mempool txids
    ctx.msg_handler->set_get_mempool_txids(
        [&ctx]() -> std::vector<rnet::uint256> {
            if (!ctx.mempool) return {};
            return ctx.mempool->get_txids();
        });

    // 14. Wire up: received headers -> feed into BlockSync
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

            // 14a. Create a temporary CPeer for BlockSync interface
            net::CPeer peer;
            peer.id = peer_id;
            ctx.block_sync->on_headers(peer, headers);
            return true;
        });

    // 15. Addr propagation — self-advertisement + relay
    //
    //     a) When a peer sends us a version message, its addr_recv field
    //        reveals what IP address the remote peer sees us as.  Feed
    //        that into MsgHandler so it knows our external address.
    //     b) After each handshake completes, push our address to the new
    //        peer so they learn about us and can relay it to others.
    //     c) Every ~30 minutes, re-advertise our address to a random
    //        peer to keep it fresh in the network.
    ctx.connman->set_external_addr_fn(
        [&ctx](const net::CNetAddr& addr) {
            if (ctx.msg_handler) {
                ctx.msg_handler->set_local_addr(addr);
            }
        });

    ctx.connman->set_addr_broadcast_fn(
        [&ctx]() {
            if (ctx.msg_handler) {
                ctx.msg_handler->advertise_local_addr();
            }
        });

    ctx.connman->on_connected.connect(
        [&ctx](net::CConnection& conn) {
            // 15a. Self-advertise to new peer after handshake.
            if (ctx.msg_handler) {
                ctx.msg_handler->push_local_addr(conn);
            }
        });

    // 16. Register all P2P message handlers
    ctx.msg_handler->register_handlers();

    // 17. Best-height update: when chainstate connects a new block, keep
    //     ConnManager's advertised height current for VERSION messages.
    //     Block relay is handled separately:
    //       - Peer-received blocks: relayed in step 11a via broadcast_except
    //       - RPC-submitted blocks: broadcast in submitblock / submittrainingblock
    if (ctx.chainstate) {
        ctx.chainstate->on_block_connected.connect(
            [&ctx](const chain::CBlockIndex* pindex) {
                if (!ctx.connman || !pindex) return;

                // 17a. Update ConnManager best height for version messages.
                ctx.connman->set_best_height(pindex->height);
            });
    }

    // 18. IBD: when a new peer connects with a higher height, start syncing
    if (ctx.block_sync && ctx.chainstate) {
        ctx.connman->on_connected.connect(
            [&ctx](net::CConnection& conn) {
                int32_t peer_height = conn.start_height();
                int32_t our_height = ctx.chainstate->height();

                LogPrintf("Peer %llu connected: their height=%d, our height=%d",
                         static_cast<unsigned long long>(conn.id()),
                         peer_height, our_height);

                if (peer_height > our_height) {
                    // 18a. This peer has blocks we need -- start IBD
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

    // 19. Start listening if configured
    if (ctx.listen) {
        auto start_res = ctx.connman->start(ctx.listen_port);
        if (start_res.is_err()) {
            LogPrintf("Warning: could not start P2P listener: %s",
                      start_res.error().c_str());
        }
    }

    // 20. Connect to -connect= and -addnode= peers
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

// ===========================================================================
// JSON-RPC
// ===========================================================================

// ---------------------------------------------------------------------------
// init_rpc
// ---------------------------------------------------------------------------
// Starts the JSON-RPC server and registers every command category
// (blockchain, control, lightning, mining, misc, network, rawtransaction,
// training, wallet).  The server binds to ctx.rpc_port and serves requests
// until shutdown.  Skipped entirely when -norpc is set.
// ---------------------------------------------------------------------------
Result<void> init_rpc(NodeContext& ctx)
{
    // 1. Check if RPC is disabled
    if (!ctx.rpc_enabled) {
        LogPrintf("JSON-RPC server disabled (-norpc)");
        return Result<void>::ok();
    }

    LogPrintf("Starting JSON-RPC server on port %u...", ctx.rpc_port);

    // 2. Create the RPC server and bind it to the node context
    ctx.rpc_server = std::make_unique<rpc::RPCServer>();
    ctx.rpc_server->set_context(&ctx);
    ctx.rpc_server->set_data_dir(ctx.data_dir);

    // 3. Register all RPC command handlers
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

    // 4. Start the server
    if (!ctx.rpc_server->start(ctx.rpc_port)) {
        return Result<void>::err("Failed to start RPC server on port " +
                                  std::to_string(ctx.rpc_port));
    }

    LogPrintf("JSON-RPC server started on port %u", ctx.rpc_port);

    // 5. Auto-load wallet if wallet.dat exists in data directory.
    auto wallet_path = ctx.data_dir / "wallet.dat";
    if (std::filesystem::exists(wallet_path)) {
        auto load_result = wallet::CWallet::load(wallet_path.string());
        if (load_result.is_ok()) {
            static std::unique_ptr<wallet::CWallet> s_auto_wallet;
            s_auto_wallet = std::move(load_result.value());
            rpc::set_rpc_wallet(s_auto_wallet.get());
            LogPrintf("Wallet loaded: %s", wallet_path.string().c_str());

            // Rescan existing blocks for wallet-relevant outputs.
            if (ctx.chainstate) {
                int tip_height = ctx.chainstate->height();
                if (tip_height > 0) {
                    int found = 0;
                    for (int h = 1; h <= tip_height; ++h) {
                        auto* idx = ctx.chainstate->get_block_by_height(h);
                        if (!idx || idx->file_number < 0) continue;
                        chain::DiskBlockPos dpos;
                        dpos.file_number = idx->file_number;
                        dpos.pos = idx->data_pos;
                        auto br = ctx.chainstate->storage().read_block(dpos);
                        if (br.is_ok()) {
                            s_auto_wallet->scan_block(br.value());
                            ++found;
                        }
                    }
                    LogPrintf("Wallet rescan: scanned %d blocks up to height %d",
                              found, tip_height);
                }
            }
        } else {
            LogPrintf("Warning: could not load wallet: %s", load_result.error().c_str());
        }
    }

    return Result<void>::ok();
}

// ===========================================================================
// Application Lifecycle
// ===========================================================================

// ---------------------------------------------------------------------------
// app_init
// ---------------------------------------------------------------------------
// Master initialisation sequence.  Subsystems are started in strict
// dependency order so that each layer can assume its prerequisites are
// already live:
//   signals -> args -> datadir -> chain -> mempool -> network -> rpc
// Any failure short-circuits and the error propagates to main().
// ---------------------------------------------------------------------------
Result<void> app_init(NodeContext& ctx, int argc, const char* const argv[])
{
    // 1. Store context pointer for signal handlers and reset shutdown state
    g_node_ctx = &ctx;
    reset_shutdown();
    install_signal_handlers();

    // 2. Parse command-line arguments
    auto args_res = parse_args(ctx, argc, argv);
    if (args_res.is_err()) return args_res;

    // 3. Handle --help / --version (early exit, no subsystem init needed)
    if (ctx.args->is_set("help") || ctx.args->is_set("version")) {
        return Result<void>::ok();
    }

    // 4. Data directory and logging
    auto dir_res = init_data_dir(ctx);
    if (dir_res.is_err()) return dir_res;

    // 5. Chain state (consensus params, LevelDB, genesis)
    auto chain_res = init_chain(ctx);
    if (chain_res.is_err()) return chain_res;

    // 6. Transaction mempool
    auto mem_res = init_mempool(ctx);
    if (mem_res.is_err()) return mem_res;

    // 7. P2P networking (addr manager, connman, block sync, msg handler)
    auto net_res = init_network(ctx);
    if (net_res.is_err()) return net_res;

    // 8. JSON-RPC server
    auto rpc_res = init_rpc(ctx);
    if (rpc_res.is_err()) return rpc_res;

    LogPrintf("Initialisation complete");
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// wait_for_shutdown
// ---------------------------------------------------------------------------
// Blocks the main thread until the global shutdown flag is set (via Ctrl+C
// or an RPC stop command).  The node continues serving P2P and RPC requests
// on background threads while this function sleeps.
// ---------------------------------------------------------------------------
void wait_for_shutdown(NodeContext& /*ctx*/)
{
    LogPrintf("Node running. Press Ctrl+C to stop.");
    rnet::node::wait_for_shutdown();
}

// ---------------------------------------------------------------------------
// app_shutdown
// ---------------------------------------------------------------------------
// Tears down all subsystems in reverse init order to ensure clean resource
// release:  RPC -> sync -> net -> mempool -> chain -> lock -> log.
// Persists the peer database before releasing the address manager.
// ---------------------------------------------------------------------------
void app_shutdown(NodeContext& ctx)
{
    LogPrintf("Shutting down...");

    // 1. Stop RPC server (no new commands accepted)
    if (ctx.rpc_server) {
        ctx.rpc_server->stop();
        ctx.rpc_server.reset();
    }

    // 2. Stop block sync (cancel in-flight IBD requests)
    if (ctx.block_sync) {
        ctx.block_sync->stop();
        ctx.block_sync.reset();
    }

    // 3. Stop connection manager (close all peer sockets)
    if (ctx.connman) {
        ctx.connman->stop();
    }

    // 4. Save peers database before releasing the address manager
    if (ctx.addrman) {
        auto peers_path = ctx.data_dir / "peers.dat";
        auto save_res = ctx.addrman->save(peers_path.string());
        if (save_res.is_err()) {
            LogPrintf("Warning: could not save peers.dat: %s",
                      save_res.error().c_str());
        }
    }

    // 5. Release subsystems in reverse order
    ctx.msg_handler.reset();
    ctx.addrman.reset();
    ctx.connman.reset();
    ctx.checkpoint_store.reset();
    ctx.mempool.reset();
    ctx.chainstate.reset();

    // 6. Release filesystem lock and close log
    core::unlock_directory(ctx.data_dir);
    core::Logger::instance().close_log_file();

    // 7. Clear global context pointer
    g_node_ctx = nullptr;

    LogPrintf("Shutdown complete.");
}

} // namespace rnet::node

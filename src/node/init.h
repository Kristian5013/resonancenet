#pragma once

#include "core/error.h"

namespace rnet::node {

struct NodeContext;

/// Full application initialisation sequence for rnetd / rnet-qt.
///
/// Steps performed by app_init():
///   1. Parse command-line args into NodeContext config
///   2. Create/verify data directory, acquire lock file
///   3. Open log file
///   4. Select consensus params (mainnet / testnet / regtest)
///   5. Open block storage (LevelDB)
///   6. Create CChainState, load genesis / block index, verify chain
///   7. Create mempool
///   8. Create network subsystems (ConnManager, AddrManager, BlockSync)
///   9. Start network threads
///  10. Return — caller enters wait_for_shutdown()
///
/// On failure at any step the function returns an error and the caller
/// must invoke app_shutdown() before exiting.
Result<void> app_init(NodeContext& ctx, int argc, const char* const argv[]);

/// Orderly shutdown: stop network, flush chain state, release lock file.
void app_shutdown(NodeContext& ctx);

/// Block until shutdown is requested (via SIGINT, RPC "stop", etc.)
void wait_for_shutdown(NodeContext& ctx);

// ── Sub-steps (exposed for tests & rnet-qt which do partial init) ───

/// Register all recognised command-line arguments on ctx.args.
void register_args(NodeContext& ctx);

/// Parse args and populate ctx configuration fields.
Result<void> parse_args(NodeContext& ctx, int argc, const char* const argv[]);

/// Initialise data directory, lock file, and log file.
Result<void> init_data_dir(NodeContext& ctx);

/// Initialise chain state (block storage + UTXO DB + genesis).
Result<void> init_chain(NodeContext& ctx);

/// Initialise mempool.
Result<void> init_mempool(NodeContext& ctx);

/// Initialise networking (ConnManager, AddrManager, BlockSync) and start
/// listener / connector threads.
Result<void> init_network(NodeContext& ctx);

/// Initialise JSON-RPC server and register all command handlers.
Result<void> init_rpc(NodeContext& ctx);

} // namespace rnet::node

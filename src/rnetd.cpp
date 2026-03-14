// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

// Project headers.
#include "chain/chainstate.h"
#include "core/config.h"
#include "core/logging.h"
#include "mempool/pool.h"
#include "net/addr_manager.h"
#include "net/conn_manager.h"
#include "net/sync.h"
#include "node/context.h"
#include "node/init.h"
#include "node/shutdown.h"

// Standard library.
#include <csignal>
#include <cstdio>

// Platform-specific Winsock header.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#endif

static rnet::node::NodeContext* g_ctx = nullptr;

// ---------------------------------------------------------------------------
// signal_handler
// ---------------------------------------------------------------------------
static void signal_handler(int sig)
{
    if (g_ctx) {
        g_ctx->request_shutdown();
    }
}

// ---------------------------------------------------------------------------
// setup_signals
// ---------------------------------------------------------------------------
// Installs signal handlers for graceful shutdown (SIGINT, SIGTERM) and
// ignores SIGHUP/SIGPIPE on Unix.
// ---------------------------------------------------------------------------
static void setup_signals()
{
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
#ifndef _WIN32
    std::signal(SIGHUP, SIG_IGN);
    std::signal(SIGPIPE, SIG_IGN);
#endif
}

#ifdef _WIN32
// ---------------------------------------------------------------------------
// init_winsock / cleanup_winsock
// ---------------------------------------------------------------------------
static bool init_winsock()
{
    WSADATA wsa_data;
    int result = WSAStartup(MAKEWORD(2, 2), &wsa_data);
    if (result != 0) {
        fprintf(stderr, "WSAStartup failed: %d\n", result);
        return false;
    }
    return true;
}

static void cleanup_winsock()
{
    WSACleanup();
}
#endif

// ===========================================================================
//  Main
// ===========================================================================

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
// Entry point for rnetd.  Initialises platform resources, creates the node
// context, runs app_init, waits for shutdown signal, then tears down.
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
#ifdef _WIN32
    if (!init_winsock()) {
        return 1;
    }
#endif

    rnet::node::NodeContext ctx;
    g_ctx = &ctx;

    setup_signals();

    // 1. Initialise the node.
    auto result = rnet::node::app_init(ctx, argc, const_cast<const char* const*>(argv));
    if (result.is_err()) {
        fprintf(stderr, "Error: %s\n", result.error().c_str());
        rnet::node::app_shutdown(ctx);
        g_ctx = nullptr;
#ifdef _WIN32
        cleanup_winsock();
#endif
        return 1;
    }

    LogPrintf("ResonanceNet daemon started. Listening on port %u, RPC on port %u.",
              ctx.listen_port, ctx.rpc_port);

    // 2. Block until shutdown is requested.
    rnet::node::wait_for_shutdown(ctx);

    LogPrintf("Shutdown initiated...");

    // 3. Tear down.
    rnet::node::app_shutdown(ctx);
    g_ctx = nullptr;

#ifdef _WIN32
    cleanup_winsock();
#endif

    LogPrintf("Shutdown complete.");
    return 0;
}

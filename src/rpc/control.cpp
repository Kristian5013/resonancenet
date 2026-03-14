// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "rpc/control.h"

#include "core/logging.h"
#include "core/time.h"
#include "node/context.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace rnet::rpc {

// ===========================================================================
//  Server startup time (set externally by rnetd)
// ===========================================================================

namespace {
int64_t g_startup_time = 0;
}  // anonymous namespace

// ---------------------------------------------------------------------------
// set_rpc_startup_time -- called once during server boot to record t0.
// ---------------------------------------------------------------------------
void set_rpc_startup_time(int64_t t) {
    g_startup_time = t;
}

// ===========================================================================
//  stop
// ===========================================================================

// ---------------------------------------------------------------------------
// Requests a graceful shutdown of the ResonanceNet daemon via the node
// context's shutdown flag.
// ---------------------------------------------------------------------------
static JsonValue rpc_stop(const RPCRequest& req,
                          node::NodeContext& ctx) {
    // 1. Log and trigger shutdown.
    LogPrintf("Shutdown requested via RPC");
    ctx.request_shutdown();
    return JsonValue(std::string("ResonanceNet server stopping"));
}

// ===========================================================================
//  help
// ===========================================================================

// ---------------------------------------------------------------------------
// Returns help text for a single command, or a summary of all commands.
// The server layer typically overrides this with RPCTable::help().
// ---------------------------------------------------------------------------
static JsonValue rpc_help(const RPCRequest& req,
                          node::NodeContext& ctx) {
    // 1. Check for an optional command-name argument.
    std::string command;
    const auto& cmd_param = get_param_optional(req, 0);
    if (cmd_param.is_string()) command = cmd_param.as_string();

    // 2. Return general or command-specific placeholder.
    if (command.empty()) {
        return JsonValue(std::string(
            "Use \"help <command>\" for help on a specific command.\n"
            "Available commands can be listed with \"help\"."));
    }

    return JsonValue(std::string("Help for \"" + command + "\" not available."));
}

// ===========================================================================
//  uptime
// ===========================================================================

// ---------------------------------------------------------------------------
// Returns the total uptime of the server in seconds since g_startup_time.
// ---------------------------------------------------------------------------
static JsonValue rpc_uptime(const RPCRequest& req,
                            node::NodeContext& ctx) {
    // 1. Compute elapsed seconds since startup.
    int64_t now = core::get_time();
    int64_t uptime = now - g_startup_time;
    if (uptime < 0) uptime = 0;
    return JsonValue(static_cast<int64_t>(uptime));
}

// ===========================================================================
//  getmemoryinfo
// ===========================================================================

// ---------------------------------------------------------------------------
// Returns process and system memory statistics.  Uses platform-specific
// APIs: GetProcessMemoryInfo / GlobalMemoryStatusEx on Windows,
// getrusage / sysconf on POSIX.
// ---------------------------------------------------------------------------
static JsonValue rpc_getmemoryinfo(const RPCRequest& req,
                                   node::NodeContext& ctx) {
    JsonValue result = JsonValue::object();

    // 1. Gather platform-specific memory counters.
    int64_t used_bytes = 0;
    int64_t free_bytes = 0;
    int64_t total_bytes = 0;

#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        used_bytes = static_cast<int64_t>(pmc.WorkingSetSize);
    }

    MEMORYSTATUSEX memstat;
    memstat.dwLength = sizeof(memstat);
    if (GlobalMemoryStatusEx(&memstat)) {
        total_bytes = static_cast<int64_t>(memstat.ullTotalPhys);
        free_bytes = static_cast<int64_t>(memstat.ullAvailPhys);
    }
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
#ifdef __APPLE__
        used_bytes = usage.ru_maxrss;  // bytes on macOS
#else
        used_bytes = usage.ru_maxrss * 1024;  // KB to bytes on Linux
#endif
    }

    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        total_bytes = pages * page_size;
    }
#ifdef _SC_AVPHYS_PAGES
    long avail = sysconf(_SC_AVPHYS_PAGES);
    if (avail > 0 && page_size > 0) {
        free_bytes = avail * page_size;
    }
#endif
#endif

    // 2. Build the "locked" sub-object (Bitcoin Core compatibility).
    JsonValue locked = JsonValue::object();
    locked.set("used", JsonValue(used_bytes));
    locked.set("free", JsonValue(free_bytes));
    locked.set("total", JsonValue(total_bytes));
    locked.set("locked", JsonValue(static_cast<int64_t>(0)));
    locked.set("chunks_used", JsonValue(static_cast<int64_t>(0)));
    locked.set("chunks_free", JsonValue(static_cast<int64_t>(0)));

    result.set("locked", std::move(locked));

    return result;
}

// ===========================================================================
//  logging
// ===========================================================================

// ---------------------------------------------------------------------------
// Gets and sets the logging configuration.  Accepts optional arrays of
// category names to include/exclude, then returns the current state of
// every known log category.
// ---------------------------------------------------------------------------
static JsonValue rpc_logging(const RPCRequest& req,
                             node::NodeContext& ctx) {
    // 1. Read optional include/exclude parameter arrays.
    const auto& include_param = get_param_optional(req, 0);
    const auto& exclude_param = get_param_optional(req, 1);

    auto& logger = core::Logger::instance();

    // 2. Enable requested categories.
    if (include_param.is_array()) {
        for (size_t i = 0; i < include_param.size(); ++i) {
            if (include_param[i].is_string()) {
                auto cat = core::log_category_from_name(
                    include_param[i].as_string());
                logger.enable_category(cat);
            }
        }
    }

    // 3. Disable excluded categories.
    if (exclude_param.is_array()) {
        for (size_t i = 0; i < exclude_param.size(); ++i) {
            if (exclude_param[i].is_string()) {
                auto cat = core::log_category_from_name(
                    exclude_param[i].as_string());
                logger.disable_category(cat);
            }
        }
    }

    // 4. Return current state of all categories.
    JsonValue result = JsonValue::object();
    for (const auto& entry : core::ALL_LOG_CATEGORIES) {
        result.set(entry.name,
                    JsonValue(logger.is_enabled(entry.cat)));
    }

    return result;
}

// ===========================================================================
//  Registration
// ===========================================================================

// ---------------------------------------------------------------------------
// register_control_rpcs -- add all control/server-management RPCs to the
// command table.
// ---------------------------------------------------------------------------
void register_control_rpcs(RPCTable& table) {
    table.register_command({
        "stop",
        rpc_stop,
        "Request a graceful shutdown of the ResonanceNet server.",
        "Control"
    });

    table.register_command({
        "help",
        rpc_help,
        "List all commands, or get help for a specified command.\n"
        "Arguments: command (string, optional)",
        "Control"
    });

    table.register_command({
        "uptime",
        rpc_uptime,
        "Returns the total uptime of the server in seconds.",
        "Control"
    });

    table.register_command({
        "getmemoryinfo",
        rpc_getmemoryinfo,
        "Returns information about memory usage.",
        "Control"
    });

    table.register_command({
        "logging",
        rpc_logging,
        "Gets and sets the logging configuration.\n"
        "Arguments: include (array of categories), exclude (array of categories)",
        "Control"
    });
}

} // namespace rnet::rpc

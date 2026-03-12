#pragma once

#include <cstdint>

#include "rpc/util.h"

namespace rnet::rpc {

/// Register control RPC commands
void register_control_rpcs(RPCTable& table);

/// Set the server startup time for the uptime command
void set_rpc_startup_time(int64_t t);

}  // namespace rnet::rpc

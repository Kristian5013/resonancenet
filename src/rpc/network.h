#pragma once

#include "rpc/util.h"

namespace rnet::rpc {

/// Register network-related RPC commands
void register_network_rpcs(RPCTable& table);

}  // namespace rnet::rpc

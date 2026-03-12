#pragma once

#include "rpc/util.h"

namespace rnet::rpc {

/// Register mining-related RPC commands
void register_mining_rpcs(RPCTable& table);

}  // namespace rnet::rpc

#pragma once

#include "rpc/util.h"

namespace rnet::rpc {

/// Register blockchain-related RPC commands
void register_blockchain_rpcs(RPCTable& table);

}  // namespace rnet::rpc

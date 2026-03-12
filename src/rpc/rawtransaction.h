#pragma once

#include "rpc/util.h"

namespace rnet::rpc {

/// Register raw transaction-related RPC commands
void register_rawtransaction_rpcs(RPCTable& table);

}  // namespace rnet::rpc

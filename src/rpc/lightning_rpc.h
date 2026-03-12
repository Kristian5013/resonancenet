#pragma once

#include "rpc/util.h"

namespace rnet::rpc {

/// Register Lightning Network RPC commands
void register_lightning_rpcs(RPCTable& table);

}  // namespace rnet::rpc

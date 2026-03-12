#pragma once

#include "rpc/util.h"

namespace rnet::rpc {

/// Register training-related RPC commands (PoT-specific)
void register_training_rpcs(RPCTable& table);

}  // namespace rnet::rpc

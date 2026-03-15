#pragma once

#include "rpc/util.h"

// Forward declaration
namespace rnet::wallet { class CWallet; }

namespace rnet::rpc {

/// Register wallet-related RPC commands
void register_wallet_rpcs(RPCTable& table);

/// Set the wallet pointer for RPC handlers
void set_rpc_wallet(wallet::CWallet* w);

/// Get the wallet pointer (for cross-module block scanning)
wallet::CWallet* get_rpc_wallet();

}  // namespace rnet::rpc

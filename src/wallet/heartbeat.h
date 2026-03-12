#pragma once

#include <cstdint>
#include <vector>

#include "core/error.h"
#include "core/types.h"
#include "primitives/transaction.h"
#include "wallet/coins.h"
#include "wallet/keys.h"

namespace rnet::wallet {

/// HeartbeatCreator: creates heartbeat (version=3) self-spend transactions
/// that prove the wallet owner is still active.
///
/// A heartbeat tx:
///   - version = 3 (TX_VERSION_HEARTBEAT)
///   - Spends one or more of the wallet's own UTXOs
///   - Sends back to the same address (self-spend)
///   - Resets the UTXO expiry timer
class HeartbeatCreator {
public:
    /// Create a heartbeat transaction.
    /// Selects a wallet UTXO, creates a self-spend with version=3.
    ///
    /// @param available_coins  Wallet UTXOs to choose from.
    /// @param change_hash      Pubkey hash for the output (same as input owner).
    /// @param fee_rate         Fee rate for the transaction.
    /// @return Unsigned heartbeat transaction.
    static Result<primitives::CMutableTransaction> create_heartbeat_tx(
        const std::vector<WalletCoin>& available_coins,
        const uint160& change_hash,
        int64_t fee_per_kvb = 1000);

    /// Sign a heartbeat transaction with wallet keys.
    static Result<void> sign_heartbeat_tx(
        const KeyStore& keys,
        primitives::CMutableTransaction& tx,
        const std::vector<WalletCoin>& spent_coins);

    /// Validate that a transaction is a proper heartbeat.
    static bool is_valid_heartbeat(const primitives::CMutableTransaction& tx);
};

}  // namespace rnet::wallet

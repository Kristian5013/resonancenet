#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/error.h"
#include "primitives/transaction.h"
#include "primitives/address.h"
#include "primitives/fees.h"
#include "wallet/coins.h"
#include "wallet/spend.h"

namespace rnet::wallet {

/// A recipient for transaction creation.
struct Recipient {
    std::string address;            ///< Destination address
    int64_t amount = 0;             ///< Amount in resonances
    bool subtract_fee = false;      ///< Subtract fee from this output
};

/// Transaction creation parameters.
struct CreateTxParams {
    std::vector<Recipient> recipients;
    primitives::CFeeRate fee_rate;
    std::string change_address;        ///< Change address (empty = auto)
    uint160 change_pubkey_hash;        ///< Change destination hash
    uint32_t locktime = 0;
    bool rbf = true;                   ///< Signal RBF via sequence
    int32_t version = primitives::TX_VERSION_DEFAULT;
    primitives::NetworkType network = primitives::NetworkType::MAINNET;
};

/// Result of transaction creation (unsigned).
struct CreateTxResult {
    primitives::CMutableTransaction tx;
    CoinSelectionResult coin_selection;
    int64_t total_fee = 0;
    std::string change_address;
};

/// Create an unsigned transaction.
/// Performs coin selection, constructs inputs/outputs, adds change.
Result<CreateTxResult> create_transaction(
    const std::vector<WalletCoin>& available_coins,
    const CreateTxParams& params);

}  // namespace rnet::wallet

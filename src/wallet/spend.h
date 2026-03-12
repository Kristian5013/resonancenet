#pragma once

#include <cstdint>
#include <vector>

#include "core/error.h"
#include "primitives/fees.h"
#include "wallet/coins.h"

namespace rnet::wallet {

/// Coin selection result.
struct CoinSelectionResult {
    std::vector<WalletCoin> selected;
    int64_t total_value = 0;         ///< Sum of selected coin values
    int64_t target_value = 0;        ///< Requested target
    int64_t fee = 0;                 ///< Estimated fee
    int64_t change = 0;              ///< Change amount (may be 0 if exact)
    bool has_change = false;         ///< Whether a change output is needed
};

/// Coin selection parameters.
struct CoinSelectionParams {
    int64_t target_value = 0;                  ///< Amount to send (excluding fee)
    primitives::CFeeRate fee_rate;             ///< Target fee rate
    int64_t min_change = 1000;                 ///< Minimum change to create output
    size_t change_output_size = 43;            ///< Estimated change output size (P2WPKH)
    size_t tx_overhead_size = 11;              ///< Version + locktime + overhead
    size_t input_size = 68;                    ///< Estimated input size (with witness)
    size_t output_size = 43;                   ///< Estimated output size
    int32_t max_inputs = 600;                  ///< Maximum number of inputs
};

/// Select coins using Branch-and-Bound algorithm (optimal exact match).
/// Falls back to Knapsack if no exact solution found.
Result<CoinSelectionResult> select_coins(
    const std::vector<WalletCoin>& available,
    const CoinSelectionParams& params);

/// Branch-and-Bound coin selection: tries to find an exact match
/// (target + fee) within a cost tolerance to avoid change output.
/// @param utxos          Available UTXOs sorted by value descending.
/// @param target         Target value (including fee for N outputs).
/// @param cost_of_change Cost of adding a change output.
/// @return Selected coins, or error if no exact match found.
Result<std::vector<WalletCoin>> select_coins_bnb(
    const std::vector<WalletCoin>& utxos,
    int64_t target,
    int64_t cost_of_change);

/// Knapsack coin selection: greedy approximation.
/// Selects the smallest set of coins >= target.
/// @param utxos     Available UTXOs.
/// @param target    Target value (including fee).
/// @return Selected coins, or error if insufficient funds.
Result<std::vector<WalletCoin>> select_coins_knapsack(
    const std::vector<WalletCoin>& utxos,
    int64_t target);

}  // namespace rnet::wallet

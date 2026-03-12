#include "wallet/spend.h"

#include "core/random.h"

#include <algorithm>
#include <numeric>

namespace rnet::wallet {

/// Estimate the fee for a transaction with the given parameters.
static int64_t estimate_tx_fee(const CoinSelectionParams& params,
                               size_t num_inputs,
                               size_t num_outputs,
                               bool has_change) {
    size_t vsize = params.tx_overhead_size +
                   num_inputs * params.input_size +
                   num_outputs * params.output_size;
    if (has_change) {
        vsize += params.change_output_size;
    }
    return params.fee_rate.get_fee(vsize);
}

Result<std::vector<WalletCoin>> select_coins_bnb(
    const std::vector<WalletCoin>& utxos,
    int64_t target,
    int64_t cost_of_change) {

    if (utxos.empty() || target <= 0) {
        return Result<std::vector<WalletCoin>>::err("invalid inputs for BnB");
    }

    // Sort by value descending
    auto sorted = utxos;
    std::sort(sorted.begin(), sorted.end(),
              [](const WalletCoin& a, const WalletCoin& b) {
                  return a.txout.value > b.txout.value;
              });

    size_t n = sorted.size();
    if (n > 200) n = 200;  // Limit search space

    // BnB with backtracking (depth-first search)
    std::vector<bool> best_selection;
    int64_t best_waste = cost_of_change + 1;  // Start worse than change

    std::vector<bool> current_selection(n, false);
    int64_t current_value = 0;

    // Pre-compute suffix sums for pruning
    std::vector<int64_t> suffix_sum(n + 1, 0);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        suffix_sum[static_cast<size_t>(i)] =
            suffix_sum[static_cast<size_t>(i) + 1] + sorted[static_cast<size_t>(i)].txout.value;
    }

    // Total available must cover target
    if (suffix_sum[0] < target) {
        return Result<std::vector<WalletCoin>>::err("insufficient funds for BnB");
    }

    // Iterative BnB with max iterations
    constexpr int MAX_ITERATIONS = 100000;
    int iterations = 0;
    size_t depth = 0;
    bool backtrack = false;

    while (iterations < MAX_ITERATIONS) {
        ++iterations;

        if (backtrack) {
            // Find the last included item and exclude it
            while (depth > 0) {
                --depth;
                if (current_selection[depth]) {
                    current_selection[depth] = false;
                    current_value -= sorted[depth].txout.value;
                    ++depth;
                    break;
                }
            }
            if (depth == 0 && !current_selection[0]) {
                break;  // Exhausted all possibilities
            }
            backtrack = false;
            continue;
        }

        if (depth >= n) {
            backtrack = true;
            continue;
        }

        // Try including this coin
        int64_t with_value = current_value + sorted[depth].txout.value;

        if (with_value == target) {
            // Perfect match
            current_selection[depth] = true;
            std::vector<WalletCoin> result;
            for (size_t i = 0; i <= depth; ++i) {
                if (current_selection[i]) {
                    result.push_back(sorted[i]);
                }
            }
            return Result<std::vector<WalletCoin>>::ok(std::move(result));
        }

        if (with_value > target) {
            int64_t waste = with_value - target;
            if (waste < best_waste) {
                best_waste = waste;
                best_selection = current_selection;
                best_selection[depth] = true;
            }
            // Don't include (skip this coin)
            current_selection[depth] = false;
            ++depth;
            continue;
        }

        // with_value < target
        // Check if remaining coins can reach target
        if (with_value + suffix_sum[depth + 1] < target) {
            backtrack = true;
            continue;
        }

        // Include this coin
        current_selection[depth] = true;
        current_value = with_value;
        ++depth;
    }

    // Return best solution if waste < cost_of_change
    if (!best_selection.empty() && best_waste <= cost_of_change) {
        std::vector<WalletCoin> result;
        for (size_t i = 0; i < best_selection.size(); ++i) {
            if (best_selection[i]) {
                result.push_back(sorted[i]);
            }
        }
        return Result<std::vector<WalletCoin>>::ok(std::move(result));
    }

    return Result<std::vector<WalletCoin>>::err("BnB found no suitable selection");
}

Result<std::vector<WalletCoin>> select_coins_knapsack(
    const std::vector<WalletCoin>& utxos,
    int64_t target) {

    if (utxos.empty() || target <= 0) {
        return Result<std::vector<WalletCoin>>::err("invalid inputs for knapsack");
    }

    // Sort by value ascending
    auto sorted = utxos;
    std::sort(sorted.begin(), sorted.end(),
              [](const WalletCoin& a, const WalletCoin& b) {
                  return a.txout.value < b.txout.value;
              });

    // Check total available
    int64_t total = 0;
    for (const auto& c : sorted) {
        total += c.txout.value;
    }
    if (total < target) {
        return Result<std::vector<WalletCoin>>::err("insufficient funds");
    }

    // Look for the smallest single coin >= target
    WalletCoin const* best_single = nullptr;
    for (const auto& c : sorted) {
        if (c.txout.value >= target) {
            best_single = &c;
            break;
        }
    }

    // Try subset sum approximation (randomized selection)
    std::vector<WalletCoin> best_set;
    int64_t best_excess = std::numeric_limits<int64_t>::max();

    // Make 3 randomized passes
    for (int pass = 0; pass < 3; ++pass) {
        auto shuffled = sorted;
        if (pass > 0) {
            core::random_shuffle(shuffled);
        }

        std::vector<WalletCoin> current_set;
        int64_t current_value = 0;

        for (const auto& coin : shuffled) {
            current_set.push_back(coin);
            current_value += coin.txout.value;
            if (current_value >= target) {
                int64_t excess = current_value - target;
                if (excess < best_excess) {
                    best_excess = excess;
                    best_set = current_set;
                }
                break;
            }
        }
    }

    // Compare single coin vs set
    if (best_single) {
        int64_t single_excess = best_single->txout.value - target;
        if (best_set.empty() || single_excess < best_excess) {
            return Result<std::vector<WalletCoin>>::ok(
                std::vector<WalletCoin>{*best_single});
        }
    }

    if (!best_set.empty()) {
        return Result<std::vector<WalletCoin>>::ok(std::move(best_set));
    }

    return Result<std::vector<WalletCoin>>::err("knapsack failed to find solution");
}

Result<CoinSelectionResult> select_coins(
    const std::vector<WalletCoin>& available,
    const CoinSelectionParams& params) {

    if (available.empty()) {
        return Result<CoinSelectionResult>::err("no coins available");
    }
    if (params.target_value <= 0) {
        return Result<CoinSelectionResult>::err("invalid target value");
    }

    // Filter out already-spent coins
    std::vector<WalletCoin> eligible;
    for (const auto& coin : available) {
        if (!coin.is_spent && coin.txout.value > 0) {
            eligible.push_back(coin);
        }
    }

    if (eligible.empty()) {
        return Result<CoinSelectionResult>::err("no eligible coins");
    }

    // Limit inputs
    if (eligible.size() > static_cast<size_t>(params.max_inputs)) {
        // Keep highest-value coins
        std::sort(eligible.begin(), eligible.end(),
                  [](const WalletCoin& a, const WalletCoin& b) {
                      return a.txout.value > b.txout.value;
                  });
        eligible.resize(static_cast<size_t>(params.max_inputs));
    }

    // Estimate fee for 1 output + change
    int64_t fee_with_change = estimate_tx_fee(params, eligible.size(), 1, true);
    int64_t target_with_fee = params.target_value + fee_with_change;
    int64_t cost_of_change = params.fee_rate.get_fee(params.change_output_size) +
                             params.min_change;

    // Try BnB first (exact match, no change)
    int64_t fee_no_change = estimate_tx_fee(params, eligible.size(), 1, false);
    int64_t target_no_change = params.target_value + fee_no_change;

    auto bnb_result = select_coins_bnb(eligible, target_no_change, cost_of_change);
    if (bnb_result) {
        CoinSelectionResult result;
        result.selected = std::move(bnb_result.value());
        result.total_value = 0;
        for (const auto& c : result.selected) {
            result.total_value += c.txout.value;
        }
        result.target_value = params.target_value;
        result.fee = result.total_value - params.target_value;
        result.change = 0;
        result.has_change = false;
        return Result<CoinSelectionResult>::ok(std::move(result));
    }

    // Fall back to Knapsack
    auto knapsack_result = select_coins_knapsack(eligible, target_with_fee);
    if (!knapsack_result) {
        return Result<CoinSelectionResult>::err("coin selection failed: " +
                                                knapsack_result.error());
    }

    CoinSelectionResult result;
    result.selected = std::move(knapsack_result.value());
    result.total_value = 0;
    for (const auto& c : result.selected) {
        result.total_value += c.txout.value;
    }
    result.target_value = params.target_value;

    // Recalculate fee with actual input count
    int64_t actual_fee = estimate_tx_fee(params, result.selected.size(), 1, true);
    result.change = result.total_value - params.target_value - actual_fee;

    if (result.change < params.min_change) {
        // Change too small, donate to fee
        result.fee = result.total_value - params.target_value;
        result.change = 0;
        result.has_change = false;
    } else {
        result.fee = actual_fee;
        result.has_change = true;
    }

    return Result<CoinSelectionResult>::ok(std::move(result));
}

}  // namespace rnet::wallet

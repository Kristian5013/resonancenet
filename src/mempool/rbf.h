#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/types.h"
#include "primitives/fees.h"
#include "primitives/transaction.h"

namespace rnet::mempool {

class CTxMemPoolEntry;

/// RBF policy result
enum class RBFResult {
    OK,                    ///< Replacement is acceptable
    NOT_REPLACEABLE,       ///< Original tx does not signal RBF
    INSUFFICIENT_FEE,      ///< New fee not high enough
    TOO_MANY_CONFLICTS,    ///< Would evict too many transactions
    NEW_UNCONFIRMED_INPUT, ///< New tx spends unconfirmed input not in original
};

/// Convert RBFResult to string
std::string rbf_result_string(RBFResult result);

/// Check if a transaction signals RBF (opt-in via sequence number)
bool signals_opt_in_rbf(const primitives::CTransaction& tx);

/// RBF policy parameters
struct RBFPolicy {
    /// Maximum number of transactions that can be evicted
    int64_t max_conflicts = 100;

    /// Minimum fee bump (absolute, in resonances)
    int64_t min_fee_bump = 1000;

    /// Minimum fee rate increase (per kVB)
    int64_t min_fee_rate_bump = 1000;
};

/// Check if a replacement transaction meets RBF policy requirements.
/// @param replacement    The new transaction attempting to replace
/// @param replaced_fee   Total fee of all transactions being replaced
/// @param replaced_size  Total vsize of all transactions being replaced
/// @param conflict_count Number of transactions that would be evicted
/// @param policy         RBF policy parameters
/// @return RBFResult indicating if the replacement is acceptable
RBFResult check_rbf_policy(const primitives::CTransaction& replacement,
                           int64_t replacement_fee,
                           int64_t replaced_fee,
                           size_t replaced_size,
                           int64_t conflict_count,
                           const RBFPolicy& policy = RBFPolicy{});

}  // namespace rnet::mempool

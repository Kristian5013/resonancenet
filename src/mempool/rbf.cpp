// Copyright (c) 2025 The ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "mempool/rbf.h"

#include "primitives/txin.h"

namespace rnet::mempool {

// ===========================================================================
//  Replace-By-Fee policy checks
// ===========================================================================

// ---------------------------------------------------------------------------
// rbf_result_string
//   Converts an RBFResult enum value to a human-readable string.
// ---------------------------------------------------------------------------
std::string rbf_result_string(RBFResult result)
{
    switch (result) {
        case RBFResult::OK:                    return "ok";
        case RBFResult::NOT_REPLACEABLE:       return "not-replaceable";
        case RBFResult::INSUFFICIENT_FEE:      return "insufficient-fee";
        case RBFResult::TOO_MANY_CONFLICTS:    return "too-many-conflicts";
        case RBFResult::NEW_UNCONFIRMED_INPUT: return "new-unconfirmed-input";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// signals_opt_in_rbf
//   Returns true if any input has the RBF sequence flag set.
// ---------------------------------------------------------------------------
bool signals_opt_in_rbf(const primitives::CTransaction& tx)
{
    for (const auto& txin : tx.vin()) {
        if (txin.is_rbf()) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// check_rbf_policy
//   Validates a replacement against the RBF policy: conflict count must not
//   exceed the limit and the new fee must beat the old fee by at least
//   min_fee_bump.
// ---------------------------------------------------------------------------
RBFResult check_rbf_policy(const primitives::CTransaction& /*replacement*/,
                           int64_t replacement_fee,
                           int64_t replaced_fee,
                           size_t /*replaced_size*/,
                           int64_t conflict_count,
                           const RBFPolicy& policy)
{
    // 1. Check conflict count
    if (conflict_count > policy.max_conflicts) {
        return RBFResult::TOO_MANY_CONFLICTS;
    }

    // 2. New fee must exceed old fee by at least min_fee_bump
    if (replacement_fee < replaced_fee + policy.min_fee_bump) {
        return RBFResult::INSUFFICIENT_FEE;
    }

    return RBFResult::OK;
}

} // namespace rnet::mempool

#include "mempool/rbf.h"

#include "primitives/txin.h"

namespace rnet::mempool {

std::string rbf_result_string(RBFResult result) {
    switch (result) {
        case RBFResult::OK:                    return "ok";
        case RBFResult::NOT_REPLACEABLE:       return "not-replaceable";
        case RBFResult::INSUFFICIENT_FEE:      return "insufficient-fee";
        case RBFResult::TOO_MANY_CONFLICTS:    return "too-many-conflicts";
        case RBFResult::NEW_UNCONFIRMED_INPUT: return "new-unconfirmed-input";
    }
    return "unknown";
}

bool signals_opt_in_rbf(const primitives::CTransaction& tx) {
    for (const auto& txin : tx.vin()) {
        if (txin.is_rbf()) return true;
    }
    return false;
}

RBFResult check_rbf_policy(const primitives::CTransaction& /*replacement*/,
                           int64_t replacement_fee,
                           int64_t replaced_fee,
                           size_t /*replaced_size*/,
                           int64_t conflict_count,
                           const RBFPolicy& policy)
{
    // Check conflict count
    if (conflict_count > policy.max_conflicts) {
        return RBFResult::TOO_MANY_CONFLICTS;
    }

    // New fee must exceed old fee by at least min_fee_bump
    if (replacement_fee < replaced_fee + policy.min_fee_bump) {
        return RBFResult::INSUFFICIENT_FEE;
    }

    return RBFResult::OK;
}

}  // namespace rnet::mempool

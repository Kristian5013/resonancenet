#pragma once

#include "consensus/validation.h"
#include "primitives/transaction.h"

namespace rnet::consensus {

/// Validate a transaction's internal consistency (context-free checks).
/// Checks:
///   - Non-empty vin and vout
///   - All output values in valid money range
///   - Total output value in valid money range
///   - No duplicate inputs
///   - Coinbase scriptSig size limits
///   - Non-coinbase inputs must not have null prevouts
bool check_transaction(const primitives::CTransaction& tx, ValidationState& state);

}  // namespace rnet::consensus

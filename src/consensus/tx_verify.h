#pragma once

#include "chain/coins.h"
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

/// Verify all inputs of a non-coinbase transaction against the UTXO set.
/// For each input:
///   1. Look up the coin in the UTXO set
///   2. Verify the coin is not spent
///   3. Run the script interpreter to verify the signature
/// @param tx           The transaction to verify
/// @param coins        The coins cache for UTXO lookups
/// @param state        Validation state for error reporting
/// @return true if all inputs are valid
bool check_inputs(const primitives::CTransaction& tx,
                  const chain::CCoinsViewCache& coins,
                  ValidationState& state);

}  // namespace rnet::consensus

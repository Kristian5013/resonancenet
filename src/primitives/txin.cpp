// Copyright (c) 2025 The ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "primitives/txin.h"

namespace rnet::primitives {

// ---------------------------------------------------------------------------
// CTxIn::to_string
//   Human-readable dump of outpoint, scriptSig size, sequence, and witness.
// ---------------------------------------------------------------------------
std::string CTxIn::to_string() const
{
    std::string result = "CTxIn(";
    result += prevout.to_string();
    result += ", scriptSig.size=" + std::to_string(script_sig.size());
    result += ", seq=" + std::to_string(sequence);
    if (!witness.is_null()) {
        result += ", witness=" + witness.to_string();
    }
    result += ")";
    return result;
}

} // namespace rnet::primitives

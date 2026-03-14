// Copyright (c) 2025 The ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "primitives/witness.h"

#include "core/hex.h"

namespace rnet::primitives {

// ---------------------------------------------------------------------------
// CScriptWitness::to_string
//   Renders each witness stack item as a hex-encoded string.
// ---------------------------------------------------------------------------
std::string CScriptWitness::to_string() const
{
    std::string result = "[";
    for (size_t i = 0; i < stack.size(); ++i) {
        if (i > 0) result += ", ";
        result += "0x";
        static constexpr char hex_chars[] = "0123456789abcdef";
        for (uint8_t b : stack[i]) {
            result += hex_chars[(b >> 4) & 0x0F];
            result += hex_chars[b & 0x0F];
        }
    }
    result += "]";
    return result;
}

} // namespace rnet::primitives

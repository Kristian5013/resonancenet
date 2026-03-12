#include "primitives/witness.h"

#include "core/hex.h"

namespace rnet::primitives {

std::string CScriptWitness::to_string() const {
    std::string result = "[";
    for (size_t i = 0; i < stack.size(); ++i) {
        if (i > 0) result += ", ";
        // Represent each stack item as hex
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

}  // namespace rnet::primitives

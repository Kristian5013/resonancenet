#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/serialize.h"

namespace rnet::primitives {

/// CScriptWitness — segregated witness data for a transaction input.
/// Contains a stack of byte vectors (witness items).
struct CScriptWitness {
    std::vector<std::vector<uint8_t>> stack;

    bool is_null() const { return stack.empty(); }

    void set_null() { stack.clear(); }

    /// Human-readable representation
    std::string to_string() const;

    bool operator==(const CScriptWitness& other) const {
        return stack == other.stack;
    }

    SERIALIZE_METHODS(
        READWRITE(self.stack);
    )
};

}  // namespace rnet::primitives

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/serialize.h"
#include "primitives/amount.h"

namespace rnet::primitives {

/// CTxOut — a transaction output: value + scriptPubKey.
struct CTxOut {
    int64_t value = -1;                     ///< Amount in resonances
    std::vector<uint8_t> script_pub_key;    ///< Output locking script

    CTxOut() = default;
    CTxOut(int64_t value_in, std::vector<uint8_t> script_in)
        : value(value_in), script_pub_key(std::move(script_in)) {}

    bool is_null() const { return value == -1; }

    void set_null() {
        value = -1;
        script_pub_key.clear();
    }

    /// Check if this is a P2WPKH output: OP_0 [20-byte hash]
    /// Format: [0x00][0x14][20 bytes]
    bool is_p2wpkh() const {
        return script_pub_key.size() == 22 &&
               script_pub_key[0] == 0x00 &&
               script_pub_key[1] == 0x14;
    }

    /// Check if this is a P2WSH output: OP_0 [32-byte hash]
    /// Format: [0x00][0x20][32 bytes]
    bool is_p2wsh() const {
        return script_pub_key.size() == 34 &&
               script_pub_key[0] == 0x00 &&
               script_pub_key[1] == 0x20;
    }

    /// Human-readable
    std::string to_string() const;

    bool operator==(const CTxOut& other) const {
        return value == other.value &&
               script_pub_key == other.script_pub_key;
    }

    SERIALIZE_METHODS(
        READWRITE(self.value);
        READWRITE(self.script_pub_key);
    )
};

/// Create a P2WPKH output script from a 20-byte Hash160.
std::vector<uint8_t> make_p2wpkh_script(const uint8_t* hash160);

/// Create a P2WSH output script from a 32-byte script hash.
std::vector<uint8_t> make_p2wsh_script(const uint8_t* hash256);

}  // namespace rnet::primitives

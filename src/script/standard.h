#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "script/script.h"

namespace rnet::script {

/// Transaction output script types.
enum class TxoutType {
    NONSTANDARD,
    PUBKEY,               ///< [pubkey] OP_CHECKSIG
    PUBKEYHASH,           ///< OP_DUP OP_HASH160 [20] OP_EQUALVERIFY OP_CHECKSIG (P2PKH)
    SCRIPTHASH,           ///< OP_HASH160 [20] OP_EQUAL (P2SH)
    WITNESS_V0_KEYHASH,   ///< OP_0 [20] (P2WPKH)
    WITNESS_V0_SCRIPTHASH,///< OP_0 [32] (P2WSH)
    WITNESS_UNKNOWN,      ///< OP_n [2..40] where n != 0
    NULL_DATA,            ///< OP_RETURN [data]
};

/// Get a human-readable name for a TxoutType.
std::string_view txout_type_name(TxoutType type);

/// Identify the script template and extract solution data.
///
/// For each type, solutions contains:
///   PUBKEY:               [pubkey]
///   PUBKEYHASH:           [hash160]
///   SCRIPTHASH:           [hash160]
///   WITNESS_V0_KEYHASH:   [hash160]
///   WITNESS_V0_SCRIPTHASH:[hash256]
///   WITNESS_UNKNOWN:      [program]
///   NULL_DATA:            [data] (may be empty)
///   NONSTANDARD:          (empty)
TxoutType solver(const CScript& script_pub_key,
                 std::vector<std::vector<uint8_t>>& solutions);

/// Construct a standard script for a given type and hash.
///
/// Supported types:
///   PUBKEYHASH:            P2PKH script from 20-byte hash
///   SCRIPTHASH:            P2SH script from 20-byte hash
///   WITNESS_V0_KEYHASH:    P2WPKH script from 20-byte hash
///   WITNESS_V0_SCRIPTHASH: P2WSH script from 32-byte hash
///   PUBKEY:                raw pubkey checksig (hash = pubkey bytes)
CScript get_script_for_destination(TxoutType type,
                                   const std::vector<uint8_t>& hash);

/// Check if a script is considered "standard" for relay policy.
bool is_standard_script(const CScript& script_pub_key);

/// Extract the destination hash from a standard script.
/// Returns the hash (20 or 32 bytes) and the type.
/// Returns NONSTANDARD if the script is not recognized.
TxoutType extract_destination(const CScript& script_pub_key,
                              std::vector<uint8_t>& hash_out);

}  // namespace rnet::script

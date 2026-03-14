// Copyright (c) 2024-2026 The ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "script/standard.h"

namespace rnet::script {

// ---------------------------------------------------------------------------
// txout_type_name
//
// Human-readable label for each standard output type.
// ---------------------------------------------------------------------------

std::string_view txout_type_name(TxoutType type) {
    switch (type) {
        case TxoutType::NONSTANDARD:          return "nonstandard";
        case TxoutType::PUBKEY:               return "pubkey";
        case TxoutType::PUBKEYHASH:           return "pubkeyhash";
        case TxoutType::SCRIPTHASH:           return "scripthash";
        case TxoutType::WITNESS_V0_KEYHASH:   return "witness_v0_keyhash";
        case TxoutType::WITNESS_V0_SCRIPTHASH:return "witness_v0_scripthash";
        case TxoutType::WITNESS_UNKNOWN:      return "witness_unknown";
        case TxoutType::NULL_DATA:            return "nulldata";
        default:                              return "unknown";
    }
}

// ---------------------------------------------------------------------------
// solver
//
// Template-match a scriptPubKey against known standard output patterns.
// Returns the type and extracts solutions (hashes, pubkeys).
//
// Pattern priority:
//   1. OP_RETURN (null data)
//   2. Witness programs (P2WPKH 20B, P2WSH 32B, unknown versions)
//   3. P2PKH: OP_DUP OP_HASH160 <20B> OP_EQUALVERIFY OP_CHECKSIG
//   4. P2SH:  OP_HASH160 <20B> OP_EQUAL
//   5. P2PK:  <33|65|32 byte pubkey> OP_CHECKSIG
// ---------------------------------------------------------------------------

TxoutType solver(const CScript& script_pub_key,
                 std::vector<std::vector<uint8_t>>& solutions) {
    solutions.clear();

    // 1. OP_RETURN [data] -- null data output.
    if (script_pub_key.is_unspendable()) {
        if (script_pub_key.size() > 1) {
            solutions.emplace_back(script_pub_key.begin() + 1,
                                   script_pub_key.end());
        }
        return TxoutType::NULL_DATA;
    }

    // 2. Witness programs: OP_n [2..40 bytes].
    int wit_version = -1;
    std::vector<uint8_t> wit_program;
    if (script_pub_key.is_witness_program(wit_version, wit_program)) {
        if (wit_version == 0 && wit_program.size() == 20) {
            solutions.push_back(std::move(wit_program));
            return TxoutType::WITNESS_V0_KEYHASH;
        }
        if (wit_version == 0 && wit_program.size() == 32) {
            solutions.push_back(std::move(wit_program));
            return TxoutType::WITNESS_V0_SCRIPTHASH;
        }
        if (wit_version != 0) {
            solutions.push_back(std::move(wit_program));
            return TxoutType::WITNESS_UNKNOWN;
        }
        return TxoutType::NONSTANDARD;
    }

    // 3. P2PKH: OP_DUP OP_HASH160 [20 bytes] OP_EQUALVERIFY OP_CHECKSIG.
    if (script_pub_key.size() == 25 &&
        script_pub_key[0] == static_cast<uint8_t>(Opcode::OP_DUP) &&
        script_pub_key[1] == static_cast<uint8_t>(Opcode::OP_HASH160) &&
        script_pub_key[2] == 0x14 &&
        script_pub_key[23] == static_cast<uint8_t>(Opcode::OP_EQUALVERIFY) &&
        script_pub_key[24] == static_cast<uint8_t>(Opcode::OP_CHECKSIG)) {
        solutions.emplace_back(script_pub_key.begin() + 3,
                               script_pub_key.begin() + 23);
        return TxoutType::PUBKEYHASH;
    }

    // 4. P2SH: OP_HASH160 [20 bytes] OP_EQUAL.
    if (script_pub_key.is_pay_to_script_hash()) {
        solutions.emplace_back(script_pub_key.begin() + 2,
                               script_pub_key.begin() + 22);
        return TxoutType::SCRIPTHASH;
    }

    // 5. P2PK: [33|65|32 byte pubkey] OP_CHECKSIG.
    //    The 32-byte variant supports Ed25519 raw pubkeys.
    if (script_pub_key.size() >= 35) {
        uint8_t pk_len = script_pub_key[0];
        if ((pk_len == 33 || pk_len == 65 || pk_len == 32) &&
            static_cast<size_t>(pk_len + 2) == script_pub_key.size() &&
            script_pub_key.back() == static_cast<uint8_t>(Opcode::OP_CHECKSIG)) {
            solutions.emplace_back(script_pub_key.begin() + 1,
                                   script_pub_key.begin() + 1 + pk_len);
            return TxoutType::PUBKEY;
        }
    }

    return TxoutType::NONSTANDARD;
}

// ---------------------------------------------------------------------------
// get_script_for_destination
//
// Build a scriptPubKey from a type and hash/pubkey bytes.
//
//   PUBKEYHASH  -->  OP_DUP OP_HASH160 <hash> OP_EQUALVERIFY OP_CHECKSIG
//   SCRIPTHASH  -->  OP_HASH160 <hash> OP_EQUAL
//   P2WPKH      -->  OP_0 <20-byte hash>
//   P2WSH       -->  OP_0 <32-byte hash>
//   PUBKEY      -->  <pubkey> OP_CHECKSIG
// ---------------------------------------------------------------------------

CScript get_script_for_destination(TxoutType type,
                                   const std::vector<uint8_t>& hash) {
    CScript script;
    switch (type) {
        case TxoutType::PUBKEYHASH:
            script << Opcode::OP_DUP
                   << Opcode::OP_HASH160
                   << hash
                   << Opcode::OP_EQUALVERIFY
                   << Opcode::OP_CHECKSIG;
            break;

        case TxoutType::SCRIPTHASH:
            script << Opcode::OP_HASH160
                   << hash
                   << Opcode::OP_EQUAL;
            break;

        case TxoutType::WITNESS_V0_KEYHASH:
            script << Opcode::OP_0
                   << hash;
            break;

        case TxoutType::WITNESS_V0_SCRIPTHASH:
            script << Opcode::OP_0
                   << hash;
            break;

        case TxoutType::PUBKEY:
            script << hash
                   << Opcode::OP_CHECKSIG;
            break;

        default:
            break;
    }
    return script;
}

// ---------------------------------------------------------------------------
// is_standard_script
//
// Returns true if the scriptPubKey matches any recognized standard pattern.
// ---------------------------------------------------------------------------

bool is_standard_script(const CScript& script_pub_key) {
    std::vector<std::vector<uint8_t>> solutions;
    TxoutType type = solver(script_pub_key, solutions);
    return type != TxoutType::NONSTANDARD;
}

// ---------------------------------------------------------------------------
// extract_destination
//
// Convenience wrapper: solve the script and return the first solution
// (the destination hash or pubkey) along with the type.
// ---------------------------------------------------------------------------

TxoutType extract_destination(const CScript& script_pub_key,
                              std::vector<uint8_t>& hash_out) {
    std::vector<std::vector<uint8_t>> solutions;
    TxoutType type = solver(script_pub_key, solutions);

    if (!solutions.empty()) {
        hash_out = solutions[0];
    } else {
        hash_out.clear();
    }
    return type;
}

} // namespace rnet::script

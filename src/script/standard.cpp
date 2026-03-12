#include "script/standard.h"

namespace rnet::script {

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

TxoutType solver(const CScript& script_pub_key,
                 std::vector<std::vector<uint8_t>>& solutions) {
    solutions.clear();

    // OP_RETURN [data]
    if (script_pub_key.is_unspendable()) {
        if (script_pub_key.size() > 1) {
            solutions.emplace_back(script_pub_key.begin() + 1,
                                   script_pub_key.end());
        }
        return TxoutType::NULL_DATA;
    }

    // Witness programs: OP_n [2..40 bytes]
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

    // P2PKH: OP_DUP OP_HASH160 [20 bytes] OP_EQUALVERIFY OP_CHECKSIG
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

    // P2SH: OP_HASH160 [20 bytes] OP_EQUAL
    if (script_pub_key.is_pay_to_script_hash()) {
        solutions.emplace_back(script_pub_key.begin() + 2,
                               script_pub_key.begin() + 22);
        return TxoutType::SCRIPTHASH;
    }

    // P2PK: [33 or 65 byte pubkey] OP_CHECKSIG
    // Also Ed25519: [32 byte pubkey prefix 0x20] OP_CHECKSIG
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
            // OP_0 [20-byte hash]
            script << Opcode::OP_0
                   << hash;
            break;

        case TxoutType::WITNESS_V0_SCRIPTHASH:
            // OP_0 [32-byte hash]
            script << Opcode::OP_0
                   << hash;
            break;

        case TxoutType::PUBKEY:
            // [pubkey] OP_CHECKSIG
            script << hash
                   << Opcode::OP_CHECKSIG;
            break;

        default:
            break;
    }
    return script;
}

bool is_standard_script(const CScript& script_pub_key) {
    std::vector<std::vector<uint8_t>> solutions;
    TxoutType type = solver(script_pub_key, solutions);
    return type != TxoutType::NONSTANDARD;
}

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

}  // namespace rnet::script

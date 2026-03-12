#include "script/recovery_script.h"

#include "crypto/hash.h"
#include "crypto/keccak.h"

namespace rnet::script {

CScript build_recovery_script(
    const std::vector<uint8_t>& owner_pubkey_hash,
    RecoveryType type,
    const RecoveryPolicy& policy) {

    CScript script;

    // Normal spend path (owner)
    script << Opcode::OP_IF;
    script << Opcode::OP_DUP;
    script << Opcode::OP_HASH160;
    script << owner_pubkey_hash;
    script << Opcode::OP_EQUALVERIFY;
    script << Opcode::OP_CHECKSIG;

    // Recovery path
    script << Opcode::OP_ELSE;

    switch (type) {
        case RecoveryType::HEARTBEAT: {
            const auto& hp = std::get<HeartbeatPolicy>(policy);

            // Require sequence lock (relative timelock)
            script << static_cast<int64_t>(hp.interval);
            script << Opcode::OP_CHECKSEQUENCEVERIFY;
            script << Opcode::OP_DROP;

            // Recovery key can spend
            script << Opcode::OP_DUP;
            script << Opcode::OP_HASH160;
            script << hp.recovery_pubkey_hash;
            script << Opcode::OP_EQUALVERIFY;
            script << Opcode::OP_CHECKSIG;
            break;
        }

        case RecoveryType::SOCIAL: {
            const auto& sp = std::get<SocialPolicy>(policy);

            // Require waiting period
            script << static_cast<int64_t>(sp.waiting_period);
            script << Opcode::OP_CHECKSEQUENCEVERIFY;
            script << Opcode::OP_DROP;

            // Multisig of guardians
            script << static_cast<int64_t>(sp.threshold);
            for (const auto& guardian_pk : sp.guardian_pubkeys) {
                std::vector<uint8_t> pk(guardian_pk.begin(), guardian_pk.end());
                script << pk;
            }
            script << static_cast<int64_t>(sp.guardian_pubkeys.size());
            script << Opcode::OP_CHECKMULTISIG;
            break;
        }

        case RecoveryType::EMISSION: {
            const auto& ep = std::get<EmissionPolicy>(policy);

            // Require inactivity period
            script << static_cast<int64_t>(ep.inactivity_period);
            script << Opcode::OP_CHECKSEQUENCEVERIFY;
            script << Opcode::OP_DROP;

            // After inactivity, anyone can spend (coins return to mining).
            // Use OP_TRUE to allow spending by miners.
            script << Opcode::OP_TRUE;
            break;
        }
    }

    script << Opcode::OP_ENDIF;
    return script;
}

bool parse_recovery_script(const CScript& script,
                           RecoveryType& type,
                           RecoveryPolicy& policy) {
    // Basic validation: must start with OP_IF and end with OP_ENDIF
    if (script.size() < 10) return false;
    if (script[0] != static_cast<uint8_t>(Opcode::OP_IF)) return false;
    if (script.back() != static_cast<uint8_t>(Opcode::OP_ENDIF)) return false;

    // Walk through the script to find the OP_ELSE boundary
    ScriptIterator it(script);
    Opcode op;
    std::vector<uint8_t> data;

    // Skip OP_IF
    if (!it.next(op, data) || op != Opcode::OP_IF) return false;

    // Skip the owner spend path until OP_ELSE
    int depth = 1;
    while (it.next(op, data)) {
        if (op == Opcode::OP_IF || op == Opcode::OP_NOTIF) ++depth;
        if (op == Opcode::OP_ENDIF) --depth;
        if (op == Opcode::OP_ELSE && depth == 1) break;
    }

    if (op != Opcode::OP_ELSE) return false;

    // Parse the recovery path
    // First element should be the timelock value
    if (!it.next(op, data)) return false;

    int64_t timelock = 0;
    if (!data.empty()) {
        timelock = scriptnum_decode(data, data.size(), false);
    } else {
        int n = decode_op_n(op);
        if (n >= 0) {
            timelock = n;
        } else {
            return false;
        }
    }

    // Next should be OP_CHECKSEQUENCEVERIFY
    if (!it.next(op, data) || op != Opcode::OP_CHECKSEQUENCEVERIFY) {
        return false;
    }

    // Next should be OP_DROP
    if (!it.next(op, data) || op != Opcode::OP_DROP) {
        return false;
    }

    // Now determine the recovery type based on what follows.
    size_t saved_pos = it.pos();
    if (!it.next(op, data)) return false;

    // Check for OP_TRUE (emission)
    if (op == Opcode::OP_TRUE || op == Opcode::OP_1) {
        type = RecoveryType::EMISSION;
        EmissionPolicy ep;
        ep.inactivity_period = static_cast<uint64_t>(timelock);
        policy = ep;
        return true;
    }

    // Check for threshold number (social recovery starts with a number)
    int threshold = decode_op_n(op);
    if (threshold > 0 && data.empty()) {
        // Could be social recovery: threshold + pubkeys + n + OP_CHECKMULTISIG
        SocialPolicy sp;
        sp.threshold = static_cast<uint8_t>(threshold);
        sp.waiting_period = static_cast<uint64_t>(timelock);

        // Read guardian pubkeys
        while (it.next(op, data)) {
            if (data.size() == 32) {
                std::array<uint8_t, 32> pk;
                std::copy(data.begin(), data.end(), pk.begin());
                sp.guardian_pubkeys.push_back(pk);
            } else if (op == Opcode::OP_CHECKMULTISIG) {
                break;
            } else {
                // This might be the N count
                int n = decode_op_n(op);
                if (n > 0) continue;  // skip the count
                break;
            }
        }

        if (op == Opcode::OP_CHECKMULTISIG && !sp.guardian_pubkeys.empty()) {
            type = RecoveryType::SOCIAL;
            policy = sp;
            return true;
        }
        return false;
    }

    // Otherwise, it's a heartbeat (P2PKH-style recovery spend)
    if (op == Opcode::OP_DUP) {
        // OP_DUP OP_HASH160 <hash> OP_EQUALVERIFY OP_CHECKSIG
        if (!it.next(op, data) || op != Opcode::OP_HASH160) return false;
        if (!it.next(op, data) || data.size() != 20) return false;

        HeartbeatPolicy hp;
        hp.interval = static_cast<uint64_t>(timelock);
        hp.recovery_pubkey_hash = data;

        type = RecoveryType::HEARTBEAT;
        policy = hp;
        return true;
    }

    return false;
}

CScript build_recovery_p2wsh(const CScript& recovery_script) {
    // P2WSH = OP_0 [32-byte keccak256d(script)]
    auto script_hash = rnet::crypto::keccak256d(
        std::span<const uint8_t>(recovery_script.data(),
                                 recovery_script.size()));

    CScript p2wsh;
    p2wsh << Opcode::OP_0;
    std::vector<uint8_t> hash_vec(script_hash.begin(), script_hash.end());
    p2wsh << hash_vec;
    return p2wsh;
}

std::vector<std::vector<uint8_t>> build_owner_spend_witness(
    const std::vector<uint8_t>& signature,
    const CScript& recovery_script) {
    std::vector<std::vector<uint8_t>> witness;

    // signature
    witness.push_back(signature);

    // OP_TRUE to take the IF branch
    witness.push_back({0x01});

    // The recovery script itself (as the witness script)
    witness.emplace_back(recovery_script.begin(), recovery_script.end());

    return witness;
}

std::vector<std::vector<uint8_t>> build_heartbeat_recovery_witness(
    const std::vector<uint8_t>& signature,
    const CScript& recovery_script) {
    std::vector<std::vector<uint8_t>> witness;

    // signature
    witness.push_back(signature);

    // OP_FALSE to take the ELSE branch
    witness.emplace_back();  // empty = false

    // The recovery script itself
    witness.emplace_back(recovery_script.begin(), recovery_script.end());

    return witness;
}

std::vector<std::vector<uint8_t>> build_social_recovery_witness(
    const std::vector<std::vector<uint8_t>>& signatures,
    const CScript& recovery_script) {
    std::vector<std::vector<uint8_t>> witness;

    // Dummy element for CHECKMULTISIG bug
    witness.emplace_back();

    // Guardian signatures
    for (const auto& sig : signatures) {
        witness.push_back(sig);
    }

    // OP_FALSE to take the ELSE branch
    witness.emplace_back();  // empty = false

    // The recovery script itself
    witness.emplace_back(recovery_script.begin(), recovery_script.end());

    return witness;
}

}  // namespace rnet::script

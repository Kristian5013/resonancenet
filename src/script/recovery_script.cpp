// Copyright (c) 2024-2026 The ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "script/recovery_script.h"

#include "crypto/hash.h"
#include "crypto/keccak.h"

namespace rnet::script {

// ---------------------------------------------------------------------------
// build_recovery_script
//
// Constructs a Bitcoin-style recovery script with two spend paths:
//
//   OP_IF
//     <owner P2PKH spend>          -- normal path
//   OP_ELSE
//     <timelock> OP_CSV OP_DROP
//     <recovery condition>         -- heartbeat / social / emission
//   OP_ENDIF
//
// Recovery types:
//   HEARTBEAT  -- single recovery key after inactivity interval
//   SOCIAL     -- M-of-N guardian multisig after waiting period
//   EMISSION   -- anyone-can-spend (coins return to mining) after inactivity
// ---------------------------------------------------------------------------

CScript build_recovery_script(
    const std::vector<uint8_t>& owner_pubkey_hash,
    RecoveryType type,
    const RecoveryPolicy& policy) {

    CScript script;

    // 1. Normal spend path (owner P2PKH).
    script << Opcode::OP_IF;
    script << Opcode::OP_DUP;
    script << Opcode::OP_HASH160;
    script << owner_pubkey_hash;
    script << Opcode::OP_EQUALVERIFY;
    script << Opcode::OP_CHECKSIG;

    // 2. Recovery path (time-locked alternative).
    script << Opcode::OP_ELSE;

    switch (type) {
        case RecoveryType::HEARTBEAT: {
            const auto& hp = std::get<HeartbeatPolicy>(policy);

            // 2a. Require relative timelock (sequence lock).
            script << static_cast<int64_t>(hp.interval);
            script << Opcode::OP_CHECKSEQUENCEVERIFY;
            script << Opcode::OP_DROP;

            // 2b. Recovery key P2PKH.
            script << Opcode::OP_DUP;
            script << Opcode::OP_HASH160;
            script << hp.recovery_pubkey_hash;
            script << Opcode::OP_EQUALVERIFY;
            script << Opcode::OP_CHECKSIG;
            break;
        }

        case RecoveryType::SOCIAL: {
            const auto& sp = std::get<SocialPolicy>(policy);

            // 2a. Require waiting period.
            script << static_cast<int64_t>(sp.waiting_period);
            script << Opcode::OP_CHECKSEQUENCEVERIFY;
            script << Opcode::OP_DROP;

            // 2b. M-of-N guardian multisig.
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

            // 2a. Require inactivity period.
            script << static_cast<int64_t>(ep.inactivity_period);
            script << Opcode::OP_CHECKSEQUENCEVERIFY;
            script << Opcode::OP_DROP;

            // 2b. Anyone-can-spend (coins return to mining rewards).
            script << Opcode::OP_TRUE;
            break;
        }
    }

    script << Opcode::OP_ENDIF;
    return script;
}

// ---------------------------------------------------------------------------
// parse_recovery_script
//
// Reverse-engineers a recovery script to extract its type and policy.
// Walks the bytecode past the OP_IF/OP_ELSE boundary, then identifies
// the recovery variant by the opcodes that follow the timelock.
//
//   OP_TRUE                     --> EMISSION
//   threshold + pubkeys + CMS   --> SOCIAL
//   OP_DUP (P2PKH pattern)      --> HEARTBEAT
// ---------------------------------------------------------------------------

bool parse_recovery_script(const CScript& script,
                           RecoveryType& type,
                           RecoveryPolicy& policy) {
    // 1. Basic structure validation.
    if (script.size() < 10) return false;
    if (script[0] != static_cast<uint8_t>(Opcode::OP_IF)) return false;
    if (script.back() != static_cast<uint8_t>(Opcode::OP_ENDIF)) return false;

    // 2. Walk to the OP_ELSE boundary.
    ScriptIterator it(script);
    Opcode op;
    std::vector<uint8_t> data;

    if (!it.next(op, data) || op != Opcode::OP_IF) return false;

    int depth = 1;
    while (it.next(op, data)) {
        if (op == Opcode::OP_IF || op == Opcode::OP_NOTIF) ++depth;
        if (op == Opcode::OP_ENDIF) --depth;
        if (op == Opcode::OP_ELSE && depth == 1) break;
    }

    if (op != Opcode::OP_ELSE) return false;

    // 3. Parse the timelock value.
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

    // 4. Expect OP_CHECKSEQUENCEVERIFY OP_DROP.
    if (!it.next(op, data) || op != Opcode::OP_CHECKSEQUENCEVERIFY) {
        return false;
    }
    if (!it.next(op, data) || op != Opcode::OP_DROP) {
        return false;
    }

    // 5. Determine recovery type from what follows.
    size_t saved_pos = it.pos();
    if (!it.next(op, data)) return false;

    // 5a. OP_TRUE -> emission (anyone-can-spend after inactivity).
    if (op == Opcode::OP_TRUE || op == Opcode::OP_1) {
        type = RecoveryType::EMISSION;
        EmissionPolicy ep;
        ep.inactivity_period = static_cast<uint64_t>(timelock);
        policy = ep;
        return true;
    }

    // 5b. Threshold number -> social recovery (M-of-N multisig).
    int threshold = decode_op_n(op);
    if (threshold > 0 && data.empty()) {
        SocialPolicy sp;
        sp.threshold = static_cast<uint8_t>(threshold);
        sp.waiting_period = static_cast<uint64_t>(timelock);

        while (it.next(op, data)) {
            if (data.size() == 32) {
                std::array<uint8_t, 32> pk;
                std::copy(data.begin(), data.end(), pk.begin());
                sp.guardian_pubkeys.push_back(pk);
            } else if (op == Opcode::OP_CHECKMULTISIG) {
                break;
            } else {
                int n = decode_op_n(op);
                if (n > 0) continue;
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

    // 5c. OP_DUP -> heartbeat (single recovery key P2PKH).
    if (op == Opcode::OP_DUP) {
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

// ---------------------------------------------------------------------------
// build_recovery_p2wsh
//
// Wrap a recovery script as P2WSH:  OP_0 [32-byte Keccak256d(script)]
// ---------------------------------------------------------------------------

CScript build_recovery_p2wsh(const CScript& recovery_script) {
    // 1. Hash the recovery script with Keccak256d.
    auto script_hash = rnet::crypto::keccak256d(
        std::span<const uint8_t>(recovery_script.data(),
                                 recovery_script.size()));

    // 2. Build OP_0 <32-byte hash>.
    CScript p2wsh;
    p2wsh << Opcode::OP_0;
    std::vector<uint8_t> hash_vec(script_hash.begin(), script_hash.end());
    p2wsh << hash_vec;
    return p2wsh;
}

// ---------------------------------------------------------------------------
// build_owner_spend_witness
//
// Witness stack for the owner (IF) branch:
//   [ signature, OP_TRUE (0x01), recovery_script ]
// ---------------------------------------------------------------------------

std::vector<std::vector<uint8_t>> build_owner_spend_witness(
    const std::vector<uint8_t>& signature,
    const CScript& recovery_script) {
    std::vector<std::vector<uint8_t>> witness;

    // 1. Signature.
    witness.push_back(signature);

    // 2. OP_TRUE to select the IF branch.
    witness.push_back({0x01});

    // 3. The recovery script (witness script for P2WSH).
    witness.emplace_back(recovery_script.begin(), recovery_script.end());

    return witness;
}

// ---------------------------------------------------------------------------
// build_heartbeat_recovery_witness
//
// Witness stack for heartbeat recovery (ELSE branch):
//   [ signature, <empty> (OP_FALSE), recovery_script ]
// ---------------------------------------------------------------------------

std::vector<std::vector<uint8_t>> build_heartbeat_recovery_witness(
    const std::vector<uint8_t>& signature,
    const CScript& recovery_script) {
    std::vector<std::vector<uint8_t>> witness;

    // 1. Signature.
    witness.push_back(signature);

    // 2. Empty element (OP_FALSE) to select the ELSE branch.
    witness.emplace_back();

    // 3. The recovery script.
    witness.emplace_back(recovery_script.begin(), recovery_script.end());

    return witness;
}

// ---------------------------------------------------------------------------
// build_social_recovery_witness
//
// Witness stack for social recovery (ELSE branch, multisig):
//   [ <dummy>, sig1, sig2, ..., <empty> (OP_FALSE), recovery_script ]
//
// The leading dummy element works around the CHECKMULTISIG off-by-one bug.
// ---------------------------------------------------------------------------

std::vector<std::vector<uint8_t>> build_social_recovery_witness(
    const std::vector<std::vector<uint8_t>>& signatures,
    const CScript& recovery_script) {
    std::vector<std::vector<uint8_t>> witness;

    // 1. Dummy element for CHECKMULTISIG bug.
    witness.emplace_back();

    // 2. Guardian signatures.
    for (const auto& sig : signatures) {
        witness.push_back(sig);
    }

    // 3. Empty element (OP_FALSE) to select the ELSE branch.
    witness.emplace_back();

    // 4. The recovery script.
    witness.emplace_back(recovery_script.begin(), recovery_script.end());

    return witness;
}

} // namespace rnet::script

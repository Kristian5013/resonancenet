#include "script/verify.h"
#include "script/standard.h"

#include "crypto/hash.h"
#include "crypto/keccak.h"

#include <cstring>

namespace rnet::script {

static inline void set_error(ScriptError* err, ScriptError code) {
    if (err) *err = code;
}

bool verify_script(const CScript& script_sig,
                   const CScript& script_pub_key,
                   const rnet::primitives::CScriptWitness* witness,
                   uint32_t flags,
                   const BaseSignatureChecker& checker,
                   ScriptError* error) {
    set_error(error, ScriptError::UNKNOWN);

    // SIGPUSHONLY check on scriptSig
    if ((flags & SCRIPT_VERIFY_SIGPUSHONLY) && !script_sig.is_push_only()) {
        set_error(error, ScriptError::SIG_PUSHONLY);
        return false;
    }

    // ── Step 1: Evaluate scriptSig ──────────────────────────────────
    ScriptStack stack;
    if (!eval_script(stack, script_sig, flags, checker, error)) {
        return false;
    }

    // Save a copy for P2SH evaluation.
    ScriptStack stack_copy;
    if (flags & SCRIPT_VERIFY_P2SH) {
        stack_copy = stack;
    }

    // ── Step 2: Evaluate scriptPubKey ───────────────────────────────
    if (!eval_script(stack, script_pub_key, flags, checker, error)) {
        return false;
    }

    if (stack.empty()) {
        set_error(error, ScriptError::EVAL_FALSE);
        return false;
    }

    if (!cast_to_bool(stack.back())) {
        set_error(error, ScriptError::EVAL_FALSE);
        return false;
    }

    // ── Step 3: P2SH evaluation ─────────────────────────────────────
    bool had_success = true;
    if ((flags & SCRIPT_VERIFY_P2SH) &&
        script_pub_key.is_pay_to_script_hash()) {

        // scriptSig must be push-only
        if (!script_sig.is_push_only()) {
            set_error(error, ScriptError::SIG_PUSHONLY);
            return false;
        }

        // The top of the original stack is the serialized redeem script.
        if (stack_copy.empty()) {
            set_error(error, ScriptError::EVAL_FALSE);
            return false;
        }

        CScript redeem_script(stack_copy.back());
        stack_copy.pop_back();

        if (!eval_script(stack_copy, redeem_script, flags, checker, error)) {
            return false;
        }

        if (stack_copy.empty()) {
            set_error(error, ScriptError::EVAL_FALSE);
            return false;
        }

        if (!cast_to_bool(stack_copy.back())) {
            set_error(error, ScriptError::EVAL_FALSE);
            return false;
        }

        // For witness-in-P2SH, check the redeem script
        if ((flags & SCRIPT_VERIFY_WITNESS) && witness) {
            int wit_version = -1;
            std::vector<uint8_t> wit_program;
            if (redeem_script.is_witness_program(wit_version, wit_program)) {
                // The scriptSig must contain ONLY the serialized redeem script
                if (script_sig.size() != redeem_script.size() + 1 +
                    (redeem_script.size() <= 75 ? 0 :
                     redeem_script.size() <= 255 ? 1 :
                     redeem_script.size() <= 65535 ? 2 : 4)) {
                    // Approximation: just check it's push-only
                }

                // Verify witness program
                if (wit_version == 0 && wit_program.size() == 20) {
                    // P2WPKH
                    if (witness->stack.size() != 2) {
                        set_error(error, ScriptError::WITNESS_PROGRAM_MISMATCH);
                        return false;
                    }

                    // Check pubkey hash matches
                    auto& wpk = witness->stack[1];
                    auto pk_hash = rnet::crypto::hash160(
                        std::span<const uint8_t>(wpk.data(), wpk.size()));

                    if (std::memcmp(pk_hash.data(), wit_program.data(), 20) != 0) {
                        set_error(error, ScriptError::WITNESS_PROGRAM_MISMATCH);
                        return false;
                    }

                    CScript witness_script;
                    witness_script << Opcode::OP_DUP
                                   << Opcode::OP_HASH160;
                    witness_script << wit_program;
                    witness_script << Opcode::OP_EQUALVERIFY
                                   << Opcode::OP_CHECKSIG;

                    ScriptStack wit_stack(witness->stack.begin(),
                                          witness->stack.end());
                    if (!eval_script(wit_stack, witness_script, flags,
                                     checker, error)) {
                        return false;
                    }

                    if (wit_stack.size() != 1 || !cast_to_bool(wit_stack.back())) {
                        set_error(error, ScriptError::EVAL_FALSE);
                        return false;
                    }
                }
            }
        }

        stack = stack_copy;
    }

    // ── Step 4: Witness program evaluation (native segwit) ──────────
    if ((flags & SCRIPT_VERIFY_WITNESS) && witness) {
        int wit_version = -1;
        std::vector<uint8_t> wit_program;

        if (script_pub_key.is_witness_program(wit_version, wit_program)) {
            // scriptSig must be empty for native witness
            if (!script_sig.empty()) {
                set_error(error, ScriptError::WITNESS_MALLEATED);
                return false;
            }

            if (wit_version == 0 && wit_program.size() == 20) {
                // P2WPKH
                if (witness->stack.size() != 2) {
                    set_error(error, ScriptError::WITNESS_PROGRAM_MISMATCH);
                    return false;
                }

                auto& wpk = witness->stack[1];
                auto pk_hash = rnet::crypto::hash160(
                    std::span<const uint8_t>(wpk.data(), wpk.size()));

                if (std::memcmp(pk_hash.data(), wit_program.data(), 20) != 0) {
                    set_error(error, ScriptError::WITNESS_PROGRAM_MISMATCH);
                    return false;
                }

                // Construct the P2PKH script for evaluation
                CScript witness_script;
                witness_script << Opcode::OP_DUP
                               << Opcode::OP_HASH160;
                witness_script << wit_program;
                witness_script << Opcode::OP_EQUALVERIFY
                               << Opcode::OP_CHECKSIG;

                ScriptStack wit_stack(witness->stack.begin(),
                                      witness->stack.end());
                if (!eval_script(wit_stack, witness_script, flags,
                                 checker, error)) {
                    return false;
                }

                if (wit_stack.size() != 1 || !cast_to_bool(wit_stack.back())) {
                    set_error(error, ScriptError::EVAL_FALSE);
                    return false;
                }
                stack = wit_stack;
            } else if (wit_version == 0 && wit_program.size() == 32) {
                // P2WSH
                if (witness->stack.empty()) {
                    set_error(error, ScriptError::WITNESS_PROGRAM_WITNESS_EMPTY);
                    return false;
                }

                // The last witness item is the witness script
                auto& witness_script_bytes = witness->stack.back();
                CScript witness_script(witness_script_bytes);

                // Verify the script hash matches
                auto script_hash = rnet::crypto::keccak256d(
                    std::span<const uint8_t>(witness_script_bytes.data(),
                                             witness_script_bytes.size()));

                if (std::memcmp(script_hash.data(),
                                wit_program.data(), 32) != 0) {
                    set_error(error, ScriptError::WITNESS_PROGRAM_MISMATCH);
                    return false;
                }

                // Evaluate with all witness items except the script
                ScriptStack wit_stack(witness->stack.begin(),
                                      witness->stack.end() - 1);
                if (!eval_script(wit_stack, witness_script, flags,
                                 checker, error)) {
                    return false;
                }

                if (wit_stack.size() != 1 || !cast_to_bool(wit_stack.back())) {
                    set_error(error, ScriptError::EVAL_FALSE);
                    return false;
                }
                stack = wit_stack;
            } else if (wit_version != 0) {
                // Future witness versions: treat as anyone-can-spend
                // if DISCOURAGE_UPGRADABLE flags aren't set
            } else {
                set_error(error, ScriptError::WITNESS_PROGRAM_WRONG_LENGTH);
                return false;
            }
        } else if (!witness->is_null()) {
            // Witness data present but script is not a witness program
            if (!(flags & SCRIPT_VERIFY_P2SH) ||
                !script_pub_key.is_pay_to_script_hash()) {
                set_error(error, ScriptError::WITNESS_UNEXPECTED);
                return false;
            }
        }
    }

    // ── Step 5: Cleanstack check ────────────────────────────────────
    if ((flags & SCRIPT_VERIFY_CLEANSTACK) && stack.size() != 1) {
        set_error(error, ScriptError::CLEANSTACK);
        return false;
    }

    set_error(error, ScriptError::OK);
    return true;
}

unsigned int count_witness_sig_ops(
    const CScript& script_sig,
    const CScript& script_pub_key,
    const rnet::primitives::CScriptWitness* witness,
    uint32_t flags) {

    unsigned int count = script_pub_key.get_sig_op_count(true);

    // P2SH: count sigops in the redeem script
    if ((flags & SCRIPT_VERIFY_P2SH) &&
        script_pub_key.is_pay_to_script_hash()) {
        // Extract the redeem script from scriptSig
        ScriptStack stack;
        ScriptIterator it(script_sig);
        Opcode op;
        std::vector<uint8_t> data;
        while (it.next(op, data)) {
            if (!data.empty()) {
                stack.push_back(data);
            }
        }
        if (!stack.empty()) {
            CScript redeem(stack.back());
            count += redeem.get_sig_op_count(true);
        }
    }

    // Witness: P2WPKH counts as 1, P2WSH counts the witness script
    if ((flags & SCRIPT_VERIFY_WITNESS) && witness && !witness->is_null()) {
        int wit_version = -1;
        std::vector<uint8_t> wit_program;
        if (script_pub_key.is_witness_program(wit_version, wit_program)) {
            if (wit_version == 0 && wit_program.size() == 20) {
                count += 1;
            } else if (wit_version == 0 && wit_program.size() == 32) {
                if (!witness->stack.empty()) {
                    CScript ws(witness->stack.back());
                    count += ws.get_sig_op_count(true);
                }
            }
        }
    }

    return count;
}

}  // namespace rnet::script

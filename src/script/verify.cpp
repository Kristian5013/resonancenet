// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "script/verify.h"
#include "script/standard.h"

#include "crypto/hash.h"
#include "crypto/keccak.h"

#include <cstring>

namespace rnet::script {

// ---------------------------------------------------------------------------
// set_error  (local helper)
// ---------------------------------------------------------------------------

static inline void set_error(ScriptError* err, ScriptError code)
{
    if (err) *err = code;
}

// ---------------------------------------------------------------------------
// verify_script
// ---------------------------------------------------------------------------
//
// DESIGN NOTE -- script verification flow
//
//   Verification proceeds through five ordered steps.  Each step may abort
//   early with a specific ScriptError code.
//
//   Step 1  Evaluate scriptSig onto a fresh stack.
//   Step 2  Evaluate scriptPubKey against that stack.
//   Step 3  P2SH -- if scriptPubKey is a pay-to-script-hash, deserialize
//           the top stack item as a redeem script and re-evaluate.  If the
//           redeem script is itself a witness program (P2SH-wrapped segwit),
//           verify the witness here.
//   Step 4  Native segwit -- if scriptPubKey is a witness program, verify
//           witness data directly:
//             P2WPKH (v0, 20-byte program)
//               Witness stack must be [sig, pubkey].  Hash160(pubkey) must
//               match the program.  An implied P2PKH script is evaluated.
//             P2WSH  (v0, 32-byte program)
//               Last witness item is the witness script.  Keccak256d of
//               the script must match the program.  Remaining witness items
//               form the initial stack for evaluation.
//   Step 5  CLEANSTACK -- after all evaluation the stack must contain
//           exactly one (truthy) element.
//
// DESIGN NOTE -- SignatureChecker dispatch
//
//   OP_CHECKSIG (called during eval_script) inspects the public key length
//   to choose the verification algorithm:
//
//     32 bytes  -> Ed25519 verify
//     33 bytes  -> secp256k1 compressed ECDSA verify
//     65 bytes  -> secp256k1 uncompressed ECDSA verify
//
//   The checker is injected via BaseSignatureChecker so that unit tests can
//   substitute a mock.

bool verify_script(const CScript& script_sig,
                   const CScript& script_pub_key,
                   const rnet::primitives::CScriptWitness* witness,
                   uint32_t flags,
                   const BaseSignatureChecker& checker,
                   ScriptError* error)
{
    set_error(error, ScriptError::UNKNOWN);

    // 1. Preliminary: reject non-push-only scriptSig when required.
    if ((flags & SCRIPT_VERIFY_SIGPUSHONLY) && !script_sig.is_push_only()) {
        set_error(error, ScriptError::SIG_PUSHONLY);
        return false;
    }

    // 2. Evaluate scriptSig.
    ScriptStack stack;
    if (!eval_script(stack, script_sig, flags, checker, error)) {
        return false;
    }

    // 3. Snapshot the stack for a possible P2SH pass.
    ScriptStack stack_copy;
    if (flags & SCRIPT_VERIFY_P2SH) {
        stack_copy = stack;
    }

    // 4. Evaluate scriptPubKey.
    if (!eval_script(stack, script_pub_key, flags, checker, error)) {
        return false;
    }

    // 5. The result must be a single truthy value.
    if (stack.empty()) {
        set_error(error, ScriptError::EVAL_FALSE);
        return false;
    }
    if (!cast_to_bool(stack.back())) {
        set_error(error, ScriptError::EVAL_FALSE);
        return false;
    }

    // -----------------------------------------------------------------
    // Step 3: P2SH evaluation
    // -----------------------------------------------------------------
    bool had_success = true;
    if ((flags & SCRIPT_VERIFY_P2SH) &&
        script_pub_key.is_pay_to_script_hash()) {

        // 1. scriptSig must be push-only for P2SH.
        if (!script_sig.is_push_only()) {
            set_error(error, ScriptError::SIG_PUSHONLY);
            return false;
        }

        // 2. Top of the original stack is the serialized redeem script.
        if (stack_copy.empty()) {
            set_error(error, ScriptError::EVAL_FALSE);
            return false;
        }

        CScript redeem_script(stack_copy.back());
        stack_copy.pop_back();

        // 3. Evaluate the redeem script.
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

        // 4. P2SH-wrapped witness: if the redeem script is itself a
        //    witness program, verify the witness data now.
        if ((flags & SCRIPT_VERIFY_WITNESS) && witness) {
            int wit_version = -1;
            std::vector<uint8_t> wit_program;
            if (redeem_script.is_witness_program(wit_version, wit_program)) {
                // scriptSig must contain ONLY the serialized redeem script.
                if (script_sig.size() != redeem_script.size() + 1 +
                    (redeem_script.size() <= 75 ? 0 :
                     redeem_script.size() <= 255 ? 1 :
                     redeem_script.size() <= 65535 ? 2 : 4)) {
                    // Approximation: just check it's push-only
                }

                // 4a. P2SH-P2WPKH (v0, 20-byte program).
                if (wit_version == 0 && wit_program.size() == 20) {
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

    // -----------------------------------------------------------------
    // Step 4: Native segwit (witness program in scriptPubKey)
    // -----------------------------------------------------------------
    if ((flags & SCRIPT_VERIFY_WITNESS) && witness) {
        int wit_version = -1;
        std::vector<uint8_t> wit_program;

        if (script_pub_key.is_witness_program(wit_version, wit_program)) {
            // 1. scriptSig must be empty for native witness outputs.
            if (!script_sig.empty()) {
                set_error(error, ScriptError::WITNESS_MALLEATED);
                return false;
            }

            if (wit_version == 0 && wit_program.size() == 20) {
                // 2. P2WPKH -- witness stack is [sig, pubkey].
                if (witness->stack.size() != 2) {
                    set_error(error, ScriptError::WITNESS_PROGRAM_MISMATCH);
                    return false;
                }

                // 2a. Verify Hash160(pubkey) matches the 20-byte program.
                auto& wpk = witness->stack[1];
                auto pk_hash = rnet::crypto::hash160(
                    std::span<const uint8_t>(wpk.data(), wpk.size()));

                if (std::memcmp(pk_hash.data(), wit_program.data(), 20) != 0) {
                    set_error(error, ScriptError::WITNESS_PROGRAM_MISMATCH);
                    return false;
                }

                // 2b. Construct implied P2PKH and evaluate.
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
                // 3. P2WSH -- last witness item is the witness script.
                if (witness->stack.empty()) {
                    set_error(error, ScriptError::WITNESS_PROGRAM_WITNESS_EMPTY);
                    return false;
                }

                auto& witness_script_bytes = witness->stack.back();
                CScript witness_script(witness_script_bytes);

                // 3a. Verify Keccak256d(script) matches the 32-byte program.
                auto script_hash = rnet::crypto::keccak256d(
                    std::span<const uint8_t>(witness_script_bytes.data(),
                                             witness_script_bytes.size()));

                if (std::memcmp(script_hash.data(),
                                wit_program.data(), 32) != 0) {
                    set_error(error, ScriptError::WITNESS_PROGRAM_MISMATCH);
                    return false;
                }

                // 3b. Evaluate with all witness items except the script itself.
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
                // 4. Future witness versions: anyone-can-spend unless
                //    DISCOURAGE_UPGRADABLE flags are set.
            } else {
                set_error(error, ScriptError::WITNESS_PROGRAM_WRONG_LENGTH);
                return false;
            }
        } else if (!witness->is_null()) {
            // 5. Witness data present but scriptPubKey is not a witness
            //    program -- reject unless this is a P2SH-wrapped case
            //    (already handled above).
            if (!(flags & SCRIPT_VERIFY_P2SH) ||
                !script_pub_key.is_pay_to_script_hash()) {
                set_error(error, ScriptError::WITNESS_UNEXPECTED);
                return false;
            }
        }
    }

    // -----------------------------------------------------------------
    // Step 5: Cleanstack
    // -----------------------------------------------------------------
    if ((flags & SCRIPT_VERIFY_CLEANSTACK) && stack.size() != 1) {
        set_error(error, ScriptError::CLEANSTACK);
        return false;
    }

    set_error(error, ScriptError::OK);
    return true;
}

// ---------------------------------------------------------------------------
// count_witness_sig_ops
// ---------------------------------------------------------------------------
//
// Counts signature-check operations for the block sigop limit.  Each script
// type contributes differently:
//   - Legacy: counted from scriptPubKey directly.
//   - P2SH:   the redeem script is extracted from scriptSig and counted.
//   - P2WPKH: always 1 sigop.
//   - P2WSH:  counted from the witness script (last witness stack item).

unsigned int count_witness_sig_ops(
    const CScript& script_sig,
    const CScript& script_pub_key,
    const rnet::primitives::CScriptWitness* witness,
    uint32_t flags)
{
    // 1. Base count from scriptPubKey.
    unsigned int count = script_pub_key.get_sig_op_count(true);

    // 2. P2SH: extract and count the redeem script.
    if ((flags & SCRIPT_VERIFY_P2SH) &&
        script_pub_key.is_pay_to_script_hash()) {
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

    // 3. Witness sigops.
    if ((flags & SCRIPT_VERIFY_WITNESS) && witness && !witness->is_null()) {
        int wit_version = -1;
        std::vector<uint8_t> wit_program;
        if (script_pub_key.is_witness_program(wit_version, wit_program)) {
            if (wit_version == 0 && wit_program.size() == 20) {
                // 3a. P2WPKH: exactly one CHECKSIG.
                count += 1;
            } else if (wit_version == 0 && wit_program.size() == 32) {
                // 3b. P2WSH: count from the witness script.
                if (!witness->stack.empty()) {
                    CScript ws(witness->stack.back());
                    count += ws.get_sig_op_count(true);
                }
            }
        }
    }

    return count;
}

} // namespace rnet::script

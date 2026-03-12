#include "script/interpreter.h"
#include "script/sign.h"

#include "crypto/ed25519.h"
#include "crypto/hash.h"
#include "crypto/keccak.h"
#include "crypto/secp256k1.h"

#include <algorithm>
#include <cstring>

namespace rnet::script {

// ── Helpers ─────────────────────────────────────────────────────────

static inline void set_error(ScriptError* err, ScriptError code) {
    if (err) *err = code;
}

bool cast_to_bool(const std::vector<uint8_t>& v) {
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i] != 0) {
            // Negative zero: all zeros except the last byte is 0x80.
            if (i == v.size() - 1 && v[i] == 0x80) {
                return false;
            }
            return true;
        }
    }
    return false;
}

/// Pop the top element from the stack.
static inline std::vector<uint8_t> stack_pop(ScriptStack& stack) {
    auto val = std::move(stack.back());
    stack.pop_back();
    return val;
}

/// Reference the top of the stack at a depth.
/// stacktop(-1) is the top element.
static inline std::vector<uint8_t>& stacktop(ScriptStack& stack, int idx) {
    return stack[static_cast<int>(stack.size()) + idx];
}

// ── Script error strings ────────────────────────────────────────────

std::string_view script_error_string(ScriptError err) {
    switch (err) {
        case ScriptError::OK:                       return "No error";
        case ScriptError::UNKNOWN:                  return "Unknown error";
        case ScriptError::EVAL_FALSE:               return "Script evaluated without error but finished with a false/empty top stack element";
        case ScriptError::OP_RETURN:                return "OP_RETURN was encountered";
        case ScriptError::SCRIPT_SIZE:              return "Script is too big";
        case ScriptError::PUSH_SIZE:                return "Push value size limit exceeded";
        case ScriptError::OP_COUNT:                 return "Operation limit exceeded";
        case ScriptError::STACK_SIZE:               return "Stack size limit exceeded";
        case ScriptError::SIG_COUNT:                return "Signature count negative or greater than pubkey count";
        case ScriptError::PUBKEY_COUNT:             return "Pubkey count negative or limit exceeded";
        case ScriptError::VERIFY:                   return "Script failed an OP_VERIFY operation";
        case ScriptError::EQUALVERIFY:              return "Script failed an OP_EQUALVERIFY operation";
        case ScriptError::CHECKMULTISIGVERIFY:      return "Script failed an OP_CHECKMULTISIGVERIFY operation";
        case ScriptError::CHECKSIGVERIFY:           return "Script failed an OP_CHECKSIGVERIFY operation";
        case ScriptError::NUMEQUALVERIFY:           return "Script failed an OP_NUMEQUALVERIFY operation";
        case ScriptError::BAD_OPCODE:               return "Opcode missing or not understood";
        case ScriptError::DISABLED_OPCODE:          return "Attempted to use a disabled opcode";
        case ScriptError::INVALID_STACK_OPERATION:  return "Operation not valid with the current stack size";
        case ScriptError::INVALID_ALTSTACK_OPERATION: return "Operation not valid with the current altstack size";
        case ScriptError::UNBALANCED_CONDITIONAL:   return "Expected OP_ENDIF";
        case ScriptError::SIG_HASHTYPE:             return "Signature hash type missing or not understood";
        case ScriptError::SIG_DER:                  return "Non-canonical DER signature";
        case ScriptError::SIG_NULLDUMMY:            return "Dummy CHECKMULTISIG argument must be zero";
        case ScriptError::SIG_NULLFAIL:             return "Signature must be zero for failed CHECK(MULTI)SIG operation";
        case ScriptError::MINIMALDATA:              return "Data push larger than necessary";
        case ScriptError::MINIMALIF:                return "OP_IF/NOTIF argument must be minimal";
        case ScriptError::SIG_PUSHONLY:             return "Only push operators allowed in scriptSig";
        case ScriptError::SIG_HIGH_S:               return "Non-canonical signature: S value is unnecessarily high";
        case ScriptError::PUBKEYTYPE:               return "Public key is neither compressed nor uncompressed";
        case ScriptError::NEGATIVE_LOCKTIME:        return "Negative locktime";
        case ScriptError::UNSATISFIED_LOCKTIME:     return "Locktime requirement not satisfied";
        case ScriptError::WITNESS_PROGRAM_WRONG_LENGTH:  return "Witness program has incorrect length";
        case ScriptError::WITNESS_PROGRAM_WITNESS_EMPTY: return "Witness program was passed an empty witness";
        case ScriptError::WITNESS_PROGRAM_MISMATCH:      return "Witness program hash mismatch";
        case ScriptError::WITNESS_MALLEATED:             return "Witness requires empty scriptSig";
        case ScriptError::WITNESS_MALLEATED_P2SH:        return "Witness requires only-redeemscript scriptSig";
        case ScriptError::WITNESS_UNEXPECTED:            return "Witness provided for non-witness script";
        case ScriptError::WITNESS_PUBKEYTYPE:            return "Using non-compressed keys in segwit";
        case ScriptError::CLEANSTACK:               return "Stack size must be exactly one after execution";
        case ScriptError::SCRIPT_EXCEPTION:          return "Exception during script evaluation";
        default:                                     return "Unknown error";
    }
}

// ── Script evaluation engine ────────────────────────────────────────

bool eval_script(ScriptStack& stack,
                 const CScript& script,
                 uint32_t flags,
                 const BaseSignatureChecker& checker,
                 ScriptError* error) {
    set_error(error, ScriptError::UNKNOWN);

    if (script.size() > MAX_SCRIPT_SIZE) {
        set_error(error, ScriptError::SCRIPT_SIZE);
        return false;
    }

    int op_count = 0;
    ScriptStack altstack;

    // Conditional execution tracking.
    // vfExec[i] == true means we are in an executing branch.
    std::vector<bool> vf_exec;

    auto fExec = [&vf_exec]() -> bool {
        for (auto v : vf_exec) {
            if (!v) return false;
        }
        return true;
    };

    ScriptIterator it(script);
    Opcode opcode;
    std::vector<uint8_t> push_data;

    while (it.next(opcode, push_data)) {
        auto op = static_cast<uint8_t>(opcode);
        bool executing = fExec();

        // Data pushes
        if (op <= 0x60 && op != 0x00) {
            // For opcodes 0x01..0x4b (direct pushes) and
            // OP_PUSHDATA1/2/4 and OP_1NEGATE and OP_1..OP_16:
            // push_data was already extracted by the iterator.
        }

        // Push data operations are always allowed (for IF/ELSE tracking)
        if (!push_data.empty() || op == 0x00) {
            if (push_data.size() > MAX_SCRIPT_ELEMENT_SIZE) {
                set_error(error, ScriptError::PUSH_SIZE);
                return false;
            }
            if (executing) {
                // Handle OP_0 through OP_16 and OP_1NEGATE
                if (op == 0x00) {
                    stack.emplace_back();  // empty = false/zero
                } else if (op >= 0x51 && op <= 0x60) {
                    // OP_1..OP_16
                    int n = op - 0x50;
                    stack.push_back(scriptnum_encode(n));
                } else if (op == 0x4f) {
                    // OP_1NEGATE
                    stack.push_back(scriptnum_encode(-1));
                } else {
                    // Direct push or PUSHDATA
                    stack.push_back(push_data);
                }
            }
            if (stack.size() + altstack.size() > static_cast<size_t>(MAX_STACK_SIZE)) {
                set_error(error, ScriptError::STACK_SIZE);
                return false;
            }
            continue;
        }

        // OP_0 handled above as a push
        if (op == 0x00) continue;

        // OP_1NEGATE, OP_1..OP_16 handled above as pushes
        if (op == 0x4f || (op >= 0x51 && op <= 0x60)) {
            if (executing) {
                if (op == 0x4f) {
                    stack.push_back(scriptnum_encode(-1));
                } else {
                    stack.push_back(scriptnum_encode(op - 0x50));
                }
            }
            if (stack.size() + altstack.size() > static_cast<size_t>(MAX_STACK_SIZE)) {
                set_error(error, ScriptError::STACK_SIZE);
                return false;
            }
            continue;
        }

        // Count non-push opcodes
        if (op > static_cast<uint8_t>(Opcode::OP_16)) {
            ++op_count;
            if (op_count > MAX_OPS_PER_SCRIPT) {
                set_error(error, ScriptError::OP_COUNT);
                return false;
            }
        }

        // Disabled opcodes always fail
        if (is_disabled_opcode(opcode)) {
            set_error(error, ScriptError::DISABLED_OPCODE);
            return false;
        }

        // Conditionals
        if (opcode == Opcode::OP_IF || opcode == Opcode::OP_NOTIF) {
            bool fValue = false;
            if (executing) {
                if (stack.empty()) {
                    set_error(error, ScriptError::UNBALANCED_CONDITIONAL);
                    return false;
                }
                auto& top = stacktop(stack, -1);
                if (flags & SCRIPT_VERIFY_MINIMALIF) {
                    if (top.size() > 1 ||
                        (top.size() == 1 && top[0] != 0 && top[0] != 1)) {
                        set_error(error, ScriptError::MINIMALIF);
                        return false;
                    }
                }
                fValue = cast_to_bool(top);
                if (opcode == Opcode::OP_NOTIF) fValue = !fValue;
                stack_pop(stack);
            }
            vf_exec.push_back(fValue);
            continue;
        }

        if (opcode == Opcode::OP_ELSE) {
            if (vf_exec.empty()) {
                set_error(error, ScriptError::UNBALANCED_CONDITIONAL);
                return false;
            }
            vf_exec.back() = !vf_exec.back();
            continue;
        }

        if (opcode == Opcode::OP_ENDIF) {
            if (vf_exec.empty()) {
                set_error(error, ScriptError::UNBALANCED_CONDITIONAL);
                return false;
            }
            vf_exec.pop_back();
            continue;
        }

        // All other opcodes only execute if we're in an executing branch
        if (!executing) continue;

        switch (opcode) {

        // ── Flow control ─────────────────────────────────────────
        case Opcode::OP_NOP:
            break;

        case Opcode::OP_VERIFY: {
            if (stack.empty()) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            if (!cast_to_bool(stacktop(stack, -1))) {
                set_error(error, ScriptError::VERIFY);
                return false;
            }
            stack_pop(stack);
            break;
        }

        case Opcode::OP_RETURN: {
            set_error(error, ScriptError::OP_RETURN);
            return false;
        }

        // ── Stack operations ─────────────────────────────────────
        case Opcode::OP_TOALTSTACK: {
            if (stack.empty()) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            altstack.push_back(stack_pop(stack));
            break;
        }

        case Opcode::OP_FROMALTSTACK: {
            if (altstack.empty()) {
                set_error(error, ScriptError::INVALID_ALTSTACK_OPERATION);
                return false;
            }
            stack.push_back(std::move(altstack.back()));
            altstack.pop_back();
            break;
        }

        case Opcode::OP_2DROP: {
            if (stack.size() < 2) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            stack_pop(stack);
            stack_pop(stack);
            break;
        }

        case Opcode::OP_2DUP: {
            if (stack.size() < 2) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            auto v1 = stacktop(stack, -2);
            auto v2 = stacktop(stack, -1);
            stack.push_back(v1);
            stack.push_back(v2);
            break;
        }

        case Opcode::OP_3DUP: {
            if (stack.size() < 3) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            auto v1 = stacktop(stack, -3);
            auto v2 = stacktop(stack, -2);
            auto v3 = stacktop(stack, -1);
            stack.push_back(v1);
            stack.push_back(v2);
            stack.push_back(v3);
            break;
        }

        case Opcode::OP_2OVER: {
            if (stack.size() < 4) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            auto v1 = stacktop(stack, -4);
            auto v2 = stacktop(stack, -3);
            stack.push_back(v1);
            stack.push_back(v2);
            break;
        }

        case Opcode::OP_2ROT: {
            if (stack.size() < 6) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            auto v1 = stacktop(stack, -6);
            auto v2 = stacktop(stack, -5);
            stack.erase(stack.end() - 6, stack.end() - 4);
            stack.push_back(v1);
            stack.push_back(v2);
            break;
        }

        case Opcode::OP_2SWAP: {
            if (stack.size() < 4) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            std::swap(stacktop(stack, -4), stacktop(stack, -2));
            std::swap(stacktop(stack, -3), stacktop(stack, -1));
            break;
        }

        case Opcode::OP_IFDUP: {
            if (stack.empty()) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            if (cast_to_bool(stacktop(stack, -1))) {
                stack.push_back(stacktop(stack, -1));
            }
            break;
        }

        case Opcode::OP_DEPTH: {
            auto sn = scriptnum_encode(static_cast<int64_t>(stack.size()));
            stack.push_back(sn);
            break;
        }

        case Opcode::OP_DROP: {
            if (stack.empty()) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            stack_pop(stack);
            break;
        }

        case Opcode::OP_DUP: {
            if (stack.empty()) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            stack.push_back(stacktop(stack, -1));
            break;
        }

        case Opcode::OP_NIP: {
            if (stack.size() < 2) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            stack.erase(stack.end() - 2);
            break;
        }

        case Opcode::OP_OVER: {
            if (stack.size() < 2) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            stack.push_back(stacktop(stack, -2));
            break;
        }

        case Opcode::OP_PICK:
        case Opcode::OP_ROLL: {
            if (stack.empty()) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            int64_t n = scriptnum_decode(stacktop(stack, -1));
            stack_pop(stack);
            if (n < 0 || static_cast<size_t>(n) >= stack.size()) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            auto val = stacktop(stack, -static_cast<int>(n) - 1);
            if (opcode == Opcode::OP_ROLL) {
                stack.erase(stack.end() - static_cast<int>(n) - 1);
            }
            stack.push_back(val);
            break;
        }

        case Opcode::OP_ROT: {
            if (stack.size() < 3) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            std::swap(stacktop(stack, -3), stacktop(stack, -2));
            std::swap(stacktop(stack, -2), stacktop(stack, -1));
            break;
        }

        case Opcode::OP_SWAP: {
            if (stack.size() < 2) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            std::swap(stacktop(stack, -2), stacktop(stack, -1));
            break;
        }

        case Opcode::OP_TUCK: {
            if (stack.size() < 2) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            auto val = stacktop(stack, -1);
            stack.insert(stack.end() - 2, val);
            break;
        }

        case Opcode::OP_SIZE: {
            if (stack.empty()) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            auto sn = scriptnum_encode(
                static_cast<int64_t>(stacktop(stack, -1).size()));
            stack.push_back(sn);
            break;
        }

        // ── Equality ─────────────────────────────────────────────
        case Opcode::OP_EQUAL:
        case Opcode::OP_EQUALVERIFY: {
            if (stack.size() < 2) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            auto v1 = stack_pop(stack);
            auto v2 = stack_pop(stack);
            bool equal = (v1 == v2);
            stack.push_back(equal ? scriptnum_encode(1)
                                  : std::vector<uint8_t>{});
            if (opcode == Opcode::OP_EQUALVERIFY) {
                if (equal) {
                    stack_pop(stack);
                } else {
                    set_error(error, ScriptError::EQUALVERIFY);
                    return false;
                }
            }
            break;
        }

        // ── Arithmetic ───────────────────────────────────────────
        case Opcode::OP_1ADD:
        case Opcode::OP_1SUB:
        case Opcode::OP_NEGATE:
        case Opcode::OP_ABS:
        case Opcode::OP_NOT:
        case Opcode::OP_0NOTEQUAL: {
            if (stack.empty()) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            int64_t bn = scriptnum_decode(stacktop(stack, -1));
            stack_pop(stack);
            switch (opcode) {
                case Opcode::OP_1ADD:       bn += 1; break;
                case Opcode::OP_1SUB:       bn -= 1; break;
                case Opcode::OP_NEGATE:     bn = -bn; break;
                case Opcode::OP_ABS:        if (bn < 0) bn = -bn; break;
                case Opcode::OP_NOT:        bn = (bn == 0) ? 1 : 0; break;
                case Opcode::OP_0NOTEQUAL:  bn = (bn != 0) ? 1 : 0; break;
                default: break;
            }
            stack.push_back(scriptnum_encode(bn));
            break;
        }

        case Opcode::OP_ADD:
        case Opcode::OP_SUB:
        case Opcode::OP_BOOLAND:
        case Opcode::OP_BOOLOR:
        case Opcode::OP_NUMEQUAL:
        case Opcode::OP_NUMEQUALVERIFY:
        case Opcode::OP_NUMNOTEQUAL:
        case Opcode::OP_LESSTHAN:
        case Opcode::OP_GREATERTHAN:
        case Opcode::OP_LESSTHANOREQUAL:
        case Opcode::OP_GREATERTHANOREQUAL:
        case Opcode::OP_MIN:
        case Opcode::OP_MAX: {
            if (stack.size() < 2) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            int64_t bn1 = scriptnum_decode(stacktop(stack, -2));
            int64_t bn2 = scriptnum_decode(stacktop(stack, -1));
            stack_pop(stack);
            stack_pop(stack);
            int64_t result = 0;
            switch (opcode) {
                case Opcode::OP_ADD:                result = bn1 + bn2; break;
                case Opcode::OP_SUB:                result = bn1 - bn2; break;
                case Opcode::OP_BOOLAND:            result = (bn1 != 0 && bn2 != 0) ? 1 : 0; break;
                case Opcode::OP_BOOLOR:             result = (bn1 != 0 || bn2 != 0) ? 1 : 0; break;
                case Opcode::OP_NUMEQUAL:           result = (bn1 == bn2) ? 1 : 0; break;
                case Opcode::OP_NUMEQUALVERIFY:     result = (bn1 == bn2) ? 1 : 0; break;
                case Opcode::OP_NUMNOTEQUAL:        result = (bn1 != bn2) ? 1 : 0; break;
                case Opcode::OP_LESSTHAN:           result = (bn1 < bn2) ? 1 : 0; break;
                case Opcode::OP_GREATERTHAN:        result = (bn1 > bn2) ? 1 : 0; break;
                case Opcode::OP_LESSTHANOREQUAL:    result = (bn1 <= bn2) ? 1 : 0; break;
                case Opcode::OP_GREATERTHANOREQUAL: result = (bn1 >= bn2) ? 1 : 0; break;
                case Opcode::OP_MIN:                result = std::min(bn1, bn2); break;
                case Opcode::OP_MAX:                result = std::max(bn1, bn2); break;
                default: break;
            }
            stack.push_back(scriptnum_encode(result));
            if (opcode == Opcode::OP_NUMEQUALVERIFY) {
                if (cast_to_bool(stacktop(stack, -1))) {
                    stack_pop(stack);
                } else {
                    set_error(error, ScriptError::NUMEQUALVERIFY);
                    return false;
                }
            }
            break;
        }

        case Opcode::OP_WITHIN: {
            if (stack.size() < 3) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            int64_t bn1 = scriptnum_decode(stacktop(stack, -3));
            int64_t bn2 = scriptnum_decode(stacktop(stack, -2));
            int64_t bn3 = scriptnum_decode(stacktop(stack, -1));
            stack_pop(stack);
            stack_pop(stack);
            stack_pop(stack);
            bool within = (bn2 <= bn1 && bn1 < bn3);
            stack.push_back(within ? scriptnum_encode(1)
                                   : std::vector<uint8_t>{});
            break;
        }

        // ── Crypto ───────────────────────────────────────────────
        case Opcode::OP_HASH160: {
            if (stack.empty()) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            auto& top = stacktop(stack, -1);
            // Hash160 = first 20 bytes of keccak256d
            auto h = rnet::crypto::hash160(
                std::span<const uint8_t>(top.data(), top.size()));
            std::vector<uint8_t> result(h.begin(), h.end());
            stack_pop(stack);
            stack.push_back(std::move(result));
            break;
        }

        case Opcode::OP_HASH256: {
            if (stack.empty()) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            auto& top = stacktop(stack, -1);
            // Hash256 = keccak256d
            auto h = rnet::crypto::keccak256d(
                std::span<const uint8_t>(top.data(), top.size()));
            std::vector<uint8_t> result(h.begin(), h.end());
            stack_pop(stack);
            stack.push_back(std::move(result));
            break;
        }

        case Opcode::OP_SHA256: {
            if (stack.empty()) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            auto& top = stacktop(stack, -1);
            auto h = rnet::crypto::sha256(
                std::span<const uint8_t>(top.data(), top.size()));
            std::vector<uint8_t> result(h.begin(), h.end());
            stack_pop(stack);
            stack.push_back(std::move(result));
            break;
        }

        case Opcode::OP_RIPEMD160: {
            if (stack.empty()) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            auto& top = stacktop(stack, -1);
            auto h = rnet::crypto::ripemd160(
                std::span<const uint8_t>(top.data(), top.size()));
            std::vector<uint8_t> result(h.begin(), h.end());
            stack_pop(stack);
            stack.push_back(std::move(result));
            break;
        }

        case Opcode::OP_CHECKSIG:
        case Opcode::OP_CHECKSIGVERIFY: {
            if (stack.size() < 2) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            auto pubkey = stacktop(stack, -1);
            auto sig = stacktop(stack, -2);
            stack_pop(stack);
            stack_pop(stack);

            // Build the script code for hashing (remove OP_CODESEPARATOR).
            CScript script_code(script.begin(), script.end());

            bool success = false;
            if (!sig.empty()) {
                success = checker.check_sig(sig, pubkey, script_code, flags);
            }

            if (!success && (flags & SCRIPT_VERIFY_NULLFAIL) && !sig.empty()) {
                set_error(error, ScriptError::SIG_NULLFAIL);
                return false;
            }

            stack.push_back(success ? scriptnum_encode(1)
                                    : std::vector<uint8_t>{});
            if (opcode == Opcode::OP_CHECKSIGVERIFY) {
                if (success) {
                    stack_pop(stack);
                } else {
                    set_error(error, ScriptError::CHECKSIGVERIFY);
                    return false;
                }
            }
            break;
        }

        case Opcode::OP_CHECKMULTISIG:
        case Opcode::OP_CHECKMULTISIGVERIFY: {
            size_t i = 1;
            if (stack.size() < i) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }

            int keys_count = static_cast<int>(
                scriptnum_decode(stacktop(stack, -static_cast<int>(i))));
            if (keys_count < 0 || keys_count > MAX_PUBKEYS_PER_MULTISIG) {
                set_error(error, ScriptError::PUBKEY_COUNT);
                return false;
            }
            op_count += keys_count;
            if (op_count > MAX_OPS_PER_SCRIPT) {
                set_error(error, ScriptError::OP_COUNT);
                return false;
            }
            size_t ikey = ++i;
            i += static_cast<size_t>(keys_count);

            if (stack.size() < i) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }

            int sigs_count = static_cast<int>(
                scriptnum_decode(stacktop(stack, -static_cast<int>(i))));
            if (sigs_count < 0 || sigs_count > keys_count) {
                set_error(error, ScriptError::SIG_COUNT);
                return false;
            }
            size_t isig = ++i;
            i += static_cast<size_t>(sigs_count);

            if (stack.size() < i) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }

            // Build script code
            CScript script_code(script.begin(), script.end());

            bool success = true;
            int sigs_remaining = sigs_count;
            int keys_remaining = keys_count;

            while (sigs_remaining > 0) {
                auto& sig_elem = stacktop(stack, -static_cast<int>(isig));
                auto& key_elem = stacktop(stack, -static_cast<int>(ikey));

                bool match = false;
                if (!sig_elem.empty()) {
                    match = checker.check_sig(sig_elem, key_elem,
                                              script_code, flags);
                }

                if (match) {
                    ++isig;
                    --sigs_remaining;
                }
                ++ikey;
                --keys_remaining;

                if (sigs_remaining > keys_remaining) {
                    success = false;
                    break;
                }
            }

            // Pop everything including the dummy element.
            // i already accounts for all elements.
            while (i > 1) {
                if (!success && (flags & SCRIPT_VERIFY_NULLFAIL)) {
                    auto& elem = stacktop(stack, -1);
                    // Check sig elements for nullfail
                }
                stack_pop(stack);
                --i;
            }

            // The dummy element (Bitcoin consensus bug).
            if (stack.empty()) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            if ((flags & SCRIPT_VERIFY_NULLDUMMY) &&
                !stacktop(stack, -1).empty()) {
                set_error(error, ScriptError::SIG_NULLDUMMY);
                return false;
            }
            stack_pop(stack);

            stack.push_back(success ? scriptnum_encode(1)
                                    : std::vector<uint8_t>{});

            if (opcode == Opcode::OP_CHECKMULTISIGVERIFY) {
                if (success) {
                    stack_pop(stack);
                } else {
                    set_error(error, ScriptError::CHECKMULTISIGVERIFY);
                    return false;
                }
            }
            break;
        }

        // ── Locktime ─────────────────────────────────────────────
        case Opcode::OP_CHECKLOCKTIMEVERIFY: {
            if (!(flags & SCRIPT_VERIFY_CHECKLOCKTIMEVERIFY)) {
                // Treat as NOP
                break;
            }
            if (stack.empty()) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            int64_t locktime = scriptnum_decode(stacktop(stack, -1), 5, false);
            if (locktime < 0) {
                set_error(error, ScriptError::NEGATIVE_LOCKTIME);
                return false;
            }
            if (!checker.check_locktime(locktime)) {
                set_error(error, ScriptError::UNSATISFIED_LOCKTIME);
                return false;
            }
            break;
        }

        case Opcode::OP_CHECKSEQUENCEVERIFY: {
            if (!(flags & SCRIPT_VERIFY_CHECKSEQUENCEVERIFY)) {
                // Treat as NOP
                break;
            }
            if (stack.empty()) {
                set_error(error, ScriptError::INVALID_STACK_OPERATION);
                return false;
            }
            int64_t sequence = scriptnum_decode(stacktop(stack, -1), 5, false);
            if (sequence < 0) {
                set_error(error, ScriptError::NEGATIVE_LOCKTIME);
                return false;
            }
            // Bit 31 set means disabled
            if (!(sequence & (1 << 31))) {
                if (!checker.check_sequence(sequence)) {
                    set_error(error, ScriptError::UNSATISFIED_LOCKTIME);
                    return false;
                }
            }
            break;
        }

        // ── NOP upgradables ──────────────────────────────────────
        case Opcode::OP_NOP1:
        case Opcode::OP_NOP4:
        case Opcode::OP_NOP5:
        case Opcode::OP_NOP6:
        case Opcode::OP_NOP7:
        case Opcode::OP_NOP8:
        case Opcode::OP_NOP9:
        case Opcode::OP_NOP10: {
            if (flags & SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_NOPS) {
                set_error(error, ScriptError::BAD_OPCODE);
                return false;
            }
            break;
        }

        case Opcode::OP_CODESEPARATOR:
            // Just a marker, skip.
            break;

        // ── Reserved / invalid ───────────────────────────────────
        case Opcode::OP_RESERVED:
        case Opcode::OP_VER:
        case Opcode::OP_VERIF:
        case Opcode::OP_VERNOTIF:
        case Opcode::OP_RESERVED1:
        case Opcode::OP_RESERVED2:
            set_error(error, ScriptError::BAD_OPCODE);
            return false;

        // ── ResonanceNet extensions (NOP for now) ────────────────
        case Opcode::OP_CHECKHEARTBEAT:
        case Opcode::OP_CHECKRECOVERY:
        case Opcode::OP_CHECKGUARDIAN:
            // TODO: Implement recovery verification logic
            break;

        default:
            set_error(error, ScriptError::BAD_OPCODE);
            return false;
        }

        if (stack.size() + altstack.size() > static_cast<size_t>(MAX_STACK_SIZE)) {
            set_error(error, ScriptError::STACK_SIZE);
            return false;
        }
    }

    if (!vf_exec.empty()) {
        set_error(error, ScriptError::UNBALANCED_CONDITIONAL);
        return false;
    }

    set_error(error, ScriptError::OK);
    return true;
}

// ── TransactionSignatureChecker ─────────────────────────────────────

TransactionSignatureChecker::TransactionSignatureChecker(
    const rnet::primitives::CTransaction* tx,
    unsigned int input_idx,
    int64_t amount)
    : tx_(tx), input_idx_(input_idx), amount_(amount) {}

bool TransactionSignatureChecker::check_sig(
    const std::vector<uint8_t>& sig,
    const std::vector<uint8_t>& pubkey,
    const CScript& script_code,
    uint32_t flags) const {
    if (sig.empty() || pubkey.empty()) return false;
    if (!tx_) return false;

    // The last byte of the signature is the hash type.
    uint8_t hash_type = sig.back();
    auto sig_data = std::vector<uint8_t>(sig.begin(), sig.end() - 1);

    // Compute the signature hash.
    auto sighash = signature_hash(*tx_, input_idx_, script_code, hash_type);

    // Try Ed25519 first (32-byte pubkey).
    if (pubkey.size() == 32) {
        auto ed_pubkey = rnet::crypto::Ed25519PublicKey::from_bytes(
            std::span<const uint8_t>(pubkey.data(), pubkey.size()));
        if (sig_data.size() == 64) {
            rnet::crypto::Ed25519Signature ed_sig;
            std::memcpy(ed_sig.data.data(), sig_data.data(), 64);
            return rnet::crypto::ed25519_verify(
                ed_pubkey,
                std::span<const uint8_t>(sighash.data(), sighash.size()),
                ed_sig);
        }
        return false;
    }

    // Fall back to secp256k1 ECDSA (33 or 65 byte pubkey).
    if (pubkey.size() == 33 || pubkey.size() == 65) {
        auto pk_result = rnet::crypto::Secp256k1PubKey::from_bytes(
            std::span<const uint8_t>(pubkey.data(), pubkey.size()));
        if (!pk_result.is_ok()) return false;

        auto sig_result = rnet::crypto::Secp256k1Signature::from_der(
            std::span<const uint8_t>(sig_data.data(), sig_data.size()));
        if (!sig_result.is_ok()) return false;

        return rnet::crypto::secp256k1_verify(
            pk_result.value(), sighash, sig_result.value());
    }

    return false;
}

bool TransactionSignatureChecker::check_locktime(int64_t locktime) const {
    if (!tx_) return false;

    // Locktime types must match (both block height or both timestamp).
    static constexpr int64_t LOCKTIME_THRESHOLD = 500000000;
    auto tx_locktime = static_cast<int64_t>(tx_->locktime());

    if ((tx_locktime < LOCKTIME_THRESHOLD) != (locktime < LOCKTIME_THRESHOLD)) {
        return false;
    }

    if (locktime > tx_locktime) return false;

    // The input must not be finalized.
    if (tx_->vin()[input_idx_].sequence == rnet::primitives::SEQUENCE_FINAL) {
        return false;
    }

    return true;
}

bool TransactionSignatureChecker::check_sequence(int64_t sequence) const {
    if (!tx_) return false;

    // Transaction version must be >= 2 for sequence locks.
    if (tx_->version() < 2) return false;

    auto tx_sequence = static_cast<int64_t>(
        tx_->vin()[input_idx_].sequence);

    // Disable flag must not be set on the input sequence.
    if (tx_sequence & rnet::primitives::SEQUENCE_LOCKTIME_DISABLE_FLAG) {
        return false;
    }

    // Type flags must match.
    static constexpr int64_t TYPE_FLAG =
        rnet::primitives::SEQUENCE_LOCKTIME_TYPE_FLAG;
    static constexpr int64_t MASK =
        rnet::primitives::SEQUENCE_LOCKTIME_MASK;

    if ((sequence & TYPE_FLAG) != (tx_sequence & TYPE_FLAG)) {
        return false;
    }

    if ((sequence & MASK) > (tx_sequence & MASK)) {
        return false;
    }

    return true;
}

// ── MutableTransactionSignatureChecker ──────────────────────────────

MutableTransactionSignatureChecker::MutableTransactionSignatureChecker(
    const rnet::primitives::CMutableTransaction* mtx,
    unsigned int input_idx,
    int64_t amount)
    : TransactionSignatureChecker(&tx_cache_, input_idx, amount)
    , tx_cache_(*mtx) {}

}  // namespace rnet::script

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"
#include "primitives/transaction.h"
#include "script/script.h"

namespace rnet::script {

// ── Script verification flags ───────────────────────────────────────

static constexpr uint32_t SCRIPT_VERIFY_NONE                    = 0;
static constexpr uint32_t SCRIPT_VERIFY_P2SH                    = (1U << 0);
static constexpr uint32_t SCRIPT_VERIFY_WITNESS                 = (1U << 1);
static constexpr uint32_t SCRIPT_VERIFY_STRICTENC               = (1U << 2);
static constexpr uint32_t SCRIPT_VERIFY_CHECKLOCKTIMEVERIFY     = (1U << 3);
static constexpr uint32_t SCRIPT_VERIFY_CHECKSEQUENCEVERIFY     = (1U << 4);
static constexpr uint32_t SCRIPT_VERIFY_DERSIG                  = (1U << 5);
static constexpr uint32_t SCRIPT_VERIFY_NULLDUMMY               = (1U << 6);
static constexpr uint32_t SCRIPT_VERIFY_SIGPUSHONLY              = (1U << 7);
static constexpr uint32_t SCRIPT_VERIFY_MINIMALDATA              = (1U << 8);
static constexpr uint32_t SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_NOPS = (1U << 9);
static constexpr uint32_t SCRIPT_VERIFY_CLEANSTACK              = (1U << 10);
static constexpr uint32_t SCRIPT_VERIFY_MINIMALIF               = (1U << 11);
static constexpr uint32_t SCRIPT_VERIFY_NULLFAIL                = (1U << 12);
static constexpr uint32_t SCRIPT_VERIFY_WITNESS_PUBKEYTYPE      = (1U << 13);

/// Standard verification flags for mainnet.
static constexpr uint32_t STANDARD_SCRIPT_VERIFY_FLAGS =
    SCRIPT_VERIFY_P2SH |
    SCRIPT_VERIFY_WITNESS |
    SCRIPT_VERIFY_STRICTENC |
    SCRIPT_VERIFY_CHECKLOCKTIMEVERIFY |
    SCRIPT_VERIFY_CHECKSEQUENCEVERIFY |
    SCRIPT_VERIFY_DERSIG |
    SCRIPT_VERIFY_NULLDUMMY |
    SCRIPT_VERIFY_MINIMALDATA |
    SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_NOPS |
    SCRIPT_VERIFY_CLEANSTACK |
    SCRIPT_VERIFY_MINIMALIF |
    SCRIPT_VERIFY_NULLFAIL;

// ── Script error codes ──────────────────────────────────────────────

enum class ScriptError {
    OK = 0,
    UNKNOWN,
    EVAL_FALSE,
    OP_RETURN,

    // Size limits
    SCRIPT_SIZE,
    PUSH_SIZE,
    OP_COUNT,
    STACK_SIZE,
    SIG_COUNT,
    PUBKEY_COUNT,

    // Verification
    VERIFY,
    EQUALVERIFY,
    CHECKMULTISIGVERIFY,
    CHECKSIGVERIFY,
    NUMEQUALVERIFY,

    // Operand errors
    BAD_OPCODE,
    DISABLED_OPCODE,
    INVALID_STACK_OPERATION,
    INVALID_ALTSTACK_OPERATION,
    UNBALANCED_CONDITIONAL,

    // Signature errors
    SIG_HASHTYPE,
    SIG_DER,
    SIG_NULLDUMMY,
    SIG_NULLFAIL,
    MINIMALDATA,
    MINIMALIF,
    SIG_PUSHONLY,
    SIG_HIGH_S,
    PUBKEYTYPE,

    // Locktime
    NEGATIVE_LOCKTIME,
    UNSATISFIED_LOCKTIME,

    // Witness
    WITNESS_PROGRAM_WRONG_LENGTH,
    WITNESS_PROGRAM_WITNESS_EMPTY,
    WITNESS_PROGRAM_MISMATCH,
    WITNESS_MALLEATED,
    WITNESS_MALLEATED_P2SH,
    WITNESS_UNEXPECTED,
    WITNESS_PUBKEYTYPE,

    // Cleanstack
    CLEANSTACK,

    // Misc
    SCRIPT_EXCEPTION,
};

/// Get a human-readable description of a script error.
std::string_view script_error_string(ScriptError err);

// ── Stack type ──────────────────────────────────────────────────────

using ScriptStack = std::vector<std::vector<uint8_t>>;

// ── Signature checker ───────────────────────────────────────────────

/// Abstract base for signature checking.
/// Allows script evaluation without knowing the transaction context.
class BaseSignatureChecker {
public:
    virtual ~BaseSignatureChecker() = default;

    /// Check a signature against a public key and script code.
    /// sig includes the hash type byte as its last byte.
    virtual bool check_sig(const std::vector<uint8_t>& sig,
                           const std::vector<uint8_t>& pubkey,
                           const CScript& script_code,
                           uint32_t flags) const {
        (void)sig; (void)pubkey; (void)script_code; (void)flags;
        return false;
    }

    /// Check OP_CHECKLOCKTIMEVERIFY.
    virtual bool check_locktime(int64_t locktime) const {
        (void)locktime;
        return false;
    }

    /// Check OP_CHECKSEQUENCEVERIFY.
    virtual bool check_sequence(int64_t sequence) const {
        (void)sequence;
        return false;
    }
};

/// Signature checker that verifies against a real transaction.
class TransactionSignatureChecker : public BaseSignatureChecker {
public:
    /// Construct with a transaction, input index, and spent amount.
    TransactionSignatureChecker(const rnet::primitives::CTransaction* tx,
                                unsigned int input_idx,
                                int64_t amount);

    bool check_sig(const std::vector<uint8_t>& sig,
                   const std::vector<uint8_t>& pubkey,
                   const CScript& script_code,
                   uint32_t flags) const override;

    bool check_locktime(int64_t locktime) const override;
    bool check_sequence(int64_t sequence) const override;

private:
    const rnet::primitives::CTransaction* tx_;
    unsigned int input_idx_;
    int64_t amount_;
};

/// Signature checker that uses a mutable transaction.
class MutableTransactionSignatureChecker : public TransactionSignatureChecker {
public:
    MutableTransactionSignatureChecker(
        const rnet::primitives::CMutableTransaction* mtx,
        unsigned int input_idx,
        int64_t amount);

private:
    rnet::primitives::CTransaction tx_cache_;
};

// ── Stack helpers ───────────────────────────────────────────────────

/// Check if a stack element evaluates to true (non-zero).
bool cast_to_bool(const std::vector<uint8_t>& v);

// ── Script evaluation ───────────────────────────────────────────────

/// Evaluate a script on the given stack.
///
/// Returns true if script execution succeeds (no errors).
/// On failure, sets *error if non-null.
bool eval_script(ScriptStack& stack,
                 const CScript& script,
                 uint32_t flags,
                 const BaseSignatureChecker& checker,
                 ScriptError* error = nullptr);

}  // namespace rnet::script

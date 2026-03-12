#pragma once

#include <cstdint>

#include "primitives/witness.h"
#include "script/interpreter.h"
#include "script/script.h"

namespace rnet::script {

/// Full script verification: evaluate scriptSig + scriptPubKey,
/// with optional P2SH and witness support.
///
/// This is the main entry point for verifying that an input correctly
/// unlocks the output it references.
///
/// @param script_sig     The input's unlocking script (scriptSig).
/// @param script_pub_key The output's locking script (scriptPubKey).
/// @param witness        The input's witness data (may be null).
/// @param flags          Verification flags (SCRIPT_VERIFY_*).
/// @param checker        Signature checker for this input.
/// @param error          Optional output for detailed error code.
/// @return true if the script pair verifies successfully.
bool verify_script(const CScript& script_sig,
                   const CScript& script_pub_key,
                   const rnet::primitives::CScriptWitness* witness,
                   uint32_t flags,
                   const BaseSignatureChecker& checker,
                   ScriptError* error = nullptr);

/// Count the total sigops in a script pair (for block sigop limit).
///
/// @param script_sig     The input's scriptSig.
/// @param script_pub_key The output's scriptPubKey.
/// @param witness        The input's witness data (may be null).
/// @param flags          Verification flags.
/// @return Total sigop count.
unsigned int count_witness_sig_ops(const CScript& script_sig,
                                   const CScript& script_pub_key,
                                   const rnet::primitives::CScriptWitness* witness,
                                   uint32_t flags);

}  // namespace rnet::script

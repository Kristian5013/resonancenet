#pragma once

#include <array>
#include <cstdint>
#include <variant>
#include <vector>

#include "core/types.h"
#include "core/error.h"
#include "script/script.h"

namespace rnet::script {

/// Recovery policy types for UTXO reclaim.
enum class RecoveryType : uint8_t {
    HEARTBEAT = 1,   ///< Owner must send heartbeat tx within interval
    SOCIAL    = 2,   ///< Guardian quorum can recover after waiting period
    EMISSION  = 3,   ///< Coins return to mining after inactivity
};

/// Heartbeat recovery: if the owner fails to send a heartbeat
/// transaction within `interval` blocks, the recovery pubkey
/// can spend the output.
struct HeartbeatPolicy {
    uint64_t interval = 0;                      ///< Blocks between heartbeats
    std::vector<uint8_t> recovery_pubkey_hash;  ///< Hash160 of recovery key
};

/// Social recovery: a threshold of guardian public keys can
/// recover the funds after a waiting period.
struct SocialPolicy {
    uint8_t threshold = 0;                                  ///< Required signatures
    std::vector<std::array<uint8_t, 32>> guardian_pubkeys;  ///< Ed25519 pubkeys
    uint64_t waiting_period = 0;                            ///< Blocks to wait
};

/// Emission recovery: coins return to the mining reward pool
/// after an inactivity period (protocol-level, not script-enforced
/// in the traditional sense, but we encode the policy in script
/// metadata for transparency).
struct EmissionPolicy {
    uint64_t inactivity_period = 0;  ///< Blocks of inactivity before reclaim
};

/// Policy variant for convenience.
using RecoveryPolicy = std::variant<HeartbeatPolicy, SocialPolicy, EmissionPolicy>;

/// Build a recovery script that encodes the owner's normal spend path
/// plus a recovery path.
///
/// The script structure depends on the recovery type:
///
/// HEARTBEAT:
///   OP_IF
///     <owner_pubkey_hash> OP_CHECKSIG          (normal spend)
///   OP_ELSE
///     <interval> OP_CHECKSEQUENCEVERIFY OP_DROP
///     <recovery_pubkey_hash> OP_CHECKSIG       (recovery spend)
///   OP_ENDIF
///
/// SOCIAL:
///   OP_IF
///     <owner_pubkey_hash> OP_CHECKSIG          (normal spend)
///   OP_ELSE
///     <waiting_period> OP_CHECKSEQUENCEVERIFY OP_DROP
///     <threshold> <pubkey1> ... <pubkeyN> <N> OP_CHECKMULTISIG
///   OP_ENDIF
///
/// EMISSION:
///   OP_IF
///     <owner_pubkey_hash> OP_CHECKSIG          (normal spend)
///   OP_ELSE
///     <inactivity_period> OP_CHECKSEQUENCEVERIFY OP_DROP
///     OP_RETURN                                 (burns to mining)
///   OP_ENDIF
///
/// @param owner_pubkey_hash  Hash160 of the owner's public key.
/// @param type               Recovery type.
/// @param policy             Policy parameters.
/// @return The complete recovery script.
CScript build_recovery_script(
    const std::vector<uint8_t>& owner_pubkey_hash,
    RecoveryType type,
    const RecoveryPolicy& policy);

/// Parse a recovery script to extract the recovery type and policy.
///
/// @param script  The script to parse.
/// @param type    Output: recovery type.
/// @param policy  Output: policy parameters.
/// @return true if the script is a valid recovery script.
bool parse_recovery_script(const CScript& script,
                           RecoveryType& type,
                           RecoveryPolicy& policy);

/// Build a P2WSH output for a recovery script.
/// Wraps the recovery script in a witness script hash.
///
/// @param recovery_script  The recovery script.
/// @return P2WSH scriptPubKey (OP_0 [32-byte hash]).
CScript build_recovery_p2wsh(const CScript& recovery_script);

/// Build the witness stack for spending a heartbeat recovery path.
///
/// @param signature  The recovery key's signature (with hash type byte).
/// @param recovery_script  The full recovery script (for the witness).
/// @return Witness stack items.
std::vector<std::vector<uint8_t>> build_heartbeat_recovery_witness(
    const std::vector<uint8_t>& signature,
    const CScript& recovery_script);

/// Build the witness stack for spending via normal owner path.
///
/// @param signature  The owner's signature (with hash type byte).
/// @param recovery_script  The full recovery script.
/// @return Witness stack items.
std::vector<std::vector<uint8_t>> build_owner_spend_witness(
    const std::vector<uint8_t>& signature,
    const CScript& recovery_script);

/// Build the witness stack for social recovery.
///
/// @param signatures  Guardian signatures (threshold count).
/// @param recovery_script  The full recovery script.
/// @return Witness stack items.
std::vector<std::vector<uint8_t>> build_social_recovery_witness(
    const std::vector<std::vector<uint8_t>>& signatures,
    const CScript& recovery_script);

}  // namespace rnet::script

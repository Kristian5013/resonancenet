#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "core/types.h"
#include "core/error.h"
#include "primitives/transaction.h"
#include "script/script.h"

namespace rnet::script {

// ── Sighash types ───────────────────────────────────────────────────

static constexpr uint8_t SIGHASH_ALL          = 0x01;
static constexpr uint8_t SIGHASH_NONE         = 0x02;
static constexpr uint8_t SIGHASH_SINGLE       = 0x03;
static constexpr uint8_t SIGHASH_ANYONECANPAY = 0x80;

/// Compute the signature hash for a transaction input.
///
/// This is the message that gets signed/verified by OP_CHECKSIG.
/// Uses keccak256d for hashing (ResonanceNet consensus).
///
/// @param tx          The transaction being signed.
/// @param input       The input index being signed.
/// @param script_code The scriptPubKey of the output being spent
///                    (or the redeem script for P2SH).
/// @param hash_type   Sighash type (SIGHASH_ALL, etc.).
/// @return 256-bit hash to be signed.
rnet::uint256 signature_hash(const rnet::primitives::CTransaction& tx,
                             unsigned int input,
                             const CScript& script_code,
                             int hash_type);

/// Compute BIP143-style witness signature hash (segwit v0).
///
/// @param tx          The transaction being signed.
/// @param input       The input index being signed.
/// @param script_code The witness script.
/// @param amount      The value of the output being spent.
/// @param hash_type   Sighash type.
/// @return 256-bit hash to be signed.
rnet::uint256 witness_signature_hash(
    const rnet::primitives::CTransaction& tx,
    unsigned int input,
    const CScript& script_code,
    int64_t amount,
    int hash_type);

// ── Key provider interface ──────────────────────────────────────────

/// Interface for providing keys during signing.
class SigningProvider {
public:
    virtual ~SigningProvider() = default;

    /// Get the private key (Ed25519 seed, 32 bytes) for a public key hash.
    virtual bool get_ed25519_key(
        const std::vector<uint8_t>& pubkey_hash,
        std::vector<uint8_t>& secret_out,
        std::vector<uint8_t>& pubkey_out) const {
        (void)pubkey_hash; (void)secret_out; (void)pubkey_out;
        return false;
    }

    /// Get the secp256k1 private key (32 bytes) for a public key hash.
    virtual bool get_secp256k1_key(
        const std::vector<uint8_t>& pubkey_hash,
        std::vector<uint8_t>& secret_out,
        std::vector<uint8_t>& pubkey_out) const {
        (void)pubkey_hash; (void)secret_out; (void)pubkey_out;
        return false;
    }

    /// Get the redeem script for a P2SH script hash.
    virtual bool get_redeem_script(
        const std::vector<uint8_t>& script_hash,
        CScript& redeem_out) const {
        (void)script_hash; (void)redeem_out;
        return false;
    }

    /// Get the witness script for a P2WSH script hash.
    virtual bool get_witness_script(
        const std::vector<uint8_t>& script_hash,
        CScript& witness_out) const {
        (void)script_hash; (void)witness_out;
        return false;
    }
};

// ── Signing functions ───────────────────────────────────────────────

/// Sign a transaction input.
///
/// Fills in script_sig and/or witness data for the specified input.
///
/// @param provider    Key provider for looking up private keys.
/// @param mtx         Mutable transaction to sign.
/// @param input       Input index to sign.
/// @param script_pub_key The scriptPubKey of the output being spent.
/// @param amount      Value of the output being spent.
/// @param hash_type   Sighash type (default SIGHASH_ALL).
/// @return true on success.
bool sign_input(const SigningProvider& provider,
                rnet::primitives::CMutableTransaction& mtx,
                unsigned int input,
                const CScript& script_pub_key,
                int64_t amount,
                int hash_type = SIGHASH_ALL);

/// Sign all inputs of a transaction that can be signed with the
/// provided keys.
///
/// @param provider        Key provider.
/// @param mtx             Mutable transaction.
/// @param prev_scripts    scriptPubKey for each input.
/// @param prev_amounts    Value for each input.
/// @param hash_type       Sighash type.
/// @return Number of successfully signed inputs.
int sign_transaction(const SigningProvider& provider,
                     rnet::primitives::CMutableTransaction& mtx,
                     const std::vector<CScript>& prev_scripts,
                     const std::vector<int64_t>& prev_amounts,
                     int hash_type = SIGHASH_ALL);

/// Create an Ed25519 signature for a sighash.
///
/// @param secret  64-byte Ed25519 secret key.
/// @param sighash 32-byte hash to sign.
/// @return 64-byte signature, or empty on failure.
std::vector<uint8_t> sign_ed25519(
    const std::vector<uint8_t>& secret,
    const rnet::uint256& sighash);

/// Create a secp256k1 ECDSA signature for a sighash.
///
/// @param secret  32-byte secret key.
/// @param sighash 32-byte hash to sign.
/// @return DER-encoded signature, or empty on failure.
std::vector<uint8_t> sign_secp256k1(
    const std::vector<uint8_t>& secret,
    const rnet::uint256& sighash);

}  // namespace rnet::script

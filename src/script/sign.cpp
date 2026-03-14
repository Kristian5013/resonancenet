// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "script/sign.h"
#include "script/standard.h"
#include "script/interpreter.h"

#include "core/stream.h"
#include "crypto/ed25519.h"
#include "crypto/hash.h"
#include "crypto/keccak.h"
#include "crypto/secp256k1.h"

#include <cstring>

namespace rnet::script {

// ---------------------------------------------------------------------------
// write_compact_size  (local helper)
// ---------------------------------------------------------------------------
//
// Serialize a length prefix in Bitcoin compact-size encoding.  Scripts are
// expected to stay well under 64 KiB, so only the 1-byte and 3-byte forms
// are emitted.

static void write_compact_size(rnet::crypto::HashWriter& hw, uint64_t len)
{
    if (len < 253) {
        uint8_t b = static_cast<uint8_t>(len);
        hw.write(&b, 1);
    } else {
        uint8_t marker = 253;
        hw.write(&marker, 1);
        uint16_t len16 = static_cast<uint16_t>(len);
        hw.write(&len16, 2);
    }
}

// ---------------------------------------------------------------------------
// signature_hash
// ---------------------------------------------------------------------------
//
// DESIGN NOTE -- sighash computation (legacy, pre-segwit)
//
//   The hash commits to different subsets of the transaction depending on the
//   SIGHASH flags:
//
//     SIGHASH_ALL (0x01)
//       Commits to every input and every output.  This is the default and
//       most restrictive mode -- nothing can be changed after signing.
//
//     SIGHASH_NONE (0x02)
//       Commits to all inputs but *no* outputs.  Sequence numbers of
//       non-signed inputs are zeroed so they can be replaced freely.
//
//     SIGHASH_SINGLE (0x03)
//       Commits to all inputs and exactly the output at the same index as
//       the input being signed.  Outputs before the signed index are
//       serialized as null (-1 value, empty script).
//
//     SIGHASH_ANYONECANPAY (0x80, ORed with the above)
//       Commits to *only* the input being signed; other inputs may be added
//       or removed without invalidating the signature.
//
//   The serialization format mirrors Bitcoin's legacy sighash algorithm:
//   version | inputs | outputs | locktime | hash_type, then Keccak256d.

rnet::uint256 signature_hash(const rnet::primitives::CTransaction& tx,
                             unsigned int input,
                             const CScript& script_code,
                             int hash_type)
{
    // 1. Validate the input index.
    if (input >= tx.vin().size()) {
        rnet::uint256 one;
        one[0] = 1;  // Return 1 as error sentinel (Bitcoin convention)
        return one;
    }

    // 2. Decompose hash_type into base type and ANYONECANPAY flag.
    bool anyone_can_pay = (hash_type & SIGHASH_ANYONECANPAY) != 0;
    int base_type = hash_type & 0x1F;

    rnet::crypto::HashWriter hasher;

    // 3. Version.
    auto ver = tx.version();
    hasher.write(&ver, sizeof(ver));

    // 4. Serialize inputs.
    if (anyone_can_pay) {
        // 4a. ANYONECANPAY -- only the input being signed.
        uint8_t count = 1;
        hasher.write(&count, 1);

        hasher.write(tx.vin()[input].prevout.hash.data(), 32);
        uint32_t n = tx.vin()[input].prevout.n;
        hasher.write(&n, sizeof(n));

        write_compact_size(hasher, script_code.size());
        hasher.write(script_code.data(), script_code.size());

        uint32_t seq = tx.vin()[input].sequence;
        hasher.write(&seq, sizeof(seq));
    } else {
        // 4b. Standard -- every input, but only the signed input gets the
        //     script code; all others receive an empty script.
        uint8_t vin_count = static_cast<uint8_t>(tx.vin().size());
        hasher.write(&vin_count, 1);

        for (size_t i = 0; i < tx.vin().size(); ++i) {
            hasher.write(tx.vin()[i].prevout.hash.data(), 32);
            uint32_t n = tx.vin()[i].prevout.n;
            hasher.write(&n, sizeof(n));

            if (i == input) {
                write_compact_size(hasher, script_code.size());
                hasher.write(script_code.data(), script_code.size());
            } else {
                uint8_t zero = 0;
                hasher.write(&zero, 1);
            }

            uint32_t seq = tx.vin()[i].sequence;
            if (i != input &&
                (base_type == SIGHASH_NONE || base_type == SIGHASH_SINGLE)) {
                seq = 0;
            }
            hasher.write(&seq, sizeof(seq));
        }
    }

    // 5. Serialize outputs according to base sighash type.
    if (base_type == SIGHASH_NONE) {
        // 5a. No outputs committed.
        uint8_t zero = 0;
        hasher.write(&zero, 1);
    } else if (base_type == SIGHASH_SINGLE) {
        // 5b. Only the output at the same index as the signed input.
        if (input >= tx.vout().size()) {
            rnet::uint256 one;
            one[0] = 1;
            return one;
        }
        uint8_t count = static_cast<uint8_t>(input + 1);
        hasher.write(&count, 1);

        for (size_t i = 0; i < input; ++i) {
            int64_t neg1 = -1;
            hasher.write(&neg1, sizeof(neg1));
            uint8_t zero = 0;
            hasher.write(&zero, 1);
        }

        hasher.write(&tx.vout()[input].value, sizeof(tx.vout()[input].value));
        write_compact_size(hasher, tx.vout()[input].script_pub_key.size());
        hasher.write(tx.vout()[input].script_pub_key.data(),
                     tx.vout()[input].script_pub_key.size());
    } else {
        // 5c. SIGHASH_ALL -- every output.
        uint8_t vout_count = static_cast<uint8_t>(tx.vout().size());
        hasher.write(&vout_count, 1);

        for (const auto& out : tx.vout()) {
            hasher.write(&out.value, sizeof(out.value));
            write_compact_size(hasher, out.script_pub_key.size());
            hasher.write(out.script_pub_key.data(), out.script_pub_key.size());
        }
    }

    // 6. Locktime.
    auto lt = tx.locktime();
    hasher.write(&lt, sizeof(lt));

    // 7. Hash type suffix.
    uint32_t ht32 = static_cast<uint32_t>(hash_type);
    hasher.write(&ht32, sizeof(ht32));

    return hasher.get_hash256();
}

// ---------------------------------------------------------------------------
// witness_signature_hash
// ---------------------------------------------------------------------------
//
// DESIGN NOTE -- BIP-143-style witness sighash
//
//   Segwit v0 replaces the quadratic-cost legacy algorithm with a linear
//   preimage that caches intermediate hashes (hashPrevouts, hashSequence,
//   hashOutputs).  The preimage layout is:
//
//     version | hashPrevouts | hashSequence | outpoint | scriptCode |
//     amount  | sequence     | hashOutputs  | locktime | hashType
//
//   ANYONECANPAY zeroes hashPrevouts and hashSequence.
//   SIGHASH_NONE/SINGLE zeroes hashSequence; SIGHASH_NONE also zeroes
//   hashOutputs, while SIGHASH_SINGLE hashes only the paired output.

rnet::uint256 witness_signature_hash(
    const rnet::primitives::CTransaction& tx,
    unsigned int input,
    const CScript& script_code,
    int64_t amount,
    int hash_type)
{
    // 1. Decompose hash_type.
    bool anyone_can_pay = (hash_type & SIGHASH_ANYONECANPAY) != 0;
    int base_type = hash_type & 0x1F;

    rnet::uint256 hash_prevouts;
    rnet::uint256 hash_sequence;
    rnet::uint256 hash_outputs;

    // 2. Compute hashPrevouts (skipped under ANYONECANPAY).
    if (!anyone_can_pay) {
        rnet::crypto::HashWriter hw;
        for (const auto& in : tx.vin()) {
            hw.write(in.prevout.hash.data(), 32);
            uint32_t n = in.prevout.n;
            hw.write(&n, sizeof(n));
        }
        hash_prevouts = hw.get_hash256();
    }

    // 3. Compute hashSequence (only for ALL).
    if (!anyone_can_pay && base_type != SIGHASH_SINGLE &&
        base_type != SIGHASH_NONE) {
        rnet::crypto::HashWriter hw;
        for (const auto& in : tx.vin()) {
            uint32_t seq = in.sequence;
            hw.write(&seq, sizeof(seq));
        }
        hash_sequence = hw.get_hash256();
    }

    // 4. Compute hashOutputs.
    if (base_type != SIGHASH_SINGLE && base_type != SIGHASH_NONE) {
        // 4a. ALL -- hash every output.
        rnet::crypto::HashWriter hw;
        for (const auto& out : tx.vout()) {
            hw.write(&out.value, sizeof(out.value));
            write_compact_size(hw, out.script_pub_key.size());
            hw.write(out.script_pub_key.data(), out.script_pub_key.size());
        }
        hash_outputs = hw.get_hash256();
    } else if (base_type == SIGHASH_SINGLE && input < tx.vout().size()) {
        // 4b. SINGLE -- hash only the output at the signed index.
        rnet::crypto::HashWriter hw;
        hw.write(&tx.vout()[input].value, sizeof(tx.vout()[input].value));
        auto& spk = tx.vout()[input].script_pub_key;
        write_compact_size(hw, spk.size());
        hw.write(spk.data(), spk.size());
        hash_outputs = hw.get_hash256();
    }

    // 5. Assemble the final preimage and hash.
    rnet::crypto::HashWriter preimage;

    auto ver = tx.version();
    preimage.write(&ver, sizeof(ver));
    preimage.write(hash_prevouts.data(), 32);
    preimage.write(hash_sequence.data(), 32);

    preimage.write(tx.vin()[input].prevout.hash.data(), 32);
    uint32_t n = tx.vin()[input].prevout.n;
    preimage.write(&n, sizeof(n));

    write_compact_size(preimage, script_code.size());
    preimage.write(script_code.data(), script_code.size());

    preimage.write(&amount, sizeof(amount));

    uint32_t seq = tx.vin()[input].sequence;
    preimage.write(&seq, sizeof(seq));

    preimage.write(hash_outputs.data(), 32);

    auto lt = tx.locktime();
    preimage.write(&lt, sizeof(lt));

    uint32_t ht32 = static_cast<uint32_t>(hash_type);
    preimage.write(&ht32, sizeof(ht32));

    return preimage.get_hash256();
}

// ---------------------------------------------------------------------------
// sign_ed25519
// ---------------------------------------------------------------------------
//
// DESIGN NOTE -- SignatureChecker key dispatch
//
//   ResonanceNet supports two signature schemes.  The signer is selected by
//   the length of the secret / public key material:
//
//     Ed25519  (32-byte compressed pubkey)
//       Secret is the 64-byte expanded Ed25519 key (seed || public).
//       Produces a fixed 64-byte signature.
//
//     secp256k1 (33-byte compressed / 65-byte uncompressed pubkey)
//       Secret is the raw 32-byte scalar.
//       Produces a variable-length DER-encoded ECDSA signature.
//
//   Both functions append SIGHASH_* byte at the call site, not here.

std::vector<uint8_t> sign_ed25519(
    const std::vector<uint8_t>& secret,
    const rnet::uint256& sighash)
{
    // 1. Validate key length.
    if (secret.size() != 64) return {};

    // 2. Copy into typed key and sign.
    rnet::crypto::Ed25519SecretKey sk;
    std::memcpy(sk.data.data(), secret.data(), 64);

    auto result = rnet::crypto::ed25519_sign(
        sk, std::span<const uint8_t>(sighash.data(), sighash.size()));

    // 3. Wipe secret material before returning.
    sk.wipe();

    if (!result.is_ok()) return {};

    return std::vector<uint8_t>(result.value().data.begin(),
                                result.value().data.end());
}

// ---------------------------------------------------------------------------
// sign_secp256k1
// ---------------------------------------------------------------------------

std::vector<uint8_t> sign_secp256k1(
    const std::vector<uint8_t>& secret,
    const rnet::uint256& sighash)
{
    // 1. Validate key length.
    if (secret.size() != 32) return {};

    // 2. Copy into typed key and sign.
    rnet::crypto::Secp256k1SecretKey sk;
    std::memcpy(sk.data.data(), secret.data(), 32);

    auto result = rnet::crypto::secp256k1_sign(sk, sighash);

    // 3. Wipe secret material before returning.
    sk.wipe();

    if (!result.is_ok()) return {};

    return result.value().der_data;
}

// ---------------------------------------------------------------------------
// sign_input
// ---------------------------------------------------------------------------
//
// Dispatches signing for a single transaction input.  The output type
// (P2PKH, P2WPKH, P2PK) determines how the sighash is computed and where
// the signature is placed (scriptSig vs. witness stack).

bool sign_input(const SigningProvider& provider,
                rnet::primitives::CMutableTransaction& mtx,
                unsigned int input,
                const CScript& script_pub_key,
                int64_t amount,
                int hash_type)
{
    // 1. Range-check the input index.
    if (input >= mtx.vin.size()) return false;

    // 2. Classify the output script.
    std::vector<std::vector<uint8_t>> solutions;
    TxoutType type = solver(script_pub_key, solutions);

    rnet::primitives::CTransaction tx(mtx);

    switch (type) {
        case TxoutType::PUBKEYHASH: {
            // 3. P2PKH: try Ed25519 first, then secp256k1.
            if (solutions.empty()) return false;
            auto& hash = solutions[0];

            std::vector<uint8_t> secret, pubkey;
            if (provider.get_ed25519_key(hash, secret, pubkey)) {
                auto sighash = signature_hash(tx, input, script_pub_key, hash_type);
                auto sig = sign_ed25519(secret, sighash);
                if (sig.empty()) return false;
                sig.push_back(static_cast<uint8_t>(hash_type));

                CScript script_sig;
                script_sig << sig << pubkey;
                mtx.vin[input].script_sig.assign(script_sig.begin(), script_sig.end());
                return true;
            }
            if (provider.get_secp256k1_key(hash, secret, pubkey)) {
                auto sighash = signature_hash(tx, input, script_pub_key, hash_type);
                auto sig = sign_secp256k1(secret, sighash);
                if (sig.empty()) return false;
                sig.push_back(static_cast<uint8_t>(hash_type));

                CScript script_sig;
                script_sig << sig << pubkey;
                mtx.vin[input].script_sig.assign(script_sig.begin(), script_sig.end());
                return true;
            }
            return false;
        }

        case TxoutType::WITNESS_V0_KEYHASH: {
            // 4. P2WPKH: signature and pubkey go in the witness stack.
            if (solutions.empty()) return false;
            auto& hash = solutions[0];

            std::vector<uint8_t> secret, pubkey;
            bool have_key = provider.get_ed25519_key(hash, secret, pubkey) ||
                            provider.get_secp256k1_key(hash, secret, pubkey);
            if (!have_key) return false;

            // 4a. Build the implied P2PKH witness script.
            CScript witness_script;
            witness_script << Opcode::OP_DUP
                           << Opcode::OP_HASH160
                           << hash
                           << Opcode::OP_EQUALVERIFY
                           << Opcode::OP_CHECKSIG;

            auto sighash = witness_signature_hash(
                tx, input, witness_script, amount, hash_type);

            // 4b. Sign with the appropriate scheme based on pubkey length.
            std::vector<uint8_t> sig;
            if (pubkey.size() == 32) {
                sig = sign_ed25519(secret, sighash);
            } else {
                sig = sign_secp256k1(secret, sighash);
            }
            if (sig.empty()) return false;
            sig.push_back(static_cast<uint8_t>(hash_type));

            // 4c. Place [sig, pubkey] in witness; scriptSig stays empty.
            mtx.vin[input].witness.stack.clear();
            mtx.vin[input].witness.stack.push_back(sig);
            mtx.vin[input].witness.stack.push_back(pubkey);
            mtx.vin[input].script_sig.clear();
            return true;
        }

        case TxoutType::PUBKEY: {
            // 5. P2PK: bare pubkey, signature only in scriptSig.
            if (solutions.empty()) return false;
            auto& pubkey = solutions[0];

            auto pk_hash = rnet::crypto::hash160(
                std::span<const uint8_t>(pubkey.data(), pubkey.size()));
            std::vector<uint8_t> hash_vec(pk_hash.begin(), pk_hash.end());

            std::vector<uint8_t> secret, pubkey_out;
            bool have_key = provider.get_ed25519_key(hash_vec, secret, pubkey_out) ||
                            provider.get_secp256k1_key(hash_vec, secret, pubkey_out);
            if (!have_key) return false;

            auto sighash = signature_hash(tx, input, script_pub_key, hash_type);

            std::vector<uint8_t> sig;
            if (pubkey.size() == 32) {
                sig = sign_ed25519(secret, sighash);
            } else {
                sig = sign_secp256k1(secret, sighash);
            }
            if (sig.empty()) return false;
            sig.push_back(static_cast<uint8_t>(hash_type));

            CScript script_sig;
            script_sig << sig;
            mtx.vin[input].script_sig.assign(script_sig.begin(), script_sig.end());
            return true;
        }

        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
// sign_transaction
// ---------------------------------------------------------------------------
//
// Iterates every input and attempts to sign it.  Returns the number of
// inputs that were successfully signed.

int sign_transaction(const SigningProvider& provider,
                     rnet::primitives::CMutableTransaction& mtx,
                     const std::vector<CScript>& prev_scripts,
                     const std::vector<int64_t>& prev_amounts,
                     int hash_type)
{
    int signed_count = 0;
    for (size_t i = 0; i < mtx.vin.size(); ++i) {
        // 1. Skip inputs that lack prevout metadata.
        if (i >= prev_scripts.size() || i >= prev_amounts.size()) break;

        // 2. Attempt to sign; count successes.
        if (sign_input(provider, mtx, static_cast<unsigned int>(i),
                       prev_scripts[i], prev_amounts[i], hash_type)) {
            ++signed_count;
        }
    }
    return signed_count;
}

} // namespace rnet::script

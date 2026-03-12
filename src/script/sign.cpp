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

// ── Signature hash (legacy) ─────────────────────────────────────────

rnet::uint256 signature_hash(const rnet::primitives::CTransaction& tx,
                             unsigned int input,
                             const CScript& script_code,
                             int hash_type) {
    // Validate input index
    if (input >= tx.vin().size()) {
        rnet::uint256 one;
        one[0] = 1;  // Return 1 as error sentinel (Bitcoin convention)
        return one;
    }

    // Build the serialized transaction for hashing
    rnet::crypto::HashWriter hasher;

    // Version
    auto ver = tx.version();
    hasher.write(&ver, sizeof(ver));

    // Handle SIGHASH_ANYONECANPAY
    bool anyone_can_pay = (hash_type & SIGHASH_ANYONECANPAY) != 0;
    int base_type = hash_type & 0x1F;

    // Inputs
    if (anyone_can_pay) {
        // Only serialize the input being signed
        uint8_t count = 1;
        hasher.write(&count, 1);

        // Outpoint
        hasher.write(tx.vin()[input].prevout.hash.data(), 32);
        uint32_t n = tx.vin()[input].prevout.n;
        hasher.write(&n, sizeof(n));

        // Script code (the subscript)
        uint64_t script_len = script_code.size();
        // Write compact size
        if (script_len < 253) {
            uint8_t b = static_cast<uint8_t>(script_len);
            hasher.write(&b, 1);
        } else {
            // Simplified: scripts shouldn't be this large
            uint8_t marker = 253;
            hasher.write(&marker, 1);
            uint16_t len16 = static_cast<uint16_t>(script_len);
            hasher.write(&len16, 2);
        }
        hasher.write(script_code.data(), script_code.size());

        // Sequence
        uint32_t seq = tx.vin()[input].sequence;
        hasher.write(&seq, sizeof(seq));
    } else {
        // Serialize all inputs
        uint8_t vin_count = static_cast<uint8_t>(tx.vin().size());
        hasher.write(&vin_count, 1);

        for (size_t i = 0; i < tx.vin().size(); ++i) {
            // Outpoint
            hasher.write(tx.vin()[i].prevout.hash.data(), 32);
            uint32_t n = tx.vin()[i].prevout.n;
            hasher.write(&n, sizeof(n));

            // Script: use script_code for the input being signed,
            // empty for others.
            if (i == input) {
                uint64_t script_len = script_code.size();
                if (script_len < 253) {
                    uint8_t b = static_cast<uint8_t>(script_len);
                    hasher.write(&b, 1);
                } else {
                    uint8_t marker = 253;
                    hasher.write(&marker, 1);
                    uint16_t len16 = static_cast<uint16_t>(script_len);
                    hasher.write(&len16, 2);
                }
                hasher.write(script_code.data(), script_code.size());
            } else {
                uint8_t zero = 0;
                hasher.write(&zero, 1);
            }

            // Sequence
            uint32_t seq = tx.vin()[i].sequence;
            if (i != input &&
                (base_type == SIGHASH_NONE || base_type == SIGHASH_SINGLE)) {
                seq = 0;
            }
            hasher.write(&seq, sizeof(seq));
        }
    }

    // Outputs
    if (base_type == SIGHASH_NONE) {
        uint8_t zero = 0;
        hasher.write(&zero, 1);
    } else if (base_type == SIGHASH_SINGLE) {
        if (input >= tx.vout().size()) {
            rnet::uint256 one;
            one[0] = 1;
            return one;
        }
        uint8_t count = static_cast<uint8_t>(input + 1);
        hasher.write(&count, 1);
        for (size_t i = 0; i < input; ++i) {
            // Empty/null outputs for indices before the signed input
            int64_t neg1 = -1;
            hasher.write(&neg1, sizeof(neg1));
            uint8_t zero = 0;
            hasher.write(&zero, 1);
        }
        // The output at the signed input index
        hasher.write(&tx.vout()[input].value, sizeof(tx.vout()[input].value));
        uint64_t spk_len = tx.vout()[input].script_pub_key.size();
        if (spk_len < 253) {
            uint8_t b = static_cast<uint8_t>(spk_len);
            hasher.write(&b, 1);
        } else {
            uint8_t marker = 253;
            hasher.write(&marker, 1);
            uint16_t len16 = static_cast<uint16_t>(spk_len);
            hasher.write(&len16, 2);
        }
        hasher.write(tx.vout()[input].script_pub_key.data(), spk_len);
    } else {
        // SIGHASH_ALL: serialize all outputs
        uint8_t vout_count = static_cast<uint8_t>(tx.vout().size());
        hasher.write(&vout_count, 1);
        for (const auto& out : tx.vout()) {
            hasher.write(&out.value, sizeof(out.value));
            uint64_t spk_len = out.script_pub_key.size();
            if (spk_len < 253) {
                uint8_t b = static_cast<uint8_t>(spk_len);
                hasher.write(&b, 1);
            } else {
                uint8_t marker = 253;
                hasher.write(&marker, 1);
                uint16_t len16 = static_cast<uint16_t>(spk_len);
                hasher.write(&len16, 2);
            }
            hasher.write(out.script_pub_key.data(), spk_len);
        }
    }

    // Locktime
    auto lt = tx.locktime();
    hasher.write(&lt, sizeof(lt));

    // Hash type
    uint32_t ht32 = static_cast<uint32_t>(hash_type);
    hasher.write(&ht32, sizeof(ht32));

    return hasher.get_hash256();
}

// ── Witness signature hash (BIP143-style) ───────────────────────────

rnet::uint256 witness_signature_hash(
    const rnet::primitives::CTransaction& tx,
    unsigned int input,
    const CScript& script_code,
    int64_t amount,
    int hash_type) {

    bool anyone_can_pay = (hash_type & SIGHASH_ANYONECANPAY) != 0;
    int base_type = hash_type & 0x1F;

    rnet::uint256 hash_prevouts;
    rnet::uint256 hash_sequence;
    rnet::uint256 hash_outputs;

    // hashPrevouts
    if (!anyone_can_pay) {
        rnet::crypto::HashWriter hw;
        for (const auto& in : tx.vin()) {
            hw.write(in.prevout.hash.data(), 32);
            uint32_t n = in.prevout.n;
            hw.write(&n, sizeof(n));
        }
        hash_prevouts = hw.get_hash256();
    }

    // hashSequence
    if (!anyone_can_pay && base_type != SIGHASH_SINGLE &&
        base_type != SIGHASH_NONE) {
        rnet::crypto::HashWriter hw;
        for (const auto& in : tx.vin()) {
            uint32_t seq = in.sequence;
            hw.write(&seq, sizeof(seq));
        }
        hash_sequence = hw.get_hash256();
    }

    // hashOutputs
    if (base_type != SIGHASH_SINGLE && base_type != SIGHASH_NONE) {
        rnet::crypto::HashWriter hw;
        for (const auto& out : tx.vout()) {
            hw.write(&out.value, sizeof(out.value));
            uint64_t spk_len = out.script_pub_key.size();
            if (spk_len < 253) {
                uint8_t b = static_cast<uint8_t>(spk_len);
                hw.write(&b, 1);
            } else {
                uint8_t marker = 253;
                hw.write(&marker, 1);
                uint16_t len16 = static_cast<uint16_t>(spk_len);
                hw.write(&len16, 2);
            }
            hw.write(out.script_pub_key.data(), spk_len);
        }
        hash_outputs = hw.get_hash256();
    } else if (base_type == SIGHASH_SINGLE && input < tx.vout().size()) {
        rnet::crypto::HashWriter hw;
        hw.write(&tx.vout()[input].value, sizeof(tx.vout()[input].value));
        auto& spk = tx.vout()[input].script_pub_key;
        uint64_t spk_len = spk.size();
        if (spk_len < 253) {
            uint8_t b = static_cast<uint8_t>(spk_len);
            hw.write(&b, 1);
        } else {
            uint8_t marker = 253;
            hw.write(&marker, 1);
            uint16_t len16 = static_cast<uint16_t>(spk_len);
            hw.write(&len16, 2);
        }
        hw.write(spk.data(), spk_len);
        hash_outputs = hw.get_hash256();
    }

    // Final preimage
    rnet::crypto::HashWriter preimage;
    auto ver = tx.version();
    preimage.write(&ver, sizeof(ver));
    preimage.write(hash_prevouts.data(), 32);
    preimage.write(hash_sequence.data(), 32);

    // This input's outpoint
    preimage.write(tx.vin()[input].prevout.hash.data(), 32);
    uint32_t n = tx.vin()[input].prevout.n;
    preimage.write(&n, sizeof(n));

    // Script code
    uint64_t sc_len = script_code.size();
    if (sc_len < 253) {
        uint8_t b = static_cast<uint8_t>(sc_len);
        preimage.write(&b, 1);
    } else {
        uint8_t marker = 253;
        preimage.write(&marker, 1);
        uint16_t len16 = static_cast<uint16_t>(sc_len);
        preimage.write(&len16, 2);
    }
    preimage.write(script_code.data(), script_code.size());

    // Amount
    preimage.write(&amount, sizeof(amount));

    // Sequence
    uint32_t seq = tx.vin()[input].sequence;
    preimage.write(&seq, sizeof(seq));

    preimage.write(hash_outputs.data(), 32);

    // Locktime
    auto lt = tx.locktime();
    preimage.write(&lt, sizeof(lt));

    // Hash type
    uint32_t ht32 = static_cast<uint32_t>(hash_type);
    preimage.write(&ht32, sizeof(ht32));

    return preimage.get_hash256();
}

// ── Signing helpers ─────────────────────────────────────────────────

std::vector<uint8_t> sign_ed25519(
    const std::vector<uint8_t>& secret,
    const rnet::uint256& sighash) {
    if (secret.size() != 64) return {};

    rnet::crypto::Ed25519SecretKey sk;
    std::memcpy(sk.data.data(), secret.data(), 64);

    auto result = rnet::crypto::ed25519_sign(
        sk, std::span<const uint8_t>(sighash.data(), sighash.size()));

    sk.wipe();

    if (!result.is_ok()) return {};

    return std::vector<uint8_t>(result.value().data.begin(),
                                result.value().data.end());
}

std::vector<uint8_t> sign_secp256k1(
    const std::vector<uint8_t>& secret,
    const rnet::uint256& sighash) {
    if (secret.size() != 32) return {};

    rnet::crypto::Secp256k1SecretKey sk;
    std::memcpy(sk.data.data(), secret.data(), 32);

    auto result = rnet::crypto::secp256k1_sign(sk, sighash);

    sk.wipe();

    if (!result.is_ok()) return {};

    return result.value().der_data;
}

// ── Transaction signing ─────────────────────────────────────────────

bool sign_input(const SigningProvider& provider,
                rnet::primitives::CMutableTransaction& mtx,
                unsigned int input,
                const CScript& script_pub_key,
                int64_t amount,
                int hash_type) {
    if (input >= mtx.vin.size()) return false;

    std::vector<std::vector<uint8_t>> solutions;
    TxoutType type = solver(script_pub_key, solutions);

    rnet::primitives::CTransaction tx(mtx);

    switch (type) {
        case TxoutType::PUBKEYHASH: {
            // P2PKH: try Ed25519, then secp256k1
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
            // P2WPKH: signature and pubkey go in the witness
            if (solutions.empty()) return false;
            auto& hash = solutions[0];

            std::vector<uint8_t> secret, pubkey;
            bool have_key = provider.get_ed25519_key(hash, secret, pubkey) ||
                            provider.get_secp256k1_key(hash, secret, pubkey);
            if (!have_key) return false;

            // Witness script for P2WPKH is P2PKH of the pubkey
            CScript witness_script;
            witness_script << Opcode::OP_DUP
                           << Opcode::OP_HASH160
                           << hash
                           << Opcode::OP_EQUALVERIFY
                           << Opcode::OP_CHECKSIG;

            auto sighash = witness_signature_hash(
                tx, input, witness_script, amount, hash_type);

            std::vector<uint8_t> sig;
            if (pubkey.size() == 32) {
                sig = sign_ed25519(secret, sighash);
            } else {
                sig = sign_secp256k1(secret, sighash);
            }
            if (sig.empty()) return false;
            sig.push_back(static_cast<uint8_t>(hash_type));

            mtx.vin[input].witness.stack.clear();
            mtx.vin[input].witness.stack.push_back(sig);
            mtx.vin[input].witness.stack.push_back(pubkey);
            mtx.vin[input].script_sig.clear();
            return true;
        }

        case TxoutType::PUBKEY: {
            if (solutions.empty()) return false;
            auto& pubkey = solutions[0];

            // Compute hash of pubkey to look up key
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

int sign_transaction(const SigningProvider& provider,
                     rnet::primitives::CMutableTransaction& mtx,
                     const std::vector<CScript>& prev_scripts,
                     const std::vector<int64_t>& prev_amounts,
                     int hash_type) {
    int signed_count = 0;
    for (size_t i = 0; i < mtx.vin.size(); ++i) {
        if (i >= prev_scripts.size() || i >= prev_amounts.size()) break;
        if (sign_input(provider, mtx, static_cast<unsigned int>(i),
                       prev_scripts[i], prev_amounts[i], hash_type)) {
            ++signed_count;
        }
    }
    return signed_count;
}

}  // namespace rnet::script

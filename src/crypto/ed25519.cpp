// Copyright (c) 2024-2026 The ResonanceNet Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "crypto/ed25519.h"

#include "core/hex.h"
#include "core/logging.h"
#include "core/random.h"
#include "crypto/hash.h"

#include <cstring>
#include <stdexcept>

#include <openssl/err.h>
#include <openssl/evp.h>

namespace rnet::crypto {

// ---------------------------------------------------------------------------
// Ed25519SecretKey::wipe
// ---------------------------------------------------------------------------
// Securely zeroes the 64-byte secret key buffer (seed + embedded pubkey)
// so that key material does not linger on the stack or heap after use.
// ---------------------------------------------------------------------------
void Ed25519SecretKey::wipe() {
    secure_wipe(data.data(), data.size());
}

// ---------------------------------------------------------------------------
// Ed25519PublicKey::to_uint256
// ---------------------------------------------------------------------------
// Converts the 32-byte compressed curve point to a uint256 for use as
// a map key, UTXO index, or serialisation target.
// ---------------------------------------------------------------------------
rnet::uint256 Ed25519PublicKey::to_uint256() const {
    rnet::uint256 result;
    std::memcpy(result.data(), data.data(), 32);  // 32 bytes = 256 bits
    return result;
}

// ---------------------------------------------------------------------------
// Ed25519PublicKey::from_uint256
// ---------------------------------------------------------------------------
Ed25519PublicKey Ed25519PublicKey::from_uint256(const rnet::uint256& val) {
    Ed25519PublicKey pk;
    std::memcpy(pk.data.data(), val.data(), 32);
    return pk;
}

// ---------------------------------------------------------------------------
// Ed25519PublicKey::from_bytes
// ---------------------------------------------------------------------------
Ed25519PublicKey Ed25519PublicKey::from_bytes(
    std::span<const uint8_t> bytes)
{
    Ed25519PublicKey pk;
    size_t copy_len = (bytes.size() < 32) ? bytes.size() : 32;
    std::memcpy(pk.data.data(), bytes.data(), copy_len);
    return pk;
}

// ---------------------------------------------------------------------------
// Ed25519PublicKey::is_zero
// ---------------------------------------------------------------------------
bool Ed25519PublicKey::is_zero() const {
    for (auto b : data) {
        if (b != 0) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Ed25519PublicKey::to_hex
// ---------------------------------------------------------------------------
std::string Ed25519PublicKey::to_hex() const {
    return rnet::core::to_hex(
        std::span<const uint8_t>(data.data(), data.size()));
}

// ---------------------------------------------------------------------------
// Ed25519PublicKey::from_hex
// ---------------------------------------------------------------------------
// Parses a 64-character hex string into a 32-byte public key.
// Returns an error Result if the input length is wrong or contains
// invalid hex characters.
// ---------------------------------------------------------------------------
rnet::Result<Ed25519PublicKey> Ed25519PublicKey::from_hex(
    std::string_view hex)
{
    // 1. Validate hex string length (32 bytes = 64 hex chars)
    if (hex.size() != 64) {
        return rnet::Result<Ed25519PublicKey>::err(
            "Ed25519 public key hex must be 64 characters");
    }

    // 2. Decode hex to raw bytes
    auto bytes = rnet::core::from_hex(hex);
    if (bytes.size() != 32) {
        return rnet::Result<Ed25519PublicKey>::err(
            "Invalid hex for Ed25519 public key");
    }

    // 3. Copy into public key struct
    Ed25519PublicKey pk;
    std::memcpy(pk.data.data(), bytes.data(), 32);
    return rnet::Result<Ed25519PublicKey>::ok(pk);
}

// ---------------------------------------------------------------------------
// Ed25519Signature::is_zero
// ---------------------------------------------------------------------------
bool Ed25519Signature::is_zero() const {
    for (auto b : data) {
        if (b != 0) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Ed25519Signature::to_hex
// ---------------------------------------------------------------------------
std::string Ed25519Signature::to_hex() const {
    return rnet::core::to_hex(
        std::span<const uint8_t>(data.data(), data.size()));
}

// ---------------------------------------------------------------------------
// Ed25519Signature::from_hex
// ---------------------------------------------------------------------------
// Parses a 128-character hex string into a 64-byte signature (R || S).
// ---------------------------------------------------------------------------
rnet::Result<Ed25519Signature> Ed25519Signature::from_hex(
    std::string_view hex)
{
    // 1. Validate hex string length (64 bytes = 128 hex chars)
    if (hex.size() != 128) {
        return rnet::Result<Ed25519Signature>::err(
            "Ed25519 signature hex must be 128 characters");
    }

    // 2. Decode hex to raw bytes
    auto bytes = rnet::core::from_hex(hex);
    if (bytes.size() != 64) {
        return rnet::Result<Ed25519Signature>::err(
            "Invalid hex for Ed25519 signature");
    }

    // 3. Copy into signature struct
    Ed25519Signature sig;
    std::memcpy(sig.data.data(), bytes.data(), 64);
    return rnet::Result<Ed25519Signature>::ok(sig);
}

// ---------------------------------------------------------------------------
// Ed25519KeyPair::wipe
// ---------------------------------------------------------------------------
void Ed25519KeyPair::wipe() {
    secret.wipe();
}

// ---------------------------------------------------------------------------
// RAII wrappers for OpenSSL types
// ---------------------------------------------------------------------------
// Custom deleters allow use with std::unique_ptr so that EVP handles
// are freed automatically when they leave scope, preventing leaks on
// early-return error paths.
// ---------------------------------------------------------------------------

struct EvpPkeyDeleter {
    void operator()(EVP_PKEY* p) const {
        if (p) EVP_PKEY_free(p);
    }
};
using EvpPkeyPtr = std::unique_ptr<EVP_PKEY, EvpPkeyDeleter>;

struct EvpMdCtxDeleter {
    void operator()(EVP_MD_CTX* p) const {
        if (p) EVP_MD_CTX_free(p);
    }
};
using EvpMdCtxPtr = std::unique_ptr<EVP_MD_CTX, EvpMdCtxDeleter>;

// ---------------------------------------------------------------------------
// make_ed25519_private_key
// ---------------------------------------------------------------------------
// Wraps a 32-byte seed into an OpenSSL EVP_PKEY suitable for signing.
// The seed is the raw private scalar before SHA-512 expansion.
// ---------------------------------------------------------------------------
static EvpPkeyPtr make_ed25519_private_key(
    std::span<const uint8_t> seed)
{
    EVP_PKEY* pkey = EVP_PKEY_new_raw_private_key(
        EVP_PKEY_ED25519,  // NID for Ed25519
        nullptr,           // no engine
        seed.data(), seed.size());
    return EvpPkeyPtr(pkey);
}

// ---------------------------------------------------------------------------
// make_ed25519_public_key
// ---------------------------------------------------------------------------
// Wraps a 32-byte compressed curve point into an OpenSSL EVP_PKEY
// suitable for signature verification.
// ---------------------------------------------------------------------------
static EvpPkeyPtr make_ed25519_public_key(
    std::span<const uint8_t> pubkey_bytes)
{
    EVP_PKEY* pkey = EVP_PKEY_new_raw_public_key(
        EVP_PKEY_ED25519,  // NID for Ed25519
        nullptr,           // no engine
        pubkey_bytes.data(), pubkey_bytes.size());
    return EvpPkeyPtr(pkey);
}

// ---------------------------------------------------------------------------
// ed25519_generate
// ---------------------------------------------------------------------------
// Generates a fresh Ed25519 keypair from 32 bytes of cryptographic
// randomness (CSPRNG).  The seed is wiped from the stack immediately
// after deriving the keypair.
// ---------------------------------------------------------------------------
rnet::Result<Ed25519KeyPair> ed25519_generate() {
    // 1. Draw 32 bytes of cryptographic randomness for the seed
    uint8_t seed[32];
    rnet::core::get_rand_bytes(std::span<uint8_t>(seed, 32));

    // 2. Derive the full keypair from the seed
    auto result = ed25519_from_seed(
        std::span<const uint8_t>(seed, 32));

    // 3. Wipe seed from stack so it does not persist
    secure_wipe(seed, 32);

    return result;
}

// ---------------------------------------------------------------------------
// ed25519_from_seed
// ---------------------------------------------------------------------------
// RFC 8032 key derivation.  Given a 32-byte seed, OpenSSL internally
// computes SHA-512(seed) to obtain the secret scalar (lower 32 bytes,
// clamped) and the nonce prefix (upper 32 bytes).  We extract the
// public key (point compression of scalar * B) and pack the keypair
// as seed[32] || pubkey[32].
//
// NOTE: The 64-byte Ed25519SecretKey layout matches NaCl/libsodium
// convention (seed || public_key), NOT the expanded scalar form.
// ---------------------------------------------------------------------------
rnet::Result<Ed25519KeyPair> ed25519_from_seed(
    std::span<const uint8_t> seed)
{
    // 1. Validate seed length
    if (seed.size() != 32) {
        return rnet::Result<Ed25519KeyPair>::err(
            "Ed25519 seed must be exactly 32 bytes");
    }

    // 2. Create an OpenSSL private key from the raw seed
    auto pkey = make_ed25519_private_key(seed);
    if (!pkey) {
        return rnet::Result<Ed25519KeyPair>::err(
            "Failed to create Ed25519 private key from seed");
    }

    // 3. Extract the 32-byte public key (compressed curve point)
    size_t pub_len = 32;
    uint8_t pub_bytes[32];
    if (EVP_PKEY_get_raw_public_key(pkey.get(), pub_bytes, &pub_len)
        != 1 || pub_len != 32) {
        return rnet::Result<Ed25519KeyPair>::err(
            "Failed to extract Ed25519 public key");
    }

    // 4. Build the keypair: secret = seed[32] || pubkey[32]
    Ed25519KeyPair kp;
    std::memcpy(kp.secret.data.data(), seed.data(), 32);
    std::memcpy(kp.secret.data.data() + 32, pub_bytes, 32);
    std::memcpy(kp.public_key.data.data(), pub_bytes, 32);

    return rnet::Result<Ed25519KeyPair>::ok(std::move(kp));
}

// ---------------------------------------------------------------------------
// ed25519_sign  (binary message)
// ---------------------------------------------------------------------------
// RFC 8032 Ed25519 signing (PureEdDSA, no context or prehash).
//
// Algorithm:
//   1. secret_scalar = SHA-512(seed)[0..31]  (clamped)
//   2. nonce_hash    = SHA-512(SHA-512(seed)[32..63] || message)
//   3. R = nonce_hash * B  (base point multiplication)
//   4. S = nonce_hash + SHA-512(R || pubkey || message) * secret_scalar
//   sig = R || S  (64 bytes)
//
// NOTE: ResonanceNet uses Ed25519 for block signing and transaction
// signatures (32-byte pubkeys).  ECDSA/secp256k1 is the fallback
// for legacy compatibility.
// ---------------------------------------------------------------------------
rnet::Result<Ed25519Signature> ed25519_sign(
    const Ed25519SecretKey& secret,
    std::span<const uint8_t> message)
{
    // 1. Reconstruct the OpenSSL private key from the 32-byte seed
    auto pkey = make_ed25519_private_key(secret.seed());
    if (!pkey) {
        return rnet::Result<Ed25519Signature>::err(
            "Failed to create Ed25519 signing key");
    }

    // 2. Allocate a digest-sign context
    EvpMdCtxPtr ctx(EVP_MD_CTX_new());
    if (!ctx) {
        return rnet::Result<Ed25519Signature>::err(
            "Failed to create EVP_MD_CTX for Ed25519 signing");
    }

    // 3. Initialise the signing operation (Ed25519 has no separate digest)
    if (EVP_DigestSignInit(ctx.get(), nullptr, nullptr, nullptr,
                           pkey.get()) != 1) {
        return rnet::Result<Ed25519Signature>::err(
            "EVP_DigestSignInit failed for Ed25519");
    }

    // 4. Produce the 64-byte signature
    Ed25519Signature sig;
    size_t sig_len = 64;  // Ed25519 signatures are always 64 bytes

    if (EVP_DigestSign(ctx.get(), sig.data.data(), &sig_len,
                       message.data(), message.size()) != 1) {
        return rnet::Result<Ed25519Signature>::err(
            "EVP_DigestSign failed for Ed25519");
    }

    // 5. Sanity-check output length
    if (sig_len != 64) {
        return rnet::Result<Ed25519Signature>::err(
            "Unexpected Ed25519 signature length");
    }

    return rnet::Result<Ed25519Signature>::ok(sig);
}

// ---------------------------------------------------------------------------
// ed25519_sign  (string_view convenience)
// ---------------------------------------------------------------------------
rnet::Result<Ed25519Signature> ed25519_sign(
    const Ed25519SecretKey& secret,
    std::string_view message)
{
    return ed25519_sign(secret,
        std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(message.data()),
            message.size()));
}

// ---------------------------------------------------------------------------
// ed25519_verify  (binary message)
// ---------------------------------------------------------------------------
// RFC 8032 Ed25519 verification.  Reconstructs the public-key EVP_PKEY,
// then delegates to OpenSSL's EVP_DigestVerify which checks:
//
//   S * B == R + SHA-512(R || pubkey || message) * A
//
// Returns true only if the signature is mathematically valid.
// ---------------------------------------------------------------------------
bool ed25519_verify(
    const Ed25519PublicKey& pubkey,
    std::span<const uint8_t> message,
    const Ed25519Signature& signature)
{
    // 1. Wrap the 32-byte public key in an OpenSSL EVP_PKEY
    auto pkey = make_ed25519_public_key(
        std::span<const uint8_t>(pubkey.data.data(), 32));
    if (!pkey) {
        return false;
    }

    // 2. Allocate a digest-verify context
    EvpMdCtxPtr ctx(EVP_MD_CTX_new());
    if (!ctx) {
        return false;
    }

    // 3. Initialise the verification operation
    if (EVP_DigestVerifyInit(ctx.get(), nullptr, nullptr, nullptr,
                             pkey.get()) != 1) {
        return false;
    }

    // 4. Verify: returns 1 on success, 0 on signature mismatch
    int rc = EVP_DigestVerify(
        ctx.get(),
        signature.data.data(), signature.data.size(),
        message.data(), message.size());

    return rc == 1;
}

// ---------------------------------------------------------------------------
// ed25519_verify  (string_view convenience)
// ---------------------------------------------------------------------------
bool ed25519_verify(
    const Ed25519PublicKey& pubkey,
    std::string_view message,
    const Ed25519Signature& signature)
{
    return ed25519_verify(pubkey,
        std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(message.data()),
            message.size()),
        signature);
}

// ---------------------------------------------------------------------------
// ed25519_batch_verify
// ---------------------------------------------------------------------------
// Verifies multiple (pubkey, message, signature) tuples in one call.
//
// OpenSSL does not yet expose a native batch-verification API for
// Ed25519, so we fall back to sequential single-verification.  The
// interface exists so that a future OpenSSL upgrade (or a libsodium
// backend) can switch to true batch mode (roughly 2x faster for
// large batches) without changing callers.
// ---------------------------------------------------------------------------
bool ed25519_batch_verify(
    const std::vector<Ed25519PublicKey>& pubkeys,
    const std::vector<std::span<const uint8_t>>& messages,
    const std::vector<Ed25519Signature>& signatures)
{
    // 1. All three vectors must have matching length
    if (pubkeys.size() != messages.size() ||
        pubkeys.size() != signatures.size()) {
        return false;
    }

    // 2. Empty batch is trivially valid
    if (pubkeys.empty()) {
        return true;
    }

    // 3. Sequential fallback — verify each signature individually
    for (size_t i = 0; i < pubkeys.size(); ++i) {
        if (!ed25519_verify(pubkeys[i], messages[i], signatures[i])) {
            return false;
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// ed25519_is_valid_pubkey
// ---------------------------------------------------------------------------
// Checks whether a 32-byte buffer encodes a valid point on the
// Ed25519 curve.  Delegates to OpenSSL: if EVP_PKEY construction
// succeeds, the point passed decompression and on-curve checks.
// ---------------------------------------------------------------------------
bool ed25519_is_valid_pubkey(std::span<const uint8_t> bytes) {
    // 1. Length must be exactly 32 bytes (compressed Edwards point)
    if (bytes.size() != 32) {
        return false;
    }

    // 2. Attempt to construct an EVP_PKEY — succeeds iff point is on curve
    auto pkey = make_ed25519_public_key(bytes);
    return pkey != nullptr;
}

// ---------------------------------------------------------------------------
// ed25519_coinbase_script
// ---------------------------------------------------------------------------
// Builds the coinbase output script for Proof-of-Training block rewards.
//
// Script format (34 bytes total):
//   [0x20]                 — OP_PUSH32: push next 32 bytes onto stack
//   [32-byte pubkey]       — compressed Ed25519 public key
//   [0xAC]                 — OP_CHECKSIG: verify signature against pubkey
//
// This is the Ed25519 analogue of Bitcoin's P2PK script.
// ---------------------------------------------------------------------------
std::vector<uint8_t> ed25519_coinbase_script(
    const Ed25519PublicKey& pubkey)
{
    std::vector<uint8_t> script;
    script.reserve(34);                // 1 + 32 + 1 = 34 bytes
    script.push_back(0x20);            // OP_PUSH32: push 32 bytes
    script.insert(script.end(), pubkey.data.begin(), pubkey.data.end());
    script.push_back(0xAC);            // OP_CHECKSIG
    return script;
}

// ---------------------------------------------------------------------------
// ed25519_parse_coinbase_script
// ---------------------------------------------------------------------------
// Parses a 34-byte coinbase script back into an Ed25519 public key.
//
// Validates:
//   1. Total length is exactly 34 bytes
//   2. First byte is 0x20 (OP_PUSH32)
//   3. Last byte is 0xAC (OP_CHECKSIG)
//   4. Embedded 32-byte key is a valid Ed25519 curve point
// ---------------------------------------------------------------------------
rnet::Result<Ed25519PublicKey> ed25519_parse_coinbase_script(
    std::span<const uint8_t> script)
{
    // 1. Validate total script length
    if (script.size() != 34) {
        return rnet::Result<Ed25519PublicKey>::err(
            "Coinbase script must be exactly 34 bytes");
    }

    // 2. Check leading opcode: 0x20 = OP_PUSH32
    if (script[0] != 0x20) {
        return rnet::Result<Ed25519PublicKey>::err(
            "Coinbase script must start with 0x20 (OP_PUSH32)");
    }

    // 3. Check trailing opcode: 0xAC = OP_CHECKSIG
    if (script[33] != 0xAC) {
        return rnet::Result<Ed25519PublicKey>::err(
            "Coinbase script must end with 0xAC (OP_CHECKSIG)");
    }

    // 4. Extract the 32-byte public key from bytes [1..32]
    Ed25519PublicKey pk;
    std::memcpy(pk.data.data(), script.data() + 1, 32);

    // 5. Verify the key is a valid Ed25519 curve point
    if (!ed25519_is_valid_pubkey(
            std::span<const uint8_t>(pk.data.data(), 32))) {
        return rnet::Result<Ed25519PublicKey>::err(
            "Invalid Ed25519 public key in coinbase script");
    }

    return rnet::Result<Ed25519PublicKey>::ok(pk);
}

} // namespace rnet::crypto

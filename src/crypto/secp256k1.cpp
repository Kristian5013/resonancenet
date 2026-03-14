// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "crypto/secp256k1.h"

#include "core/hex.h"
#include "core/logging.h"
#include "core/random.h"
#include "crypto/hash.h"

#include <cstring>
#include <mutex>

// Conditionally include secp256k1 if available.
// If not available, we provide stub implementations that return errors.
#if __has_include(<secp256k1.h>)
    #include <secp256k1.h>
    #include <secp256k1_ecdh.h>
    #define HAVE_SECP256K1 1
#else
    #define HAVE_SECP256K1 0
#endif

namespace rnet::crypto {

// ---------------------------------------------------------------------------
// Secp256k1PubKey::is_zero
// ---------------------------------------------------------------------------

bool Secp256k1PubKey::is_zero() const
{
    for (auto b : data) {
        if (b != 0) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Secp256k1PubKey::to_hex
// ---------------------------------------------------------------------------

std::string Secp256k1PubKey::to_hex() const
{
    return rnet::core::to_hex(
        std::span<const uint8_t>(data.data(), data.size()));
}

// ---------------------------------------------------------------------------
// Secp256k1PubKey::from_hex
// ---------------------------------------------------------------------------
//
// Design note — compressed public key hex parsing:
//   A SEC1-compressed secp256k1 public key is 33 bytes, which encodes to
//   66 hex characters.  The first byte is the parity prefix (0x02 or 0x03)
//   followed by the 32-byte x-coordinate.

rnet::Result<Secp256k1PubKey> Secp256k1PubKey::from_hex(
    std::string_view hex)
{
    // 1. Validate hex length (33 bytes = 66 hex chars).
    if (hex.size() != 66) {
        return rnet::Result<Secp256k1PubKey>::err(
            "secp256k1 compressed pubkey hex must be 66 characters");
    }

    // 2. Decode hex to bytes.
    auto bytes = rnet::core::from_hex(hex);
    if (bytes.size() != 33) {
        return rnet::Result<Secp256k1PubKey>::err(
            "Invalid hex for secp256k1 pubkey");
    }

    // 3. Delegate to from_bytes for curve-point validation.
    return from_bytes(std::span<const uint8_t>(bytes));
}

// ---------------------------------------------------------------------------
// Secp256k1PubKey::from_bytes
// ---------------------------------------------------------------------------
//
// Design note — SEC1 public key formats:
//   Compressed   (33 bytes): 0x02 | x   if y is even
//                            0x03 | x   if y is odd
//   Uncompressed (65 bytes): 0x04 | x | y
//
// Uncompressed keys are re-serialised to compressed form for storage.

rnet::Result<Secp256k1PubKey> Secp256k1PubKey::from_bytes(
    std::span<const uint8_t> bytes)
{
    // 1. Verify input length matches a known SEC1 encoding.
    if (bytes.size() != 33 && bytes.size() != 65) {
        return rnet::Result<Secp256k1PubKey>::err(
            "secp256k1 pubkey must be 33 (compressed) or "
            "65 (uncompressed) bytes");
    }

    Secp256k1PubKey pk;

    if (bytes.size() == 33) {
        // 2a. Already compressed — copy directly and validate on curve.
        std::memcpy(pk.data.data(), bytes.data(), 33);
        pk.valid = secp256k1_is_valid_pubkey(bytes);
    } else {
        // 2b. Uncompressed — compress via library round-trip.
        auto compressed = secp256k1_compress_pubkey(bytes);
        if (compressed.is_err()) {
            return rnet::Result<Secp256k1PubKey>::err(
                compressed.error());
        }
        pk = compressed.value();
    }

    return rnet::Result<Secp256k1PubKey>::ok(pk);
}

// ---------------------------------------------------------------------------
// Secp256k1SecretKey::wipe
// ---------------------------------------------------------------------------

void Secp256k1SecretKey::wipe()
{
    secure_wipe(data.data(), data.size());
}

// ---------------------------------------------------------------------------
// Secp256k1SecretKey::is_zero
// ---------------------------------------------------------------------------

bool Secp256k1SecretKey::is_zero() const
{
    for (auto b : data) {
        if (b != 0) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Secp256k1Signature::to_hex
// ---------------------------------------------------------------------------

std::string Secp256k1Signature::to_hex() const
{
    return rnet::core::to_hex(
        std::span<const uint8_t>(der_data));
}

// ---------------------------------------------------------------------------
// Secp256k1Signature::to_compact
// ---------------------------------------------------------------------------
//
// Design note — ECDSA compact signature format:
//   64 bytes = R (32 bytes big-endian) || S (32 bytes big-endian).
//   No DER length prefixes; fixed width.  Used in compact-sig wire formats.

std::array<uint8_t, 64> Secp256k1Signature::to_compact() const
{
    std::array<uint8_t, 64> compact{};
#if HAVE_SECP256K1
    // 1. Obtain the secp256k1 context.
    auto& ctx_inst = Secp256k1Context::instance();
    if (!ctx_inst.is_initialized()) return compact;

    auto* ctx = static_cast<secp256k1_context*>(ctx_inst.raw_ctx());

    // 2. Parse the internal DER representation.
    secp256k1_ecdsa_signature sig;
    if (secp256k1_ecdsa_signature_parse_der(
            ctx, &sig, der_data.data(), der_data.size()) != 1) {
        return compact;
    }

    // 3. Serialise to fixed-width compact form.
    secp256k1_ecdsa_signature_serialize_compact(ctx, compact.data(), &sig);
#endif
    return compact;
}

// ---------------------------------------------------------------------------
// Secp256k1Signature::from_compact
// ---------------------------------------------------------------------------
//
// Design note — DER encoding of ECDSA signatures:
//   0x30 <total-len> 0x02 <r-len> <R> 0x02 <s-len> <S>
//   Maximum 72 bytes (when both R and S need a leading 0x00 pad).

rnet::Result<Secp256k1Signature> Secp256k1Signature::from_compact(
    std::span<const uint8_t> compact)
{
    // 1. Validate compact signature length.
    if (compact.size() != 64) {
        return rnet::Result<Secp256k1Signature>::err(
            "Compact signature must be 64 bytes");
    }

#if HAVE_SECP256K1
    // 2. Obtain the secp256k1 context.
    auto& ctx_inst = Secp256k1Context::instance();
    if (!ctx_inst.is_initialized()) {
        return rnet::Result<Secp256k1Signature>::err(
            "secp256k1 not initialized");
    }
    auto* ctx = static_cast<secp256k1_context*>(ctx_inst.raw_ctx());

    // 3. Parse compact bytes into internal representation.
    secp256k1_ecdsa_signature sig;
    if (secp256k1_ecdsa_signature_parse_compact(
            ctx, &sig, compact.data()) != 1) {
        return rnet::Result<Secp256k1Signature>::err(
            "Failed to parse compact signature");
    }

    // 4. Re-serialise to DER for canonical storage.
    Secp256k1Signature result;
    result.der_data.resize(72);
    size_t der_len = 72;
    if (secp256k1_ecdsa_signature_serialize_der(
            ctx, result.der_data.data(), &der_len, &sig) != 1) {
        return rnet::Result<Secp256k1Signature>::err(
            "Failed to serialize signature to DER");
    }
    result.der_data.resize(der_len);
    return rnet::Result<Secp256k1Signature>::ok(std::move(result));
#else
    return rnet::Result<Secp256k1Signature>::err(
        "secp256k1 library not available");
#endif
}

// ---------------------------------------------------------------------------
// Secp256k1Signature::from_der
// ---------------------------------------------------------------------------

rnet::Result<Secp256k1Signature> Secp256k1Signature::from_der(
    std::span<const uint8_t> der)
{
    Secp256k1Signature sig;
    sig.der_data.assign(der.begin(), der.end());
    return rnet::Result<Secp256k1Signature>::ok(std::move(sig));
}

// ---------------------------------------------------------------------------
// Secp256k1KeyPair::wipe
// ---------------------------------------------------------------------------

void Secp256k1KeyPair::wipe()
{
    secret.wipe();
}

// ---------------------------------------------------------------------------
// Secp256k1Context (pimpl)
// ---------------------------------------------------------------------------

struct Secp256k1Context::Impl {
#if HAVE_SECP256K1
    secp256k1_context* ctx = nullptr;
#endif
    bool initialized = false;
    std::mutex mutex;
};

// ---------------------------------------------------------------------------
// Secp256k1Context::Secp256k1Context
// ---------------------------------------------------------------------------

Secp256k1Context::Secp256k1Context()
    : impl_(std::make_unique<Impl>()) {}

// ---------------------------------------------------------------------------
// Secp256k1Context::~Secp256k1Context
// ---------------------------------------------------------------------------

Secp256k1Context::~Secp256k1Context()
{
    shutdown();
}

// ---------------------------------------------------------------------------
// Secp256k1Context::instance
// ---------------------------------------------------------------------------

Secp256k1Context& Secp256k1Context::instance()
{
    static Secp256k1Context inst;
    return inst;
}

// ---------------------------------------------------------------------------
// Secp256k1Context::init
// ---------------------------------------------------------------------------
//
// Design note — context initialisation:
//   libsecp256k1 requires a context object for all operations.  We create
//   one with both SIGN and VERIFY capabilities, then randomise it with
//   32 bytes of entropy to harden against side-channel attacks.

rnet::Result<void> Secp256k1Context::init()
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->initialized) {
        return rnet::Result<void>::ok();
    }

#if HAVE_SECP256K1
    // 1. Create context with signing and verification capabilities.
    impl_->ctx = secp256k1_context_create(
        SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    if (!impl_->ctx) {
        return rnet::Result<void>::err(
            "Failed to create secp256k1 context");
    }

    // 2. Randomise context for side-channel protection.
    uint8_t seed[32];
    rnet::core::get_rand_bytes(std::span<uint8_t>(seed, 32));
    secp256k1_context_randomize(impl_->ctx, seed);
    secure_wipe(seed, 32);

    // 3. Mark initialised.
    impl_->initialized = true;
    return rnet::Result<void>::ok();
#else
    return rnet::Result<void>::err(
        "secp256k1 library not available at compile time");
#endif
}

// ---------------------------------------------------------------------------
// Secp256k1Context::shutdown
// ---------------------------------------------------------------------------

void Secp256k1Context::shutdown()
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
#if HAVE_SECP256K1
    if (impl_->ctx) {
        secp256k1_context_destroy(impl_->ctx);
        impl_->ctx = nullptr;
    }
#endif
    impl_->initialized = false;
}

// ---------------------------------------------------------------------------
// Secp256k1Context::is_initialized
// ---------------------------------------------------------------------------

bool Secp256k1Context::is_initialized() const
{
    return impl_->initialized;
}

// ---------------------------------------------------------------------------
// Secp256k1Context::raw_ctx
// ---------------------------------------------------------------------------

void* Secp256k1Context::raw_ctx() const
{
#if HAVE_SECP256K1
    return impl_->ctx;
#else
    return nullptr;
#endif
}

// ---------------------------------------------------------------------------
// secp256k1_generate
// ---------------------------------------------------------------------------
//
// Design note — key generation:
//   A valid secp256k1 secret key is a 32-byte scalar k where
//   0 < k < n, with curve order
//   n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141.
//   We draw random bytes and let libsecp256k1 verify validity,
//   retrying up to 100 times (probability of needing even 2 is negligible).

rnet::Result<Secp256k1KeyPair> secp256k1_generate()
{
#if HAVE_SECP256K1
    // 1. Ensure context is initialised.
    auto& ctx_inst = Secp256k1Context::instance();
    if (!ctx_inst.is_initialized()) {
        auto init_res = ctx_inst.init();
        if (init_res.is_err()) {
            return rnet::Result<Secp256k1KeyPair>::err(init_res.error());
        }
    }

    Secp256k1KeyPair kp;

    // 2. Generate random secret key, retry if outside [1, n-1].
    for (int attempt = 0; attempt < 100; ++attempt) {
        rnet::core::get_rand_bytes(
            std::span<uint8_t>(kp.secret.data.data(), 32));
        if (secp256k1_ec_seckey_verify(
                static_cast<secp256k1_context*>(ctx_inst.raw_ctx()),
                kp.secret.data.data()) == 1) {
            // 3. Derive the corresponding compressed public key.
            auto pk = secp256k1_pubkey_from_secret(kp.secret);
            if (pk.is_ok()) {
                kp.public_key = pk.value();
                return rnet::Result<Secp256k1KeyPair>::ok(std::move(kp));
            }
        }
    }

    return rnet::Result<Secp256k1KeyPair>::err(
        "Failed to generate valid secp256k1 key after 100 attempts");
#else
    return rnet::Result<Secp256k1KeyPair>::err(
        "secp256k1 library not available");
#endif
}

// ---------------------------------------------------------------------------
// secp256k1_pubkey_from_secret
// ---------------------------------------------------------------------------
//
// Design note — public key derivation:
//   Given secret scalar k, the public key is the curve point P = k * G
//   where G is the secp256k1 generator.  The result is serialised in
//   SEC1 compressed form: prefix byte (0x02 even y, 0x03 odd y) || x.

rnet::Result<Secp256k1PubKey> secp256k1_pubkey_from_secret(
    const Secp256k1SecretKey& secret)
{
#if HAVE_SECP256K1
    // 1. Validate context.
    auto& ctx_inst = Secp256k1Context::instance();
    if (!ctx_inst.is_initialized()) {
        return rnet::Result<Secp256k1PubKey>::err(
            "secp256k1 not initialized");
    }
    auto* ctx = static_cast<secp256k1_context*>(ctx_inst.raw_ctx());

    // 2. Compute P = k * G.
    secp256k1_pubkey pubkey;
    if (secp256k1_ec_pubkey_create(
            ctx, &pubkey, secret.data.data()) != 1) {
        return rnet::Result<Secp256k1PubKey>::err(
            "Failed to create public key from secret");
    }

    // 3. Serialise to compressed SEC1 (33 bytes).
    Secp256k1PubKey result;
    size_t out_len = 33;
    if (secp256k1_ec_pubkey_serialize(
            ctx, result.data.data(), &out_len, &pubkey,
            SECP256K1_EC_COMPRESSED) != 1) {
        return rnet::Result<Secp256k1PubKey>::err(
            "Failed to serialize public key");
    }
    result.valid = true;
    return rnet::Result<Secp256k1PubKey>::ok(result);
#else
    (void)secret;
    return rnet::Result<Secp256k1PubKey>::err(
        "secp256k1 library not available");
#endif
}

// ---------------------------------------------------------------------------
// secp256k1_secret_tweak_add
// ---------------------------------------------------------------------------
//
// Design note — BIP-32 child key derivation (private):
//   child_secret = (parent_secret + tweak) mod n
//   where n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141.
//   The tweak is the 32-byte HMAC-SHA512 left-half from the BIP-32 CKD step.

rnet::Result<Secp256k1SecretKey> secp256k1_secret_tweak_add(
    const Secp256k1SecretKey& secret,
    std::span<const uint8_t> tweak)
{
#if HAVE_SECP256K1
    // 1. Validate tweak length.
    if (tweak.size() != 32) {
        return rnet::Result<Secp256k1SecretKey>::err(
            "Tweak must be 32 bytes");
    }

    // 2. Validate context.
    auto& ctx_inst = Secp256k1Context::instance();
    if (!ctx_inst.is_initialized()) {
        return rnet::Result<Secp256k1SecretKey>::err(
            "secp256k1 not initialized");
    }
    auto* ctx = static_cast<secp256k1_context*>(ctx_inst.raw_ctx());

    // 3. Copy secret and add tweak in-place (mod n).
    Secp256k1SecretKey result;
    std::memcpy(result.data.data(), secret.data.data(), 32);

    if (secp256k1_ec_seckey_tweak_add(
            ctx, result.data.data(), tweak.data()) != 1) {
        return rnet::Result<Secp256k1SecretKey>::err(
            "Secret key tweak add failed");
    }
    return rnet::Result<Secp256k1SecretKey>::ok(result);
#else
    (void)secret; (void)tweak;
    return rnet::Result<Secp256k1SecretKey>::err(
        "secp256k1 library not available");
#endif
}

// ---------------------------------------------------------------------------
// secp256k1_pubkey_tweak_add
// ---------------------------------------------------------------------------
//
// Design note — BIP-32 child key derivation (public):
//   child_pubkey = parent_pubkey + tweak * G
//   This is the EC point addition that mirrors secret tweak-add without
//   needing the private key.

rnet::Result<Secp256k1PubKey> secp256k1_pubkey_tweak_add(
    const Secp256k1PubKey& pubkey,
    std::span<const uint8_t> tweak)
{
#if HAVE_SECP256K1
    // 1. Validate tweak length.
    if (tweak.size() != 32) {
        return rnet::Result<Secp256k1PubKey>::err(
            "Tweak must be 32 bytes");
    }

    // 2. Validate context.
    auto& ctx_inst = Secp256k1Context::instance();
    if (!ctx_inst.is_initialized()) {
        return rnet::Result<Secp256k1PubKey>::err(
            "secp256k1 not initialized");
    }
    auto* ctx = static_cast<secp256k1_context*>(ctx_inst.raw_ctx());

    // 3. Parse the compressed public key into internal representation.
    secp256k1_pubkey pk;
    if (secp256k1_ec_pubkey_parse(
            ctx, &pk, pubkey.data.data(), 33) != 1) {
        return rnet::Result<Secp256k1PubKey>::err(
            "Failed to parse public key for tweak");
    }

    // 4. Add tweak * G to the point.
    if (secp256k1_ec_pubkey_tweak_add(
            ctx, &pk, tweak.data()) != 1) {
        return rnet::Result<Secp256k1PubKey>::err(
            "Public key tweak add failed");
    }

    // 5. Re-serialise to compressed SEC1.
    Secp256k1PubKey result;
    size_t out_len = 33;
    secp256k1_ec_pubkey_serialize(
        ctx, result.data.data(), &out_len, &pk,
        SECP256K1_EC_COMPRESSED);
    result.valid = true;
    return rnet::Result<Secp256k1PubKey>::ok(result);
#else
    (void)pubkey; (void)tweak;
    return rnet::Result<Secp256k1PubKey>::err(
        "secp256k1 library not available");
#endif
}

// ---------------------------------------------------------------------------
// secp256k1_sign
// ---------------------------------------------------------------------------
//
// Design note — ECDSA signing algorithm:
//   1. Choose random nonce k (handled internally by libsecp256k1 via
//      RFC 6979 deterministic nonce generation).
//   2. Compute R = k * G;  r = R.x mod n.
//   3. Compute s = k^{-1} * (hash + r * secret) mod n.
//   4. Signature is (r, s), serialised to DER:
//      0x30 <len> 0x02 <r-len> <R> 0x02 <s-len> <S>
//
// Curve order n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141.

rnet::Result<Secp256k1Signature> secp256k1_sign(
    const Secp256k1SecretKey& secret,
    const rnet::uint256& hash)
{
#if HAVE_SECP256K1
    // 1. Validate context.
    auto& ctx_inst = Secp256k1Context::instance();
    if (!ctx_inst.is_initialized()) {
        return rnet::Result<Secp256k1Signature>::err(
            "secp256k1 not initialized");
    }
    auto* ctx = static_cast<secp256k1_context*>(ctx_inst.raw_ctx());

    // 2. Produce ECDSA signature (RFC 6979 nonce).
    secp256k1_ecdsa_signature sig;
    if (secp256k1_ecdsa_sign(ctx, &sig, hash.data(),
                             secret.data.data(), nullptr, nullptr) != 1) {
        return rnet::Result<Secp256k1Signature>::err(
            "ECDSA signing failed");
    }

    // 3. Serialise to DER encoding.
    Secp256k1Signature result;
    result.der_data.resize(72);
    size_t der_len = 72;
    if (secp256k1_ecdsa_signature_serialize_der(
            ctx, result.der_data.data(), &der_len, &sig) != 1) {
        return rnet::Result<Secp256k1Signature>::err(
            "Failed to serialize ECDSA signature");
    }
    result.der_data.resize(der_len);
    return rnet::Result<Secp256k1Signature>::ok(std::move(result));
#else
    (void)secret; (void)hash;
    return rnet::Result<Secp256k1Signature>::err(
        "secp256k1 library not available");
#endif
}

// ---------------------------------------------------------------------------
// secp256k1_verify
// ---------------------------------------------------------------------------
//
// Design note — ECDSA verification algorithm:
//   1. Parse the DER-encoded signature to recover (r, s).
//   2. Normalise to low-S form (BIP-62 malleability fix):
//      if s > n/2, replace s with n - s.
//   3. Compute u1 = hash * s^{-1} mod n,  u2 = r * s^{-1} mod n.
//   4. Compute R' = u1 * G + u2 * pubkey.
//   5. Accept iff R'.x mod n == r.

bool secp256k1_verify(
    const Secp256k1PubKey& pubkey,
    const rnet::uint256& hash,
    const Secp256k1Signature& signature)
{
#if HAVE_SECP256K1
    // 1. Validate context.
    auto& ctx_inst = Secp256k1Context::instance();
    if (!ctx_inst.is_initialized()) return false;
    auto* ctx = static_cast<secp256k1_context*>(ctx_inst.raw_ctx());

    // 2. Parse the compressed public key.
    secp256k1_pubkey pk;
    if (secp256k1_ec_pubkey_parse(
            ctx, &pk, pubkey.data.data(), 33) != 1) {
        return false;
    }

    // 3. Parse the DER-encoded signature.
    secp256k1_ecdsa_signature sig;
    if (secp256k1_ecdsa_signature_parse_der(
            ctx, &sig,
            signature.der_data.data(),
            signature.der_data.size()) != 1) {
        return false;
    }

    // 4. Normalise to low-S form (BIP-62).
    secp256k1_ecdsa_signature_normalize(ctx, &sig, &sig);

    // 5. Verify the signature against the hash and public key.
    return secp256k1_ecdsa_verify(ctx, &sig, hash.data(), &pk) == 1;
#else
    (void)pubkey; (void)hash; (void)signature;
    return false;
#endif
}

// ---------------------------------------------------------------------------
// secp256k1_ecdh
// ---------------------------------------------------------------------------
//
// Design note — ECDH shared secret:
//   shared = SHA-256( x-coordinate of secret * pubkey )
//   libsecp256k1 performs the scalar multiplication and hashing internally.

rnet::Result<rnet::uint256> secp256k1_ecdh(
    const Secp256k1SecretKey& secret,
    const Secp256k1PubKey& pubkey)
{
#if HAVE_SECP256K1
    // 1. Validate context.
    auto& ctx_inst = Secp256k1Context::instance();
    if (!ctx_inst.is_initialized()) {
        return rnet::Result<rnet::uint256>::err(
            "secp256k1 not initialized");
    }
    auto* ctx = static_cast<secp256k1_context*>(ctx_inst.raw_ctx());

    // 2. Parse the peer's compressed public key.
    secp256k1_pubkey pk;
    if (secp256k1_ec_pubkey_parse(
            ctx, &pk, pubkey.data.data(), 33) != 1) {
        return rnet::Result<rnet::uint256>::err(
            "Failed to parse public key for ECDH");
    }

    // 3. Compute the shared secret.
    rnet::uint256 result;
    if (secp256k1_ecdh(ctx, result.data(), &pk,
                       secret.data.data(), nullptr, nullptr) != 1) {
        return rnet::Result<rnet::uint256>::err("ECDH failed");
    }
    return rnet::Result<rnet::uint256>::ok(result);
#else
    (void)secret; (void)pubkey;
    return rnet::Result<rnet::uint256>::err(
        "secp256k1 library not available");
#endif
}

// ---------------------------------------------------------------------------
// secp256k1_is_valid_secret
// ---------------------------------------------------------------------------
//
// Design note — secret key validity:
//   A valid secret key is a 32-byte big-endian integer k satisfying
//   0 < k < n, where n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141.
//   Zero and values >= n are rejected.

bool secp256k1_is_valid_secret(std::span<const uint8_t> bytes)
{
    if (bytes.size() != 32) return false;

#if HAVE_SECP256K1
    // 1. Validate context.
    auto& ctx_inst = Secp256k1Context::instance();
    if (!ctx_inst.is_initialized()) return false;
    auto* ctx = static_cast<secp256k1_context*>(ctx_inst.raw_ctx());

    // 2. Let libsecp256k1 check the scalar range.
    return secp256k1_ec_seckey_verify(ctx, bytes.data()) == 1;
#else
    // Fallback: reject all-zeros.
    bool all_zero = true;
    for (size_t i = 0; i < 32; ++i) {
        if (bytes[i] != 0) { all_zero = false; break; }
    }
    return !all_zero;
#endif
}

// ---------------------------------------------------------------------------
// secp256k1_is_valid_pubkey
// ---------------------------------------------------------------------------
//
// Design note — public key validation:
//   Compressed keys must start with 0x02 or 0x03 (SEC1 even/odd y parity).
//   Uncompressed keys must start with 0x04.  libsecp256k1 additionally
//   verifies that the decoded point lies on the curve y^2 = x^3 + 7.

bool secp256k1_is_valid_pubkey(std::span<const uint8_t> bytes)
{
    if (bytes.size() != 33 && bytes.size() != 65) return false;

#if HAVE_SECP256K1
    // 1. Validate context.
    auto& ctx_inst = Secp256k1Context::instance();
    if (!ctx_inst.is_initialized()) return false;
    auto* ctx = static_cast<secp256k1_context*>(ctx_inst.raw_ctx());

    // 2. Attempt to parse — succeeds only for valid on-curve points.
    secp256k1_pubkey pk;
    return secp256k1_ec_pubkey_parse(
        ctx, &pk, bytes.data(), bytes.size()) == 1;
#else
    // Fallback: check prefix byte only.
    if (bytes.size() == 33) {
        return bytes[0] == 0x02 || bytes[0] == 0x03;
    }
    if (bytes.size() == 65) {
        return bytes[0] == 0x04;
    }
    return false;
#endif
}

// ---------------------------------------------------------------------------
// secp256k1_decompress_pubkey
// ---------------------------------------------------------------------------
//
// Design note — public key decompression:
//   Given compressed key (prefix || x), recover the full point (x, y) by
//   solving y^2 = x^3 + 7 (mod p) and selecting the root matching the
//   parity prefix.  Result is SEC1 uncompressed: 0x04 || x || y (65 bytes).

rnet::Result<std::array<uint8_t, 65>> secp256k1_decompress_pubkey(
    const Secp256k1PubKey& pubkey)
{
#if HAVE_SECP256K1
    // 1. Validate context.
    auto& ctx_inst = Secp256k1Context::instance();
    if (!ctx_inst.is_initialized()) {
        return rnet::Result<std::array<uint8_t, 65>>::err(
            "secp256k1 not initialized");
    }
    auto* ctx = static_cast<secp256k1_context*>(ctx_inst.raw_ctx());

    // 2. Parse the 33-byte compressed key.
    secp256k1_pubkey pk;
    if (secp256k1_ec_pubkey_parse(
            ctx, &pk, pubkey.data.data(), 33) != 1) {
        return rnet::Result<std::array<uint8_t, 65>>::err(
            "Failed to parse compressed pubkey");
    }

    // 3. Serialise to 65-byte uncompressed form.
    std::array<uint8_t, 65> uncompressed;
    size_t out_len = 65;
    secp256k1_ec_pubkey_serialize(
        ctx, uncompressed.data(), &out_len, &pk,
        SECP256K1_EC_UNCOMPRESSED);

    return rnet::Result<std::array<uint8_t, 65>>::ok(uncompressed);
#else
    (void)pubkey;
    return rnet::Result<std::array<uint8_t, 65>>::err(
        "secp256k1 library not available");
#endif
}

// ---------------------------------------------------------------------------
// secp256k1_compress_pubkey
// ---------------------------------------------------------------------------
//
// Design note — public key compression (SEC1):
//   Given uncompressed key 0x04 || x || y (65 bytes), produce compressed
//   form: (0x02 if y is even, 0x03 if y is odd) || x (33 bytes).

rnet::Result<Secp256k1PubKey> secp256k1_compress_pubkey(
    std::span<const uint8_t> uncompressed)
{
    // 1. Validate input length.
    if (uncompressed.size() != 65) {
        return rnet::Result<Secp256k1PubKey>::err(
            "Uncompressed pubkey must be 65 bytes");
    }

#if HAVE_SECP256K1
    // 2. Validate context.
    auto& ctx_inst = Secp256k1Context::instance();
    if (!ctx_inst.is_initialized()) {
        return rnet::Result<Secp256k1PubKey>::err(
            "secp256k1 not initialized");
    }
    auto* ctx = static_cast<secp256k1_context*>(ctx_inst.raw_ctx());

    // 3. Parse the 65-byte uncompressed key.
    secp256k1_pubkey pk;
    if (secp256k1_ec_pubkey_parse(
            ctx, &pk, uncompressed.data(), 65) != 1) {
        return rnet::Result<Secp256k1PubKey>::err(
            "Failed to parse uncompressed pubkey");
    }

    // 4. Re-serialise to 33-byte compressed form.
    Secp256k1PubKey result;
    size_t out_len = 33;
    secp256k1_ec_pubkey_serialize(
        ctx, result.data.data(), &out_len, &pk,
        SECP256K1_EC_COMPRESSED);
    result.valid = true;
    return rnet::Result<Secp256k1PubKey>::ok(result);
#else
    (void)uncompressed;
    return rnet::Result<Secp256k1PubKey>::err(
        "secp256k1 library not available");
#endif
}

} // namespace rnet::crypto

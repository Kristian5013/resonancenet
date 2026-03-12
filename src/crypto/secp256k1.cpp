#include "crypto/secp256k1.h"
#include "crypto/hash.h"

#include <cstring>
#include <mutex>

#include "core/hex.h"
#include "core/logging.h"
#include "core/random.h"

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

// -----------------------------------------------------------------------
// Secp256k1PubKey
// -----------------------------------------------------------------------

bool Secp256k1PubKey::is_zero() const {
    for (auto b : data) {
        if (b != 0) return false;
    }
    return true;
}

std::string Secp256k1PubKey::to_hex() const {
    return rnet::core::to_hex(
        std::span<const uint8_t>(data.data(), data.size()));
}

rnet::Result<Secp256k1PubKey> Secp256k1PubKey::from_hex(
    std::string_view hex)
{
    if (hex.size() != 66) {
        return rnet::Result<Secp256k1PubKey>::err(
            "secp256k1 compressed pubkey hex must be 66 characters");
    }
    auto bytes = rnet::core::from_hex(hex);
    if (bytes.size() != 33) {
        return rnet::Result<Secp256k1PubKey>::err(
            "Invalid hex for secp256k1 pubkey");
    }
    return from_bytes(std::span<const uint8_t>(bytes));
}

rnet::Result<Secp256k1PubKey> Secp256k1PubKey::from_bytes(
    std::span<const uint8_t> bytes)
{
    if (bytes.size() != 33 && bytes.size() != 65) {
        return rnet::Result<Secp256k1PubKey>::err(
            "secp256k1 pubkey must be 33 (compressed) or "
            "65 (uncompressed) bytes");
    }

    Secp256k1PubKey pk;
    if (bytes.size() == 33) {
        std::memcpy(pk.data.data(), bytes.data(), 33);
        pk.valid = secp256k1_is_valid_pubkey(bytes);
    } else {
        // Compress
        auto compressed = secp256k1_compress_pubkey(bytes);
        if (compressed.is_err()) {
            return rnet::Result<Secp256k1PubKey>::err(
                compressed.error());
        }
        pk = compressed.value();
    }
    return rnet::Result<Secp256k1PubKey>::ok(pk);
}

// -----------------------------------------------------------------------
// Secp256k1SecretKey
// -----------------------------------------------------------------------

void Secp256k1SecretKey::wipe() {
    secure_wipe(data.data(), data.size());
}

bool Secp256k1SecretKey::is_zero() const {
    for (auto b : data) {
        if (b != 0) return false;
    }
    return true;
}

// -----------------------------------------------------------------------
// Secp256k1Signature
// -----------------------------------------------------------------------

std::string Secp256k1Signature::to_hex() const {
    return rnet::core::to_hex(
        std::span<const uint8_t>(der_data));
}

std::array<uint8_t, 64> Secp256k1Signature::to_compact() const {
    std::array<uint8_t, 64> compact{};
#if HAVE_SECP256K1
    auto& ctx_inst = Secp256k1Context::instance();
    if (!ctx_inst.is_initialized()) return compact;

    auto* ctx = static_cast<secp256k1_context*>(ctx_inst.raw_ctx());
    secp256k1_ecdsa_signature sig;
    if (secp256k1_ecdsa_signature_parse_der(
            ctx, &sig, der_data.data(), der_data.size()) != 1) {
        return compact;
    }
    secp256k1_ecdsa_signature_serialize_compact(ctx, compact.data(), &sig);
#endif
    return compact;
}

rnet::Result<Secp256k1Signature> Secp256k1Signature::from_compact(
    std::span<const uint8_t> compact)
{
    if (compact.size() != 64) {
        return rnet::Result<Secp256k1Signature>::err(
            "Compact signature must be 64 bytes");
    }
#if HAVE_SECP256K1
    auto& ctx_inst = Secp256k1Context::instance();
    if (!ctx_inst.is_initialized()) {
        return rnet::Result<Secp256k1Signature>::err(
            "secp256k1 not initialized");
    }
    auto* ctx = static_cast<secp256k1_context*>(ctx_inst.raw_ctx());

    secp256k1_ecdsa_signature sig;
    if (secp256k1_ecdsa_signature_parse_compact(
            ctx, &sig, compact.data()) != 1) {
        return rnet::Result<Secp256k1Signature>::err(
            "Failed to parse compact signature");
    }

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

rnet::Result<Secp256k1Signature> Secp256k1Signature::from_der(
    std::span<const uint8_t> der)
{
    Secp256k1Signature sig;
    sig.der_data.assign(der.begin(), der.end());
    return rnet::Result<Secp256k1Signature>::ok(std::move(sig));
}

// -----------------------------------------------------------------------
// Secp256k1KeyPair
// -----------------------------------------------------------------------

void Secp256k1KeyPair::wipe() {
    secret.wipe();
}

// -----------------------------------------------------------------------
// Context implementation
// -----------------------------------------------------------------------

struct Secp256k1Context::Impl {
#if HAVE_SECP256K1
    secp256k1_context* ctx = nullptr;
#endif
    bool initialized = false;
    std::mutex mutex;
};

Secp256k1Context::Secp256k1Context()
    : impl_(std::make_unique<Impl>()) {}

Secp256k1Context::~Secp256k1Context() {
    shutdown();
}

Secp256k1Context& Secp256k1Context::instance() {
    static Secp256k1Context inst;
    return inst;
}

rnet::Result<void> Secp256k1Context::init() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->initialized) {
        return rnet::Result<void>::ok();
    }

#if HAVE_SECP256K1
    impl_->ctx = secp256k1_context_create(
        SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    if (!impl_->ctx) {
        return rnet::Result<void>::err(
            "Failed to create secp256k1 context");
    }

    // Randomize the context for side-channel protection
    uint8_t seed[32];
    rnet::core::get_rand_bytes(std::span<uint8_t>(seed, 32));
    secp256k1_context_randomize(impl_->ctx, seed);
    secure_wipe(seed, 32);

    impl_->initialized = true;
    return rnet::Result<void>::ok();
#else
    return rnet::Result<void>::err(
        "secp256k1 library not available at compile time");
#endif
}

void Secp256k1Context::shutdown() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
#if HAVE_SECP256K1
    if (impl_->ctx) {
        secp256k1_context_destroy(impl_->ctx);
        impl_->ctx = nullptr;
    }
#endif
    impl_->initialized = false;
}

bool Secp256k1Context::is_initialized() const {
    return impl_->initialized;
}

void* Secp256k1Context::raw_ctx() const {
#if HAVE_SECP256K1
    return impl_->ctx;
#else
    return nullptr;
#endif
}

// -----------------------------------------------------------------------
// Key generation
// -----------------------------------------------------------------------

rnet::Result<Secp256k1KeyPair> secp256k1_generate() {
#if HAVE_SECP256K1
    auto& ctx_inst = Secp256k1Context::instance();
    if (!ctx_inst.is_initialized()) {
        auto init_res = ctx_inst.init();
        if (init_res.is_err()) {
            return rnet::Result<Secp256k1KeyPair>::err(init_res.error());
        }
    }

    Secp256k1KeyPair kp;

    // Generate random secret key, retry if invalid
    for (int attempt = 0; attempt < 100; ++attempt) {
        rnet::core::get_rand_bytes(
            std::span<uint8_t>(kp.secret.data.data(), 32));
        if (secp256k1_ec_seckey_verify(
                static_cast<secp256k1_context*>(ctx_inst.raw_ctx()),
                kp.secret.data.data()) == 1) {
            // Valid secret key found
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

rnet::Result<Secp256k1PubKey> secp256k1_pubkey_from_secret(
    const Secp256k1SecretKey& secret)
{
#if HAVE_SECP256K1
    auto& ctx_inst = Secp256k1Context::instance();
    if (!ctx_inst.is_initialized()) {
        return rnet::Result<Secp256k1PubKey>::err(
            "secp256k1 not initialized");
    }
    auto* ctx = static_cast<secp256k1_context*>(ctx_inst.raw_ctx());

    secp256k1_pubkey pubkey;
    if (secp256k1_ec_pubkey_create(
            ctx, &pubkey, secret.data.data()) != 1) {
        return rnet::Result<Secp256k1PubKey>::err(
            "Failed to create public key from secret");
    }

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

rnet::Result<Secp256k1SecretKey> secp256k1_secret_tweak_add(
    const Secp256k1SecretKey& secret,
    std::span<const uint8_t> tweak)
{
#if HAVE_SECP256K1
    if (tweak.size() != 32) {
        return rnet::Result<Secp256k1SecretKey>::err(
            "Tweak must be 32 bytes");
    }

    auto& ctx_inst = Secp256k1Context::instance();
    if (!ctx_inst.is_initialized()) {
        return rnet::Result<Secp256k1SecretKey>::err(
            "secp256k1 not initialized");
    }
    auto* ctx = static_cast<secp256k1_context*>(ctx_inst.raw_ctx());

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

rnet::Result<Secp256k1PubKey> secp256k1_pubkey_tweak_add(
    const Secp256k1PubKey& pubkey,
    std::span<const uint8_t> tweak)
{
#if HAVE_SECP256K1
    if (tweak.size() != 32) {
        return rnet::Result<Secp256k1PubKey>::err(
            "Tweak must be 32 bytes");
    }

    auto& ctx_inst = Secp256k1Context::instance();
    if (!ctx_inst.is_initialized()) {
        return rnet::Result<Secp256k1PubKey>::err(
            "secp256k1 not initialized");
    }
    auto* ctx = static_cast<secp256k1_context*>(ctx_inst.raw_ctx());

    // Parse the compressed key
    secp256k1_pubkey pk;
    if (secp256k1_ec_pubkey_parse(
            ctx, &pk, pubkey.data.data(), 33) != 1) {
        return rnet::Result<Secp256k1PubKey>::err(
            "Failed to parse public key for tweak");
    }

    if (secp256k1_ec_pubkey_tweak_add(
            ctx, &pk, tweak.data()) != 1) {
        return rnet::Result<Secp256k1PubKey>::err(
            "Public key tweak add failed");
    }

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

// -----------------------------------------------------------------------
// ECDSA signing / verification
// -----------------------------------------------------------------------

rnet::Result<Secp256k1Signature> secp256k1_sign(
    const Secp256k1SecretKey& secret,
    const rnet::uint256& hash)
{
#if HAVE_SECP256K1
    auto& ctx_inst = Secp256k1Context::instance();
    if (!ctx_inst.is_initialized()) {
        return rnet::Result<Secp256k1Signature>::err(
            "secp256k1 not initialized");
    }
    auto* ctx = static_cast<secp256k1_context*>(ctx_inst.raw_ctx());

    secp256k1_ecdsa_signature sig;
    if (secp256k1_ecdsa_sign(ctx, &sig, hash.data(),
                             secret.data.data(), nullptr, nullptr) != 1) {
        return rnet::Result<Secp256k1Signature>::err(
            "ECDSA signing failed");
    }

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

bool secp256k1_verify(
    const Secp256k1PubKey& pubkey,
    const rnet::uint256& hash,
    const Secp256k1Signature& signature)
{
#if HAVE_SECP256K1
    auto& ctx_inst = Secp256k1Context::instance();
    if (!ctx_inst.is_initialized()) return false;
    auto* ctx = static_cast<secp256k1_context*>(ctx_inst.raw_ctx());

    secp256k1_pubkey pk;
    if (secp256k1_ec_pubkey_parse(
            ctx, &pk, pubkey.data.data(), 33) != 1) {
        return false;
    }

    secp256k1_ecdsa_signature sig;
    if (secp256k1_ecdsa_signature_parse_der(
            ctx, &sig,
            signature.der_data.data(),
            signature.der_data.size()) != 1) {
        return false;
    }

    // Normalize to low-S form
    secp256k1_ecdsa_signature_normalize(ctx, &sig, &sig);

    return secp256k1_ecdsa_verify(ctx, &sig, hash.data(), &pk) == 1;
#else
    (void)pubkey; (void)hash; (void)signature;
    return false;
#endif
}

// -----------------------------------------------------------------------
// ECDH
// -----------------------------------------------------------------------

rnet::Result<rnet::uint256> secp256k1_ecdh(
    const Secp256k1SecretKey& secret,
    const Secp256k1PubKey& pubkey)
{
#if HAVE_SECP256K1
    auto& ctx_inst = Secp256k1Context::instance();
    if (!ctx_inst.is_initialized()) {
        return rnet::Result<rnet::uint256>::err(
            "secp256k1 not initialized");
    }
    auto* ctx = static_cast<secp256k1_context*>(ctx_inst.raw_ctx());

    secp256k1_pubkey pk;
    if (secp256k1_ec_pubkey_parse(
            ctx, &pk, pubkey.data.data(), 33) != 1) {
        return rnet::Result<rnet::uint256>::err(
            "Failed to parse public key for ECDH");
    }

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

// -----------------------------------------------------------------------
// Key validation
// -----------------------------------------------------------------------

bool secp256k1_is_valid_secret(std::span<const uint8_t> bytes) {
    if (bytes.size() != 32) return false;
#if HAVE_SECP256K1
    auto& ctx_inst = Secp256k1Context::instance();
    if (!ctx_inst.is_initialized()) return false;
    auto* ctx = static_cast<secp256k1_context*>(ctx_inst.raw_ctx());
    return secp256k1_ec_seckey_verify(ctx, bytes.data()) == 1;
#else
    // Basic check: not zero
    bool all_zero = true;
    for (size_t i = 0; i < 32; ++i) {
        if (bytes[i] != 0) { all_zero = false; break; }
    }
    return !all_zero;
#endif
}

bool secp256k1_is_valid_pubkey(std::span<const uint8_t> bytes) {
    if (bytes.size() != 33 && bytes.size() != 65) return false;
#if HAVE_SECP256K1
    auto& ctx_inst = Secp256k1Context::instance();
    if (!ctx_inst.is_initialized()) return false;
    auto* ctx = static_cast<secp256k1_context*>(ctx_inst.raw_ctx());

    secp256k1_pubkey pk;
    return secp256k1_ec_pubkey_parse(
        ctx, &pk, bytes.data(), bytes.size()) == 1;
#else
    // Basic prefix check for compressed keys
    if (bytes.size() == 33) {
        return bytes[0] == 0x02 || bytes[0] == 0x03;
    }
    if (bytes.size() == 65) {
        return bytes[0] == 0x04;
    }
    return false;
#endif
}

rnet::Result<std::array<uint8_t, 65>> secp256k1_decompress_pubkey(
    const Secp256k1PubKey& pubkey)
{
#if HAVE_SECP256K1
    auto& ctx_inst = Secp256k1Context::instance();
    if (!ctx_inst.is_initialized()) {
        return rnet::Result<std::array<uint8_t, 65>>::err(
            "secp256k1 not initialized");
    }
    auto* ctx = static_cast<secp256k1_context*>(ctx_inst.raw_ctx());

    secp256k1_pubkey pk;
    if (secp256k1_ec_pubkey_parse(
            ctx, &pk, pubkey.data.data(), 33) != 1) {
        return rnet::Result<std::array<uint8_t, 65>>::err(
            "Failed to parse compressed pubkey");
    }

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

rnet::Result<Secp256k1PubKey> secp256k1_compress_pubkey(
    std::span<const uint8_t> uncompressed)
{
    if (uncompressed.size() != 65) {
        return rnet::Result<Secp256k1PubKey>::err(
            "Uncompressed pubkey must be 65 bytes");
    }
#if HAVE_SECP256K1
    auto& ctx_inst = Secp256k1Context::instance();
    if (!ctx_inst.is_initialized()) {
        return rnet::Result<Secp256k1PubKey>::err(
            "secp256k1 not initialized");
    }
    auto* ctx = static_cast<secp256k1_context*>(ctx_inst.raw_ctx());

    secp256k1_pubkey pk;
    if (secp256k1_ec_pubkey_parse(
            ctx, &pk, uncompressed.data(), 65) != 1) {
        return rnet::Result<Secp256k1PubKey>::err(
            "Failed to parse uncompressed pubkey");
    }

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

}  // namespace rnet::crypto

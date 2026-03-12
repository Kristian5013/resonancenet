#include "crypto/ed25519.h"
#include "crypto/hash.h"

#include <cstring>
#include <stdexcept>

#include <openssl/evp.h>
#include <openssl/err.h>

#include "core/hex.h"
#include "core/logging.h"
#include "core/random.h"

namespace rnet::crypto {

// -----------------------------------------------------------------------
// Ed25519SecretKey
// -----------------------------------------------------------------------

void Ed25519SecretKey::wipe() {
    secure_wipe(data.data(), data.size());
}

// -----------------------------------------------------------------------
// Ed25519PublicKey
// -----------------------------------------------------------------------

rnet::uint256 Ed25519PublicKey::to_uint256() const {
    rnet::uint256 result;
    std::memcpy(result.data(), data.data(), 32);
    return result;
}

Ed25519PublicKey Ed25519PublicKey::from_uint256(const rnet::uint256& val) {
    Ed25519PublicKey pk;
    std::memcpy(pk.data.data(), val.data(), 32);
    return pk;
}

Ed25519PublicKey Ed25519PublicKey::from_bytes(
    std::span<const uint8_t> bytes)
{
    Ed25519PublicKey pk;
    size_t copy_len = (bytes.size() < 32) ? bytes.size() : 32;
    std::memcpy(pk.data.data(), bytes.data(), copy_len);
    return pk;
}

bool Ed25519PublicKey::is_zero() const {
    for (auto b : data) {
        if (b != 0) return false;
    }
    return true;
}

std::string Ed25519PublicKey::to_hex() const {
    return rnet::core::to_hex(
        std::span<const uint8_t>(data.data(), data.size()));
}

rnet::Result<Ed25519PublicKey> Ed25519PublicKey::from_hex(
    std::string_view hex)
{
    if (hex.size() != 64) {
        return rnet::Result<Ed25519PublicKey>::err(
            "Ed25519 public key hex must be 64 characters");
    }
    auto bytes = rnet::core::from_hex(hex);
    if (bytes.size() != 32) {
        return rnet::Result<Ed25519PublicKey>::err(
            "Invalid hex for Ed25519 public key");
    }
    Ed25519PublicKey pk;
    std::memcpy(pk.data.data(), bytes.data(), 32);
    return rnet::Result<Ed25519PublicKey>::ok(pk);
}

// -----------------------------------------------------------------------
// Ed25519Signature
// -----------------------------------------------------------------------

bool Ed25519Signature::is_zero() const {
    for (auto b : data) {
        if (b != 0) return false;
    }
    return true;
}

std::string Ed25519Signature::to_hex() const {
    return rnet::core::to_hex(
        std::span<const uint8_t>(data.data(), data.size()));
}

rnet::Result<Ed25519Signature> Ed25519Signature::from_hex(
    std::string_view hex)
{
    if (hex.size() != 128) {
        return rnet::Result<Ed25519Signature>::err(
            "Ed25519 signature hex must be 128 characters");
    }
    auto bytes = rnet::core::from_hex(hex);
    if (bytes.size() != 64) {
        return rnet::Result<Ed25519Signature>::err(
            "Invalid hex for Ed25519 signature");
    }
    Ed25519Signature sig;
    std::memcpy(sig.data.data(), bytes.data(), 64);
    return rnet::Result<Ed25519Signature>::ok(sig);
}

// -----------------------------------------------------------------------
// Ed25519KeyPair
// -----------------------------------------------------------------------

void Ed25519KeyPair::wipe() {
    secret.wipe();
}

// -----------------------------------------------------------------------
// Helper: RAII wrapper for EVP_PKEY
// -----------------------------------------------------------------------

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

// -----------------------------------------------------------------------
// Create EVP_PKEY from raw seed (private key)
// -----------------------------------------------------------------------

static EvpPkeyPtr make_ed25519_private_key(
    std::span<const uint8_t> seed)
{
    EVP_PKEY* pkey = EVP_PKEY_new_raw_private_key(
        EVP_PKEY_ED25519, nullptr,
        seed.data(), seed.size());
    return EvpPkeyPtr(pkey);
}

// -----------------------------------------------------------------------
// Create EVP_PKEY from raw public key
// -----------------------------------------------------------------------

static EvpPkeyPtr make_ed25519_public_key(
    std::span<const uint8_t> pubkey_bytes)
{
    EVP_PKEY* pkey = EVP_PKEY_new_raw_public_key(
        EVP_PKEY_ED25519, nullptr,
        pubkey_bytes.data(), pubkey_bytes.size());
    return EvpPkeyPtr(pkey);
}

// -----------------------------------------------------------------------
// Key generation
// -----------------------------------------------------------------------

rnet::Result<Ed25519KeyPair> ed25519_generate() {
    // Generate 32 bytes of random seed
    uint8_t seed[32];
    rnet::core::get_rand_bytes(std::span<uint8_t>(seed, 32));

    auto result = ed25519_from_seed(
        std::span<const uint8_t>(seed, 32));

    // Wipe seed from stack
    secure_wipe(seed, 32);

    return result;
}

rnet::Result<Ed25519KeyPair> ed25519_from_seed(
    std::span<const uint8_t> seed)
{
    if (seed.size() != 32) {
        return rnet::Result<Ed25519KeyPair>::err(
            "Ed25519 seed must be exactly 32 bytes");
    }

    // Create private key from seed
    auto pkey = make_ed25519_private_key(seed);
    if (!pkey) {
        return rnet::Result<Ed25519KeyPair>::err(
            "Failed to create Ed25519 private key from seed");
    }

    // Extract the public key bytes
    size_t pub_len = 32;
    uint8_t pub_bytes[32];
    if (EVP_PKEY_get_raw_public_key(pkey.get(), pub_bytes, &pub_len)
        != 1 || pub_len != 32) {
        return rnet::Result<Ed25519KeyPair>::err(
            "Failed to extract Ed25519 public key");
    }

    // Build the keypair
    Ed25519KeyPair kp;

    // Secret = seed || public_key
    std::memcpy(kp.secret.data.data(), seed.data(), 32);
    std::memcpy(kp.secret.data.data() + 32, pub_bytes, 32);

    // Public key
    std::memcpy(kp.public_key.data.data(), pub_bytes, 32);

    return rnet::Result<Ed25519KeyPair>::ok(std::move(kp));
}

// -----------------------------------------------------------------------
// Signing
// -----------------------------------------------------------------------

rnet::Result<Ed25519Signature> ed25519_sign(
    const Ed25519SecretKey& secret,
    std::span<const uint8_t> message)
{
    // Extract the 32-byte seed
    auto pkey = make_ed25519_private_key(secret.seed());
    if (!pkey) {
        return rnet::Result<Ed25519Signature>::err(
            "Failed to create Ed25519 signing key");
    }

    EvpMdCtxPtr ctx(EVP_MD_CTX_new());
    if (!ctx) {
        return rnet::Result<Ed25519Signature>::err(
            "Failed to create EVP_MD_CTX for Ed25519 signing");
    }

    if (EVP_DigestSignInit(ctx.get(), nullptr, nullptr, nullptr,
                           pkey.get()) != 1) {
        return rnet::Result<Ed25519Signature>::err(
            "EVP_DigestSignInit failed for Ed25519");
    }

    Ed25519Signature sig;
    size_t sig_len = 64;

    if (EVP_DigestSign(ctx.get(), sig.data.data(), &sig_len,
                       message.data(), message.size()) != 1) {
        return rnet::Result<Ed25519Signature>::err(
            "EVP_DigestSign failed for Ed25519");
    }

    if (sig_len != 64) {
        return rnet::Result<Ed25519Signature>::err(
            "Unexpected Ed25519 signature length");
    }

    return rnet::Result<Ed25519Signature>::ok(sig);
}

rnet::Result<Ed25519Signature> ed25519_sign(
    const Ed25519SecretKey& secret,
    std::string_view message)
{
    return ed25519_sign(secret,
        std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(message.data()),
            message.size()));
}

// -----------------------------------------------------------------------
// Verification
// -----------------------------------------------------------------------

bool ed25519_verify(
    const Ed25519PublicKey& pubkey,
    std::span<const uint8_t> message,
    const Ed25519Signature& signature)
{
    auto pkey = make_ed25519_public_key(
        std::span<const uint8_t>(pubkey.data.data(), 32));
    if (!pkey) {
        return false;
    }

    EvpMdCtxPtr ctx(EVP_MD_CTX_new());
    if (!ctx) {
        return false;
    }

    if (EVP_DigestVerifyInit(ctx.get(), nullptr, nullptr, nullptr,
                             pkey.get()) != 1) {
        return false;
    }

    int rc = EVP_DigestVerify(
        ctx.get(),
        signature.data.data(), signature.data.size(),
        message.data(), message.size());

    return rc == 1;
}

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

// -----------------------------------------------------------------------
// Batch verification
// -----------------------------------------------------------------------

bool ed25519_batch_verify(
    const std::vector<Ed25519PublicKey>& pubkeys,
    const std::vector<std::span<const uint8_t>>& messages,
    const std::vector<Ed25519Signature>& signatures)
{
    if (pubkeys.size() != messages.size() ||
        pubkeys.size() != signatures.size()) {
        return false;
    }

    if (pubkeys.empty()) {
        return true;
    }

    // OpenSSL doesn't expose batch verification for Ed25519 yet.
    // Fall back to sequential verification.
    // This is still correct; batch would just be faster.
    for (size_t i = 0; i < pubkeys.size(); ++i) {
        if (!ed25519_verify(pubkeys[i], messages[i], signatures[i])) {
            return false;
        }
    }

    return true;
}

// -----------------------------------------------------------------------
// Utility
// -----------------------------------------------------------------------

bool ed25519_is_valid_pubkey(std::span<const uint8_t> bytes) {
    if (bytes.size() != 32) {
        return false;
    }

    // Try to construct an EVP_PKEY from the bytes. If it succeeds,
    // the point is on the curve.
    auto pkey = make_ed25519_public_key(bytes);
    return pkey != nullptr;
}

std::vector<uint8_t> ed25519_coinbase_script(
    const Ed25519PublicKey& pubkey)
{
    // Coinbase script format: [0x20][32-byte pubkey][0xAC]
    // 0x20 = OP_PUSH32 (push 32 bytes)
    // 0xAC = OP_CHECKSIG
    std::vector<uint8_t> script;
    script.reserve(34);
    script.push_back(0x20);  // Push 32 bytes
    script.insert(script.end(), pubkey.data.begin(), pubkey.data.end());
    script.push_back(0xAC);  // OP_CHECKSIG
    return script;
}

rnet::Result<Ed25519PublicKey> ed25519_parse_coinbase_script(
    std::span<const uint8_t> script)
{
    if (script.size() != 34) {
        return rnet::Result<Ed25519PublicKey>::err(
            "Coinbase script must be exactly 34 bytes");
    }
    if (script[0] != 0x20) {
        return rnet::Result<Ed25519PublicKey>::err(
            "Coinbase script must start with 0x20 (OP_PUSH32)");
    }
    if (script[33] != 0xAC) {
        return rnet::Result<Ed25519PublicKey>::err(
            "Coinbase script must end with 0xAC (OP_CHECKSIG)");
    }

    Ed25519PublicKey pk;
    std::memcpy(pk.data.data(), script.data() + 1, 32);

    if (!ed25519_is_valid_pubkey(
            std::span<const uint8_t>(pk.data.data(), 32))) {
        return rnet::Result<Ed25519PublicKey>::err(
            "Invalid Ed25519 public key in coinbase script");
    }

    return rnet::Result<Ed25519PublicKey>::ok(pk);
}

}  // namespace rnet::crypto

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"
#include "core/error.h"
#include "lightning/channel_state.h"
#include "crypto/ed25519.h"

namespace rnet::lightning {

// ── Invoice HRP prefixes ────────────────────────────────────────────

inline constexpr const char* INVOICE_HRP_MAINNET = "rnt";
inline constexpr const char* INVOICE_HRP_TESTNET = "trnt";
inline constexpr const char* INVOICE_HRP_REGTEST = "rrnt";

// ── Invoice field tags (BOLT11-style) ───────────────────────────────

enum class InvoiceTag : uint8_t {
    PAYMENT_HASH    = 1,   // p: 256-bit payment hash
    DESCRIPTION     = 13,  // d: short description
    PAYEE_PUBKEY    = 19,  // n: payee node public key
    DESC_HASH       = 23,  // h: hash of long description
    EXPIRY          = 6,   // x: expiry time in seconds
    MIN_CLTV_EXPIRY = 24,  // c: min CLTV expiry delta
    FALLBACK_ADDR   = 9,   // f: fallback on-chain address
    ROUTE_HINT      = 3,   // r: routing information
    FEATURE_BITS    = 5,   // 9: feature flags
};

// ── Route hint ──────────────────────────────────────────────────────

struct RouteHint {
    crypto::Ed25519PublicKey  node_id;
    uint64_t                  short_channel_id = 0;
    int64_t                   fee_base = 0;
    int64_t                   fee_rate_ppm = 0;
    uint32_t                  cltv_expiry_delta = 0;
};

// ── Lightning Invoice ───────────────────────────────────────────────

/// BOLT11-style invoice with "rnt"/"trnt" HRP
class Invoice {
public:
    Invoice() = default;

    // ── Builder pattern ─────────────────────────────────────────────

    Invoice& set_payment_hash(const uint256& hash);
    Invoice& set_amount(int64_t amount_resonances);
    Invoice& set_description(std::string desc);
    Invoice& set_description_hash(const uint256& hash);
    Invoice& set_payee(const crypto::Ed25519PublicKey& pubkey);
    Invoice& set_expiry(uint32_t seconds);
    Invoice& set_min_cltv_expiry(uint32_t delta);
    Invoice& set_timestamp(uint64_t unix_time);
    Invoice& set_testnet(bool is_testnet);
    Invoice& set_regtest(bool is_regtest);
    Invoice& add_route_hint(RouteHint hint);
    Invoice& set_preimage(const uint256& preimage);

    // ── Encoding ────────────────────────────────────────────────────

    /// Encode to BOLT11 string and sign with the payee's private key
    Result<std::string> encode(const crypto::Ed25519SecretKey& signing_key) const;

    // ── Decoding ────────────────────────────────────────────────────

    /// Decode a BOLT11 invoice string
    static Result<Invoice> decode(std::string_view invoice_str);

    // ── Accessors ───────────────────────────────────────────────────

    const uint256& payment_hash() const { return payment_hash_; }
    std::optional<int64_t> amount() const { return amount_; }
    const std::string& description() const { return description_; }
    const std::optional<uint256>& description_hash() const { return description_hash_; }
    const crypto::Ed25519PublicKey& payee() const { return payee_; }
    uint32_t expiry() const { return expiry_; }
    uint32_t min_cltv_expiry() const { return min_cltv_expiry_; }
    uint64_t timestamp() const { return timestamp_; }
    bool is_testnet() const { return is_testnet_; }
    bool is_regtest() const { return is_regtest_; }
    const std::vector<RouteHint>& route_hints() const { return route_hints_; }
    const crypto::Ed25519Signature& signature() const { return signature_; }
    bool has_preimage() const { return has_preimage_; }
    const uint256& preimage() const { return preimage_; }

    /// Check if the invoice has expired
    bool is_expired(uint64_t current_time) const;

    /// Verify the invoice signature
    bool verify_signature() const;

    /// Get the human-readable prefix
    std::string_view hrp() const;

private:
    uint256                         payment_hash_;
    std::optional<int64_t>          amount_;
    std::string                     description_;
    std::optional<uint256>          description_hash_;
    crypto::Ed25519PublicKey        payee_;
    uint32_t                        expiry_ = 3600;        // 1 hour default
    uint32_t                        min_cltv_expiry_ = DEFAULT_CLTV_EXPIRY_DELTA;
    uint64_t                        timestamp_ = 0;
    bool                            is_testnet_ = false;
    bool                            is_regtest_ = false;
    std::vector<RouteHint>          route_hints_;
    crypto::Ed25519Signature        signature_;
    bool                            has_preimage_ = false;
    uint256                         preimage_;
};

}  // namespace rnet::lightning

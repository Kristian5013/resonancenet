#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"
#include "core/serialize.h"
#include "crypto/ed25519.h"
#include "primitives/amount.h"
#include "primitives/outpoint.h"

namespace rnet::lightning {

// ── Lightning constants ─────────────────────────────────────────────

/// Default ports
inline constexpr uint16_t LIGHTNING_PORT_MAINNET  = 9556;
inline constexpr uint16_t LIGHTNING_PORT_TESTNET  = 19556;
inline constexpr uint16_t LIGHTNING_PORT_REGTEST  = 29556;

/// Channel size limits
inline constexpr int64_t MIN_CHANNEL_CAPACITY     = 100'000;          // 0.001 RNT
inline constexpr int64_t MAX_CHANNEL_CAPACITY      = 1000 * primitives::COIN;

/// HTLC limits
inline constexpr uint32_t MAX_HTLCS_PER_CHANNEL   = 483;
inline constexpr int64_t  MIN_HTLC_VALUE           = 1;               // 1 resonance
inline constexpr int64_t  MAX_HTLC_VALUE           = MAX_CHANNEL_CAPACITY;

/// Timelock
inline constexpr uint32_t DEFAULT_CSV_DELAY        = 144;             // ~1 day
inline constexpr uint32_t MIN_CSV_DELAY            = 6;
inline constexpr uint32_t MAX_CSV_DELAY            = 2016;

/// Routing
inline constexpr uint32_t MAX_ROUTE_HOPS           = 20;
inline constexpr uint32_t DEFAULT_CLTV_EXPIRY_DELTA = 40;

/// Dust limit for lightning outputs
inline constexpr int64_t  LIGHTNING_DUST_LIMIT      = 546;

/// Fee rate defaults (resonances per 1000 weight units)
inline constexpr int64_t  DEFAULT_FEE_BASE_MSAT     = 1000;
inline constexpr int64_t  DEFAULT_FEE_RATE_PPM      = 1;              // 1 ppm

// ── Channel state machine ───────────────────────────────────────────

enum class ChannelState : uint8_t {
    PREOPENING       = 0,
    FUNDING_CREATED  = 1,
    FUNDING_BROADCAST = 2,
    FUNDING_LOCKED   = 3,
    NORMAL           = 4,
    SHUTDOWN         = 5,
    FORCE_CLOSING    = 6,
    CLOSED           = 7,
};

std::string_view channel_state_name(ChannelState state);

// ── Channel ID types ────────────────────────────────────────────────

/// Temporary channel ID: random 32 bytes before funding tx is known
using TempChannelId = uint256;

/// Permanent channel ID: XOR(funding_txid, funding_output_index)
using ChannelId = uint256;

/// Derive permanent channel ID from funding outpoint
ChannelId make_channel_id(const primitives::COutPoint& funding_outpoint);

// ── Channel config ──────────────────────────────────────────────────

struct ChannelConfig {
    int64_t  dust_limit           = LIGHTNING_DUST_LIMIT;
    int64_t  max_htlc_value       = MAX_HTLC_VALUE;
    int64_t  channel_reserve      = 0;         // Set to 1% of capacity
    uint32_t csv_delay            = DEFAULT_CSV_DELAY;
    uint32_t max_accepted_htlcs   = MAX_HTLCS_PER_CHANNEL;
    int64_t  min_htlc_value       = MIN_HTLC_VALUE;
    int64_t  fee_base             = DEFAULT_FEE_BASE_MSAT;
    int64_t  fee_rate_ppm         = DEFAULT_FEE_RATE_PPM;

    SERIALIZE_METHODS(
        READWRITE(self.dust_limit);
        READWRITE(self.max_htlc_value);
        READWRITE(self.channel_reserve);
        READWRITE(self.csv_delay);
        READWRITE(self.max_accepted_htlcs);
        READWRITE(self.min_htlc_value);
        READWRITE(self.fee_base);
        READWRITE(self.fee_rate_ppm);
    )
};

// ── Channel keys ────────────────────────────────────────────────────

/// Per-commitment keys for a channel side
struct ChannelKeys {
    crypto::Ed25519PublicKey funding_pubkey;
    crypto::Ed25519PublicKey revocation_basepoint;
    crypto::Ed25519PublicKey payment_basepoint;
    crypto::Ed25519PublicKey delayed_payment_basepoint;
    crypto::Ed25519PublicKey htlc_basepoint;

    template<typename Stream>
    void serialize(Stream& s) const {
        s.write(funding_pubkey.data.data(), 32);
        s.write(revocation_basepoint.data.data(), 32);
        s.write(payment_basepoint.data.data(), 32);
        s.write(delayed_payment_basepoint.data.data(), 32);
        s.write(htlc_basepoint.data.data(), 32);
    }

    template<typename Stream>
    void unserialize(Stream& s) {
        s.read(funding_pubkey.data.data(), 32);
        s.read(revocation_basepoint.data.data(), 32);
        s.read(payment_basepoint.data.data(), 32);
        s.read(delayed_payment_basepoint.data.data(), 32);
        s.read(htlc_basepoint.data.data(), 32);
    }
};

// ── Commitment number ───────────────────────────────────────────────

/// Obscured commitment number for privacy on-chain
struct CommitmentNumber {
    uint64_t number = 0;

    /// XOR with obscuring factor for on-chain encoding
    uint64_t obscured(const uint256& obscure_factor) const;

    SERIALIZE_METHODS(
        READWRITE(self.number);
    )
};

// ── Channel balance ─────────────────────────────────────────────────

struct ChannelBalance {
    int64_t local  = 0;    // Our balance in resonances
    int64_t remote = 0;    // Their balance in resonances

    int64_t total() const { return local + remote; }

    SERIALIZE_METHODS(
        READWRITE(self.local);
        READWRITE(self.remote);
    )
};

// ── Per-side channel state ──────────────────────────────────────────

struct ChannelSideState {
    ChannelKeys         keys;
    ChannelConfig       config;
    uint64_t            next_commitment_number = 0;
    uint256             last_per_commitment_secret;
    crypto::Ed25519PublicKey current_per_commitment_point;

    template<typename Stream>
    void serialize(Stream& s) const {
        keys.serialize(s);
        config.serialize(s);
        core::Serialize(s, next_commitment_number);
        core::Serialize(s, last_per_commitment_secret);
        s.write(current_per_commitment_point.data.data(), 32);
    }

    template<typename Stream>
    void unserialize(Stream& s) {
        keys.unserialize(s);
        config.unserialize(s);
        core::Unserialize(s, next_commitment_number);
        core::Unserialize(s, last_per_commitment_secret);
        s.read(current_per_commitment_point.data.data(), 32);
    }
};

}  // namespace rnet::lightning

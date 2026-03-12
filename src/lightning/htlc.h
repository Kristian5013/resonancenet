#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/types.h"
#include "core/error.h"
#include "core/serialize.h"
#include "primitives/amount.h"

namespace rnet::lightning {

// ── HTLC direction ──────────────────────────────────────────────────

enum class HtlcDirection : uint8_t {
    OFFERED  = 0,   // We offered this HTLC (outgoing)
    RECEIVED = 1,   // We received this HTLC (incoming)
};

// ── HTLC state ──────────────────────────────────────────────────────

enum class HtlcState : uint8_t {
    PENDING          = 0,
    COMMITTED_LOCAL  = 1,   // In our latest commitment
    COMMITTED_REMOTE = 2,   // In their latest commitment
    COMMITTED_BOTH   = 3,   // In both commitments
    FULFILLED        = 4,
    FAILED           = 5,
    TIMED_OUT        = 6,
};

std::string_view htlc_state_name(HtlcState state);

// ── HTLC ────────────────────────────────────────────────────────────

/// Hash Time-Locked Contract using Keccak256d payment hashes.
struct Htlc {
    uint64_t       id = 0;               // Per-channel sequential ID
    HtlcDirection  direction = HtlcDirection::OFFERED;
    HtlcState      state = HtlcState::PENDING;
    int64_t        amount = 0;           // Value in resonances
    uint256        payment_hash;         // Keccak256d(preimage)
    uint32_t       cltv_expiry = 0;      // Absolute block height timeout
    uint256        onion_routing_packet; // Truncated for storage

    /// Check if this HTLC is dust (below dust limit)
    bool is_dust(int64_t dust_limit) const { return amount < dust_limit; }

    /// Check if this HTLC has timed out at the given block height
    bool is_expired(uint32_t current_height) const {
        return current_height >= cltv_expiry;
    }

    template<typename Stream>
    void serialize(Stream& s) const {
        core::Serialize(s, id);
        core::ser_write_u8(s, static_cast<uint8_t>(direction));
        core::ser_write_u8(s, static_cast<uint8_t>(state));
        core::Serialize(s, amount);
        core::Serialize(s, payment_hash);
        core::Serialize(s, cltv_expiry);
        core::Serialize(s, onion_routing_packet);
    }

    template<typename Stream>
    void unserialize(Stream& s) {
        core::Unserialize(s, id);
        direction = static_cast<HtlcDirection>(core::ser_read_u8(s));
        state = static_cast<HtlcState>(core::ser_read_u8(s));
        core::Unserialize(s, amount);
        core::Unserialize(s, payment_hash);
        core::Unserialize(s, cltv_expiry);
        core::Unserialize(s, onion_routing_packet);
    }
};

// ── HTLC operations ─────────────────────────────────────────────────

/// Validate an HTLC before adding to a channel
Result<void> validate_htlc(const Htlc& htlc,
                            int64_t channel_capacity,
                            int64_t current_balance,
                            uint32_t current_htlc_count,
                            uint32_t max_htlcs,
                            int64_t min_htlc_value);

/// Generate a payment preimage (random 32 bytes)
uint256 generate_preimage();

/// Compute payment hash from preimage: Keccak256d(preimage)
uint256 compute_payment_hash(const uint256& preimage);

/// Verify that a preimage matches a payment hash
bool verify_preimage(const uint256& preimage, const uint256& payment_hash);

/// Compute the fee for forwarding an HTLC
/// fee = base_fee + (amount * proportional_fee / 1,000,000)
int64_t compute_htlc_fee(int64_t amount, int64_t base_fee, int64_t fee_rate_ppm);

// ── HTLC set ────────────────────────────────────────────────────────

/// Collection of HTLCs for a channel commitment
class HtlcSet {
public:
    /// Add an HTLC to the set
    Result<void> add(Htlc htlc);

    /// Fulfill an HTLC by ID with the preimage
    Result<void> fulfill(uint64_t htlc_id, const uint256& preimage);

    /// Fail an HTLC by ID
    Result<void> fail(uint64_t htlc_id);

    /// Mark timed-out HTLCs at the given block height
    uint32_t mark_expired(uint32_t current_height);

    /// Get an HTLC by ID
    const Htlc* find(uint64_t htlc_id) const;
    Htlc* find(uint64_t htlc_id);

    /// Get all pending HTLCs
    std::vector<const Htlc*> pending() const;

    /// Get all offered pending HTLCs
    std::vector<const Htlc*> offered_pending() const;

    /// Get all received pending HTLCs
    std::vector<const Htlc*> received_pending() const;

    /// Count of pending HTLCs
    uint32_t pending_count() const;

    /// Total value of pending offered HTLCs
    int64_t offered_pending_value() const;

    /// Total value of pending received HTLCs
    int64_t received_pending_value() const;

    /// Remove settled (fulfilled/failed/timed_out) HTLCs
    void prune_settled();

    /// Get all HTLCs
    const std::vector<Htlc>& all() const { return htlcs_; }

    /// Next available HTLC ID
    uint64_t next_id() const { return next_id_; }

private:
    std::vector<Htlc> htlcs_;
    uint64_t next_id_ = 0;
};

}  // namespace rnet::lightning

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/types.h"
#include "core/error.h"
#include "core/sync.h"
#include "crypto/ed25519.h"
#include "primitives/transaction.h"
#include "primitives/outpoint.h"
#include "lightning/channel_state.h"
#include "lightning/htlc.h"

namespace rnet::lightning {

// ── Commitment transaction ──────────────────────────────────────────

struct CommitmentTx {
    primitives::CMutableTransaction tx;
    uint64_t                        commitment_number = 0;
    int64_t                         fee = 0;
    ChannelBalance                  balance;
    std::vector<Htlc>               htlcs;

    SERIALIZE_METHODS(
        READWRITE(self.commitment_number);
        READWRITE(self.fee);
        READWRITE(self.balance);
    )
};

// ── Channel close types ─────────────────────────────────────────────

enum class CloseType : uint8_t {
    COOPERATIVE = 0,   // Mutual close
    FORCE_LOCAL = 1,   // We force-closed
    FORCE_REMOTE = 2,  // They force-closed
    BREACH = 3,        // They published a revoked commitment
};

std::string_view close_type_name(CloseType type);

// ── Lightning Channel ───────────────────────────────────────────────

/// Full lifecycle channel state machine.
/// Thread-safe via internal mutex.
class LightningChannel {
public:
    /// Create a new outbound channel (we are the funder)
    static Result<LightningChannel> create_outbound(
        const crypto::Ed25519KeyPair& local_keys,
        const crypto::Ed25519PublicKey& remote_node_id,
        int64_t capacity,
        int64_t push_amount,
        const ChannelConfig& config);

    /// Create a new inbound channel (they are the funder)
    static Result<LightningChannel> create_inbound(
        const crypto::Ed25519KeyPair& local_keys,
        const crypto::Ed25519PublicKey& remote_node_id,
        int64_t capacity,
        int64_t push_amount,
        const ChannelConfig& local_config,
        const ChannelConfig& remote_config);

    LightningChannel() = default;
    ~LightningChannel() = default;

    // Move-only (contains mutex)
    LightningChannel(LightningChannel&& other) noexcept;
    LightningChannel& operator=(LightningChannel&& other) noexcept;
    LightningChannel(const LightningChannel&) = delete;
    LightningChannel& operator=(const LightningChannel&) = delete;

    // ── State queries ───────────────────────────────────────────────

    ChannelState state() const;
    ChannelId channel_id() const;
    TempChannelId temp_channel_id() const;
    int64_t capacity() const;
    ChannelBalance balance() const;
    bool is_funder() const;
    const crypto::Ed25519PublicKey& local_node_id() const;
    const crypto::Ed25519PublicKey& remote_node_id() const;
    uint64_t local_commitment_number() const;
    uint64_t remote_commitment_number() const;

    // ── State transitions ───────────────────────────────────────────

    /// PREOPENING -> FUNDING_CREATED: funding tx created
    Result<void> funding_created(const primitives::COutPoint& funding_outpoint,
                                  const crypto::Ed25519Signature& remote_sig);

    /// FUNDING_CREATED -> FUNDING_BROADCAST: funding tx broadcast
    Result<void> funding_broadcast(const uint256& funding_txid);

    /// FUNDING_BROADCAST -> FUNDING_LOCKED: funding tx confirmed
    Result<void> funding_locked(uint32_t confirmation_height);

    /// NORMAL -> SHUTDOWN: initiate cooperative close
    Result<void> initiate_shutdown();

    /// NORMAL/SHUTDOWN -> FORCE_CLOSING: force-close with latest commitment
    Result<CommitmentTx> force_close();

    /// any -> CLOSED: channel is fully closed
    Result<void> mark_closed(CloseType type, const uint256& closing_txid);

    // ── HTLC operations (NORMAL state only) ─────────────────────────

    /// Add an outgoing HTLC (we are sending)
    Result<uint64_t> add_htlc(int64_t amount,
                               const uint256& payment_hash,
                               uint32_t cltv_expiry);

    /// Fulfill an incoming HTLC with preimage
    Result<void> fulfill_htlc(uint64_t htlc_id, const uint256& preimage);

    /// Fail an incoming HTLC
    Result<void> fail_htlc(uint64_t htlc_id);

    /// Receive an HTLC from the remote side
    Result<uint64_t> receive_htlc(int64_t amount,
                                   const uint256& payment_hash,
                                   uint32_t cltv_expiry);

    // ── Commitment signing ──────────────────────────────────────────

    /// Sign our latest local commitment transaction
    Result<crypto::Ed25519Signature> sign_local_commitment();

    /// Receive and validate a remote commitment signature
    Result<void> receive_commitment_sig(const crypto::Ed25519Signature& sig);

    /// Revoke the previous local commitment (send revocation secret)
    Result<uint256> revoke_and_ack();

    /// Receive a revocation from the remote side
    Result<void> receive_revocation(const uint256& per_commitment_secret,
                                     const crypto::Ed25519PublicKey& next_per_commitment_point);

    // ── Queries ─────────────────────────────────────────────────────

    /// Get the current HTLC set
    const HtlcSet& htlc_set() const;

    /// Get the local channel config
    const ChannelConfig& local_config() const;

    /// Get the remote channel config
    const ChannelConfig& remote_config() const;

    /// Get the funding outpoint
    const primitives::COutPoint& funding_outpoint() const;

    /// Get number of pending HTLCs
    uint32_t pending_htlc_count() const;

    /// Human-readable summary
    std::string to_string() const;

private:
    /// Validate state transition
    Result<void> check_state(ChannelState expected) const;
    Result<void> check_state_any(std::initializer_list<ChannelState> expected) const;

    /// Update balances after HTLC settlement
    void apply_htlc_settlement();

    mutable core::Mutex       mutex_;
    ChannelState              state_ = ChannelState::PREOPENING;
    bool                      is_funder_ = false;

    // Channel identifiers
    TempChannelId             temp_channel_id_;
    ChannelId                 channel_id_;
    primitives::COutPoint     funding_outpoint_;

    // Capacity and balance
    int64_t                   capacity_ = 0;
    ChannelBalance            balance_;

    // Local and remote state
    ChannelSideState          local_state_;
    ChannelSideState          remote_state_;

    // HTLC tracking
    HtlcSet                   htlcs_;

    // Keys
    crypto::Ed25519KeyPair    local_keypair_;
    crypto::Ed25519PublicKey  remote_node_id_;

    // Close tracking
    CloseType                 close_type_ = CloseType::COOPERATIVE;
    uint256                   closing_txid_;
    uint32_t                  confirmation_height_ = 0;
};

}  // namespace rnet::lightning

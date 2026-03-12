#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/types.h"
#include "core/error.h"
#include "core/sync.h"
#include "crypto/ed25519.h"
#include "primitives/transaction.h"
#include "lightning/channel_state.h"

namespace rnet::lightning {

// ── Breach hint ─────────────────────────────────────────────────────

/// Compact breach remedy hint: first 16 bytes of txid XOR'd with
/// encrypted justice transaction data.
struct BreachHint {
    uint256             hint;           // First 16 bytes of commitment txid hash
    std::vector<uint8_t> encrypted_blob; // Encrypted justice transaction

    SERIALIZE_METHODS(
        READWRITE(self.hint);
        READWRITE(self.encrypted_blob);
    )
};

// ── Breach remedy ───────────────────────────────────────────────────

/// Information needed to construct a justice transaction
struct BreachRemedy {
    ChannelId                    channel_id;
    uint64_t                     commitment_number = 0;
    uint256                      revocation_secret;
    primitives::CMutableTransaction justice_tx;

    SERIALIZE_METHODS(
        READWRITE(self.channel_id);
        READWRITE(self.commitment_number);
        READWRITE(self.revocation_secret);
    )
};

// ── Watchtower state ────────────────────────────────────────────────

/// Per-channel state tracked by the watchtower
struct WatchedChannel {
    ChannelId                    channel_id;
    crypto::Ed25519PublicKey     our_pubkey;
    crypto::Ed25519PublicKey     their_pubkey;
    uint64_t                     latest_commitment = 0;

    /// Map from hint to breach remedy
    std::unordered_map<uint256, BreachRemedy> remedies;
};

// ── Watchtower callback ─────────────────────────────────────────────

/// Called when a breach is detected with the justice transaction
using BreachCallback = std::function<void(
    const ChannelId& channel_id,
    const primitives::CMutableTransaction& justice_tx)>;

// ── Watchtower ──────────────────────────────────────────────────────

/// Monitors the blockchain for revoked commitment transactions
/// and constructs justice transactions to penalize cheating.
class Watchtower {
public:
    Watchtower() = default;

    /// Start watching a channel
    Result<void> watch_channel(
        const ChannelId& channel_id,
        const crypto::Ed25519PublicKey& our_pubkey,
        const crypto::Ed25519PublicKey& their_pubkey);

    /// Stop watching a channel (cooperative close)
    Result<void> unwatch_channel(const ChannelId& channel_id);

    /// Register a revoked commitment state
    /// The watchtower will watch for this commitment being broadcast
    /// and react with the justice transaction.
    Result<void> add_revocation(
        const ChannelId& channel_id,
        uint64_t commitment_number,
        const uint256& revocation_secret,
        const primitives::CMutableTransaction& justice_tx);

    /// Process a new block: check all transactions for breaches
    /// Returns the number of breaches detected
    uint32_t process_block(
        const std::vector<primitives::CTransaction>& transactions,
        uint32_t block_height);

    /// Set callback for breach detection
    void set_breach_callback(BreachCallback callback);

    /// Check if a specific transaction is a known revoked commitment
    Result<BreachRemedy> check_transaction(
        const uint256& txid,
        const ChannelId& channel_id) const;

    /// Get the number of watched channels
    size_t watched_channel_count() const;

    /// Get the number of stored revocations across all channels
    size_t total_revocation_count() const;

    /// Get the watched state for a channel
    const WatchedChannel* get_watched(const ChannelId& channel_id) const;

    /// Clear all watched channels
    void clear();

private:
    /// Compute the hint for a transaction ID
    static uint256 compute_hint(const uint256& txid);

    mutable core::Mutex mutex_;
    std::unordered_map<ChannelId, WatchedChannel> watched_;
    BreachCallback breach_callback_;
};

}  // namespace rnet::lightning

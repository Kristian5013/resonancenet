#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/types.h"
#include "core/error.h"
#include "core/sync.h"
#include "crypto/ed25519.h"
#include "lightning/channel_state.h"
#include "lightning/htlc.h"
#include "lightning/invoice.h"
#include "lightning/router.h"

namespace rnet::lightning {

// ── Payment status ──────────────────────────────────────────────────

enum class PaymentStatus : uint8_t {
    PENDING     = 0,
    IN_FLIGHT   = 1,
    SUCCEEDED   = 2,
    FAILED      = 3,
    CANCELLED   = 4,
};

std::string_view payment_status_name(PaymentStatus status);

// ── Payment attempt ─────────────────────────────────────────────────

struct PaymentAttempt {
    Route               route;
    uint64_t             htlc_id = 0;          // HTLC ID in the first-hop channel
    ChannelId            first_hop_channel;
    uint64_t             created_at = 0;       // Unix timestamp ms
    uint64_t             resolved_at = 0;      // Unix timestamp ms
    PaymentStatus        status = PaymentStatus::PENDING;
    std::string          failure_reason;
    uint32_t             attempt_number = 0;
};

// ── Payment record ──────────────────────────────────────────────────

struct Payment {
    uint256                      payment_hash;
    uint256                      preimage;           // Set on success
    int64_t                      amount = 0;         // Payment amount in resonances
    int64_t                      total_fees = 0;     // Total routing fees paid
    crypto::Ed25519PublicKey     destination;
    std::string                  description;
    PaymentStatus                status = PaymentStatus::PENDING;
    uint64_t                     created_at = 0;     // Unix timestamp ms
    uint64_t                     resolved_at = 0;    // Unix timestamp ms
    uint32_t                     max_retries = 3;
    std::vector<PaymentAttempt>  attempts;

    /// Total elapsed time in milliseconds
    uint64_t elapsed_ms() const;

    /// Number of attempts made
    uint32_t attempt_count() const {
        return static_cast<uint32_t>(attempts.size());
    }
};

// ── Incoming payment ────────────────────────────────────────────────

struct IncomingPayment {
    uint256                      payment_hash;
    uint256                      preimage;
    int64_t                      amount = 0;
    uint64_t                     htlc_id = 0;
    ChannelId                    channel_id;
    PaymentStatus                status = PaymentStatus::PENDING;
    uint64_t                     received_at = 0;
    uint32_t                     cltv_expiry = 0;
};

// ── Payment callbacks ───────────────────────────────────────────────

using PaymentResultFn = std::function<void(
    const uint256& payment_hash,
    PaymentStatus status,
    const std::string& detail)>;

// ── Payment manager ─────────────────────────────────────────────────

/// Manages the lifecycle of outgoing and incoming payments.
class PaymentManager {
public:
    explicit PaymentManager(Router& router);

    // ── Outgoing payments ───────────────────────────────────────────

    /// Initiate a payment from an invoice
    Result<uint256> send_payment(
        const Invoice& invoice,
        const crypto::Ed25519PublicKey& our_node_id,
        uint32_t current_block_height);

    /// Initiate a keysend (spontaneous) payment
    Result<uint256> send_keysend(
        const crypto::Ed25519PublicKey& destination,
        int64_t amount,
        const crypto::Ed25519PublicKey& our_node_id,
        uint32_t current_block_height);

    /// Record that a payment HTLC was fulfilled (preimage received)
    Result<void> payment_fulfilled(const uint256& payment_hash,
                                    const uint256& preimage);

    /// Record that a payment HTLC failed
    Result<void> payment_failed(const uint256& payment_hash,
                                 const std::string& reason,
                                 uint32_t failing_hop = 0);

    /// Cancel a pending payment
    Result<void> cancel_payment(const uint256& payment_hash);

    /// Retry a failed payment
    Result<void> retry_payment(const uint256& payment_hash,
                                const crypto::Ed25519PublicKey& our_node_id,
                                uint32_t current_block_height);

    // ── Incoming payments ───────────────────────────────────────────

    /// Register an expected incoming payment (from an invoice we created)
    Result<void> register_incoming(const uint256& payment_hash,
                                    const uint256& preimage,
                                    int64_t expected_amount);

    /// Process a received HTLC that may be an incoming payment
    Result<uint256> receive_htlc(const uint256& payment_hash,
                                  int64_t amount,
                                  uint64_t htlc_id,
                                  const ChannelId& channel_id,
                                  uint32_t cltv_expiry);

    /// Mark an incoming payment as settled
    Result<void> settle_incoming(const uint256& payment_hash);

    // ── Queries ─────────────────────────────────────────────────────

    /// Get a payment by hash
    const Payment* get_payment(const uint256& payment_hash) const;

    /// Get an incoming payment by hash
    const IncomingPayment* get_incoming(const uint256& payment_hash) const;

    /// Get all payments
    std::vector<Payment> get_all_payments() const;

    /// Get all payments with a specific status
    std::vector<Payment> get_payments_by_status(PaymentStatus status) const;

    /// Get all incoming payments
    std::vector<IncomingPayment> get_all_incoming() const;

    /// Total amount sent (completed payments only)
    int64_t total_sent() const;

    /// Total amount received (completed incoming only)
    int64_t total_received() const;

    /// Total fees paid
    int64_t total_fees_paid() const;

    /// Set callback for payment results
    void set_result_callback(PaymentResultFn fn);

    /// Number of in-flight payments
    size_t in_flight_count() const;

    /// Get current time in milliseconds
    static uint64_t now_ms();

private:
    /// Helper to create a payment attempt with routing
    Result<PaymentAttempt> create_attempt(
        const Payment& payment,
        const crypto::Ed25519PublicKey& our_node_id,
        uint32_t current_block_height,
        const std::vector<uint64_t>& excluded_channels);

    Router&                     router_;
    mutable core::Mutex         mutex_;
    PaymentResultFn             result_fn_;

    std::unordered_map<uint256, Payment>         payments_;
    std::unordered_map<uint256, IncomingPayment> incoming_;
};

}  // namespace rnet::lightning

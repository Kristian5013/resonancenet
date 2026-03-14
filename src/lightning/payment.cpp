// Copyright (c) 2024-2026 The ResonanceNet Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "lightning/payment.h"

#include "core/logging.h"
#include "core/random.h"

#include <chrono>

namespace rnet::lightning {

// ===========================================================================
// Status Names
// ===========================================================================

// ---------------------------------------------------------------------------
// payment_status_name
// ---------------------------------------------------------------------------
// Maps each PaymentStatus enum value to its human-readable string
// representation for logging, RPC output, and diagnostic display.
// ---------------------------------------------------------------------------
std::string_view payment_status_name(PaymentStatus status) {
    switch (status) {
        case PaymentStatus::PENDING:   return "PENDING";
        case PaymentStatus::IN_FLIGHT: return "IN_FLIGHT";
        case PaymentStatus::SUCCEEDED: return "SUCCEEDED";
        case PaymentStatus::FAILED:    return "FAILED";
        case PaymentStatus::CANCELLED: return "CANCELLED";
        default:                       return "UNKNOWN";
    }
}

// ===========================================================================
// Payment Lifecycle
// ===========================================================================

// ---------------------------------------------------------------------------
// Payment::elapsed_ms
// ---------------------------------------------------------------------------
// Returns the wall-clock duration of the payment in milliseconds.  If the
// payment has already resolved, the interval is [created_at, resolved_at].
// If still in-flight, the interval is [created_at, now].
// ---------------------------------------------------------------------------
uint64_t Payment::elapsed_ms() const {
    // 1. Resolved payment — return stored duration
    if (resolved_at > 0 && created_at > 0) {
        return resolved_at - created_at;
    }
    // 2. In-flight payment — compute live duration
    if (created_at > 0) {
        return PaymentManager::now_ms() - created_at;
    }
    // 3. Not yet started
    return 0;
}

// ---------------------------------------------------------------------------
// PaymentManager::PaymentManager
// ---------------------------------------------------------------------------
// Stores a reference to the Router used for Dijkstra-based pathfinding
// across all payment attempts managed by this instance.
// ---------------------------------------------------------------------------
PaymentManager::PaymentManager(Router& router)
    : router_(router) {}

// ---------------------------------------------------------------------------
// PaymentManager::now_ms
// ---------------------------------------------------------------------------
// Returns the current UNIX epoch time in milliseconds.  Used as the
// canonical clock source for all payment and attempt timestamps.
// ---------------------------------------------------------------------------
uint64_t PaymentManager::now_ms() {
    auto now = std::chrono::system_clock::now();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count());
}

// ---------------------------------------------------------------------------
// send_payment
// ---------------------------------------------------------------------------
// Invoice-based payment flow.  Extracts the payment hash, amount, and
// destination from a BOLT-11-style invoice, finds a route via the router,
// and creates the first HTLC attempt.  The payment is tracked in payments_
// keyed by payment_hash so that fulfillment or failure can be matched later.
// ---------------------------------------------------------------------------
Result<uint256> PaymentManager::send_payment(
    const Invoice& invoice,
    const crypto::Ed25519PublicKey& our_node_id,
    uint32_t current_block_height) {

    LOCK(mutex_);

    // 1. Validate payment hash
    if (invoice.payment_hash().is_zero()) {
        return Result<uint256>::err("Invoice has no payment hash");
    }

    // 2. Check for duplicate payment
    auto ph = invoice.payment_hash();
    if (payments_.count(ph)) {
        return Result<uint256>::err("Payment already exists for this hash");
    }

    // 3. Validate amount
    int64_t amount = invoice.amount().value_or(0);
    if (amount <= 0) {
        return Result<uint256>::err("Invoice has no amount specified");
    }

    // 4. Build payment record
    Payment payment;
    payment.payment_hash = ph;
    payment.amount = amount;
    payment.destination = invoice.payee();
    payment.description = invoice.description();
    payment.status = PaymentStatus::PENDING;
    payment.created_at = now_ms();

    // 5. Find a route and create the first attempt
    auto attempt_result = create_attempt(payment, our_node_id,
                                          current_block_height, {});
    if (!attempt_result) {
        return Result<uint256>::err(attempt_result.error());
    }

    // 6. Mark attempt and payment as in-flight
    auto& attempt = attempt_result.value();
    attempt.status = PaymentStatus::IN_FLIGHT;
    attempt.created_at = now_ms();
    payment.total_fees = attempt.route.total_fees;
    payment.status = PaymentStatus::IN_FLIGHT;
    payment.attempts.push_back(std::move(attempt));

    // 7. Store and log
    payments_[ph] = std::move(payment);

    LogPrint(LIGHTNING, "Sending payment %s, amount=%lld",
             ph.to_hex().c_str(), amount);

    return Result<uint256>::ok(ph);
}

// ---------------------------------------------------------------------------
// send_keysend
// ---------------------------------------------------------------------------
// Spontaneous payment where the sender generates the preimage locally
// rather than receiving it from a payee invoice.  The payment hash is
// computed as Keccak256d(preimage), allowing the recipient to settle the
// HTLC immediately upon receiving the preimage in the onion payload.
// ---------------------------------------------------------------------------
Result<uint256> PaymentManager::send_keysend(
    const crypto::Ed25519PublicKey& destination,
    int64_t amount,
    const crypto::Ed25519PublicKey& our_node_id,
    uint32_t current_block_height) {

    LOCK(mutex_);

    // 1. Validate amount
    if (amount <= 0) {
        return Result<uint256>::err("Amount must be positive");
    }

    // 2. Generate preimage and derive payment hash via Keccak256d
    uint256 preimage = generate_preimage();
    uint256 ph = compute_payment_hash(preimage);

    // 3. Build payment record
    Payment payment;
    payment.payment_hash = ph;
    payment.preimage = preimage;
    payment.amount = amount;
    payment.destination = destination;
    payment.description = "keysend";
    payment.status = PaymentStatus::PENDING;
    payment.created_at = now_ms();

    // 4. Find a route and create the first attempt
    auto attempt_result = create_attempt(payment, our_node_id,
                                          current_block_height, {});
    if (!attempt_result) {
        return Result<uint256>::err(attempt_result.error());
    }

    // 5. Mark attempt and payment as in-flight
    auto& attempt = attempt_result.value();
    attempt.status = PaymentStatus::IN_FLIGHT;
    attempt.created_at = now_ms();
    payment.total_fees = attempt.route.total_fees;
    payment.status = PaymentStatus::IN_FLIGHT;
    payment.attempts.push_back(std::move(attempt));

    // 6. Store and log
    payments_[ph] = std::move(payment);

    LogPrint(LIGHTNING, "Sending keysend payment %s to %s, amount=%lld",
             ph.to_hex().c_str(), destination.to_hex().c_str(), amount);

    return Result<uint256>::ok(ph);
}

// ---------------------------------------------------------------------------
// payment_fulfilled
// ---------------------------------------------------------------------------
// Called when the preimage for a payment HTLC is received back from the
// payee.  Verifies the preimage by checking that Keccak256d(preimage)
// equals the stored payment_hash.  On success, marks both the payment and
// the latest attempt as SUCCEEDED and fires the result callback.
// ---------------------------------------------------------------------------
Result<void> PaymentManager::payment_fulfilled(
    const uint256& payment_hash,
    const uint256& preimage) {

    LOCK(mutex_);

    // 1. Look up the payment
    auto it = payments_.find(payment_hash);
    if (it == payments_.end()) {
        return Result<void>::err("Payment not found");
    }

    auto& payment = it->second;

    // 2. Verify preimage: Keccak256d(preimage) == payment_hash
    if (!verify_preimage(preimage, payment_hash)) {
        return Result<void>::err("Invalid preimage");
    }

    // 3. Record the preimage and mark as succeeded
    payment.preimage = preimage;
    payment.status = PaymentStatus::SUCCEEDED;
    payment.resolved_at = now_ms();

    // 4. Mark latest attempt as succeeded
    if (!payment.attempts.empty()) {
        payment.attempts.back().status = PaymentStatus::SUCCEEDED;
        payment.attempts.back().resolved_at = now_ms();
    }

    // 5. Log completion time
    LogPrint(LIGHTNING, "Payment %s succeeded in %llu ms",
             payment_hash.to_hex().c_str(), payment.elapsed_ms());

    // 6. Fire result callback
    if (result_fn_) {
        result_fn_(payment_hash, PaymentStatus::SUCCEEDED, "");
    }

    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// payment_failed
// ---------------------------------------------------------------------------
// Records a payment HTLC failure.  If the attempt count is below
// max_retries, the payment transitions back to PENDING so that
// retry_payment can try an alternative route.  Failed channels from
// previous attempts are excluded in subsequent routing to achieve
// progressive route diversity.
// ---------------------------------------------------------------------------
Result<void> PaymentManager::payment_failed(
    const uint256& payment_hash,
    const std::string& reason,
    uint32_t /*failing_hop*/) {

    LOCK(mutex_);

    // 1. Look up the payment
    auto it = payments_.find(payment_hash);
    if (it == payments_.end()) {
        return Result<void>::err("Payment not found");
    }

    auto& payment = it->second;

    // 2. Mark latest attempt as failed
    if (!payment.attempts.empty()) {
        payment.attempts.back().status = PaymentStatus::FAILED;
        payment.attempts.back().failure_reason = reason;
        payment.attempts.back().resolved_at = now_ms();
    }

    // 3. Check if we should retry or permanently fail
    if (payment.attempt_count() < payment.max_retries) {
        payment.status = PaymentStatus::PENDING;
        LogPrint(LIGHTNING, "Payment %s failed (attempt %u/%u): %s",
                 payment_hash.to_hex().c_str(),
                 payment.attempt_count(), payment.max_retries,
                 reason.c_str());
    } else {
        payment.status = PaymentStatus::FAILED;
        payment.resolved_at = now_ms();

        LogPrint(LIGHTNING, "Payment %s permanently failed after %u attempts",
                 payment_hash.to_hex().c_str(), payment.attempt_count());

        // 4. Fire result callback on permanent failure
        if (result_fn_) {
            result_fn_(payment_hash, PaymentStatus::FAILED, reason);
        }
    }

    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// cancel_payment
// ---------------------------------------------------------------------------
// User-initiated cancellation of a payment that has not yet succeeded.
// Prevents further retry attempts and fires the result callback with
// a CANCELLED status.
// ---------------------------------------------------------------------------
Result<void> PaymentManager::cancel_payment(const uint256& payment_hash) {
    LOCK(mutex_);

    // 1. Look up the payment
    auto it = payments_.find(payment_hash);
    if (it == payments_.end()) {
        return Result<void>::err("Payment not found");
    }

    // 2. Reject cancellation of already-completed payments
    if (it->second.status == PaymentStatus::SUCCEEDED) {
        return Result<void>::err("Cannot cancel a completed payment");
    }

    // 3. Mark as cancelled
    it->second.status = PaymentStatus::CANCELLED;
    it->second.resolved_at = now_ms();

    // 4. Fire result callback
    if (result_fn_) {
        result_fn_(payment_hash, PaymentStatus::CANCELLED, "User cancelled");
    }

    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// retry_payment
// ---------------------------------------------------------------------------
// Retries a payment that is in PENDING or FAILED state by finding a new
// route.  Achieves progressive route diversity by collecting the first-hop
// short_channel_id from every previously failed attempt and passing them
// as exclusions to the router.  If no alternative route exists, the
// payment is permanently marked FAILED.
// ---------------------------------------------------------------------------
Result<void> PaymentManager::retry_payment(
    const uint256& payment_hash,
    const crypto::Ed25519PublicKey& our_node_id,
    uint32_t current_block_height) {

    LOCK(mutex_);

    // 1. Look up the payment
    auto it = payments_.find(payment_hash);
    if (it == payments_.end()) {
        return Result<void>::err("Payment not found");
    }

    auto& payment = it->second;

    // 2. Verify payment is in a retryable state
    if (payment.status != PaymentStatus::PENDING &&
        payment.status != PaymentStatus::FAILED) {
        return Result<void>::err("Payment not in retryable state");
    }

    // 3. Collect channels to exclude from failed attempts
    std::vector<uint64_t> excluded;
    for (const auto& attempt : payment.attempts) {
        if (attempt.status == PaymentStatus::FAILED &&
            !attempt.route.hops.empty()) {
            excluded.push_back(attempt.route.hops[0].short_channel_id);
        }
    }

    // 4. Find a new route excluding failed channels
    auto attempt_result = create_attempt(payment, our_node_id,
                                          current_block_height, excluded);
    if (!attempt_result) {
        payment.status = PaymentStatus::FAILED;
        payment.resolved_at = now_ms();
        return Result<void>::err(attempt_result.error());
    }

    // 5. Mark attempt and payment as in-flight
    auto& attempt = attempt_result.value();
    attempt.status = PaymentStatus::IN_FLIGHT;
    attempt.created_at = now_ms();
    attempt.attempt_number = payment.attempt_count();
    payment.status = PaymentStatus::IN_FLIGHT;
    payment.total_fees = attempt.route.total_fees;
    payment.attempts.push_back(std::move(attempt));

    return Result<void>::ok();
}

// ===========================================================================
// Incoming Payments
// ===========================================================================

// ---------------------------------------------------------------------------
// register_incoming
// ---------------------------------------------------------------------------
// Registers an expected incoming payment from an invoice that we created.
// Stores the preimage so it can be revealed when the matching HTLC arrives,
// and records the expected amount for underpayment detection.
// ---------------------------------------------------------------------------
Result<void> PaymentManager::register_incoming(
    const uint256& payment_hash,
    const uint256& preimage,
    int64_t expected_amount) {

    LOCK(mutex_);

    // 1. Reject duplicate registration
    if (incoming_.count(payment_hash)) {
        return Result<void>::err("Incoming payment already registered");
    }

    // 2. Build incoming payment record
    IncomingPayment inc;
    inc.payment_hash = payment_hash;
    inc.preimage = preimage;
    inc.amount = expected_amount;
    inc.status = PaymentStatus::PENDING;
    inc.received_at = 0;

    // 3. Store
    incoming_[payment_hash] = std::move(inc);
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// receive_htlc
// ---------------------------------------------------------------------------
// Processes an incoming HTLC that matches a previously registered invoice.
// Verifies the payment amount meets the invoice minimum, records the HTLC
// metadata (id, channel, CLTV expiry), and returns the preimage so the
// caller can fulfill the HTLC on the channel.
// ---------------------------------------------------------------------------
Result<uint256> PaymentManager::receive_htlc(
    const uint256& payment_hash,
    int64_t amount,
    uint64_t htlc_id,
    const ChannelId& channel_id,
    uint32_t cltv_expiry) {

    LOCK(mutex_);

    // 1. Look up the registered incoming payment
    auto it = incoming_.find(payment_hash);
    if (it == incoming_.end()) {
        return Result<uint256>::err("No pending invoice for this payment hash");
    }

    auto& inc = it->second;

    // 2. Verify the payment has not already been settled
    if (inc.status != PaymentStatus::PENDING) {
        return Result<uint256>::err("Incoming payment already settled");
    }

    // 3. Check amount meets minimum (underpayment protection)
    if (inc.amount > 0 && amount < inc.amount) {
        return Result<uint256>::err("Underpayment: expected " +
                                     std::to_string(inc.amount) +
                                     ", got " + std::to_string(amount));
    }

    // 4. Record HTLC metadata
    inc.amount = amount;
    inc.htlc_id = htlc_id;
    inc.channel_id = channel_id;
    inc.cltv_expiry = cltv_expiry;
    inc.received_at = now_ms();
    inc.status = PaymentStatus::IN_FLIGHT;

    // 5. Log receipt
    LogPrint(LIGHTNING, "Received HTLC for payment %s, amount=%lld",
             payment_hash.to_hex().c_str(), amount);

    // 6. Return the preimage so the caller can fulfill the HTLC
    return Result<uint256>::ok(inc.preimage);
}

// ---------------------------------------------------------------------------
// settle_incoming
// ---------------------------------------------------------------------------
// Marks an incoming payment as fully settled after the HTLC fulfill message
// has been acknowledged by the channel peer.
// ---------------------------------------------------------------------------
Result<void> PaymentManager::settle_incoming(const uint256& payment_hash) {
    LOCK(mutex_);

    // 1. Look up the incoming payment
    auto it = incoming_.find(payment_hash);
    if (it == incoming_.end()) {
        return Result<void>::err("Incoming payment not found");
    }

    // 2. Mark as succeeded
    it->second.status = PaymentStatus::SUCCEEDED;
    return Result<void>::ok();
}

// ===========================================================================
// Queries
// ===========================================================================

// ---------------------------------------------------------------------------
// get_payment
// ---------------------------------------------------------------------------
// Returns a pointer to the outgoing Payment record for the given hash,
// or nullptr if no such payment exists.
// ---------------------------------------------------------------------------
const Payment* PaymentManager::get_payment(const uint256& payment_hash) const {
    LOCK(mutex_);
    auto it = payments_.find(payment_hash);
    return it != payments_.end() ? &it->second : nullptr;
}

// ---------------------------------------------------------------------------
// get_incoming
// ---------------------------------------------------------------------------
// Returns a pointer to the IncomingPayment record for the given hash,
// or nullptr if no such incoming payment exists.
// ---------------------------------------------------------------------------
const IncomingPayment* PaymentManager::get_incoming(
    const uint256& payment_hash) const {
    LOCK(mutex_);
    auto it = incoming_.find(payment_hash);
    return it != incoming_.end() ? &it->second : nullptr;
}

// ---------------------------------------------------------------------------
// get_all_payments
// ---------------------------------------------------------------------------
// Returns a snapshot copy of all outgoing payment records.  The copy is
// taken under the lock so callers receive a consistent view.
// ---------------------------------------------------------------------------
std::vector<Payment> PaymentManager::get_all_payments() const {
    LOCK(mutex_);
    std::vector<Payment> result;
    result.reserve(payments_.size());
    for (const auto& [_, p] : payments_) {
        result.push_back(p);
    }
    return result;
}

// ---------------------------------------------------------------------------
// get_payments_by_status
// ---------------------------------------------------------------------------
// Returns all outgoing payments matching the given status filter.
// ---------------------------------------------------------------------------
std::vector<Payment> PaymentManager::get_payments_by_status(
    PaymentStatus status) const {
    LOCK(mutex_);
    std::vector<Payment> result;
    for (const auto& [_, p] : payments_) {
        if (p.status == status) {
            result.push_back(p);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// get_all_incoming
// ---------------------------------------------------------------------------
// Returns a snapshot copy of all incoming payment records.
// ---------------------------------------------------------------------------
std::vector<IncomingPayment> PaymentManager::get_all_incoming() const {
    LOCK(mutex_);
    std::vector<IncomingPayment> result;
    result.reserve(incoming_.size());
    for (const auto& [_, p] : incoming_) {
        result.push_back(p);
    }
    return result;
}

// ===========================================================================
// Statistics
// ===========================================================================

// ---------------------------------------------------------------------------
// total_sent
// ---------------------------------------------------------------------------
// Sums the amount field of all SUCCEEDED outgoing payments.
// ---------------------------------------------------------------------------
int64_t PaymentManager::total_sent() const {
    LOCK(mutex_);
    int64_t total = 0;
    for (const auto& [_, p] : payments_) {
        if (p.status == PaymentStatus::SUCCEEDED) {
            total += p.amount;
        }
    }
    return total;
}

// ---------------------------------------------------------------------------
// total_received
// ---------------------------------------------------------------------------
// Sums the amount field of all SUCCEEDED incoming payments.
// ---------------------------------------------------------------------------
int64_t PaymentManager::total_received() const {
    LOCK(mutex_);
    int64_t total = 0;
    for (const auto& [_, p] : incoming_) {
        if (p.status == PaymentStatus::SUCCEEDED) {
            total += p.amount;
        }
    }
    return total;
}

// ---------------------------------------------------------------------------
// total_fees_paid
// ---------------------------------------------------------------------------
// Sums the routing fees from all SUCCEEDED outgoing payments.
// ---------------------------------------------------------------------------
int64_t PaymentManager::total_fees_paid() const {
    LOCK(mutex_);
    int64_t total = 0;
    for (const auto& [_, p] : payments_) {
        if (p.status == PaymentStatus::SUCCEEDED) {
            total += p.total_fees;
        }
    }
    return total;
}

// ---------------------------------------------------------------------------
// set_result_callback
// ---------------------------------------------------------------------------
// Registers a callback invoked when a payment reaches a terminal state
// (SUCCEEDED, FAILED, or CANCELLED).
// ---------------------------------------------------------------------------
void PaymentManager::set_result_callback(PaymentResultFn fn) {
    LOCK(mutex_);
    result_fn_ = std::move(fn);
}

// ---------------------------------------------------------------------------
// in_flight_count
// ---------------------------------------------------------------------------
// Returns the number of outgoing payments currently in IN_FLIGHT state.
// ---------------------------------------------------------------------------
size_t PaymentManager::in_flight_count() const {
    LOCK(mutex_);
    size_t count = 0;
    for (const auto& [_, p] : payments_) {
        if (p.status == PaymentStatus::IN_FLIGHT) {
            ++count;
        }
    }
    return count;
}

// ===========================================================================
// Internal Helpers
// ===========================================================================

// ---------------------------------------------------------------------------
// create_attempt
// ---------------------------------------------------------------------------
// Builds a PaymentAttempt by running Dijkstra-based route finding from
// our_node_id to the payment destination.  The final CLTV expiry is
// computed as current_block_height + DEFAULT_CLTV_EXPIRY_DELTA, and each
// intermediate hop adds its own cltv_expiry_delta to the cumulative
// timelock.  Channels in excluded_channels are skipped during pathfinding
// to avoid routes that previously failed.
// ---------------------------------------------------------------------------
Result<PaymentAttempt> PaymentManager::create_attempt(
    const Payment& payment,
    const crypto::Ed25519PublicKey& our_node_id,
    uint32_t current_block_height,
    const std::vector<uint64_t>& excluded_channels) {

    // 1. Compute final CLTV expiry for the destination hop
    uint32_t final_cltv = current_block_height + DEFAULT_CLTV_EXPIRY_DELTA;

    // 2. Run Dijkstra route finding with channel exclusions
    auto route_result = router_.find_route(
        our_node_id, payment.destination, payment.amount,
        final_cltv, MAX_ROUTE_HOPS, excluded_channels);

    if (!route_result) {
        return Result<PaymentAttempt>::err("No route found: " +
                                            route_result.error());
    }

    // 3. Build the attempt from the discovered route
    PaymentAttempt attempt;
    attempt.route = std::move(route_result.value());

    // 4. Record first-hop channel (resolved by caller when adding HTLC)
    if (!attempt.route.hops.empty()) {
        attempt.first_hop_channel = ChannelId();  // Would be resolved by caller
    }

    return Result<PaymentAttempt>::ok(std::move(attempt));
}

} // namespace rnet::lightning

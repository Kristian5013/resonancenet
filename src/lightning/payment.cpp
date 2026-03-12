#include "lightning/payment.h"

#include <chrono>

#include "core/logging.h"
#include "core/random.h"

namespace rnet::lightning {

// ── Status names ────────────────────────────────────────────────────

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

// ── Payment ─────────────────────────────────────────────────────────

uint64_t Payment::elapsed_ms() const {
    if (resolved_at > 0 && created_at > 0) {
        return resolved_at - created_at;
    }
    if (created_at > 0) {
        return PaymentManager::now_ms() - created_at;
    }
    return 0;
}

// ── PaymentManager ──────────────────────────────────────────────────

PaymentManager::PaymentManager(Router& router)
    : router_(router) {}

uint64_t PaymentManager::now_ms() {
    auto now = std::chrono::system_clock::now();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count());
}

Result<uint256> PaymentManager::send_payment(
    const Invoice& invoice,
    const crypto::Ed25519PublicKey& our_node_id,
    uint32_t current_block_height) {

    LOCK(mutex_);

    if (invoice.payment_hash().is_zero()) {
        return Result<uint256>::err("Invoice has no payment hash");
    }

    auto ph = invoice.payment_hash();
    if (payments_.count(ph)) {
        return Result<uint256>::err("Payment already exists for this hash");
    }

    int64_t amount = invoice.amount().value_or(0);
    if (amount <= 0) {
        return Result<uint256>::err("Invoice has no amount specified");
    }

    Payment payment;
    payment.payment_hash = ph;
    payment.amount = amount;
    payment.destination = invoice.payee();
    payment.description = invoice.description();
    payment.status = PaymentStatus::PENDING;
    payment.created_at = now_ms();

    // Find a route
    auto attempt_result = create_attempt(payment, our_node_id,
                                          current_block_height, {});
    if (!attempt_result) {
        return Result<uint256>::err(attempt_result.error());
    }

    auto& attempt = attempt_result.value();
    attempt.status = PaymentStatus::IN_FLIGHT;
    attempt.created_at = now_ms();
    payment.total_fees = attempt.route.total_fees;
    payment.status = PaymentStatus::IN_FLIGHT;
    payment.attempts.push_back(std::move(attempt));

    payments_[ph] = std::move(payment);

    LogPrint(LIGHTNING, "Sending payment %s, amount=%lld",
             ph.to_hex().c_str(), amount);

    return Result<uint256>::ok(ph);
}

Result<uint256> PaymentManager::send_keysend(
    const crypto::Ed25519PublicKey& destination,
    int64_t amount,
    const crypto::Ed25519PublicKey& our_node_id,
    uint32_t current_block_height) {

    LOCK(mutex_);

    if (amount <= 0) {
        return Result<uint256>::err("Amount must be positive");
    }

    // Generate preimage and payment hash for keysend
    uint256 preimage = generate_preimage();
    uint256 ph = compute_payment_hash(preimage);

    Payment payment;
    payment.payment_hash = ph;
    payment.preimage = preimage;
    payment.amount = amount;
    payment.destination = destination;
    payment.description = "keysend";
    payment.status = PaymentStatus::PENDING;
    payment.created_at = now_ms();

    auto attempt_result = create_attempt(payment, our_node_id,
                                          current_block_height, {});
    if (!attempt_result) {
        return Result<uint256>::err(attempt_result.error());
    }

    auto& attempt = attempt_result.value();
    attempt.status = PaymentStatus::IN_FLIGHT;
    attempt.created_at = now_ms();
    payment.total_fees = attempt.route.total_fees;
    payment.status = PaymentStatus::IN_FLIGHT;
    payment.attempts.push_back(std::move(attempt));

    payments_[ph] = std::move(payment);

    LogPrint(LIGHTNING, "Sending keysend payment %s to %s, amount=%lld",
             ph.to_hex().c_str(), destination.to_hex().c_str(), amount);

    return Result<uint256>::ok(ph);
}

Result<void> PaymentManager::payment_fulfilled(
    const uint256& payment_hash,
    const uint256& preimage) {

    LOCK(mutex_);

    auto it = payments_.find(payment_hash);
    if (it == payments_.end()) {
        return Result<void>::err("Payment not found");
    }

    auto& payment = it->second;

    if (!verify_preimage(preimage, payment_hash)) {
        return Result<void>::err("Invalid preimage");
    }

    payment.preimage = preimage;
    payment.status = PaymentStatus::SUCCEEDED;
    payment.resolved_at = now_ms();

    // Mark latest attempt as succeeded
    if (!payment.attempts.empty()) {
        payment.attempts.back().status = PaymentStatus::SUCCEEDED;
        payment.attempts.back().resolved_at = now_ms();
    }

    LogPrint(LIGHTNING, "Payment %s succeeded in %llu ms",
             payment_hash.to_hex().c_str(), payment.elapsed_ms());

    if (result_fn_) {
        result_fn_(payment_hash, PaymentStatus::SUCCEEDED, "");
    }

    return Result<void>::ok();
}

Result<void> PaymentManager::payment_failed(
    const uint256& payment_hash,
    const std::string& reason,
    uint32_t /*failing_hop*/) {

    LOCK(mutex_);

    auto it = payments_.find(payment_hash);
    if (it == payments_.end()) {
        return Result<void>::err("Payment not found");
    }

    auto& payment = it->second;

    // Mark latest attempt as failed
    if (!payment.attempts.empty()) {
        payment.attempts.back().status = PaymentStatus::FAILED;
        payment.attempts.back().failure_reason = reason;
        payment.attempts.back().resolved_at = now_ms();
    }

    // Check if we should retry
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

        if (result_fn_) {
            result_fn_(payment_hash, PaymentStatus::FAILED, reason);
        }
    }

    return Result<void>::ok();
}

Result<void> PaymentManager::cancel_payment(const uint256& payment_hash) {
    LOCK(mutex_);

    auto it = payments_.find(payment_hash);
    if (it == payments_.end()) {
        return Result<void>::err("Payment not found");
    }

    if (it->second.status == PaymentStatus::SUCCEEDED) {
        return Result<void>::err("Cannot cancel a completed payment");
    }

    it->second.status = PaymentStatus::CANCELLED;
    it->second.resolved_at = now_ms();

    if (result_fn_) {
        result_fn_(payment_hash, PaymentStatus::CANCELLED, "User cancelled");
    }

    return Result<void>::ok();
}

Result<void> PaymentManager::retry_payment(
    const uint256& payment_hash,
    const crypto::Ed25519PublicKey& our_node_id,
    uint32_t current_block_height) {

    LOCK(mutex_);

    auto it = payments_.find(payment_hash);
    if (it == payments_.end()) {
        return Result<void>::err("Payment not found");
    }

    auto& payment = it->second;
    if (payment.status != PaymentStatus::PENDING &&
        payment.status != PaymentStatus::FAILED) {
        return Result<void>::err("Payment not in retryable state");
    }

    // Collect channels to exclude from failed attempts
    std::vector<uint64_t> excluded;
    for (const auto& attempt : payment.attempts) {
        if (attempt.status == PaymentStatus::FAILED &&
            !attempt.route.hops.empty()) {
            excluded.push_back(attempt.route.hops[0].short_channel_id);
        }
    }

    auto attempt_result = create_attempt(payment, our_node_id,
                                          current_block_height, excluded);
    if (!attempt_result) {
        payment.status = PaymentStatus::FAILED;
        payment.resolved_at = now_ms();
        return Result<void>::err(attempt_result.error());
    }

    auto& attempt = attempt_result.value();
    attempt.status = PaymentStatus::IN_FLIGHT;
    attempt.created_at = now_ms();
    attempt.attempt_number = payment.attempt_count();
    payment.status = PaymentStatus::IN_FLIGHT;
    payment.total_fees = attempt.route.total_fees;
    payment.attempts.push_back(std::move(attempt));

    return Result<void>::ok();
}

// ── Incoming payments ───────────────────────────────────────────────

Result<void> PaymentManager::register_incoming(
    const uint256& payment_hash,
    const uint256& preimage,
    int64_t expected_amount) {

    LOCK(mutex_);

    if (incoming_.count(payment_hash)) {
        return Result<void>::err("Incoming payment already registered");
    }

    IncomingPayment inc;
    inc.payment_hash = payment_hash;
    inc.preimage = preimage;
    inc.amount = expected_amount;
    inc.status = PaymentStatus::PENDING;
    inc.received_at = 0;

    incoming_[payment_hash] = std::move(inc);
    return Result<void>::ok();
}

Result<uint256> PaymentManager::receive_htlc(
    const uint256& payment_hash,
    int64_t amount,
    uint64_t htlc_id,
    const ChannelId& channel_id,
    uint32_t cltv_expiry) {

    LOCK(mutex_);

    auto it = incoming_.find(payment_hash);
    if (it == incoming_.end()) {
        return Result<uint256>::err("No pending invoice for this payment hash");
    }

    auto& inc = it->second;
    if (inc.status != PaymentStatus::PENDING) {
        return Result<uint256>::err("Incoming payment already settled");
    }

    if (inc.amount > 0 && amount < inc.amount) {
        return Result<uint256>::err("Underpayment: expected " +
                                     std::to_string(inc.amount) +
                                     ", got " + std::to_string(amount));
    }

    inc.amount = amount;
    inc.htlc_id = htlc_id;
    inc.channel_id = channel_id;
    inc.cltv_expiry = cltv_expiry;
    inc.received_at = now_ms();
    inc.status = PaymentStatus::IN_FLIGHT;

    LogPrint(LIGHTNING, "Received HTLC for payment %s, amount=%lld",
             payment_hash.to_hex().c_str(), amount);

    // Return the preimage so the caller can fulfill the HTLC
    return Result<uint256>::ok(inc.preimage);
}

Result<void> PaymentManager::settle_incoming(const uint256& payment_hash) {
    LOCK(mutex_);

    auto it = incoming_.find(payment_hash);
    if (it == incoming_.end()) {
        return Result<void>::err("Incoming payment not found");
    }

    it->second.status = PaymentStatus::SUCCEEDED;
    return Result<void>::ok();
}

// ── Queries ─────────────────────────────────────────────────────────

const Payment* PaymentManager::get_payment(const uint256& payment_hash) const {
    LOCK(mutex_);
    auto it = payments_.find(payment_hash);
    return it != payments_.end() ? &it->second : nullptr;
}

const IncomingPayment* PaymentManager::get_incoming(
    const uint256& payment_hash) const {
    LOCK(mutex_);
    auto it = incoming_.find(payment_hash);
    return it != incoming_.end() ? &it->second : nullptr;
}

std::vector<Payment> PaymentManager::get_all_payments() const {
    LOCK(mutex_);
    std::vector<Payment> result;
    result.reserve(payments_.size());
    for (const auto& [_, p] : payments_) {
        result.push_back(p);
    }
    return result;
}

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

std::vector<IncomingPayment> PaymentManager::get_all_incoming() const {
    LOCK(mutex_);
    std::vector<IncomingPayment> result;
    result.reserve(incoming_.size());
    for (const auto& [_, p] : incoming_) {
        result.push_back(p);
    }
    return result;
}

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

void PaymentManager::set_result_callback(PaymentResultFn fn) {
    LOCK(mutex_);
    result_fn_ = std::move(fn);
}

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

Result<PaymentAttempt> PaymentManager::create_attempt(
    const Payment& payment,
    const crypto::Ed25519PublicKey& our_node_id,
    uint32_t current_block_height,
    const std::vector<uint64_t>& excluded_channels) {

    uint32_t final_cltv = current_block_height + DEFAULT_CLTV_EXPIRY_DELTA;

    auto route_result = router_.find_route(
        our_node_id, payment.destination, payment.amount,
        final_cltv, MAX_ROUTE_HOPS, excluded_channels);

    if (!route_result) {
        return Result<PaymentAttempt>::err("No route found: " +
                                            route_result.error());
    }

    PaymentAttempt attempt;
    attempt.route = std::move(route_result.value());

    if (!attempt.route.hops.empty()) {
        // The first hop channel would be used to add the HTLC
        attempt.first_hop_channel = ChannelId();  // Would be resolved by caller
    }

    return Result<PaymentAttempt>::ok(std::move(attempt));
}

}  // namespace rnet::lightning

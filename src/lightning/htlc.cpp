// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "lightning/htlc.h"

#include "core/random.h"
#include "crypto/keccak.h"

#include <algorithm>
#include <string>

// ---------------------------------------------------------------------------
// Design note — Keccak256d payment hash, timelock, preimage reveal
//
//   payment_hash  = Keccak256d(preimage)       -- 32-byte hash commitment
//   preimage      = random 32 bytes            -- revealed to settle
//   cltv_expiry   = absolute block height      -- timelock for refund
//
// The sender locks funds behind payment_hash.  The receiver proves
// knowledge of preimage to claim.  If cltv_expiry passes without
// settlement the HTLC times out and funds return to the sender.
//
// Fee model:  fee = base_fee + (amount * fee_rate_ppm / 1,000,000)
// ---------------------------------------------------------------------------

namespace rnet::lightning {

// ---------------------------------------------------------------------------
// htlc_state_name
// ---------------------------------------------------------------------------

std::string_view htlc_state_name(HtlcState state) {
    switch (state) {
        case HtlcState::PENDING:          return "PENDING";
        case HtlcState::COMMITTED_LOCAL:  return "COMMITTED_LOCAL";
        case HtlcState::COMMITTED_REMOTE: return "COMMITTED_REMOTE";
        case HtlcState::COMMITTED_BOTH:   return "COMMITTED_BOTH";
        case HtlcState::FULFILLED:        return "FULFILLED";
        case HtlcState::FAILED:           return "FAILED";
        case HtlcState::TIMED_OUT:        return "TIMED_OUT";
        default:                          return "UNKNOWN";
    }
}

// ---------------------------------------------------------------------------
// validate_htlc
// ---------------------------------------------------------------------------

Result<void> validate_htlc(const Htlc& htlc,
                            int64_t channel_capacity,
                            int64_t current_balance,
                            uint32_t current_htlc_count,
                            uint32_t max_htlcs,
                            int64_t min_htlc_value) {
    // 1. Amount must be positive.
    if (htlc.amount <= 0) {
        return Result<void>::err("HTLC amount must be positive");
    }

    // 2. Amount must meet the minimum.
    if (htlc.amount < min_htlc_value) {
        return Result<void>::err("HTLC amount below minimum: " +
                                  std::to_string(htlc.amount) + " < " +
                                  std::to_string(min_htlc_value));
    }

    // 3. Amount must not exceed channel capacity.
    if (htlc.amount > channel_capacity) {
        return Result<void>::err("HTLC amount exceeds channel capacity");
    }

    // 4. Amount must not exceed available balance.
    if (htlc.amount > current_balance) {
        return Result<void>::err("HTLC amount exceeds available balance");
    }

    // 5. HTLC count limit.
    if (current_htlc_count >= max_htlcs) {
        return Result<void>::err("Maximum HTLCs per channel reached: " +
                                  std::to_string(max_htlcs));
    }

    // 6. Payment hash must be non-zero.
    if (htlc.payment_hash.is_zero()) {
        return Result<void>::err("HTLC payment hash cannot be zero");
    }

    // 7. CLTV expiry must be set.
    if (htlc.cltv_expiry == 0) {
        return Result<void>::err("HTLC CLTV expiry must be set");
    }

    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// generate_preimage
// ---------------------------------------------------------------------------

uint256 generate_preimage() {
    return core::get_rand_hash();
}

// ---------------------------------------------------------------------------
// compute_payment_hash
// ---------------------------------------------------------------------------

uint256 compute_payment_hash(const uint256& preimage) {
    return crypto::keccak256d(preimage.span());
}

// ---------------------------------------------------------------------------
// verify_preimage
// ---------------------------------------------------------------------------

bool verify_preimage(const uint256& preimage, const uint256& payment_hash) {
    return compute_payment_hash(preimage) == payment_hash;
}

// ---------------------------------------------------------------------------
// compute_htlc_fee
// ---------------------------------------------------------------------------

int64_t compute_htlc_fee(int64_t amount, int64_t base_fee, int64_t fee_rate_ppm) {
    // fee = base_fee + (amount * fee_rate_ppm / 1,000,000)
    int64_t proportional = (amount * fee_rate_ppm) / 1'000'000;
    return base_fee + proportional;
}

// ---------------------------------------------------------------------------
// HtlcSet::add
// ---------------------------------------------------------------------------

Result<void> HtlcSet::add(Htlc htlc) {
    // 1. Assign sequential id and initial state.
    htlc.id = next_id_++;
    htlc.state = HtlcState::PENDING;

    // 2. Append to the set.
    htlcs_.push_back(std::move(htlc));
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// HtlcSet::fulfill
// ---------------------------------------------------------------------------

Result<void> HtlcSet::fulfill(uint64_t htlc_id, const uint256& preimage) {
    // 1. Look up the HTLC by id.
    auto* h = find(htlc_id);
    if (!h) {
        return Result<void>::err("HTLC not found: " + std::to_string(htlc_id));
    }

    // 2. Reject if already settled.
    if (h->state == HtlcState::FULFILLED) {
        return Result<void>::err("HTLC already fulfilled");
    }
    if (h->state == HtlcState::FAILED || h->state == HtlcState::TIMED_OUT) {
        return Result<void>::err("HTLC already settled");
    }

    // 3. Verify preimage against payment hash (Keccak256d).
    if (!verify_preimage(preimage, h->payment_hash)) {
        return Result<void>::err("Invalid preimage for HTLC");
    }

    // 4. Transition to FULFILLED.
    h->state = HtlcState::FULFILLED;
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// HtlcSet::fail
// ---------------------------------------------------------------------------

Result<void> HtlcSet::fail(uint64_t htlc_id) {
    // 1. Look up the HTLC by id.
    auto* h = find(htlc_id);
    if (!h) {
        return Result<void>::err("HTLC not found: " + std::to_string(htlc_id));
    }

    // 2. Reject if already settled.
    if (h->state == HtlcState::FULFILLED || h->state == HtlcState::FAILED ||
        h->state == HtlcState::TIMED_OUT) {
        return Result<void>::err("HTLC already settled");
    }

    // 3. Transition to FAILED.
    h->state = HtlcState::FAILED;
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// HtlcSet::mark_expired
// ---------------------------------------------------------------------------

uint32_t HtlcSet::mark_expired(uint32_t current_height) {
    uint32_t count = 0;
    for (auto& h : htlcs_) {
        // 1. Skip already-settled HTLCs.
        if (h.state != HtlcState::FULFILLED && h.state != HtlcState::FAILED &&
            h.state != HtlcState::TIMED_OUT && h.is_expired(current_height)) {
            // 2. Timelock has passed — mark as timed out.
            h.state = HtlcState::TIMED_OUT;
            ++count;
        }
    }
    return count;
}

// ---------------------------------------------------------------------------
// HtlcSet::find  (const)
// ---------------------------------------------------------------------------

const Htlc* HtlcSet::find(uint64_t htlc_id) const {
    for (const auto& h : htlcs_) {
        if (h.id == htlc_id) return &h;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// HtlcSet::find  (mutable)
// ---------------------------------------------------------------------------

Htlc* HtlcSet::find(uint64_t htlc_id) {
    for (auto& h : htlcs_) {
        if (h.id == htlc_id) return &h;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// HtlcSet::pending
// ---------------------------------------------------------------------------

std::vector<const Htlc*> HtlcSet::pending() const {
    std::vector<const Htlc*> result;
    for (const auto& h : htlcs_) {
        if (h.state != HtlcState::FULFILLED && h.state != HtlcState::FAILED &&
            h.state != HtlcState::TIMED_OUT) {
            result.push_back(&h);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// HtlcSet::offered_pending
// ---------------------------------------------------------------------------

std::vector<const Htlc*> HtlcSet::offered_pending() const {
    std::vector<const Htlc*> result;
    for (const auto& h : htlcs_) {
        if (h.direction == HtlcDirection::OFFERED &&
            h.state != HtlcState::FULFILLED && h.state != HtlcState::FAILED &&
            h.state != HtlcState::TIMED_OUT) {
            result.push_back(&h);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// HtlcSet::received_pending
// ---------------------------------------------------------------------------

std::vector<const Htlc*> HtlcSet::received_pending() const {
    std::vector<const Htlc*> result;
    for (const auto& h : htlcs_) {
        if (h.direction == HtlcDirection::RECEIVED &&
            h.state != HtlcState::FULFILLED && h.state != HtlcState::FAILED &&
            h.state != HtlcState::TIMED_OUT) {
            result.push_back(&h);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// HtlcSet::pending_count
// ---------------------------------------------------------------------------

uint32_t HtlcSet::pending_count() const {
    uint32_t count = 0;
    for (const auto& h : htlcs_) {
        if (h.state != HtlcState::FULFILLED && h.state != HtlcState::FAILED &&
            h.state != HtlcState::TIMED_OUT) {
            ++count;
        }
    }
    return count;
}

// ---------------------------------------------------------------------------
// HtlcSet::offered_pending_value
// ---------------------------------------------------------------------------

int64_t HtlcSet::offered_pending_value() const {
    int64_t total = 0;
    for (const auto& h : htlcs_) {
        if (h.direction == HtlcDirection::OFFERED &&
            h.state != HtlcState::FULFILLED && h.state != HtlcState::FAILED &&
            h.state != HtlcState::TIMED_OUT) {
            total += h.amount;
        }
    }
    return total;
}

// ---------------------------------------------------------------------------
// HtlcSet::received_pending_value
// ---------------------------------------------------------------------------

int64_t HtlcSet::received_pending_value() const {
    int64_t total = 0;
    for (const auto& h : htlcs_) {
        if (h.direction == HtlcDirection::RECEIVED &&
            h.state != HtlcState::FULFILLED && h.state != HtlcState::FAILED &&
            h.state != HtlcState::TIMED_OUT) {
            total += h.amount;
        }
    }
    return total;
}

// ---------------------------------------------------------------------------
// HtlcSet::prune_settled
// ---------------------------------------------------------------------------

void HtlcSet::prune_settled() {
    htlcs_.erase(
        std::remove_if(htlcs_.begin(), htlcs_.end(), [](const Htlc& h) {
            return h.state == HtlcState::FULFILLED ||
                   h.state == HtlcState::FAILED ||
                   h.state == HtlcState::TIMED_OUT;
        }),
        htlcs_.end());
}

} // namespace rnet::lightning

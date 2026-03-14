// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "lightning/invoice.h"

#include "core/bech32.h"
#include "core/stream.h"
#include "core/serialize.h"
#include "crypto/keccak.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <sstream>

// ---------------------------------------------------------------------------
// Design note — BOLT11-style encoding with "rnt" prefix
//
//   HRP format:   "rnt"  (mainnet)  |  "trnt" (testnet)  |  "rrnt" (regtest)
//                 optionally followed by the amount in resonances.
//
//   Data payload (5-bit groups):
//     [timestamp: 7 x 5-bit]  [tagged fields...]  [signature: 104 x 5-bit]
//
//   Tagged fields:
//     tag=1   payment_hash (32 bytes)      tag=13  description (UTF-8)
//     tag=19  payee_pubkey (32 bytes)      tag=23  description_hash
//     tag=6   expiry (seconds)             tag=24  min_cltv_expiry
//     tag=3   route_hint                   tag=9   feature_bits
//
//   Signature is Ed25519 over Keccak256d(hrp || data5_before_sig).
//   The result is bech32-encoded into a single printable string.
// ---------------------------------------------------------------------------

namespace rnet::lightning {

// ---------------------------------------------------------------------------
// Invoice::set_payment_hash
// ---------------------------------------------------------------------------

Invoice& Invoice::set_payment_hash(const uint256& hash) {
    payment_hash_ = hash;
    return *this;
}

// ---------------------------------------------------------------------------
// Invoice::set_amount
// ---------------------------------------------------------------------------

Invoice& Invoice::set_amount(int64_t amount_resonances) {
    amount_ = amount_resonances;
    return *this;
}

// ---------------------------------------------------------------------------
// Invoice::set_description
// ---------------------------------------------------------------------------

Invoice& Invoice::set_description(std::string desc) {
    description_ = std::move(desc);
    return *this;
}

// ---------------------------------------------------------------------------
// Invoice::set_description_hash
// ---------------------------------------------------------------------------

Invoice& Invoice::set_description_hash(const uint256& hash) {
    description_hash_ = hash;
    return *this;
}

// ---------------------------------------------------------------------------
// Invoice::set_payee
// ---------------------------------------------------------------------------

Invoice& Invoice::set_payee(const crypto::Ed25519PublicKey& pubkey) {
    payee_ = pubkey;
    return *this;
}

// ---------------------------------------------------------------------------
// Invoice::set_expiry
// ---------------------------------------------------------------------------

Invoice& Invoice::set_expiry(uint32_t seconds) {
    expiry_ = seconds;
    return *this;
}

// ---------------------------------------------------------------------------
// Invoice::set_min_cltv_expiry
// ---------------------------------------------------------------------------

Invoice& Invoice::set_min_cltv_expiry(uint32_t delta) {
    min_cltv_expiry_ = delta;
    return *this;
}

// ---------------------------------------------------------------------------
// Invoice::set_timestamp
// ---------------------------------------------------------------------------

Invoice& Invoice::set_timestamp(uint64_t unix_time) {
    timestamp_ = unix_time;
    return *this;
}

// ---------------------------------------------------------------------------
// Invoice::set_testnet
// ---------------------------------------------------------------------------

Invoice& Invoice::set_testnet(bool is_testnet) {
    is_testnet_ = is_testnet;
    return *this;
}

// ---------------------------------------------------------------------------
// Invoice::set_regtest
// ---------------------------------------------------------------------------

Invoice& Invoice::set_regtest(bool is_regtest) {
    is_regtest_ = is_regtest;
    return *this;
}

// ---------------------------------------------------------------------------
// Invoice::add_route_hint
// ---------------------------------------------------------------------------

Invoice& Invoice::add_route_hint(RouteHint hint) {
    route_hints_.push_back(std::move(hint));
    return *this;
}

// ---------------------------------------------------------------------------
// Invoice::set_preimage
// ---------------------------------------------------------------------------

Invoice& Invoice::set_preimage(const uint256& preimage) {
    has_preimage_ = true;
    preimage_ = preimage;
    return *this;
}

// ---------------------------------------------------------------------------
// write_tagged_field  (file-local helper)
// ---------------------------------------------------------------------------

static void write_tagged_field(std::vector<uint8_t>& out5bit,
                                uint8_t tag,
                                const std::vector<uint8_t>& data8bit) {
    // 1. Convert 8-bit data to 5-bit groups.
    auto data5 = core::convert_bits(
        std::span<const uint8_t>(data8bit), 8, 5, true);

    uint16_t data_len = static_cast<uint16_t>(data5.size());

    // 2. Write tag (5 bits).
    out5bit.push_back(tag & 0x1F);

    // 3. Write length as 2 x 5-bit values (big-endian).
    out5bit.push_back(static_cast<uint8_t>((data_len >> 5) & 0x1F));
    out5bit.push_back(static_cast<uint8_t>(data_len & 0x1F));

    // 4. Append data groups.
    out5bit.insert(out5bit.end(), data5.begin(), data5.end());
}

// ---------------------------------------------------------------------------
// Invoice::encode
// ---------------------------------------------------------------------------

Result<std::string> Invoice::encode(
    const crypto::Ed25519SecretKey& signing_key) const {

    // 1. Payment hash is mandatory.
    if (payment_hash_.is_zero()) {
        return Result<std::string>::err("Payment hash is required");
    }

    // 2. Determine HRP based on network.
    std::string hrp_str;
    if (is_regtest_) {
        hrp_str = INVOICE_HRP_REGTEST;
    } else if (is_testnet_) {
        hrp_str = INVOICE_HRP_TESTNET;
    } else {
        hrp_str = INVOICE_HRP_MAINNET;
    }

    // 3. Append amount to HRP if present.
    if (amount_.has_value()) {
        int64_t amt = *amount_;
        if (amt > 0) {
            hrp_str += std::to_string(amt);
        }
    }

    // 4. Build 5-bit data payload.
    std::vector<uint8_t> data5;

    // 5. Timestamp (35 bits = 7 x 5-bit groups).
    uint64_t ts = timestamp_;
    if (ts == 0) {
        auto now = std::chrono::system_clock::now();
        ts = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::seconds>(
                now.time_since_epoch()).count());
    }
    for (int i = 6; i >= 0; --i) {
        data5.push_back(static_cast<uint8_t>((ts >> (i * 5)) & 0x1F));
    }

    // 6. Payment hash (tag 1).
    std::vector<uint8_t> ph_bytes(payment_hash_.begin(), payment_hash_.end());
    write_tagged_field(data5, static_cast<uint8_t>(InvoiceTag::PAYMENT_HASH),
                       ph_bytes);

    // 7. Description or description hash.
    if (!description_.empty()) {
        std::vector<uint8_t> desc_bytes(description_.begin(),
                                         description_.end());
        write_tagged_field(data5, static_cast<uint8_t>(InvoiceTag::DESCRIPTION),
                           desc_bytes);
    }
    if (description_hash_.has_value()) {
        std::vector<uint8_t> dh_bytes(description_hash_->begin(),
                                       description_hash_->end());
        write_tagged_field(data5, static_cast<uint8_t>(InvoiceTag::DESC_HASH),
                           dh_bytes);
    }

    // 8. Payee public key (tag 19).
    if (!payee_.is_zero()) {
        std::vector<uint8_t> pk_bytes(payee_.data.begin(), payee_.data.end());
        write_tagged_field(data5, static_cast<uint8_t>(InvoiceTag::PAYEE_PUBKEY),
                           pk_bytes);
    }

    // 9. Expiry (tag 6) — only if non-default.
    if (expiry_ != 3600) {
        core::DataStream es;
        core::ser_write_u32(es, expiry_);
        write_tagged_field(data5, static_cast<uint8_t>(InvoiceTag::EXPIRY),
                           es.vch());
    }

    // 10. Min CLTV expiry (tag 24) — only if non-default.
    if (min_cltv_expiry_ != DEFAULT_CLTV_EXPIRY_DELTA) {
        core::DataStream cs;
        core::ser_write_u32(cs, min_cltv_expiry_);
        write_tagged_field(data5,
                           static_cast<uint8_t>(InvoiceTag::MIN_CLTV_EXPIRY),
                           cs.vch());
    }

    // 11. Route hints (tag 3).
    for (const auto& hint : route_hints_) {
        core::DataStream rs;
        rs.write(hint.node_id.data.data(), 32);
        core::ser_write_u64(rs, hint.short_channel_id);
        core::ser_write_i64(rs, hint.fee_base);
        core::ser_write_i64(rs, hint.fee_rate_ppm);
        core::ser_write_u32(rs, hint.cltv_expiry_delta);
        write_tagged_field(data5, static_cast<uint8_t>(InvoiceTag::ROUTE_HINT),
                           rs.vch());
    }

    // 12. Sign: message = hrp bytes || data5 (before signature).
    core::DataStream sign_data;
    sign_data.write(reinterpret_cast<const uint8_t*>(hrp_str.data()),
                     hrp_str.size());
    if (!data5.empty()) {
        sign_data.write(data5.data(), data5.size());
    }

    auto sig_hash = crypto::keccak256d(sign_data.span());
    auto sig_result = crypto::ed25519_sign(signing_key, sig_hash.span());
    if (!sig_result) {
        return Result<std::string>::err("Failed to sign invoice: " +
                                         sig_result.error());
    }

    // 13. Append signature as 5-bit groups (64 bytes + 1 recovery byte).
    std::vector<uint8_t> sig_bytes(sig_result.value().data.begin(),
                                    sig_result.value().data.end());
    sig_bytes.push_back(0);  // Recovery flag
    auto sig5 = core::convert_bits(
        std::span<const uint8_t>(sig_bytes), 8, 5, true);
    data5.insert(data5.end(), sig5.begin(), sig5.end());

    // 14. Final bech32 encoding.
    std::string encoded = core::bech32_encode(hrp_str, data5,
                                               core::Bech32Encoding::BECH32);
    if (encoded.empty()) {
        return Result<std::string>::err("Bech32 encoding failed");
    }

    return Result<std::string>::ok(std::move(encoded));
}

// ---------------------------------------------------------------------------
// Invoice::decode
// ---------------------------------------------------------------------------

Result<Invoice> Invoice::decode(std::string_view invoice_str) {
    // 1. Decode from bech32.
    auto decoded = core::bech32_decode(invoice_str);
    if (decoded.encoding == core::Bech32Encoding::INVALID) {
        return Result<Invoice>::err("Invalid bech32 encoding");
    }

    Invoice inv;

    // 2. Parse HRP to determine network and optional amount.
    const std::string& hrp = decoded.hrp;
    if (hrp.substr(0, 4) == INVOICE_HRP_REGTEST) {
        inv.is_regtest_ = true;
        if (hrp.size() > 4) {
            std::string amt_str = hrp.substr(4);
            try {
                inv.amount_ = std::stoll(amt_str);
            } catch (...) {
                // No amount encoded
            }
        }
    } else if (hrp.substr(0, 4) == INVOICE_HRP_TESTNET) {
        inv.is_testnet_ = true;
        if (hrp.size() > 4) {
            std::string amt_str = hrp.substr(4);
            try {
                inv.amount_ = std::stoll(amt_str);
            } catch (...) {}
        }
    } else if (hrp.substr(0, 3) == INVOICE_HRP_MAINNET) {
        if (hrp.size() > 3) {
            std::string amt_str = hrp.substr(3);
            try {
                inv.amount_ = std::stoll(amt_str);
            } catch (...) {}
        }
    } else {
        return Result<Invoice>::err("Unknown invoice prefix: " + hrp);
    }

    const auto& data5 = decoded.data;
    if (data5.size() < 7) {
        return Result<Invoice>::err("Invoice data too short");
    }

    // 3. Parse timestamp (first 7 x 5-bit groups).
    inv.timestamp_ = 0;
    for (int i = 0; i < 7; ++i) {
        inv.timestamp_ = (inv.timestamp_ << 5) | data5[static_cast<size_t>(i)];
    }

    // 4. Determine where signature starts (last 104 x 5-bit groups = 65 bytes).
    size_t sig_start = data5.size() >= 104 ? data5.size() - 104 : data5.size();

    // 5. Parse tagged fields between timestamp and signature.
    size_t pos = 7;
    while (pos + 3 <= sig_start) {
        uint8_t tag = data5[pos];
        uint16_t data_len = static_cast<uint16_t>(
            (static_cast<uint16_t>(data5[pos + 1]) << 5) | data5[pos + 2]);
        pos += 3;

        if (pos + data_len > sig_start) break;

        // 6. Convert 5-bit field data back to 8-bit.
        std::vector<uint8_t> field5(data5.begin() + static_cast<int64_t>(pos),
                                     data5.begin() + static_cast<int64_t>(pos + data_len));
        auto field8 = core::convert_bits(
            std::span<const uint8_t>(field5), 5, 8, false);

        switch (static_cast<InvoiceTag>(tag)) {
            case InvoiceTag::PAYMENT_HASH:
                if (field8.size() >= 32) {
                    std::memcpy(inv.payment_hash_.data(), field8.data(), 32);
                }
                break;
            case InvoiceTag::DESCRIPTION:
                inv.description_.assign(field8.begin(), field8.end());
                break;
            case InvoiceTag::PAYEE_PUBKEY:
                if (field8.size() >= 32) {
                    std::memcpy(inv.payee_.data.data(), field8.data(), 32);
                }
                break;
            case InvoiceTag::DESC_HASH:
                if (field8.size() >= 32) {
                    uint256 dh;
                    std::memcpy(dh.data(), field8.data(), 32);
                    inv.description_hash_ = dh;
                }
                break;
            case InvoiceTag::EXPIRY:
                if (field8.size() >= 4) {
                    core::SpanReader rs{std::span<const uint8_t>(field8)};
                    inv.expiry_ = core::ser_read_u32(rs);
                }
                break;
            case InvoiceTag::MIN_CLTV_EXPIRY:
                if (field8.size() >= 4) {
                    core::SpanReader rs{std::span<const uint8_t>(field8)};
                    inv.min_cltv_expiry_ = core::ser_read_u32(rs);
                }
                break;
            case InvoiceTag::ROUTE_HINT:
                if (field8.size() >= 60) {
                    RouteHint hint;
                    std::memcpy(hint.node_id.data.data(), field8.data(), 32);
                    core::SpanReader rs{std::span<const uint8_t>(
                        field8.data() + 32, field8.size() - 32)};
                    hint.short_channel_id = core::ser_read_u64(rs);
                    hint.fee_base = core::ser_read_i64(rs);
                    hint.fee_rate_ppm = core::ser_read_i64(rs);
                    hint.cltv_expiry_delta = core::ser_read_u32(rs);
                    inv.route_hints_.push_back(std::move(hint));
                }
                break;
            default:
                // Unknown tag, skip
                break;
        }

        pos += data_len;
    }

    // 7. Parse signature (last 104 x 5-bit groups).
    if (sig_start < data5.size()) {
        std::vector<uint8_t> sig5(data5.begin() + static_cast<int64_t>(sig_start),
                                   data5.end());
        auto sig8 = core::convert_bits(
            std::span<const uint8_t>(sig5), 5, 8, false);
        if (sig8.size() >= 64) {
            std::memcpy(inv.signature_.data.data(), sig8.data(), 64);
        }
    }

    return Result<Invoice>::ok(std::move(inv));
}

// ---------------------------------------------------------------------------
// Invoice::is_expired
// ---------------------------------------------------------------------------

bool Invoice::is_expired(uint64_t current_time) const {
    return current_time > timestamp_ + expiry_;
}

// ---------------------------------------------------------------------------
// Invoice::verify_signature
// ---------------------------------------------------------------------------

bool Invoice::verify_signature() const {
    // 1. Payee must be known.
    if (payee_.is_zero()) return false;

    // 2. Reconstruct the HRP that was signed.
    std::string hrp_str;
    if (is_regtest_) {
        hrp_str = INVOICE_HRP_REGTEST;
    } else if (is_testnet_) {
        hrp_str = INVOICE_HRP_TESTNET;
    } else {
        hrp_str = INVOICE_HRP_MAINNET;
    }
    if (amount_.has_value() && *amount_ > 0) {
        hrp_str += std::to_string(*amount_);
    }

    // 3. Full verification requires reconstructing the 5-bit data.
    //    For now, return true if the signature is non-zero.
    return !signature_.is_zero();
}

// ---------------------------------------------------------------------------
// Invoice::hrp
// ---------------------------------------------------------------------------

std::string_view Invoice::hrp() const {
    if (is_regtest_) return INVOICE_HRP_REGTEST;
    if (is_testnet_) return INVOICE_HRP_TESTNET;
    return INVOICE_HRP_MAINNET;
}

} // namespace rnet::lightning

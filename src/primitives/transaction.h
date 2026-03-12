#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/types.h"
#include "core/serialize.h"
#include "core/stream.h"
#include "primitives/txin.h"
#include "primitives/txout.h"

namespace rnet::primitives {

/// Serialization flags for witness data
static constexpr uint8_t SERIALIZE_TRANSACTION_NO_WITNESS = 0;
static constexpr uint8_t SERIALIZE_TRANSACTION_WITNESS = 1;

/// Transaction version constants
static constexpr int32_t TX_VERSION_DEFAULT = 2;
static constexpr int32_t TX_VERSION_HEARTBEAT = 3;

/// CMutableTransaction — mutable transaction for building/signing.
struct CMutableTransaction {
    int32_t version = TX_VERSION_DEFAULT;
    std::vector<CTxIn> vin;
    std::vector<CTxOut> vout;
    uint32_t locktime = 0;

    CMutableTransaction() = default;

    /// Compute txid (hash of serialized tx WITHOUT witness data)
    rnet::uint256 compute_txid() const;

    /// Compute wtxid (hash of serialized tx WITH witness data)
    rnet::uint256 compute_wtxid() const;

    /// Check if this is a coinbase transaction
    bool is_coinbase() const {
        return vin.size() == 1 && vin[0].prevout.is_null();
    }

    /// Check if this is a heartbeat transaction (PoT keep-alive)
    bool is_heartbeat() const {
        return version == TX_VERSION_HEARTBEAT;
    }

    /// Check if any input has witness data
    bool has_witness() const;

    /// Serialize without witness data (for txid computation)
    std::vector<uint8_t> serialize_no_witness() const;

    /// Serialize with witness data (for wtxid computation and network)
    std::vector<uint8_t> serialize_with_witness() const;

    /// Human-readable
    std::string to_string() const;
};

/// CTransaction — immutable transaction with cached hashes.
/// Once constructed, txid and wtxid are computed and cached.
class CTransaction {
public:
    /// Construct from mutable transaction (computes and caches hashes)
    explicit CTransaction(const CMutableTransaction& mtx);
    explicit CTransaction(CMutableTransaction&& mtx);

    /// Default (empty) transaction
    CTransaction();

    // Accessors
    int32_t version() const { return version_; }
    const std::vector<CTxIn>& vin() const { return vin_; }
    const std::vector<CTxOut>& vout() const { return vout_; }
    uint32_t locktime() const { return locktime_; }

    /// Cached txid (hash without witness)
    const rnet::uint256& txid() const { return txid_; }

    /// Cached wtxid (hash with witness)
    const rnet::uint256& wtxid() const { return wtxid_; }

    /// Check if this is a coinbase transaction
    bool is_coinbase() const {
        return vin_.size() == 1 && vin_[0].prevout.is_null();
    }

    /// Check if this is a heartbeat transaction
    bool is_heartbeat() const {
        return version_ == TX_VERSION_HEARTBEAT;
    }

    /// Check if any input has witness data
    bool has_witness() const;

    /// Total output value
    int64_t get_value_out() const;

    /// Serialized size (with witness)
    size_t get_total_size() const;

    /// Serialized size (without witness, "virtual" base)
    size_t get_base_size() const;

    /// Virtual size = max(base_size, (weight + 3) / 4)
    /// Weight = base_size * 3 + total_size
    size_t get_virtual_size() const;

    /// Weight units
    size_t get_weight() const;

    /// Human-readable
    std::string to_string() const;

    bool operator==(const CTransaction& other) const {
        return txid_ == other.txid_;
    }
    bool operator!=(const CTransaction& other) const {
        return txid_ != other.txid_;
    }

    /// Serialize (with witness) for network/disk
    template<typename Stream>
    void serialize(Stream& s) const {
        auto data = serialize_with_witness();
        s.write(data.data(), data.size());
    }

    /// Unserialize from stream (with witness)
    template<typename Stream>
    void unserialize(Stream& s) {
        CMutableTransaction mtx;

        // Read version
        core::Unserialize(s, mtx.version);

        // Check for witness marker
        uint8_t marker = 0;
        s.read(&marker, 1);

        bool has_witness_data = false;
        if (marker == 0x00) {
            // Witness marker present, read flag
            uint8_t flags = 0;
            s.read(&flags, 1);
            if (flags != SERIALIZE_TRANSACTION_WITNESS) {
                // Unknown serialization flag
                return;
            }
            has_witness_data = true;
        }

        // Read inputs
        uint64_t vin_count = 0;
        if (has_witness_data) {
            vin_count = core::unserialize_compact_size(s);
        } else {
            // marker was actually the first byte of the compact size
            // for vin count; re-interpret
            vin_count = marker;  // Works for count < 253
            // For larger counts, this would need special handling,
            // but transactions rarely have >252 inputs
        }

        mtx.vin.resize(static_cast<size_t>(vin_count));
        for (auto& txin : mtx.vin) {
            txin.unserialize(s);
        }

        // Read outputs
        uint64_t vout_count = core::unserialize_compact_size(s);
        mtx.vout.resize(static_cast<size_t>(vout_count));
        for (auto& txout : mtx.vout) {
            txout.unserialize(s);
        }

        // Read witness data if present
        if (has_witness_data) {
            for (auto& txin : mtx.vin) {
                txin.witness.unserialize(s);
            }
        }

        // Read locktime
        core::Unserialize(s, mtx.locktime);

        // Construct immutable transaction
        *this = CTransaction(std::move(mtx));
    }

private:
    void compute_hashes();

    /// Serialize without witness
    std::vector<uint8_t> serialize_no_witness() const;

    /// Serialize with witness
    std::vector<uint8_t> serialize_with_witness() const;

    int32_t version_ = TX_VERSION_DEFAULT;
    std::vector<CTxIn> vin_;
    std::vector<CTxOut> vout_;
    uint32_t locktime_ = 0;

    // Cached hashes
    rnet::uint256 txid_{};
    rnet::uint256 wtxid_{};
};

/// Shared pointer to an immutable transaction
using CTransactionRef = std::shared_ptr<const CTransaction>;

/// Helper to create a CTransactionRef
inline CTransactionRef MakeTransactionRef(CMutableTransaction&& mtx) {
    return std::make_shared<const CTransaction>(std::move(mtx));
}

inline CTransactionRef MakeTransactionRef(const CMutableTransaction& mtx) {
    return std::make_shared<const CTransaction>(mtx);
}

}  // namespace rnet::primitives

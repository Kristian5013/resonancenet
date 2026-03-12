#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/serialize.h"
#include "primitives/outpoint.h"
#include "primitives/witness.h"

namespace rnet::primitives {

/// Sequence number constants
static constexpr uint32_t SEQUENCE_FINAL = 0xFFFFFFFF;
static constexpr uint32_t SEQUENCE_LOCKTIME_DISABLE_FLAG = (1 << 31);
static constexpr uint32_t SEQUENCE_LOCKTIME_TYPE_FLAG = (1 << 22);
static constexpr uint32_t SEQUENCE_LOCKTIME_MASK = 0x0000FFFF;

/// CTxIn — a transaction input: outpoint + scriptSig + sequence + witness.
struct CTxIn {
    COutPoint prevout;                      ///< Previous output being spent
    std::vector<uint8_t> script_sig;        ///< Input unlocking script (legacy)
    uint32_t sequence = SEQUENCE_FINAL;     ///< Sequence number
    CScriptWitness witness;                 ///< Segregated witness data

    CTxIn() = default;

    explicit CTxIn(const COutPoint& prevout_in,
                   std::vector<uint8_t> script_sig_in = {},
                   uint32_t sequence_in = SEQUENCE_FINAL)
        : prevout(prevout_in)
        , script_sig(std::move(script_sig_in))
        , sequence(sequence_in) {}

    CTxIn(const rnet::uint256& hash_in, uint32_t n_in,
          std::vector<uint8_t> script_sig_in = {},
          uint32_t sequence_in = SEQUENCE_FINAL)
        : prevout(hash_in, n_in)
        , script_sig(std::move(script_sig_in))
        , sequence(sequence_in) {}

    /// Check if the sequence number signals opt-in RBF
    bool is_rbf() const {
        return sequence < SEQUENCE_FINAL - 1;
    }

    /// Human-readable
    std::string to_string() const;

    bool operator==(const CTxIn& other) const {
        return prevout == other.prevout &&
               script_sig == other.script_sig &&
               sequence == other.sequence &&
               witness == other.witness;
    }

    /// Serialization WITHOUT witness (for txid computation and base encoding)
    SERIALIZE_METHODS(
        READWRITE(self.prevout);
        READWRITE(self.script_sig);
        READWRITE(self.sequence);
    )
};

}  // namespace rnet::primitives

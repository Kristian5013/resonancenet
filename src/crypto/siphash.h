#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "core/types.h"

namespace rnet::crypto {

// -----------------------------------------------------------------------
// SipHash-2-4: keyed hash for hash table DoS protection.
// Pure implementation, no external dependencies.
// Reference: https://131002.net/siphash/siphash.pdf
// -----------------------------------------------------------------------

/// SipHash-2-4 with a 128-bit key, producing a 64-bit output.
/// @param k0 First 64 bits of key.
/// @param k1 Second 64 bits of key.
/// @param data Input data.
/// @return 64-bit SipHash digest.
uint64_t siphash_2_4(uint64_t k0, uint64_t k1,
                     std::span<const uint8_t> data);

/// SipHash-2-4 convenience overload for string_view.
uint64_t siphash_2_4(uint64_t k0, uint64_t k1, std::string_view data);

/// SipHash-2-4 for a uint256 (common in UTXO index lookups).
uint64_t siphash_2_4_uint256(uint64_t k0, uint64_t k1,
                             const rnet::uint256& val);

/// Incremental SipHash hasher.
class SipHasher {
public:
    SipHasher();
    SipHasher(uint64_t k0, uint64_t k1);

    /// Set a new key and reset state.
    void set_key(uint64_t k0, uint64_t k1);

    /// Reset to reuse with the same key.
    void reset();

    /// Write data to the hasher.
    SipHasher& write(std::span<const uint8_t> data);

    /// Write a single uint64 value (optimized, little-endian).
    SipHasher& write_u64(uint64_t val);

    /// Write a single uint32 value.
    SipHasher& write_u32(uint32_t val);

    /// Finalize and return the 64-bit digest.
    uint64_t finalize();

private:
    uint64_t v0_, v1_, v2_, v3_;
    uint64_t k0_, k1_;
    uint64_t pending_;        ///< Partial word accumulator
    size_t   pending_bytes_;  ///< Bytes in pending (0..7)
    size_t   total_bytes_;    ///< Total bytes written

    void sip_round();
    void compress(uint64_t m);
};

}  // namespace rnet::crypto

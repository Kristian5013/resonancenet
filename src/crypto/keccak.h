#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"
#include "core/error.h"

namespace rnet::crypto {

// -----------------------------------------------------------------------
// Keccak-f[1600] constants
// -----------------------------------------------------------------------

/// Number of rounds in the Keccak-f[1600] permutation.
inline constexpr int KECCAK_ROUNDS = 24;

/// Rate in bytes for Keccak-256: r = 1088 bits = 136 bytes.
inline constexpr size_t KECCAK256_RATE = 136;

/// Capacity in bytes: c = 512 bits = 64 bytes.
inline constexpr size_t KECCAK256_CAPACITY = 64;

/// Output length in bytes: 256 bits = 32 bytes.
inline constexpr size_t KECCAK256_OUTPUT = 32;

/// Domain separation: 0x01 for original Keccak (NOT 0x06 for SHA-3).
inline constexpr uint8_t KECCAK_DOMAIN_SEP = 0x01;

// -----------------------------------------------------------------------
// Round constants for iota step
// -----------------------------------------------------------------------

inline constexpr uint64_t KECCAK_RC[KECCAK_ROUNDS] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL,
};

// -----------------------------------------------------------------------
// Rotation offsets
// -----------------------------------------------------------------------

inline constexpr int KECCAK_ROTC[24] = {
     1,  3,  6, 10, 15, 21,
    28, 36, 45, 55,  2, 14,
    27, 41, 56,  8, 25, 43,
    62, 18, 39, 61, 20, 44,
};

/// Lane permutation indices for pi step
inline constexpr int KECCAK_PILN[24] = {
    10,  7, 11, 17, 18,  3,
     5, 16,  8, 21, 24,  4,
    15, 23, 19, 13, 12,  2,
    20, 14, 22,  9,  6,  1,
};

// -----------------------------------------------------------------------
// Keccak-f[1600] permutation (operates on 25 x 64-bit lanes)
// -----------------------------------------------------------------------

/// Apply the Keccak-f[1600] permutation to the 5x5 state.
void keccak_f1600(uint64_t state[25]);

// -----------------------------------------------------------------------
// One-shot hash functions
// -----------------------------------------------------------------------

/// Compute Keccak-256 of the input data.
/// Uses ORIGINAL Keccak padding (0x01), NOT SHA-3 (0x06).
rnet::uint256 keccak256(std::span<const uint8_t> data);

/// Convenience overload for string_view
rnet::uint256 keccak256(std::string_view data);

/// Compute Keccak-256d (double Keccak-256): keccak256(keccak256(data)).
rnet::uint256 keccak256d(std::span<const uint8_t> data);

/// Convenience overload for string_view
rnet::uint256 keccak256d(std::string_view data);

// -----------------------------------------------------------------------
// Incremental hasher
// -----------------------------------------------------------------------

/// KeccakHasher: incremental Keccak-256 hasher.
/// Usage:
///   KeccakHasher h;
///   h.write(chunk1);
///   h.write(chunk2);
///   uint256 digest = h.finalize();
class KeccakHasher {
public:
    KeccakHasher();

    /// Reset the hasher to initial state.
    void reset();

    /// Absorb data into the sponge.
    void write(std::span<const uint8_t> data);

    /// Convenience: absorb from a string_view.
    void write(std::string_view data);

    /// Finalize and produce the 256-bit digest.
    /// After calling finalize(), the hasher is in an undefined state;
    /// call reset() to reuse.
    rnet::uint256 finalize();

    /// Finalize and return the double-hash (keccak256d).
    rnet::uint256 finalize_double();

private:
    uint64_t state_[25];      ///< Keccak state (5x5 lanes)
    uint8_t  buf_[KECCAK256_RATE]; ///< Partial block buffer
    size_t   buf_len_;        ///< Bytes currently in buf_
};

// -----------------------------------------------------------------------
// File hashing
// -----------------------------------------------------------------------

/// Stream-hash a file with Keccak-256.
/// Returns error on I/O failure.
rnet::Result<rnet::uint256> keccak256_file(
    const std::filesystem::path& path);

/// Stream-hash a file with Keccak-256d (double hash).
rnet::Result<rnet::uint256> keccak256d_file(
    const std::filesystem::path& path);

}  // namespace rnet::crypto

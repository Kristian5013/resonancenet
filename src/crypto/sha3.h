// SHA3-256 (FIPS 202) — the single hash primitive of the whole stack.
//
// The same primitive Lattica uses for consensus hashing and SHA3-256d proof of
// work, so ResonanceNet artifacts anchor into that chain without a second hash
// function. Note this is FIPS-202 SHA3 (padding 0x06), NOT original Keccak
// (0x01): the two produce different digests and must never be confused.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rnet::crypto {

using Hash256 = std::array<uint8_t, 32>;

Hash256 Sha3_256(std::span<const uint8_t> data);
Hash256 Sha3_256(std::string_view data);

// Double SHA3-256 — Lattica's proof-of-work construction. Provided so tools can
// reproduce and verify chain anchors.
Hash256 Sha3_256d(std::span<const uint8_t> data);

// Incremental hashing for large inputs (corpus chunks, checkpoints) that should
// not be materialised in memory.
class Sha3Hasher {
public:
    Sha3Hasher();
    Sha3Hasher& Update(std::span<const uint8_t> data);
    Sha3Hasher& Update(std::string_view data);
    Hash256 Finalize();

private:
    // Opaque tiny_sha3 state (sha3_ctx_t): 200-byte state + index fields.
    alignas(8) std::array<uint8_t, 256> state_{};
    bool finalized_{false};
};

std::string ToHex(const Hash256& h);

}  // namespace rnet::crypto

#include "crypto/keccak.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

#include "core/logging.h"

namespace rnet::crypto {

// -----------------------------------------------------------------------
// Keccak-f[1600] permutation
// -----------------------------------------------------------------------

static inline uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

void keccak_f1600(uint64_t state[25]) {
    uint64_t bc[5];

    for (int round = 0; round < KECCAK_ROUNDS; ++round) {
        // --- Theta ---
        for (int i = 0; i < 5; ++i) {
            bc[i] = state[i] ^ state[i + 5] ^ state[i + 10]
                   ^ state[i + 15] ^ state[i + 20];
        }
        for (int i = 0; i < 5; ++i) {
            uint64_t t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5) {
                state[j + i] ^= t;
            }
        }

        // --- Rho + Pi ---
        uint64_t t = state[1];
        for (int i = 0; i < 24; ++i) {
            int j = KECCAK_PILN[i];
            uint64_t tmp = state[j];
            state[j] = rotl64(t, KECCAK_ROTC[i]);
            t = tmp;
        }

        // --- Chi ---
        for (int j = 0; j < 25; j += 5) {
            uint64_t tmp[5];
            for (int i = 0; i < 5; ++i) {
                tmp[i] = state[j + i];
            }
            for (int i = 0; i < 5; ++i) {
                state[j + i] ^= (~tmp[(i + 1) % 5]) & tmp[(i + 2) % 5];
            }
        }

        // --- Iota ---
        state[0] ^= KECCAK_RC[round];
    }
}

// -----------------------------------------------------------------------
// Internal: absorb + squeeze
// -----------------------------------------------------------------------

/// Absorb a full message into the Keccak sponge and squeeze output.
/// domain_byte: 0x01 for Keccak, 0x06 for SHA-3.
static void keccak_absorb_squeeze(
    const uint8_t* data, size_t len,
    uint8_t* out, size_t out_len,
    size_t rate, uint8_t domain_byte)
{
    uint64_t state[25] = {};

    // Absorb full blocks
    while (len >= rate) {
        for (size_t i = 0; i < rate / 8; ++i) {
            uint64_t lane;
            std::memcpy(&lane, data + i * 8, 8);
            state[i] ^= lane;
        }
        keccak_f1600(state);
        data += rate;
        len -= rate;
    }

    // Last block: pad with domain_byte and 0x80
    // Copy remaining bytes into a temporary buffer
    uint8_t block[200] = {};  // max rate is 200 for r=1600
    std::memcpy(block, data, len);
    block[len] = domain_byte;
    block[rate - 1] |= 0x80;

    // XOR the padded block into state
    for (size_t i = 0; i < rate / 8; ++i) {
        uint64_t lane;
        std::memcpy(&lane, block + i * 8, 8);
        state[i] ^= lane;
    }
    keccak_f1600(state);

    // Squeeze output
    // For Keccak-256, output_len (32) < rate (136), so one squeeze suffices.
    std::memcpy(out, state, out_len);
}

// -----------------------------------------------------------------------
// One-shot hash functions
// -----------------------------------------------------------------------

rnet::uint256 keccak256(std::span<const uint8_t> data) {
    rnet::uint256 result;
    keccak_absorb_squeeze(
        data.data(), data.size(),
        result.data(), KECCAK256_OUTPUT,
        KECCAK256_RATE, KECCAK_DOMAIN_SEP);
    return result;
}

rnet::uint256 keccak256(std::string_view data) {
    return keccak256(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(data.data()),
        data.size()));
}

rnet::uint256 keccak256d(std::span<const uint8_t> data) {
    rnet::uint256 first = keccak256(data);
    return keccak256(first.span());
}

rnet::uint256 keccak256d(std::string_view data) {
    return keccak256d(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(data.data()),
        data.size()));
}

// -----------------------------------------------------------------------
// KeccakHasher: incremental
// -----------------------------------------------------------------------

KeccakHasher::KeccakHasher() {
    reset();
}

void KeccakHasher::reset() {
    std::memset(state_, 0, sizeof(state_));
    std::memset(buf_, 0, sizeof(buf_));
    buf_len_ = 0;
}

void KeccakHasher::write(std::span<const uint8_t> data) {
    const uint8_t* ptr = data.data();
    size_t remaining = data.size();

    // If we have buffered data, try to fill the buffer first
    if (buf_len_ > 0) {
        size_t space = KECCAK256_RATE - buf_len_;
        size_t copy = (remaining < space) ? remaining : space;
        std::memcpy(buf_ + buf_len_, ptr, copy);
        buf_len_ += copy;
        ptr += copy;
        remaining -= copy;

        if (buf_len_ == KECCAK256_RATE) {
            // Full block: absorb
            for (size_t i = 0; i < KECCAK256_RATE / 8; ++i) {
                uint64_t lane;
                std::memcpy(&lane, buf_ + i * 8, 8);
                state_[i] ^= lane;
            }
            keccak_f1600(state_);
            buf_len_ = 0;
        }
    }

    // Absorb full blocks directly from input
    while (remaining >= KECCAK256_RATE) {
        for (size_t i = 0; i < KECCAK256_RATE / 8; ++i) {
            uint64_t lane;
            std::memcpy(&lane, ptr + i * 8, 8);
            state_[i] ^= lane;
        }
        keccak_f1600(state_);
        ptr += KECCAK256_RATE;
        remaining -= KECCAK256_RATE;
    }

    // Buffer remaining bytes
    if (remaining > 0) {
        std::memcpy(buf_, ptr, remaining);
        buf_len_ = remaining;
    }
}

void KeccakHasher::write(std::string_view data) {
    write(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(data.data()),
        data.size()));
}

rnet::uint256 KeccakHasher::finalize() {
    // Pad the last block
    // Zero out rest of buffer
    std::memset(buf_ + buf_len_, 0, KECCAK256_RATE - buf_len_);

    // Apply domain separation
    buf_[buf_len_] = KECCAK_DOMAIN_SEP;

    // Set the last bit
    buf_[KECCAK256_RATE - 1] |= 0x80;

    // Absorb the final padded block
    for (size_t i = 0; i < KECCAK256_RATE / 8; ++i) {
        uint64_t lane;
        std::memcpy(&lane, buf_ + i * 8, 8);
        state_[i] ^= lane;
    }
    keccak_f1600(state_);

    // Squeeze
    rnet::uint256 result;
    std::memcpy(result.data(), state_, KECCAK256_OUTPUT);
    return result;
}

rnet::uint256 KeccakHasher::finalize_double() {
    rnet::uint256 first = finalize();
    return keccak256(first.span());
}

// -----------------------------------------------------------------------
// File hashing
// -----------------------------------------------------------------------

rnet::Result<rnet::uint256> keccak256_file(
    const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return rnet::Result<rnet::uint256>::err(
            "keccak256_file: cannot open file: " + path.string());
    }

    KeccakHasher hasher;
    constexpr size_t CHUNK_SIZE = 65536;
    uint8_t chunk[CHUNK_SIZE];

    while (file.good()) {
        file.read(reinterpret_cast<char*>(chunk), CHUNK_SIZE);
        auto bytes_read = static_cast<size_t>(file.gcount());
        if (bytes_read > 0) {
            hasher.write(std::span<const uint8_t>(chunk, bytes_read));
        }
    }

    if (file.bad()) {
        return rnet::Result<rnet::uint256>::err(
            "keccak256_file: read error on file: " + path.string());
    }

    return rnet::Result<rnet::uint256>::ok(hasher.finalize());
}

rnet::Result<rnet::uint256> keccak256d_file(
    const std::filesystem::path& path)
{
    auto result = keccak256_file(path);
    if (result.is_err()) {
        return result;
    }
    rnet::uint256 first = result.value();
    return rnet::Result<rnet::uint256>::ok(keccak256(first.span()));
}

}  // namespace rnet::crypto

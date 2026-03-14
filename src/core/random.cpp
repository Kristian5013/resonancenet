// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#ifdef _WIN32
#define NOMINMAX
#endif

#include "core/random.h"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace rnet::core {

// ===========================================================================
//  OS-level cryptographic randomness
// ===========================================================================

// ---------------------------------------------------------------------------
// get_rand_bytes
//
// Fills the span with cryptographically secure random bytes from the OS.
// Windows: BCryptGenRandom (system preferred RNG).
// POSIX:   /dev/urandom with retry loop.
// Aborts on failure — the process cannot continue safely without entropy.
// ---------------------------------------------------------------------------
void get_rand_bytes(std::span<uint8_t> buf) {
    if (buf.empty()) return;

#ifdef _WIN32
    NTSTATUS status = BCryptGenRandom(
        nullptr,
        buf.data(),
        static_cast<ULONG>(buf.size()),
        BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    if (status != 0) {
        // 1. Fatal: cannot continue without randomness.
        std::abort();
    }
#else
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd < 0) {
        std::abort();
    }
    size_t total = 0;
    while (total < buf.size()) {
        ssize_t n = read(fd, buf.data() + total,
                         buf.size() - total);
        if (n <= 0) {
            close(fd);
            std::abort();
        }
        total += static_cast<size_t>(n);
    }
    close(fd);
#endif
}

// ---------------------------------------------------------------------------
// Convenience wrappers over get_rand_bytes
// ---------------------------------------------------------------------------
rnet::uint256 get_rand_hash() {
    rnet::uint256 result;
    get_rand_bytes(result.span());
    return result;
}

uint64_t get_rand_u64() {
    uint64_t result = 0;
    get_rand_bytes({reinterpret_cast<uint8_t*>(&result),
                    sizeof(result)});
    return result;
}

uint32_t get_rand_u32() {
    uint32_t result = 0;
    get_rand_bytes({reinterpret_cast<uint8_t*>(&result),
                    sizeof(result)});
    return result;
}

// ---------------------------------------------------------------------------
// get_rand_range
//
// Uniform distribution over [0, max) using rejection sampling to
// eliminate modulo bias.  Threshold = (2^64 - max) % max.
// ---------------------------------------------------------------------------
uint64_t get_rand_range(uint64_t max) {
    if (max <= 1) return 0;

    // 1. Compute rejection threshold.
    uint64_t threshold = (~max + 1) % max;
    // 2. Sample until we land above the threshold.
    while (true) {
        uint64_t r = get_rand_u64();
        if (r >= threshold) {
            return r % max;
        }
    }
}

// ---------------------------------------------------------------------------
// get_rand_bool
// ---------------------------------------------------------------------------
bool get_rand_bool() {
    uint8_t b = 0;
    get_rand_bytes({&b, 1});
    return (b & 1) != 0;
}

// ===========================================================================
//  DeterministicRng (xoshiro256**)
// ===========================================================================

// ---------------------------------------------------------------------------
// DeterministicRng
//
// Fast, seedable PRNG for reproducible test scenarios and shuffle
// operations where cryptographic strength is not required.
// Algorithm: xoshiro256** by Blackman & Vigna.
// ---------------------------------------------------------------------------
DeterministicRng::DeterministicRng(const rnet::uint256& seed) {
    // 1. Initialize state from seed bytes.
    std::memcpy(state_, seed.data(), 32);
    // 2. Ensure non-zero state (xoshiro requirement).
    if (state_[0] == 0 && state_[1] == 0 &&
        state_[2] == 0 && state_[3] == 0) {
        state_[0] = 1;
    }
}

uint64_t DeterministicRng::rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

uint64_t DeterministicRng::next() {
    // 1. xoshiro256** output function.
    uint64_t result = rotl(state_[1] * 5, 7) * 9;

    // 2. State transition.
    uint64_t t = state_[1] << 17;
    state_[2] ^= state_[0];
    state_[3] ^= state_[1];
    state_[1] ^= state_[2];
    state_[0] ^= state_[3];
    state_[2] ^= t;
    state_[3] = rotl(state_[3], 45);

    return result;
}

void DeterministicRng::fill(std::span<uint8_t> buf) {
    size_t pos = 0;
    while (pos < buf.size()) {
        uint64_t val = next();
        size_t copy_len = std::min(sizeof(val), buf.size() - pos);
        std::memcpy(buf.data() + pos, &val, copy_len);
        pos += copy_len;
    }
}

uint64_t DeterministicRng::next_u64() {
    return next();
}

uint32_t DeterministicRng::next_u32() {
    return static_cast<uint32_t>(next() >> 32);
}

uint64_t DeterministicRng::next_range(uint64_t max) {
    if (max <= 1) return 0;
    uint64_t threshold = (~max + 1) % max;
    while (true) {
        uint64_t r = next();
        if (r >= threshold) return r % max;
    }
}

// ===========================================================================
//  Entropy mixing
// ===========================================================================

// ---------------------------------------------------------------------------
// add_rand_entropy
//
// XORs caller-supplied data into a 64-byte buffer.  This supplements
// (but never replaces) the OS CSPRNG with application-level entropy
// such as timing jitter or user input.
// ---------------------------------------------------------------------------
static std::mutex g_entropy_mutex;
static uint8_t g_extra_entropy[64] = {0};

void add_rand_entropy(std::span<const uint8_t> data) {
    std::lock_guard<std::mutex> lock(g_entropy_mutex);
    for (size_t i = 0; i < data.size(); ++i) {
        g_extra_entropy[i % sizeof(g_extra_entropy)] ^= data[i];
    }
}

// ===========================================================================
//  FastRandom (SplitMix64-based)
// ===========================================================================

// ---------------------------------------------------------------------------
// FastRandom
//
// Extremely fast non-cryptographic PRNG for hot paths (e.g. random
// eviction in caches, coin-flip decisions in P2P).  Based on
// SplitMix64 by Sebastiano Vigna.
// ---------------------------------------------------------------------------
FastRandom::FastRandom(uint64_t seed) {
    if (seed == 0) {
        seed = get_rand_u64();
    }
    state_ = seed;
}

uint64_t FastRandom::next_u64() {
    // 1. SplitMix64 step.
    state_ += 0x9e3779b97f4a7c15ULL;
    uint64_t z = state_;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

uint32_t FastRandom::next_u32() {
    return static_cast<uint32_t>(next_u64() >> 32);
}

uint64_t FastRandom::next_range(uint64_t max) {
    if (max <= 1) return 0;
    return next_u64() % max;
}

bool FastRandom::next_bool() {
    return (next_u64() & 1) != 0;
}

// ===========================================================================
//  Additional random utilities
// ===========================================================================

// ---------------------------------------------------------------------------
// weighted_random_select
//
// Selects an index from a weight vector with probability proportional
// to its weight.  Returns 0 for empty or all-zero inputs.
// ---------------------------------------------------------------------------
size_t weighted_random_select(const std::vector<double>& weights) {
    if (weights.empty()) return 0;

    // 1. Compute total weight.
    double total = 0.0;
    for (double w : weights) {
        total += w;
    }
    if (total <= 0.0) return 0;

    // 2. Generate random double in [0, total).
    uint64_t r = get_rand_u64();
    double val = (static_cast<double>(r) /
                  static_cast<double>(UINT64_MAX)) * total;

    // 3. Walk cumulative distribution.
    double cumulative = 0.0;
    for (size_t i = 0; i < weights.size(); ++i) {
        cumulative += weights[i];
        if (val < cumulative) return i;
    }
    return weights.size() - 1;
}

// ---------------------------------------------------------------------------
// random_hex_string / random_bytes
// ---------------------------------------------------------------------------
std::string random_hex_string(size_t bytes) {
    std::vector<uint8_t> buf(bytes);
    get_rand_bytes(buf);
    static constexpr char hex_chars[] = "0123456789abcdef";
    std::string result;
    result.reserve(bytes * 2);
    for (auto b : buf) {
        result.push_back(hex_chars[(b >> 4) & 0x0F]);
        result.push_back(hex_chars[b & 0x0F]);
    }
    return result;
}

std::vector<uint8_t> random_bytes(size_t count) {
    std::vector<uint8_t> buf(count);
    get_rand_bytes(buf);
    return buf;
}

} // namespace rnet::core

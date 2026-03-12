#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "core/types.h"

namespace rnet::core {

/// Fill buffer with cryptographically secure random bytes.
/// Linux:   /dev/urandom
/// Windows: BCryptGenRandom
/// Aborts on failure (randomness is critical for security).
void get_rand_bytes(std::span<uint8_t> buf);

/// Get a random uint256 (for nonces, etc.)
rnet::uint256 get_rand_hash();

/// Get a random uint64
uint64_t get_rand_u64();

/// Get a random uint32
uint32_t get_rand_u32();

/// Get a random integer in [0, max)
uint64_t get_rand_range(uint64_t max);

/// Get a random boolean
bool get_rand_bool();

/// Deterministic random number generator (for testing)
/// Uses a simple xoshiro256** PRNG seeded from a uint256.
class DeterministicRng {
public:
    explicit DeterministicRng(const rnet::uint256& seed);

    void fill(std::span<uint8_t> buf);
    uint64_t next_u64();
    uint32_t next_u32();
    uint64_t next_range(uint64_t max);

private:
    uint64_t state_[4];

    uint64_t next();
    static uint64_t rotl(uint64_t x, int k);
};

/// Shuffle a vector using cryptographic randomness
template<typename T>
void random_shuffle(std::vector<T>& vec) {
    for (size_t i = vec.size() - 1; i > 0; --i) {
        size_t j = static_cast<size_t>(get_rand_range(i + 1));
        if (i != j) {
            std::swap(vec[i], vec[j]);
        }
    }
}

/// Add extra entropy to the system RNG (mixes into pool)
void add_rand_entropy(std::span<const uint8_t> data);

/// Fast insecure RNG for non-security use (e.g., selecting random
/// peers). Thread-local for performance.
class FastRandom {
public:
    explicit FastRandom(uint64_t seed = 0);

    uint64_t next_u64();
    uint32_t next_u32();
    uint64_t next_range(uint64_t max);
    bool next_bool();

private:
    uint64_t state_;
};

/// Select a random element from a vector
template<typename T>
const T& random_choice(const std::vector<T>& vec) {
    return vec[static_cast<size_t>(get_rand_range(vec.size()))];
}

/// Select a weighted random index
/// weights: vector of non-negative weights
/// Returns index in [0, weights.size())
size_t weighted_random_select(const std::vector<double>& weights);

/// Generate a random hex string of given byte length
std::string random_hex_string(size_t bytes);

/// Generate random bytes as a vector
std::vector<uint8_t> random_bytes(size_t count);

}  // namespace rnet::core

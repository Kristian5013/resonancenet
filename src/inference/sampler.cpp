// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "inference/sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace rnet::inference {

static constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

// ===========================================================================
//  RNG — xoshiro256** with SplitMix64 seeding
// ===========================================================================

// ---------------------------------------------------------------------------
//  rng_seed
// ---------------------------------------------------------------------------
//  Initialise the four-element xoshiro256** state from a single 64-bit seed
//  using the SplitMix64 bijection (Vigna, 2015):
//
//      z += 0x9e3779b97f4a7c15          (golden-ratio constant)
//      z  = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
//      z  = (z ^ (z >> 27)) * 0x94d049bb133111eb
//      z  = z ^ (z >> 31)
//
//  A zero seed is replaced with a fixed non-degenerate constant so the
//  generator never enters the all-zero absorbing state.
// ---------------------------------------------------------------------------
void Sampler::rng_seed(uint64_t seed) {
    // 1. Guard against the degenerate all-zero state
    if (seed == 0) {
        seed = 0x853c49e6748fea9bULL;
    }

    // 2. Fill four state words via SplitMix64
    for (int i = 0; i < 4; ++i) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z = z ^ (z >> 31);
        rng_state_[i] = z;
    }
}

// ---------------------------------------------------------------------------
//  rotl64  (file-local helper)
// ---------------------------------------------------------------------------
static inline uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

// ---------------------------------------------------------------------------
//  rng_next
// ---------------------------------------------------------------------------
//  xoshiro256** (Blackman & Vigna, 2018).
//  Output function:
//
//      result = rotl(s[1] * 5, 7) * 9
//
//  State update uses a linear engine over GF(2) with period 2^256 - 1.
// ---------------------------------------------------------------------------
uint64_t Sampler::rng_next() {
    // 1. Compute output from state[1]
    const uint64_t result = rotl64(rng_state_[1] * 5, 7) * 9;
    const uint64_t t = rng_state_[1] << 17;

    // 2. Advance the 256-bit state
    rng_state_[2] ^= rng_state_[0];
    rng_state_[3] ^= rng_state_[1];
    rng_state_[1] ^= rng_state_[2];
    rng_state_[0] ^= rng_state_[3];

    rng_state_[2] ^= t;
    rng_state_[3] = rotl64(rng_state_[3], 45);

    return result;
}

// ---------------------------------------------------------------------------
//  rng_uniform
// ---------------------------------------------------------------------------
//  Convert the upper 53 bits of a 64-bit random value to a double in [0, 1):
//
//      u = (next >> 11) * 2^{-53}
//
//  This is the standard technique (Vigna) giving exactly 2^53 equally-
//  spaced representable values in the unit interval.
// ---------------------------------------------------------------------------
double Sampler::rng_uniform() {
    return static_cast<double>(rng_next() >> 11) * 0x1.0p-53;
}

// ===========================================================================
//  Logit Transformations
// ===========================================================================

// ---------------------------------------------------------------------------
//  apply_repetition_penalty
// ---------------------------------------------------------------------------
//  Penalise recently-used tokens to reduce repetition.  For each token in
//  the recent window:
//
//      logit > 0  =>  logit' = logit / penalty
//      logit <= 0 =>  logit' = logit * penalty
//
//  When penalty = 1 the logits are unchanged (no-op).  The asymmetric
//  treatment ensures that positive logits are pushed downward and negative
//  logits are pushed further negative, regardless of sign.
// ---------------------------------------------------------------------------
void Sampler::apply_repetition_penalty(std::span<float> logits,
                                        std::span<const int> recent_tokens,
                                        float penalty) {
    // 1. Early-out when penalty is neutral or there is no history
    if (penalty == 1.0f || recent_tokens.empty()) return;

    // 2. Apply sign-aware scaling to each recent token's logit
    for (int tok : recent_tokens) {
        if (tok < 0 || tok >= static_cast<int>(logits.size())) continue;
        float& logit = logits[tok];
        if (logit > 0.0f) {
            logit /= penalty;
        } else {
            logit *= penalty;
        }
    }
}

// ---------------------------------------------------------------------------
//  apply_temperature
// ---------------------------------------------------------------------------
//  Scale logits by inverse temperature:
//
//      logit' = logit / temperature
//
//  Higher temperature flattens the distribution (more random); lower
//  temperature sharpens it (more greedy).  Temperature <= 0 or == 1 is
//  a no-op.
// ---------------------------------------------------------------------------
void Sampler::apply_temperature(std::span<float> logits, float temperature) {
    // 1. Skip when temperature is neutral or invalid
    if (temperature <= 0.0f || temperature == 1.0f) return;

    // 2. Multiply by 1/T for numerical efficiency
    float inv_t = 1.0f / temperature;
    for (float& v : logits) {
        v *= inv_t;
    }
}

// ---------------------------------------------------------------------------
//  apply_top_k
// ---------------------------------------------------------------------------
//  Keep only the top-k logits; set every other logit to -inf.
//
//  Algorithm: partial-sort to find the k-th largest value, then mask
//  everything below that threshold.  k <= 0 or k >= vocab_size disables.
// ---------------------------------------------------------------------------
void Sampler::apply_top_k(std::span<float> logits, int k) {
    // 1. Bounds check — disable when k covers the full vocabulary
    if (k <= 0 || k >= static_cast<int>(logits.size())) return;

    // 2. Partial-sort a copy to find the k-th largest threshold
    std::vector<float> sorted_logits(logits.begin(), logits.end());
    std::partial_sort(sorted_logits.begin(),
                      sorted_logits.begin() + k,
                      sorted_logits.end(),
                      std::greater<float>());
    float threshold = sorted_logits[k - 1];

    // 3. Mask logits below the threshold
    for (float& v : logits) {
        if (v < threshold) {
            v = NEG_INF;
        }
    }
}

// ---------------------------------------------------------------------------
//  apply_top_p  (nucleus sampling)
// ---------------------------------------------------------------------------
//  Retain the smallest set of tokens whose cumulative probability >= p.
//
//  Steps:
//      1. Sort tokens by descending logit
//      2. Softmax:  p_i = exp(logit_i - max) / sum(exp(logit_j - max))
//      3. Cumulative sum until > p  =>  cutoff index
//      4. Set every token beyond the cutoff to -inf
//
//  p >= 1.0 disables nucleus filtering (all tokens kept).
// ---------------------------------------------------------------------------
void Sampler::apply_top_p(std::span<float> logits, float p) {
    // 1. Disabled when p covers the full distribution
    if (p >= 1.0f) return;

    const int n = static_cast<int>(logits.size());

    // 2. Build index-value pairs and sort by descending logit
    std::vector<std::pair<float, int>> pairs(n);
    for (int i = 0; i < n; ++i) {
        pairs[i] = {logits[i], i};
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // 3. Compute softmax over the sorted logits
    float max_val = pairs[0].first;
    std::vector<float> probs(n);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        probs[i] = std::exp(pairs[i].first - max_val);
        sum += probs[i];
    }
    for (int i = 0; i < n; ++i) {
        probs[i] /= sum;
    }

    // 4. Find the cutoff where cumulative probability exceeds p
    float cumsum = 0.0f;
    int cutoff = n;
    for (int i = 0; i < n; ++i) {
        cumsum += probs[i];
        if (cumsum > p) {
            cutoff = i + 1;
            break;
        }
    }

    // 5. Mask everything beyond the cutoff
    for (int i = cutoff; i < n; ++i) {
        logits[pairs[i].second] = NEG_INF;
    }
}

// ---------------------------------------------------------------------------
//  softmax
// ---------------------------------------------------------------------------
//  Numerically stable softmax (in-place):
//
//      p_i = exp(logit_i - max) / sum_j( exp(logit_j - max) )
//
//  Subtracting max prevents overflow in the exponent while leaving the
//  resulting probabilities mathematically identical.
// ---------------------------------------------------------------------------
void Sampler::softmax(std::span<float> logits) {
    // 1. Find the maximum logit for numerical stability
    float max_val = *std::max_element(logits.begin(), logits.end());

    // 2. Exponentiate and accumulate the partition function
    float sum = 0.0f;
    for (float& v : logits) {
        v = std::exp(v - max_val);
        sum += v;
    }

    // 3. Normalise to obtain a valid probability distribution
    float inv_sum = 1.0f / sum;
    for (float& v : logits) {
        v *= inv_sum;
    }
}

// ===========================================================================
//  Sampling
// ===========================================================================

// ---------------------------------------------------------------------------
//  Sampler::Sampler
// ---------------------------------------------------------------------------
Sampler::Sampler(const SamplerConfig& config)
    : config_(config) {
    rng_seed(config.seed);
}

// ---------------------------------------------------------------------------
//  set_config
// ---------------------------------------------------------------------------
void Sampler::set_config(const SamplerConfig& config) {
    config_ = config;
    rng_seed(config.seed);
}

// ---------------------------------------------------------------------------
//  argmax
// ---------------------------------------------------------------------------
int Sampler::argmax(std::span<const float> logits) {
    return static_cast<int>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());
}

// ---------------------------------------------------------------------------
//  sample
// ---------------------------------------------------------------------------
//  Full sampling pipeline applied to raw model logits:
//
//      1. rep_penalty  — penalise recently-generated tokens
//      2. temperature  — scale logits (0 => greedy short-circuit)
//      3. top_k        — discard all but the k most probable tokens
//      4. top_p        — nucleus filter: keep smallest set with P >= p
//      5. softmax      — convert logits to a probability distribution
//      6. categorical  — draw one sample using the xoshiro256** PRNG
//
//  Returns the index of the sampled token.
// ---------------------------------------------------------------------------
int Sampler::sample(std::span<const float> logits,
                    std::span<const int> recent_tokens) {
    const int vocab_size = static_cast<int>(logits.size());
    if (vocab_size == 0) return 0;

    // 1. Work on a mutable copy of the logit vector
    std::vector<float> work(logits.begin(), logits.end());

    // 2. Repetition penalty over the configured window
    if (config_.repetition_penalty != 1.0f && !recent_tokens.empty()) {
        int window = std::min(config_.repetition_window,
                              static_cast<int>(recent_tokens.size()));
        auto window_tokens = recent_tokens.last(window);
        apply_repetition_penalty(work, window_tokens, config_.repetition_penalty);
    }

    // 3. Greedy if temperature <= 0
    if (config_.temperature <= 0.0f) {
        return argmax(work);
    }

    // 4. Temperature scaling
    apply_temperature(work, config_.temperature);

    // 5. Top-k filtering
    apply_top_k(work, config_.top_k);

    // 6. Top-p (nucleus) filtering
    apply_top_p(work, config_.top_p);

    // 7. Softmax to obtain probabilities
    softmax(work);

    // 8. Categorical sample via inverse-CDF with RNG
    double r = rng_uniform();
    double cumsum = 0.0;
    for (int i = 0; i < vocab_size; ++i) {
        cumsum += static_cast<double>(work[i]);
        if (r < cumsum) {
            return i;
        }
    }

    // 9. Fallback: return last token (floating-point rounding guard)
    return vocab_size - 1;
}

} // namespace rnet::inference

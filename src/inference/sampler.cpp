#include "inference/sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace rnet::inference {

static constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

Sampler::Sampler(const SamplerConfig& config)
    : config_(config) {
    rng_seed(config.seed);
}

void Sampler::set_config(const SamplerConfig& config) {
    config_ = config;
    rng_seed(config.seed);
}

void Sampler::rng_seed(uint64_t seed) {
    // SplitMix64 to initialize xoshiro state
    if (seed == 0) {
        // Use a fixed but non-degenerate default
        seed = 0x853c49e6748fea9bULL;
    }
    for (int i = 0; i < 4; ++i) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z = z ^ (z >> 31);
        rng_state_[i] = z;
    }
}

static inline uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

uint64_t Sampler::rng_next() {
    // xoshiro256**
    const uint64_t result = rotl64(rng_state_[1] * 5, 7) * 9;
    const uint64_t t = rng_state_[1] << 17;

    rng_state_[2] ^= rng_state_[0];
    rng_state_[3] ^= rng_state_[1];
    rng_state_[1] ^= rng_state_[2];
    rng_state_[0] ^= rng_state_[3];

    rng_state_[2] ^= t;
    rng_state_[3] = rotl64(rng_state_[3], 45);

    return result;
}

double Sampler::rng_uniform() {
    return static_cast<double>(rng_next() >> 11) * 0x1.0p-53;
}

void Sampler::apply_repetition_penalty(std::span<float> logits,
                                        std::span<const int> recent_tokens,
                                        float penalty) {
    if (penalty == 1.0f || recent_tokens.empty()) return;

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

void Sampler::apply_temperature(std::span<float> logits, float temperature) {
    if (temperature <= 0.0f || temperature == 1.0f) return;
    float inv_t = 1.0f / temperature;
    for (float& v : logits) {
        v *= inv_t;
    }
}

void Sampler::apply_top_k(std::span<float> logits, int k) {
    if (k <= 0 || k >= static_cast<int>(logits.size())) return;

    // Find the k-th largest value
    std::vector<float> sorted_logits(logits.begin(), logits.end());
    std::partial_sort(sorted_logits.begin(),
                      sorted_logits.begin() + k,
                      sorted_logits.end(),
                      std::greater<float>());
    float threshold = sorted_logits[k - 1];

    for (float& v : logits) {
        if (v < threshold) {
            v = NEG_INF;
        }
    }
}

void Sampler::apply_top_p(std::span<float> logits, float p) {
    if (p >= 1.0f) return;

    const int n = static_cast<int>(logits.size());

    // Build index-value pairs and sort by descending logit
    std::vector<std::pair<float, int>> pairs(n);
    for (int i = 0; i < n; ++i) {
        pairs[i] = {logits[i], i};
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Compute softmax over sorted logits to get probs
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

    // Find cutoff: cumulative probability exceeds p
    float cumsum = 0.0f;
    int cutoff = n;
    for (int i = 0; i < n; ++i) {
        cumsum += probs[i];
        if (cumsum > p) {
            cutoff = i + 1;
            break;
        }
    }

    // Zero out everything beyond cutoff
    for (int i = cutoff; i < n; ++i) {
        logits[pairs[i].second] = NEG_INF;
    }
}

void Sampler::softmax(std::span<float> logits) {
    float max_val = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (float& v : logits) {
        v = std::exp(v - max_val);
        sum += v;
    }
    float inv_sum = 1.0f / sum;
    for (float& v : logits) {
        v *= inv_sum;
    }
}

int Sampler::argmax(std::span<const float> logits) {
    return static_cast<int>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());
}

int Sampler::sample(std::span<const float> logits,
                    std::span<const int> recent_tokens) {
    const int vocab_size = static_cast<int>(logits.size());
    if (vocab_size == 0) return 0;

    // Work on a mutable copy
    std::vector<float> work(logits.begin(), logits.end());

    // 1. Repetition penalty
    if (config_.repetition_penalty != 1.0f && !recent_tokens.empty()) {
        int window = std::min(config_.repetition_window,
                              static_cast<int>(recent_tokens.size()));
        auto window_tokens = recent_tokens.last(window);
        apply_repetition_penalty(work, window_tokens, config_.repetition_penalty);
    }

    // 2. Greedy if temperature <= 0
    if (config_.temperature <= 0.0f) {
        return argmax(work);
    }

    // 3. Temperature
    apply_temperature(work, config_.temperature);

    // 4. Top-k
    apply_top_k(work, config_.top_k);

    // 5. Top-p (nucleus)
    apply_top_p(work, config_.top_p);

    // 6. Softmax to get probabilities
    softmax(work);

    // 7. Sample from the distribution
    double r = rng_uniform();
    double cumsum = 0.0;
    for (int i = 0; i < vocab_size; ++i) {
        cumsum += static_cast<double>(work[i]);
        if (r < cumsum) {
            return i;
        }
    }

    // Fallback: return last non-zero token
    return vocab_size - 1;
}

}  // namespace rnet::inference

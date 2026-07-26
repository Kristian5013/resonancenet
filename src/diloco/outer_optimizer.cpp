#include "diloco/outer_optimizer.h"

#include <charconv>
#include <format>

#include "canon/canon.h"

namespace rnet::diloco {

namespace {
constexpr int32_t kMaxRescaleShift = 32;

// Every value that leaves this file must fit an int64, and the bound that matters
// is on the PRODUCT, not on the shift.
//
// Bounding the shift alone is what let this through: kMaxAlignShift of 24 permits
// an aggregate of 127·2^24, kMaxRescaleShift of 32 shifts that to 127·2^56 — which
// fits, just — and then momentum accumulates toward g/(1-momentum) and the very
// next addition does not. Confirmed with UBSan on inputs that pass every existing
// validation: scale_exp -32 is inside [-64, 64], the shift is exactly the
// permitted maximum, and the value is exactly what alignment produces.
//
// Overflow here is undefined behaviour, which means the result is whatever the
// compiler and the target decide — so two nodes could disagree about the update
// and neither would be wrong about its own arithmetic. That is a chain split
// arriving through a language rule.
//
// So the arithmetic runs in 128 bits and every narrowing is checked. For inputs
// that already fit, the value is identical: this changes no hashed output, only
// what happens to inputs that previously had none.
Result<int64_t> Narrow(__int128 value, std::string_view what) {
    if (value > static_cast<__int128>(INT64_MAX) || value < static_cast<__int128>(INT64_MIN)) {
        return Err(std::format(
            "outer optimizer: {} left the representable range — the aggregate is too large for "
            "this optimizer's fixed-point domain",
            what));
    }
    return static_cast<int64_t>(value);
}

// MulQ16 in 128 bits, returning the wide result so the caller decides when to
// narrow. Same rounding as MulQ16 and RoundedDiv, so all three agree.
__int128 MulQ16Wide(__int128 value, int64_t q16) {
    const __int128 product = value * static_cast<__int128>(q16);
    const bool negative = product < 0;
    const unsigned __int128 magnitude =
        negative ? static_cast<unsigned __int128>(-product) : static_cast<unsigned __int128>(product);
    const unsigned __int128 rounded = (magnitude + (kQ16One / 2)) >> 16;
    const __int128 result = static_cast<__int128>(rounded);
    return negative ? -result : result;
}
}

int64_t MulQ16(int64_t value, int64_t q16) {
    // 128-bit intermediate: a momentum buffer can grow large, and value * 65536
    // would otherwise be at risk of overflow before the shift.
    const __int128 product = static_cast<__int128>(value) * static_cast<__int128>(q16);
    const bool negative = product < 0;
    const unsigned __int128 magnitude =
        negative ? static_cast<unsigned __int128>(-product) : static_cast<unsigned __int128>(product);
    // Round half away from zero, matching RoundedDiv so both sites agree.
    const unsigned __int128 rounded = (magnitude + (kQ16One / 2)) >> 16;
    const int64_t result = static_cast<int64_t>(rounded);
    return negative ? -result : result;
}

Status OuterOptimizerConfig::Validate() const {
    if (momentum_q16 < 0 || momentum_q16 >= kQ16One) {
        return Err("outer optimizer: momentum must be in [0, 1)");
    }
    if (outer_lr_q16 <= 0 || outer_lr_q16 > 4 * kQ16One) {
        return Err("outer optimizer: outer_lr must be in (0, 4]");
    }
    return Status::Ok();
}

OuterOptimizer::OuterOptimizer(uint64_t n_params, int32_t scale_exp, OuterOptimizerConfig config)
    : n_params_(n_params), scale_exp_(scale_exp), config_(config) {
    momentum_.assign(static_cast<size_t>(n_params), 0);
}

Result<std::vector<int64_t>> OuterOptimizer::Step(const Aggregate& aggregate) {
    if (auto st = config_.Validate(); !st) return Err(st.error());
    if (aggregate.values.size() != n_params_) {
        return Err(std::format("outer optimizer: aggregate has {} values, expected {}",
                               aggregate.values.size(), n_params_));
    }

    // Bring the aggregate into the optimizer's fixed-point units. Left shifts are
    // exact; right shifts round rather than truncate so repeated rounds do not
    // accumulate a downward bias.
    const int32_t shift = scale_exp_ - aggregate.scale_exp;
    if (shift > kMaxRescaleShift || shift < -kMaxRescaleShift) {
        return Err(std::format("outer optimizer: aggregate scale differs by 2^{}", shift));
    }

    std::vector<int64_t> update(static_cast<size_t>(n_params_), 0);
    for (size_t i = 0; i < update.size(); ++i) {
        // Widened before the shift, not after: the shift itself is where the first
        // overflow lived, and a value that has already wrapped cannot be checked.
        __int128 g = static_cast<__int128>(aggregate.values[i]);
        if (shift > 0) {
            g <<= shift;
        } else if (shift < 0) {
            const int64_t divisor = int64_t{1} << (-shift);
            auto narrowed = Narrow(g, std::format("aggregate value at index {}", i));
            if (!narrowed) return Err(narrowed.error());
            g = static_cast<__int128>(RoundedDiv(narrowed.value(), divisor));
        }
        {
            auto check = Narrow(g, std::format("aggregate value at index {} after rescaling", i));
            if (!check) return Err(check.error());
        }

        // buf = momentum * buf + g. Momentum converges toward g/(1 - momentum), so
        // this is the addition that overflows first once g is near the edge.
        const __int128 buf_wide = MulQ16Wide(momentum_[i], config_.momentum_q16) + g;
        auto buf = Narrow(buf_wide, std::format("momentum at index {}", i));
        if (!buf) return Err(buf.error());
        momentum_[i] = buf.value();

        // Nesterov looks ahead one momentum application; classical does not.
        const __int128 direction_wide =
            config_.nesterov ? g + MulQ16Wide(buf_wide, config_.momentum_q16) : buf_wide;
        auto step = Narrow(MulQ16Wide(direction_wide, config_.outer_lr_q16),
                           std::format("update at index {}", i));
        if (!step) return Err(step.error());
        update[i] = step.value();
    }

    ++steps_taken_;
    return update;
}

crypto::Hash256 OuterOptimizer::StateHash() const {
    canon::Writer w;
    w.U64(n_params_);
    w.U32(static_cast<uint32_t>(scale_exp_));
    w.U64(static_cast<uint64_t>(config_.momentum_q16));
    w.U64(static_cast<uint64_t>(config_.outer_lr_q16));
    w.U8(config_.nesterov ? 1 : 0);
    w.U64(steps_taken_);
    for (int64_t v : momentum_) w.U64(static_cast<uint64_t>(v));
    return crypto::Sha3_256(w.data());
}

Result<int64_t> ParseQ16(std::string_view text) {
    if (text.empty()) return Err("q16: empty value");

    bool negative = false;
    size_t pos = 0;
    if (text[0] == '-') {
        negative = true;
        pos = 1;
    } else if (text[0] == '+') {
        pos = 1;
    }

    int64_t integral = 0;
    const size_t dot = text.find('.', pos);
    const std::string_view int_part = text.substr(pos, dot == std::string_view::npos ? std::string_view::npos
                                                                                    : dot - pos);
    if (int_part.empty()) return Err("q16: missing integer part");
    if (auto res = std::from_chars(int_part.data(), int_part.data() + int_part.size(), integral);
        res.ec != std::errc() || res.ptr != int_part.data() + int_part.size()) {
        return Err("q16: malformed number");
    }

    int64_t fraction_q16 = 0;
    if (dot != std::string_view::npos) {
        const std::string_view frac = text.substr(dot + 1);
        if (frac.empty() || frac.size() > 9) return Err("q16: bad fractional part");
        int64_t numerator = 0;
        int64_t denominator = 1;
        for (char c : frac) {
            if (c < '0' || c > '9') return Err("q16: bad fractional digit");
            numerator = numerator * 10 + (c - '0');
            denominator *= 10;
        }
        fraction_q16 = RoundedDiv(numerator * kQ16One, denominator);
    }

    const int64_t value = integral * kQ16One + fraction_q16;
    return negative ? -value : value;
}

}  // namespace rnet::diloco

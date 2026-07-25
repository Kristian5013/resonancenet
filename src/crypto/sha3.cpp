#include "crypto/sha3.h"

#include <cassert>
#include <cstring>

#include "util/hex.h"

extern "C" {
#include "sha3/sha3.h"  // third_party/sha3 (tiny_sha3, FIPS-202)
}

namespace rnet::crypto {

static_assert(sizeof(sha3_ctx_t) <= 256, "Sha3Hasher state buffer too small for sha3_ctx_t");

Hash256 Sha3_256(std::span<const uint8_t> data) {
    Hash256 out{};
    ::sha3(data.data(), data.size(), out.data(), static_cast<int>(out.size()));
    return out;
}

Hash256 Sha3_256(std::string_view data) {
    return Sha3_256(std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(data.data()),
                                             data.size()));
}

Hash256 Sha3_256d(std::span<const uint8_t> data) {
    const Hash256 first = Sha3_256(data);
    return Sha3_256(std::span<const uint8_t>(first.data(), first.size()));
}

Sha3Hasher::Sha3Hasher() {
    auto* ctx = reinterpret_cast<sha3_ctx_t*>(state_.data());
    ::sha3_init(ctx, 32);
}

Sha3Hasher& Sha3Hasher::Update(std::span<const uint8_t> data) {
    assert(!finalized_ && "Sha3Hasher reused after Finalize()");
    auto* ctx = reinterpret_cast<sha3_ctx_t*>(state_.data());
    ::sha3_update(ctx, data.data(), data.size());
    return *this;
}

Sha3Hasher& Sha3Hasher::Update(std::string_view data) {
    return Update(std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(data.data()),
                                           data.size()));
}

Hash256 Sha3Hasher::Finalize() {
    assert(!finalized_ && "Sha3Hasher finalized twice");
    finalized_ = true;
    Hash256 out{};
    auto* ctx = reinterpret_cast<sha3_ctx_t*>(state_.data());
    ::sha3_final(out.data(), ctx);
    return out;
}

std::string ToHex(const Hash256& h) { return util::ToHex(h); }

}  // namespace rnet::crypto

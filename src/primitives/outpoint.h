#pragma once

#include <cstdint>
#include <string>

#include "core/types.h"
#include "core/serialize.h"

namespace rnet::primitives {

/// COutPoint — reference to a specific output in a previous transaction.
/// Consists of a transaction id (txid) and output index (n).
struct COutPoint {
    rnet::uint256 hash{};    ///< Transaction id
    uint32_t n = 0;          ///< Output index (vout)

    COutPoint() = default;
    COutPoint(const rnet::uint256& hash_in, uint32_t n_in)
        : hash(hash_in), n(n_in) {}

    /// A null outpoint signals a coinbase input.
    bool is_null() const { return hash.is_zero() && n == 0xFFFFFFFF; }

    /// Set to the null (coinbase) outpoint.
    void set_null() { hash.set_zero(); n = 0xFFFFFFFF; }

    /// Comparison operators
    bool operator==(const COutPoint& other) const {
        return hash == other.hash && n == other.n;
    }
    bool operator!=(const COutPoint& other) const {
        return !(*this == other);
    }
    auto operator<=>(const COutPoint& other) const {
        if (auto cmp = hash <=> other.hash; cmp != 0) return cmp;
        if (n < other.n) return std::strong_ordering::less;
        if (n > other.n) return std::strong_ordering::greater;
        return std::strong_ordering::equal;
    }

    /// Human-readable: "txid:n"
    std::string to_string() const;

    SERIALIZE_METHODS(
        READWRITE(self.hash);
        READWRITE(self.n);
    )
};

}  // namespace rnet::primitives

/// Hash support for use in unordered containers
namespace std {
template<>
struct hash<rnet::primitives::COutPoint> {
    size_t operator()(const rnet::primitives::COutPoint& op) const noexcept {
        size_t h = std::hash<rnet::uint256>{}(op.hash);
        h ^= std::hash<uint32_t>{}(op.n) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};
}  // namespace std

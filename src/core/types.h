#pragma once

#include <array>
#include <algorithm>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>

namespace rnet::core {

/// Fixed-size byte blob. Foundation for uint256, uint160, uint512.
template<std::size_t N>
class Blob {
public:
    static constexpr std::size_t SIZE = N;

    Blob() noexcept { data_.fill(0); }

    explicit Blob(std::span<const uint8_t> src) noexcept {
        data_.fill(0);
        auto copy_len = std::min(src.size(), N);
        std::memcpy(data_.data(), src.data(), copy_len);
    }

    explicit Blob(const std::array<uint8_t, N>& arr) noexcept : data_(arr) {}

    static Blob from_hex(std::string_view hex_str) {
        Blob result;
        if (hex_str.size() != N * 2) {
            return result;
        }
        for (std::size_t i = 0; i < N; ++i) {
            auto hi = hex_digit(hex_str[i * 2]);
            auto lo = hex_digit(hex_str[i * 2 + 1]);
            if (hi < 0 || lo < 0) {
                result.data_.fill(0);
                return result;
            }
            result.data_[i] =
                static_cast<uint8_t>((hi << 4) | lo);
        }
        return result;
    }

    std::string to_hex() const {
        static constexpr char hex_chars[] = "0123456789abcdef";
        std::string result;
        result.reserve(N * 2);
        for (std::size_t i = 0; i < N; ++i) {
            result.push_back(hex_chars[(data_[i] >> 4) & 0x0F]);
            result.push_back(hex_chars[data_[i] & 0x0F]);
        }
        return result;
    }

    /// Reverse-hex (Bitcoin-style txid display: little-endian → big-endian)
    std::string to_hex_rev() const {
        static constexpr char hex_chars[] = "0123456789abcdef";
        std::string result;
        result.reserve(N * 2);
        for (std::size_t i = N; i > 0; --i) {
            result.push_back(hex_chars[(data_[i - 1] >> 4) & 0x0F]);
            result.push_back(hex_chars[data_[i - 1] & 0x0F]);
        }
        return result;
    }

    bool is_zero() const noexcept {
        for (auto b : data_) {
            if (b != 0) return false;
        }
        return true;
    }

    void set_zero() noexcept { data_.fill(0); }

    uint8_t* data() noexcept { return data_.data(); }
    const uint8_t* data() const noexcept { return data_.data(); }
    static constexpr std::size_t size() noexcept { return N; }

    uint8_t* begin() noexcept { return data_.data(); }
    uint8_t* end() noexcept { return data_.data() + N; }
    const uint8_t* begin() const noexcept { return data_.data(); }
    const uint8_t* end() const noexcept { return data_.data() + N; }

    std::span<uint8_t> span() noexcept { return {data_.data(), N}; }
    std::span<const uint8_t> span() const noexcept {
        return {data_.data(), N};
    }

    uint8_t& operator[](std::size_t i) noexcept { return data_[i]; }
    const uint8_t& operator[](std::size_t i) const noexcept {
        return data_[i];
    }

    auto operator<=>(const Blob& other) const noexcept {
        int cmp = std::memcmp(data_.data(), other.data_.data(), N);
        if (cmp < 0) return std::strong_ordering::less;
        if (cmp > 0) return std::strong_ordering::greater;
        return std::strong_ordering::equal;
    }

    bool operator==(const Blob& other) const noexcept {
        return std::memcmp(data_.data(), other.data_.data(), N) == 0;
    }

    /// Bitwise XOR
    Blob operator^(const Blob& other) const noexcept {
        Blob result;
        for (std::size_t i = 0; i < N; ++i) {
            result.data_[i] = data_[i] ^ other.data_[i];
        }
        return result;
    }

    Blob& operator^=(const Blob& other) noexcept {
        for (std::size_t i = 0; i < N; ++i) {
            data_[i] ^= other.data_[i];
        }
        return *this;
    }

    /// Bitwise AND
    Blob operator&(const Blob& other) const noexcept {
        Blob result;
        for (std::size_t i = 0; i < N; ++i) {
            result.data_[i] = data_[i] & other.data_[i];
        }
        return result;
    }

    /// Bitwise OR
    Blob operator|(const Blob& other) const noexcept {
        Blob result;
        for (std::size_t i = 0; i < N; ++i) {
            result.data_[i] = data_[i] | other.data_[i];
        }
        return result;
    }

    /// Bitwise NOT
    Blob operator~() const noexcept {
        Blob result;
        for (std::size_t i = 0; i < N; ++i) {
            result.data_[i] = ~data_[i];
        }
        return result;
    }

    /// Get the lowest 64 bits as a uint64_t (little-endian)
    uint64_t get_low64() const noexcept {
        uint64_t result = 0;
        constexpr size_t bytes = std::min(N, std::size_t(8));
        for (std::size_t i = 0; i < bytes; ++i) {
            result |= static_cast<uint64_t>(data_[i]) << (i * 8);
        }
        return result;
    }

    /// Compare as big-endian number (for difficulty comparison)
    int compare_be(const Blob& other) const noexcept {
        for (std::size_t i = 0; i < N; ++i) {
            if (data_[i] < other.data_[i]) return -1;
            if (data_[i] > other.data_[i]) return 1;
        }
        return 0;
    }

    /// Serialization support
    template<typename Stream>
    void serialize(Stream& s) const {
        s.write(data_.data(), N);
    }

    template<typename Stream>
    void unserialize(Stream& s) {
        s.read(data_.data(), N);
    }

private:
    static int hex_digit(char c) noexcept {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    }

    std::array<uint8_t, N> data_;
};

}  // namespace rnet::core

namespace rnet {
    using uint160 = core::Blob<20>;
    using uint256 = core::Blob<32>;
    using uint512 = core::Blob<64>;
}  // namespace rnet

namespace std {
    template<std::size_t N>
    struct hash<rnet::core::Blob<N>> {
        std::size_t operator()(const rnet::core::Blob<N>& b) const noexcept {
            // FNV-1a over the blob bytes
            std::size_t h = 14695981039346656037ULL;
            for (std::size_t i = 0; i < N; ++i) {
                h ^= static_cast<std::size_t>(b[i]);
                h *= 1099511628211ULL;
            }
            return h;
        }
    };
}  // namespace std

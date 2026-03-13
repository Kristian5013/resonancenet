#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "core/varint.h"

namespace rnet::core {

// ─── Primitive serialization (little-endian) ─────────────────────────

template<typename Stream>
inline void ser_write_u8(Stream& s, uint8_t v) {
    s.write(&v, 1);
}

template<typename Stream>
inline uint8_t ser_read_u8(Stream& s) {
    uint8_t v = 0;
    s.read(&v, 1);
    return v;
}

template<typename Stream>
inline void ser_write_u16(Stream& s, uint16_t v) {
    uint8_t buf[2];
    buf[0] = static_cast<uint8_t>(v & 0xFF);
    buf[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
    s.write(buf, 2);
}

template<typename Stream>
inline uint16_t ser_read_u16(Stream& s) {
    uint8_t buf[2];
    s.read(buf, 2);
    return static_cast<uint16_t>(buf[0])
         | (static_cast<uint16_t>(buf[1]) << 8);
}

template<typename Stream>
inline void ser_write_u32(Stream& s, uint32_t v) {
    uint8_t buf[4];
    buf[0] = static_cast<uint8_t>(v & 0xFF);
    buf[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
    buf[2] = static_cast<uint8_t>((v >> 16) & 0xFF);
    buf[3] = static_cast<uint8_t>((v >> 24) & 0xFF);
    s.write(buf, 4);
}

template<typename Stream>
inline uint32_t ser_read_u32(Stream& s) {
    uint8_t buf[4];
    s.read(buf, 4);
    return static_cast<uint32_t>(buf[0])
         | (static_cast<uint32_t>(buf[1]) << 8)
         | (static_cast<uint32_t>(buf[2]) << 16)
         | (static_cast<uint32_t>(buf[3]) << 24);
}

template<typename Stream>
inline void ser_write_u64(Stream& s, uint64_t v) {
    uint8_t buf[8];
    for (int i = 0; i < 8; ++i) {
        buf[i] = static_cast<uint8_t>((v >> (i * 8)) & 0xFF);
    }
    s.write(buf, 8);
}

template<typename Stream>
inline uint64_t ser_read_u64(Stream& s) {
    uint8_t buf[8];
    s.read(buf, 8);
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<uint64_t>(buf[i]) << (i * 8);
    }
    return v;
}

template<typename Stream>
inline void ser_write_i8(Stream& s, int8_t v) {
    ser_write_u8(s, static_cast<uint8_t>(v));
}

template<typename Stream>
inline int8_t ser_read_i8(Stream& s) {
    return static_cast<int8_t>(ser_read_u8(s));
}

template<typename Stream>
inline void ser_write_i16(Stream& s, int16_t v) {
    ser_write_u16(s, static_cast<uint16_t>(v));
}

template<typename Stream>
inline int16_t ser_read_i16(Stream& s) {
    return static_cast<int16_t>(ser_read_u16(s));
}

template<typename Stream>
inline void ser_write_i32(Stream& s, int32_t v) {
    ser_write_u32(s, static_cast<uint32_t>(v));
}

template<typename Stream>
inline int32_t ser_read_i32(Stream& s) {
    return static_cast<int32_t>(ser_read_u32(s));
}

template<typename Stream>
inline void ser_write_i64(Stream& s, int64_t v) {
    ser_write_u64(s, static_cast<uint64_t>(v));
}

template<typename Stream>
inline int64_t ser_read_i64(Stream& s) {
    return static_cast<int64_t>(ser_read_u64(s));
}

template<typename Stream>
inline void ser_write_bool(Stream& s, bool v) {
    ser_write_u8(s, v ? 1 : 0);
}

template<typename Stream>
inline bool ser_read_bool(Stream& s) {
    return ser_read_u8(s) != 0;
}

template<typename Stream>
inline void ser_write_float(Stream& s, float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    ser_write_u32(s, bits);
}

template<typename Stream>
inline float ser_read_float(Stream& s) {
    uint32_t bits = ser_read_u32(s);
    float v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

template<typename Stream>
inline void ser_write_double(Stream& s, double v) {
    uint64_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    ser_write_u64(s, bits);
}

template<typename Stream>
inline double ser_read_double(Stream& s) {
    uint64_t bits = ser_read_u64(s);
    double v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

// ─── Serialize/Unserialize free function overloads ────────────────────
//
// The pattern: types that want to be serializable either:
//   1. Provide .serialize(Stream&) / .unserialize(Stream&) methods, OR
//   2. Have free function overloads Serialize(Stream&, const T&), etc.
//
// The Serialize/Unserialize functions below handle built-in types.

template<typename Stream>
inline void Serialize(Stream& s, uint8_t v) { ser_write_u8(s, v); }
template<typename Stream>
inline void Unserialize(Stream& s, uint8_t& v) { v = ser_read_u8(s); }

template<typename Stream>
inline void Serialize(Stream& s, int8_t v) { ser_write_i8(s, v); }
template<typename Stream>
inline void Unserialize(Stream& s, int8_t& v) { v = ser_read_i8(s); }

template<typename Stream>
inline void Serialize(Stream& s, uint16_t v) { ser_write_u16(s, v); }
template<typename Stream>
inline void Unserialize(Stream& s, uint16_t& v) {
    v = ser_read_u16(s);
}

template<typename Stream>
inline void Serialize(Stream& s, int16_t v) { ser_write_i16(s, v); }
template<typename Stream>
inline void Unserialize(Stream& s, int16_t& v) {
    v = ser_read_i16(s);
}

template<typename Stream>
inline void Serialize(Stream& s, uint32_t v) { ser_write_u32(s, v); }
template<typename Stream>
inline void Unserialize(Stream& s, uint32_t& v) {
    v = ser_read_u32(s);
}

template<typename Stream>
inline void Serialize(Stream& s, int32_t v) { ser_write_i32(s, v); }
template<typename Stream>
inline void Unserialize(Stream& s, int32_t& v) {
    v = ser_read_i32(s);
}

template<typename Stream>
inline void Serialize(Stream& s, uint64_t v) { ser_write_u64(s, v); }
template<typename Stream>
inline void Unserialize(Stream& s, uint64_t& v) {
    v = ser_read_u64(s);
}

template<typename Stream>
inline void Serialize(Stream& s, int64_t v) { ser_write_i64(s, v); }
template<typename Stream>
inline void Unserialize(Stream& s, int64_t& v) {
    v = ser_read_i64(s);
}

template<typename Stream>
inline void Serialize(Stream& s, bool v) { ser_write_bool(s, v); }
template<typename Stream>
inline void Unserialize(Stream& s, bool& v) { v = ser_read_bool(s); }

template<typename Stream>
inline void Serialize(Stream& s, float v) { ser_write_float(s, v); }
template<typename Stream>
inline void Unserialize(Stream& s, float& v) {
    v = ser_read_float(s);
}

template<typename Stream>
inline void Serialize(Stream& s, double v) {
    ser_write_double(s, v);
}
template<typename Stream>
inline void Unserialize(Stream& s, double& v) {
    v = ser_read_double(s);
}

// ─── String ──────────────────────────────────────────────────────────

template<typename Stream>
inline void Serialize(Stream& s, const std::string& str) {
    serialize_compact_size(s, str.size());
    if (!str.empty()) {
        s.write(reinterpret_cast<const uint8_t*>(str.data()),
                str.size());
    }
}

template<typename Stream>
inline void Unserialize(Stream& s, std::string& str) {
    uint64_t len = unserialize_compact_size(s);
    if (len > 0x02000000) {  // 32 MB sanity limit
        throw std::runtime_error("string too long");
    }
    str.resize(static_cast<size_t>(len));
    if (len > 0) {
        s.read(reinterpret_cast<uint8_t*>(str.data()),
               static_cast<size_t>(len));
    }
}

// ─── Vector ──────────────────────────────────────────────────────────

template<typename Stream, typename T>
inline void Serialize(Stream& s, const std::vector<T>& v) {
    serialize_compact_size(s, v.size());
    for (const auto& elem : v) {
        Serialize(s, elem);
    }
}

template<typename Stream, typename T>
inline void Unserialize(Stream& s, std::vector<T>& v) {
    uint64_t count = unserialize_compact_size(s);
    if (count > 0x02000000) {
        throw std::runtime_error("vector too large");
    }
    v.resize(static_cast<size_t>(count));
    for (auto& elem : v) {
        Unserialize(s, elem);
    }
}

// Specialization for vector<uint8_t> — raw byte copy
template<typename Stream>
inline void Serialize(Stream& s, const std::vector<uint8_t>& v) {
    serialize_compact_size(s, v.size());
    if (!v.empty()) {
        s.write(v.data(), v.size());
    }
}

template<typename Stream>
inline void Unserialize(Stream& s, std::vector<uint8_t>& v) {
    uint64_t count = unserialize_compact_size(s);
    if (count > 0x02000000) {
        throw std::runtime_error("byte vector too large");
    }
    v.resize(static_cast<size_t>(count));
    if (count > 0) {
        s.read(v.data(), static_cast<size_t>(count));
    }
}

// ─── Array ───────────────────────────────────────────────────────────

template<typename Stream, typename T, size_t N>
inline void Serialize(Stream& s, const std::array<T, N>& arr) {
    for (const auto& elem : arr) {
        Serialize(s, elem);
    }
}

template<typename Stream, typename T, size_t N>
inline void Unserialize(Stream& s, std::array<T, N>& arr) {
    for (auto& elem : arr) {
        Unserialize(s, elem);
    }
}

// Specialization for array<uint8_t, N> — raw byte copy
template<typename Stream, size_t N>
inline void Serialize(Stream& s, const std::array<uint8_t, N>& arr) {
    s.write(arr.data(), N);
}

template<typename Stream, size_t N>
inline void Unserialize(Stream& s, std::array<uint8_t, N>& arr) {
    s.read(arr.data(), N);
}

// ─── Pair ────────────────────────────────────────────────────────────

template<typename Stream, typename K, typename V>
inline void Serialize(Stream& s, const std::pair<K, V>& p) {
    Serialize(s, p.first);
    Serialize(s, p.second);
}

template<typename Stream, typename K, typename V>
inline void Unserialize(Stream& s, std::pair<K, V>& p) {
    Unserialize(s, p.first);
    Unserialize(s, p.second);
}

// ─── Map ─────────────────────────────────────────────────────────────

template<typename Stream, typename K, typename V>
inline void Serialize(Stream& s, const std::map<K, V>& m) {
    serialize_compact_size(s, m.size());
    for (const auto& [key, val] : m) {
        Serialize(s, key);
        Serialize(s, val);
    }
}

template<typename Stream, typename K, typename V>
inline void Unserialize(Stream& s, std::map<K, V>& m) {
    m.clear();
    uint64_t count = unserialize_compact_size(s);
    if (count > 0x02000000) {
        throw std::runtime_error("map too large");
    }
    for (uint64_t i = 0; i < count; ++i) {
        K key;
        V val;
        Unserialize(s, key);
        Unserialize(s, val);
        m.emplace(std::move(key), std::move(val));
    }
}

// ─── Set ─────────────────────────────────────────────────────────────

template<typename Stream, typename T>
inline void Serialize(Stream& s, const std::set<T>& st) {
    serialize_compact_size(s, st.size());
    for (const auto& elem : st) {
        Serialize(s, elem);
    }
}

template<typename Stream, typename T>
inline void Unserialize(Stream& s, std::set<T>& st) {
    st.clear();
    uint64_t count = unserialize_compact_size(s);
    if (count > 0x02000000) {
        throw std::runtime_error("set too large");
    }
    for (uint64_t i = 0; i < count; ++i) {
        T elem;
        Unserialize(s, elem);
        st.insert(std::move(elem));
    }
}

// ─── Optional ────────────────────────────────────────────────────────

template<typename Stream, typename T>
inline void Serialize(Stream& s, const std::optional<T>& opt) {
    ser_write_bool(s, opt.has_value());
    if (opt.has_value()) {
        Serialize(s, *opt);
    }
}

template<typename Stream, typename T>
inline void Unserialize(Stream& s, std::optional<T>& opt) {
    bool has = ser_read_bool(s);
    if (has) {
        T val;
        Unserialize(s, val);
        opt = std::move(val);
    } else {
        opt.reset();
    }
}

// ─── Variant ─────────────────────────────────────────────────────────

namespace detail {

template<typename Stream, typename... Ts>
struct VariantSerializer;

template<typename Stream>
struct VariantSerializer<Stream> {
    static void serialize(Stream&, size_t, const auto&) {}
    static void unserialize(Stream&, size_t, auto&) {}
};

template<typename Stream, typename T, typename... Rest>
struct VariantSerializer<Stream, T, Rest...> {
    static void serialize(Stream& s, size_t idx,
                          const std::variant<T, Rest...>& var) {
        if (idx == 0) {
            Serialize(s, std::get<T>(var));
        } else {
            // This shouldn't happen due to index check above
        }
    }
};

}  // namespace detail

template<typename Stream, typename... Ts>
inline void Serialize(Stream& s, const std::variant<Ts...>& var) {
    ser_write_u32(s, static_cast<uint32_t>(var.index()));
    std::visit([&s](const auto& val) {
        Serialize(s, val);
    }, var);
}

template<typename Stream, typename... Ts>
inline void Unserialize(Stream& s, std::variant<Ts...>& var) {
    uint32_t idx = ser_read_u32(s);
    if (idx >= sizeof...(Ts)) {
        throw std::runtime_error("variant index out of range");
    }
    // Use a lookup table to construct the right alternative
    unserialize_variant_by_index<Stream, Ts...>(s, var, idx);
}

// Helper to unserialize variant by runtime index
template<typename Stream, typename T, typename Variant>
bool try_unserialize_variant(Stream& s, Variant& var, uint32_t idx, uint32_t& current) {
    if (current == idx) {
        T val;
        Unserialize(s, val);
        var = std::move(val);
        ++current;
        return true;  // done
    }
    ++current;
    return false;
}

template<typename Stream, typename... Ts>
void unserialize_variant_by_index(Stream& s,
                                  std::variant<Ts...>& var,
                                  uint32_t idx) {
    uint32_t current = 0;
    (void)(try_unserialize_variant<Stream, Ts>(s, var, idx, current) || ...);
}

// ─── Serializable concept (for objects with serialize/unserialize) ───

template<typename T, typename Stream>
concept HasSerialize = requires(const T& t, Stream& s) {
    { t.serialize(s) };
};

template<typename T, typename Stream>
concept HasUnserialize = requires(T& t, Stream& s) {
    { t.unserialize(s) };
};

/// Catch-all for types with .serialize()/.unserialize() methods
template<typename Stream, typename T>
    requires HasSerialize<T, Stream>
inline void Serialize(Stream& s, const T& obj) {
    obj.serialize(s);
}

template<typename Stream, typename T>
    requires HasUnserialize<T, Stream>
inline void Unserialize(Stream& s, T& obj) {
    obj.unserialize(s);
}

// ─── SERIALIZE_METHODS macro ─────────────────────────────────────────
// Makes it easy to define both serialize and unserialize in one go.

#define SERIALIZE_METHODS(...)                                         \
    template<typename Stream>                                          \
    void serialize(Stream& s) const {                                  \
        serialize_impl(*this, s);                                      \
    }                                                                  \
    template<typename Stream>                                          \
    void unserialize(Stream& s) {                                      \
        serialize_impl(*this, s);                                      \
    }                                                                  \
    template<typename Self, typename Stream>                            \
    static void serialize_impl(Self& self, Stream& s) {                \
        __VA_ARGS__                                                    \
    }

/// READWRITE helper — used inside SERIALIZE_METHODS
#define READWRITE(...)                                                  \
    ::rnet::core::SerReadWrite(s, __VA_ARGS__)

/// SerReadWrite dispatches based on const-ness
template<typename Stream, typename T>
inline void SerReadWrite(Stream& s, const T& obj) {
    Serialize(s, obj);
}

template<typename Stream, typename T>
inline void SerReadWrite(Stream& s, T& obj) {
    Unserialize(s, obj);
}

/// Get serialized size of an object
template<typename T>
size_t GetSerializeSize(const T& obj) {
    // Use a counting stream
    struct Counter {
        size_t n = 0;
        void write(const void*, size_t len) { n += len; }
        void write_byte(uint8_t) { n += 1; }
    };
    Counter c;
    Serialize(c, obj);
    return c.n;
}

// ─── Big-endian serialization (for network byte order) ──────────────

template<typename Stream>
inline void ser_write_u16_be(Stream& s, uint16_t v) {
    uint8_t buf[2];
    buf[0] = static_cast<uint8_t>((v >> 8) & 0xFF);
    buf[1] = static_cast<uint8_t>(v & 0xFF);
    s.write(buf, 2);
}

template<typename Stream>
inline uint16_t ser_read_u16_be(Stream& s) {
    uint8_t buf[2];
    s.read(buf, 2);
    return (static_cast<uint16_t>(buf[0]) << 8)
         | static_cast<uint16_t>(buf[1]);
}

template<typename Stream>
inline void ser_write_u32_be(Stream& s, uint32_t v) {
    uint8_t buf[4];
    buf[0] = static_cast<uint8_t>((v >> 24) & 0xFF);
    buf[1] = static_cast<uint8_t>((v >> 16) & 0xFF);
    buf[2] = static_cast<uint8_t>((v >> 8) & 0xFF);
    buf[3] = static_cast<uint8_t>(v & 0xFF);
    s.write(buf, 4);
}

template<typename Stream>
inline uint32_t ser_read_u32_be(Stream& s) {
    uint8_t buf[4];
    s.read(buf, 4);
    return (static_cast<uint32_t>(buf[0]) << 24)
         | (static_cast<uint32_t>(buf[1]) << 16)
         | (static_cast<uint32_t>(buf[2]) << 8)
         | static_cast<uint32_t>(buf[3]);
}

template<typename Stream>
inline void ser_write_u64_be(Stream& s, uint64_t v) {
    uint8_t buf[8];
    for (int i = 0; i < 8; ++i) {
        buf[i] = static_cast<uint8_t>((v >> ((7 - i) * 8)) & 0xFF);
    }
    s.write(buf, 8);
}

template<typename Stream>
inline uint64_t ser_read_u64_be(Stream& s) {
    uint8_t buf[8];
    s.read(buf, 8);
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<uint64_t>(buf[i]) << ((7 - i) * 8);
    }
    return v;
}

// ─── Fixed-size wrapper (serialize length without CompactSize) ──────

/// Wrapper to serialize a vector with a fixed-size length prefix
template<typename T, typename LenType = uint32_t>
struct FixedLenVec {
    std::vector<T>& vec;
    explicit FixedLenVec(std::vector<T>& v) : vec(v) {}
};

template<typename Stream, typename T, typename LenType>
inline void Serialize(Stream& s, const FixedLenVec<T, LenType>& w) {
    auto len = static_cast<LenType>(w.vec.size());
    Serialize(s, len);
    for (const auto& elem : w.vec) {
        Serialize(s, elem);
    }
}

template<typename Stream, typename T, typename LenType>
inline void Unserialize(Stream& s, FixedLenVec<T, LenType>& w) {
    LenType len;
    Unserialize(s, len);
    w.vec.resize(static_cast<size_t>(len));
    for (auto& elem : w.vec) {
        Unserialize(s, elem);
    }
}

// ─── Limited string (with max length enforcement) ───────────────────

template<size_t MAX_LEN>
struct LimitedString {
    std::string& str;
    explicit LimitedString(std::string& s) : str(s) {}
};

template<typename Stream, size_t MAX_LEN>
inline void Serialize(Stream& s, const LimitedString<MAX_LEN>& ls) {
    std::string truncated = ls.str.substr(0, MAX_LEN);
    Serialize(s, truncated);
}

template<typename Stream, size_t MAX_LEN>
inline void Unserialize(Stream& s, LimitedString<MAX_LEN>& ls) {
    Unserialize(s, ls.str);
    if (ls.str.size() > MAX_LEN) {
        throw std::runtime_error("string exceeds maximum length");
    }
}

// ─── Variadic READWRITE support ─────────────────────────────────────

template<typename Stream, typename T, typename... Rest>
inline void SerReadWriteMany(Stream& s, const T& first,
                              const Rest&... rest) {
    SerReadWrite(s, first);
    if constexpr (sizeof...(rest) > 0) {
        SerReadWriteMany(s, rest...);
    }
}

template<typename Stream, typename T, typename... Rest>
inline void SerReadWriteMany(Stream& s, T& first, Rest&... rest) {
    SerReadWrite(s, first);
    if constexpr (sizeof...(rest) > 0) {
        SerReadWriteMany(s, rest...);
    }
}

/// READWRITEMANY — serialize/unserialize multiple fields at once
#define READWRITEMANY(...)                                              \
    ::rnet::core::SerReadWriteMany(s, __VA_ARGS__)

// ─── Versioned serialization support ────────────────────────────────

/// Wrapper for version-aware serialization
template<typename T>
struct Versioned {
    int version;
    T& obj;
    Versioned(int v, T& o) : version(v), obj(o) {}
};

template<typename T>
Versioned<T> make_versioned(int version, T& obj) {
    return Versioned<T>(version, obj);
}

// ─── Checksum wrapper ───────────────────────────────────────────────

/// Serialize with a trailing 4-byte checksum placeholder.
/// The actual checksum computation is done by the crypto layer.
struct SerializeChecksum {
    uint32_t checksum = 0;

    template<typename Stream>
    void serialize(Stream& s) const {
        ser_write_u32(s, checksum);
    }

    template<typename Stream>
    void unserialize(Stream& s) {
        checksum = ser_read_u32(s);
    }
};

}  // namespace rnet::core

#pragma once

#include <cstdint>
#include <cstddef>
#include <span>

namespace rnet::core {

/// CompactSize encoding (Bitcoin-compatible):
///   0-252:       1 byte
///   253-65535:   0xfd + 2 bytes LE
///   65536-2^32:  0xfe + 4 bytes LE
///   larger:      0xff + 8 bytes LE

/// Returns the number of bytes needed to encode value as CompactSize
size_t compact_size_len(uint64_t value);

/// Encode value into buf. Returns number of bytes written.
/// buf must have at least 9 bytes available.
size_t write_compact_size(uint8_t* buf, uint64_t value);

/// Decode compact size from buf. Returns bytes consumed via out_len.
/// Returns the decoded value, or 0 with out_len=0 on error.
uint64_t read_compact_size(const uint8_t* buf, size_t buf_len,
                           size_t& out_len);

/// Stream-based compact size operations
template<typename Stream>
void serialize_compact_size(Stream& s, uint64_t value) {
    uint8_t buf[9];
    size_t len = write_compact_size(buf, value);
    s.write(buf, len);
}

template<typename Stream>
uint64_t unserialize_compact_size(Stream& s) {
    uint8_t first = 0;
    s.read(&first, 1);

    if (first < 253) {
        return first;
    }

    uint64_t result = 0;
    if (first == 253) {
        uint8_t buf[2];
        s.read(buf, 2);
        result = static_cast<uint64_t>(buf[0])
               | (static_cast<uint64_t>(buf[1]) << 8);
        // Canonical check: must be >= 253
    } else if (first == 254) {
        uint8_t buf[4];
        s.read(buf, 4);
        result = static_cast<uint64_t>(buf[0])
               | (static_cast<uint64_t>(buf[1]) << 8)
               | (static_cast<uint64_t>(buf[2]) << 16)
               | (static_cast<uint64_t>(buf[3]) << 24);
    } else {  // 0xff
        uint8_t buf[8];
        s.read(buf, 8);
        result = static_cast<uint64_t>(buf[0])
               | (static_cast<uint64_t>(buf[1]) << 8)
               | (static_cast<uint64_t>(buf[2]) << 16)
               | (static_cast<uint64_t>(buf[3]) << 24)
               | (static_cast<uint64_t>(buf[4]) << 32)
               | (static_cast<uint64_t>(buf[5]) << 40)
               | (static_cast<uint64_t>(buf[6]) << 48)
               | (static_cast<uint64_t>(buf[7]) << 56);
    }
    return result;
}

/// Variable-length integer encoding (more compact than CompactSize for
/// certain patterns). Uses continuation bits.
/// Bit 7 of each byte signals "more bytes follow".
size_t write_varint(uint8_t* buf, uint64_t value);
uint64_t read_varint(const uint8_t* buf, size_t buf_len, size_t& out_len);

template<typename Stream>
void serialize_varint(Stream& s, uint64_t value) {
    uint8_t buf[10];
    size_t len = write_varint(buf, value);
    s.write(buf, len);
}

template<typename Stream>
uint64_t unserialize_varint(Stream& s) {
    uint64_t result = 0;
    int shift = 0;
    while (true) {
        uint8_t byte = 0;
        s.read(&byte, 1);
        result |= static_cast<uint64_t>(byte & 0x7F) << shift;
        if ((byte & 0x80) == 0) break;
        shift += 7;
        if (shift > 63) break;  // Overflow protection
    }
    return result;
}

}  // namespace rnet::core

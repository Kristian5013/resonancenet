// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "core/varint.h"

namespace rnet::core {

// ===========================================================================
//  CompactSize encoding (Bitcoin-style)
// ===========================================================================

// ---------------------------------------------------------------------------
// compact_size_len / write_compact_size / read_compact_size
//
// Bitcoin's CompactSize encoding uses a 1-byte prefix to signal the
// width of the value that follows:
//   0x00..0xFC  -> 1 byte  (value is the byte itself)
//   0xFD        -> 3 bytes (uint16 LE follows)
//   0xFE        -> 5 bytes (uint32 LE follows)
//   0xFF        -> 9 bytes (uint64 LE follows)
// ---------------------------------------------------------------------------
size_t compact_size_len(uint64_t value) {
    if (value < 253)              return 1;
    if (value <= 0xFFFF)          return 3;
    if (value <= 0xFFFFFFFF)      return 5;
    return 9;
}

size_t write_compact_size(uint8_t* buf, uint64_t value) {
    if (value < 253) {
        buf[0] = static_cast<uint8_t>(value);
        return 1;
    }
    if (value <= 0xFFFF) {
        buf[0] = 253;
        buf[1] = static_cast<uint8_t>(value & 0xFF);
        buf[2] = static_cast<uint8_t>((value >> 8) & 0xFF);
        return 3;
    }
    if (value <= 0xFFFFFFFF) {
        buf[0] = 254;
        buf[1] = static_cast<uint8_t>(value & 0xFF);
        buf[2] = static_cast<uint8_t>((value >> 8) & 0xFF);
        buf[3] = static_cast<uint8_t>((value >> 16) & 0xFF);
        buf[4] = static_cast<uint8_t>((value >> 24) & 0xFF);
        return 5;
    }
    buf[0] = 255;
    buf[1] = static_cast<uint8_t>(value & 0xFF);
    buf[2] = static_cast<uint8_t>((value >> 8) & 0xFF);
    buf[3] = static_cast<uint8_t>((value >> 16) & 0xFF);
    buf[4] = static_cast<uint8_t>((value >> 24) & 0xFF);
    buf[5] = static_cast<uint8_t>((value >> 32) & 0xFF);
    buf[6] = static_cast<uint8_t>((value >> 40) & 0xFF);
    buf[7] = static_cast<uint8_t>((value >> 48) & 0xFF);
    buf[8] = static_cast<uint8_t>((value >> 56) & 0xFF);
    return 9;
}

// ---------------------------------------------------------------------------
// read_compact_size
//
// Reads a CompactSize value from buf.  Sets out_len to the number of
// bytes consumed (0 on underflow).
// ---------------------------------------------------------------------------
uint64_t read_compact_size(const uint8_t* buf, size_t buf_len,
                           size_t& out_len) {
    out_len = 0;
    if (buf_len < 1) return 0;

    uint8_t first = buf[0];
    if (first < 253) {
        out_len = 1;
        return first;
    }

    if (first == 253) {
        if (buf_len < 3) return 0;
        out_len = 3;
        return static_cast<uint64_t>(buf[1])
             | (static_cast<uint64_t>(buf[2]) << 8);
    }

    if (first == 254) {
        if (buf_len < 5) return 0;
        out_len = 5;
        return static_cast<uint64_t>(buf[1])
             | (static_cast<uint64_t>(buf[2]) << 8)
             | (static_cast<uint64_t>(buf[3]) << 16)
             | (static_cast<uint64_t>(buf[4]) << 24);
    }

    // 1. first == 255: read full uint64 LE.
    if (buf_len < 9) return 0;
    out_len = 9;
    return static_cast<uint64_t>(buf[1])
         | (static_cast<uint64_t>(buf[2]) << 8)
         | (static_cast<uint64_t>(buf[3]) << 16)
         | (static_cast<uint64_t>(buf[4]) << 24)
         | (static_cast<uint64_t>(buf[5]) << 32)
         | (static_cast<uint64_t>(buf[6]) << 40)
         | (static_cast<uint64_t>(buf[7]) << 48)
         | (static_cast<uint64_t>(buf[8]) << 56);
}

// ===========================================================================
//  LEB128 varint encoding
// ===========================================================================

// ---------------------------------------------------------------------------
// write_varint / read_varint
//
// Standard LEB128 (Little-Endian Base 128) variable-length integer
// encoding.  Each byte stores 7 data bits; the high bit signals
// continuation.  Maximum 10 bytes for a 64-bit value.
// ---------------------------------------------------------------------------
size_t write_varint(uint8_t* buf, uint64_t value) {
    size_t i = 0;
    while (value > 0x7F) {
        buf[i++] = static_cast<uint8_t>((value & 0x7F) | 0x80);
        value >>= 7;
    }
    buf[i++] = static_cast<uint8_t>(value & 0x7F);
    return i;
}

uint64_t read_varint(const uint8_t* buf, size_t buf_len,
                     size_t& out_len) {
    out_len = 0;
    uint64_t result = 0;
    int shift = 0;

    for (size_t i = 0; i < buf_len && shift <= 63; ++i) {
        result |= static_cast<uint64_t>(buf[i] & 0x7F) << shift;
        out_len = i + 1;
        if ((buf[i] & 0x80) == 0) {
            return result;
        }
        shift += 7;
    }

    return result;
}

} // namespace rnet::core

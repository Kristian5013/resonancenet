// Copyright (c) 2024-2026 The ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "crypto/siphash.h"

#include <cstring>

namespace rnet::crypto {

// ---------------------------------------------------------------------------
// SipHash-2-4 compression round (SipRound)
//
// One round of the SipHash ARX permutation on four 64-bit state words:
//
//   v0 += v1;  v1 = rotl(v1, 13);  v1 ^= v0;  v0 = rotl(v0, 32);
//   v2 += v3;  v3 = rotl(v3, 16);  v3 ^= v2;
//   v0 += v3;  v3 = rotl(v3, 21);  v3 ^= v0;
//   v2 += v1;  v1 = rotl(v1, 17);  v1 ^= v2;  v2 = rotl(v2, 32);
//
// "2-4" means 2 compression rounds per message word, 4 finalization rounds.
// ---------------------------------------------------------------------------

static inline uint64_t rotl64(uint64_t x, int b) {
    return (x << b) | (x >> (64 - b));
}

static inline void sip_round(uint64_t& v0, uint64_t& v1,
                              uint64_t& v2, uint64_t& v3) {
    v0 += v1;
    v1 = rotl64(v1, 13);
    v1 ^= v0;
    v0 = rotl64(v0, 32);
    v2 += v3;
    v3 = rotl64(v3, 16);
    v3 ^= v2;
    v0 += v3;
    v3 = rotl64(v3, 21);
    v3 ^= v0;
    v2 += v1;
    v1 = rotl64(v1, 17);
    v1 ^= v2;
    v2 = rotl64(v2, 32);
}

// ---------------------------------------------------------------------------
// siphash_2_4  (one-shot, span)
//
// SipHash-2-4 with 128-bit key (k0, k1).
//
// Initialization:
//   v0 = k0 ^ "somepseudorandomlygeneratedbytes"[0..7]
//   v1 = k1 ^ "somepseudorandomlygeneratedbytes"[8..15]
//   v2 = k0 ^ "somepseudorandomlygeneratedbytes"[16..23]
//   v3 = k1 ^ "somepseudorandomlygeneratedbytes"[24..31]
//
// Processing: for each 8-byte word m:  v3^=m, 2x SipRound, v0^=m
// Finalization: v2^=0xff, 4x SipRound, return v0^v1^v2^v3
// ---------------------------------------------------------------------------

uint64_t siphash_2_4(uint64_t k0, uint64_t k1,
                     std::span<const uint8_t> data) {
    // 1. Initialize state from key.
    uint64_t v0 = k0 ^ 0x736f6d6570736575ULL;
    uint64_t v1 = k1 ^ 0x646f72616e646f6dULL;
    uint64_t v2 = k0 ^ 0x6c7967656e657261ULL;
    uint64_t v3 = k1 ^ 0x7465646279746573ULL;

    const uint8_t* ptr = data.data();
    size_t len = data.size();
    size_t blocks = len / 8;

    // 2. Process full 8-byte blocks (2 compression rounds each).
    for (size_t i = 0; i < blocks; ++i) {
        uint64_t m;
        std::memcpy(&m, ptr + i * 8, 8);
        v3 ^= m;
        sip_round(v0, v1, v2, v3);
        sip_round(v0, v1, v2, v3);
        v0 ^= m;
    }

    // 3. Pad remaining bytes + encode total length in the high byte.
    uint64_t last = static_cast<uint64_t>(len) << 56;
    const uint8_t* tail = ptr + blocks * 8;
    size_t tail_len = len & 7;

    switch (tail_len) {
        case 7: last |= static_cast<uint64_t>(tail[6]) << 48; [[fallthrough]];
        case 6: last |= static_cast<uint64_t>(tail[5]) << 40; [[fallthrough]];
        case 5: last |= static_cast<uint64_t>(tail[4]) << 32; [[fallthrough]];
        case 4: last |= static_cast<uint64_t>(tail[3]) << 24; [[fallthrough]];
        case 3: last |= static_cast<uint64_t>(tail[2]) << 16; [[fallthrough]];
        case 2: last |= static_cast<uint64_t>(tail[1]) <<  8; [[fallthrough]];
        case 1: last |= static_cast<uint64_t>(tail[0]);        break;
        case 0: break;
    }

    // 4. Compress the final padded word.
    v3 ^= last;
    sip_round(v0, v1, v2, v3);
    sip_round(v0, v1, v2, v3);
    v0 ^= last;

    // 5. Finalization: 4 rounds with v2 ^= 0xff.
    v2 ^= 0xff;
    sip_round(v0, v1, v2, v3);
    sip_round(v0, v1, v2, v3);
    sip_round(v0, v1, v2, v3);
    sip_round(v0, v1, v2, v3);

    return v0 ^ v1 ^ v2 ^ v3;
}

// ---------------------------------------------------------------------------
// siphash_2_4  (string_view overload)
// ---------------------------------------------------------------------------

uint64_t siphash_2_4(uint64_t k0, uint64_t k1, std::string_view data) {
    return siphash_2_4(k0, k1, std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(data.data()),
        data.size()));
}

// ---------------------------------------------------------------------------
// siphash_2_4_uint256
// ---------------------------------------------------------------------------

uint64_t siphash_2_4_uint256(uint64_t k0, uint64_t k1,
                             const rnet::uint256& val) {
    return siphash_2_4(k0, k1, val.span());
}

// ---------------------------------------------------------------------------
// SipHasher  --  incremental SipHash-2-4
//
// Buffers partial words in pending_ and compresses complete 8-byte words
// as they arrive.  finalize() pads and runs 4 finalization rounds.
// ---------------------------------------------------------------------------

SipHasher::SipHasher()
    : k0_(0), k1_(0), pending_(0), pending_bytes_(0),
      total_bytes_(0)
{
    set_key(0, 0);
}

SipHasher::SipHasher(uint64_t k0, uint64_t k1)
    : k0_(k0), k1_(k1), pending_(0), pending_bytes_(0),
      total_bytes_(0)
{
    v0_ = k0_ ^ 0x736f6d6570736575ULL;
    v1_ = k1_ ^ 0x646f72616e646f6dULL;
    v2_ = k0_ ^ 0x6c7967656e657261ULL;
    v3_ = k1_ ^ 0x7465646279746573ULL;
}

// ---------------------------------------------------------------------------
// SipHasher::set_key / reset
// ---------------------------------------------------------------------------

void SipHasher::set_key(uint64_t k0, uint64_t k1) {
    k0_ = k0;
    k1_ = k1;
    reset();
}

void SipHasher::reset() {
    v0_ = k0_ ^ 0x736f6d6570736575ULL;
    v1_ = k1_ ^ 0x646f72616e646f6dULL;
    v2_ = k0_ ^ 0x6c7967656e657261ULL;
    v3_ = k1_ ^ 0x7465646279746573ULL;
    pending_ = 0;
    pending_bytes_ = 0;
    total_bytes_ = 0;
}

// ---------------------------------------------------------------------------
// SipHasher::sip_round / compress
// ---------------------------------------------------------------------------

void SipHasher::sip_round() {
    rnet::crypto::sip_round(v0_, v1_, v2_, v3_);
}

void SipHasher::compress(uint64_t m) {
    v3_ ^= m;
    sip_round();
    sip_round();
    v0_ ^= m;
}

// ---------------------------------------------------------------------------
// SipHasher::write
//
// Feed arbitrary-length data.  Buffers partial words until 8 bytes
// accumulate, then compresses.
// ---------------------------------------------------------------------------

SipHasher& SipHasher::write(std::span<const uint8_t> data) {
    const uint8_t* ptr = data.data();
    size_t len = data.size();
    total_bytes_ += len;

    // 1. Fill pending word first.
    if (pending_bytes_ > 0) {
        size_t need = 8 - pending_bytes_;
        size_t take = (len < need) ? len : need;
        for (size_t i = 0; i < take; ++i) {
            pending_ |= static_cast<uint64_t>(ptr[i])
                        << ((pending_bytes_ + i) * 8);
        }
        pending_bytes_ += take;
        ptr += take;
        len -= take;

        if (pending_bytes_ == 8) {
            compress(pending_);
            pending_ = 0;
            pending_bytes_ = 0;
        }
    }

    // 2. Process full 8-byte words.
    while (len >= 8) {
        uint64_t m;
        std::memcpy(&m, ptr, 8);
        compress(m);
        ptr += 8;
        len -= 8;
    }

    // 3. Buffer remaining bytes.
    for (size_t i = 0; i < len; ++i) {
        pending_ |= static_cast<uint64_t>(ptr[i])
                    << ((pending_bytes_ + i) * 8);
    }
    pending_bytes_ += len;

    return *this;
}

// ---------------------------------------------------------------------------
// SipHasher::write_u64 / write_u32
// ---------------------------------------------------------------------------

SipHasher& SipHasher::write_u64(uint64_t val) {
    uint8_t buf[8];
    std::memcpy(buf, &val, 8);
    return write(std::span<const uint8_t>(buf, 8));
}

SipHasher& SipHasher::write_u32(uint32_t val) {
    uint8_t buf[4];
    std::memcpy(buf, &val, 4);
    return write(std::span<const uint8_t>(buf, 4));
}

// ---------------------------------------------------------------------------
// SipHasher::finalize
//
// Pad the final word with length in the high byte, compress, then run
// 4 finalization rounds.
// ---------------------------------------------------------------------------

uint64_t SipHasher::finalize() {
    // 1. Construct final word: length in high byte + pending bytes.
    uint64_t last = static_cast<uint64_t>(total_bytes_) << 56;
    last |= pending_;

    // 2. Compress the final word.
    v3_ ^= last;
    sip_round();
    sip_round();
    v0_ ^= last;

    // 3. Finalization: v2 ^= 0xff, then 4 rounds.
    v2_ ^= 0xff;
    sip_round();
    sip_round();
    sip_round();
    sip_round();

    return v0_ ^ v1_ ^ v2_ ^ v3_;
}

} // namespace rnet::crypto

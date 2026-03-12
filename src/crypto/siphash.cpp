#include "crypto/siphash.h"

#include <cstring>

namespace rnet::crypto {

// -----------------------------------------------------------------------
// SipHash-2-4 implementation
// -----------------------------------------------------------------------

static inline uint64_t rotl64(uint64_t x, int b) {
    return (x << b) | (x >> (64 - b));
}

// One SipRound
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

uint64_t siphash_2_4(uint64_t k0, uint64_t k1,
                     std::span<const uint8_t> data) {
    uint64_t v0 = k0 ^ 0x736f6d6570736575ULL;
    uint64_t v1 = k1 ^ 0x646f72616e646f6dULL;
    uint64_t v2 = k0 ^ 0x6c7967656e657261ULL;
    uint64_t v3 = k1 ^ 0x7465646279746573ULL;

    const uint8_t* ptr = data.data();
    size_t len = data.size();
    size_t blocks = len / 8;

    // Process full 8-byte blocks
    for (size_t i = 0; i < blocks; ++i) {
        uint64_t m;
        std::memcpy(&m, ptr + i * 8, 8);
        v3 ^= m;
        sip_round(v0, v1, v2, v3);
        sip_round(v0, v1, v2, v3);
        v0 ^= m;
    }

    // Process remaining bytes + length byte
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

    v3 ^= last;
    sip_round(v0, v1, v2, v3);
    sip_round(v0, v1, v2, v3);
    v0 ^= last;

    // Finalization
    v2 ^= 0xff;
    sip_round(v0, v1, v2, v3);
    sip_round(v0, v1, v2, v3);
    sip_round(v0, v1, v2, v3);
    sip_round(v0, v1, v2, v3);

    return v0 ^ v1 ^ v2 ^ v3;
}

uint64_t siphash_2_4(uint64_t k0, uint64_t k1, std::string_view data) {
    return siphash_2_4(k0, k1, std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(data.data()),
        data.size()));
}

uint64_t siphash_2_4_uint256(uint64_t k0, uint64_t k1,
                             const rnet::uint256& val) {
    return siphash_2_4(k0, k1, val.span());
}

// -----------------------------------------------------------------------
// SipHasher: incremental
// -----------------------------------------------------------------------

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

void SipHasher::sip_round() {
    rnet::crypto::sip_round(v0_, v1_, v2_, v3_);
}

void SipHasher::compress(uint64_t m) {
    v3_ ^= m;
    sip_round();
    sip_round();
    v0_ ^= m;
}

SipHasher& SipHasher::write(std::span<const uint8_t> data) {
    const uint8_t* ptr = data.data();
    size_t len = data.size();
    total_bytes_ += len;

    // Fill pending word first
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

    // Process full 8-byte words
    while (len >= 8) {
        uint64_t m;
        std::memcpy(&m, ptr, 8);
        compress(m);
        ptr += 8;
        len -= 8;
    }

    // Buffer remaining
    for (size_t i = 0; i < len; ++i) {
        pending_ |= static_cast<uint64_t>(ptr[i])
                    << ((pending_bytes_ + i) * 8);
    }
    pending_bytes_ += len;

    return *this;
}

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

uint64_t SipHasher::finalize() {
    // Construct the final word with length byte
    uint64_t last = static_cast<uint64_t>(total_bytes_) << 56;
    last |= pending_;

    v3_ ^= last;
    sip_round();
    sip_round();
    v0_ ^= last;

    v2_ ^= 0xff;
    sip_round();
    sip_round();
    sip_round();
    sip_round();

    return v0_ ^ v1_ ^ v2_ ^ v3_;
}

}  // namespace rnet::crypto

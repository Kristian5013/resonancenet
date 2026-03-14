// Copyright (c) 2024-2026 The ResonanceNet Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "crypto/keccak.h"

#include "core/logging.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

// ===========================================================================
//
//  Keccak-256 / Keccak-256d   --   ResonanceNet core hash
//
//  Design notes
//  ------------
//  This file implements the *original* Keccak hash as submitted to the
//  SHA-3 competition, NOT the final FIPS-202 SHA-3 standard.  The only
//  difference is the domain-separation / padding byte:
//
//      Keccak (this code) :  pad byte = 0x01
//      SHA-3  (FIPS-202)  :  pad byte = 0x06
//
//  Sponge parameters for Keccak-256:
//
//      state width   b = 1600 bits  = 25 x 64-bit lanes  (5x5 matrix)
//      rate          r = 1088 bits  = 136 bytes
//      capacity      c =  512 bits  =  64 bytes
//      output        d =  256 bits  =  32 bytes
//      rounds        n =  24
//
//      security level = c/2 = 256 bits  (collision resistance)
//
//  Derived hash functions used throughout ResonanceNet:
//
//      Keccak-256d(m) = Keccak-256( Keccak-256(m) )
//
//          Double-hashing prevents length-extension attacks and mirrors
//          the Bitcoin SHA-256d convention.  Used for block hashes, txids,
//          Merkle trees, and proof-of-training target checks.
//
//      Hash160(m) = first 20 bytes of Keccak-256d(m)
//
//          Used for P2WPKH address derivation:
//              scriptPubKey = [0x00][0x14][Hash160(pubkey)]
//
//  Padding rule (multi-rate padding, pad10*1):
//
//      last_block[pos]     |=  domain_byte   (0x01 for Keccak)
//      last_block[rate-1]  |=  0x80
//
//      If pos == rate-1 the two OR operations collapse into a single
//      byte: 0x01 | 0x80 = 0x81.
//
// ===========================================================================

namespace rnet::crypto {

// ---------------------------------------------------------------------------
// rotl64  --  64-bit left rotation
// ---------------------------------------------------------------------------

static inline uint64_t rotl64(uint64_t x, int n)
{
    return (x << n) | (x >> (64 - n));
}

// ---------------------------------------------------------------------------
// keccak_f1600  --  the Keccak-f[1600] permutation
// ---------------------------------------------------------------------------
//
//  Operates in-place on the 5x5 state matrix (25 x uint64_t lanes).
//  The five steps per round are:
//
//      theta   --  column-parity diffusion
//      rho     --  intra-lane bit rotation
//      pi      --  lane position permutation
//      chi     --  non-linear row mixing
//      iota    --  round-constant XOR (breaks symmetry)
//
//  Total: 24 rounds.
//

void keccak_f1600(uint64_t state[25])
{
    uint64_t bc[5];

    for (int round = 0; round < KECCAK_ROUNDS; ++round) {

        // 1. Theta -- compute column parities, then XOR in neighbours.
        //
        //    C[x]    = state[x,0] ^ state[x,1] ^ ... ^ state[x,4]
        //    D[x]    = C[x-1] ^ ROT(C[x+1], 1)
        //    state  ^= D   (column-wise)
        //
        for (int i = 0; i < 5; ++i) {
            bc[i] = state[i] ^ state[i + 5] ^ state[i + 10]
                   ^ state[i + 15] ^ state[i + 20];
        }
        for (int i = 0; i < 5; ++i) {
            uint64_t t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5) {
                state[j + i] ^= t;
            }
        }

        // 2. Rho + Pi -- rotate each lane, then move it to its new position.
        //
        //    B[y, 2x+3y] = ROT( A[x,y], rho_offset(x,y) )
        //
        //    Combined into a single loop using the precomputed PILN and
        //    ROTC tables.
        //
        uint64_t t = state[1];
        for (int i = 0; i < 24; ++i) {
            int j = KECCAK_PILN[i];
            uint64_t tmp = state[j];
            state[j] = rotl64(t, KECCAK_ROTC[i]);
            t = tmp;
        }

        // 3. Chi -- non-linear step (the only non-linear operation).
        //
        //    A[x,y] ^= (~A[x+1, y]) & A[x+2, y]
        //
        for (int j = 0; j < 25; j += 5) {
            uint64_t tmp[5];
            for (int i = 0; i < 5; ++i) {
                tmp[i] = state[j + i];
            }
            for (int i = 0; i < 5; ++i) {
                state[j + i] ^= (~tmp[(i + 1) % 5]) & tmp[(i + 2) % 5];
            }
        }

        // 4. Iota -- XOR a round constant into lane (0,0).
        state[0] ^= KECCAK_RC[round];
    }
}

// ---------------------------------------------------------------------------
// keccak_absorb_squeeze  --  sponge absorb + squeeze (internal)
// ---------------------------------------------------------------------------
//
//  Implements the full sponge construction:
//
//      ABSORB:  for each r-byte block of (padded) input,
//               XOR block into state[0..r/8-1], then apply f1600.
//
//      SQUEEZE: copy first `out_len` bytes from state.
//               (For Keccak-256 out_len=32 < rate=136, so one squeeze
//                is always sufficient.)
//
//  Padding (pad10*1):
//
//      block[len]      =  domain_byte          (0x01 for Keccak)
//      block[rate-1]  |=  0x80
//

static void keccak_absorb_squeeze(
    const uint8_t* data, size_t len,
    uint8_t* out, size_t out_len,
    size_t rate, uint8_t domain_byte)
{
    uint64_t state[25] = {};

    // 1. Absorb full r-byte blocks.
    while (len >= rate) {
        for (size_t i = 0; i < rate / 8; ++i) {
            uint64_t lane;
            std::memcpy(&lane, data + i * 8, 8);
            state[i] ^= lane;
        }
        keccak_f1600(state);
        data += rate;
        len -= rate;
    }

    // 2. Pad the final block (pad10*1 with domain byte).
    uint8_t block[200] = {};  // max state width = 1600 bits = 200 bytes
    std::memcpy(block, data, len);
    block[len] = domain_byte;
    block[rate - 1] |= 0x80;

    // 3. Absorb the padded final block.
    for (size_t i = 0; i < rate / 8; ++i) {
        uint64_t lane;
        std::memcpy(&lane, block + i * 8, 8);
        state[i] ^= lane;
    }
    keccak_f1600(state);

    // 4. Squeeze -- copy first out_len bytes from the state.
    //    For Keccak-256: out_len = 32 < rate = 136, single squeeze.
    std::memcpy(out, state, out_len);
}

// ---------------------------------------------------------------------------
// keccak256  --  one-shot Keccak-256 (span overload)
// ---------------------------------------------------------------------------

rnet::uint256 keccak256(std::span<const uint8_t> data)
{
    rnet::uint256 result;
    keccak_absorb_squeeze(
        data.data(), data.size(),
        result.data(), KECCAK256_OUTPUT,
        KECCAK256_RATE, KECCAK_DOMAIN_SEP);
    return result;
}

// ---------------------------------------------------------------------------
// keccak256  --  one-shot Keccak-256 (string_view overload)
// ---------------------------------------------------------------------------

rnet::uint256 keccak256(std::string_view data)
{
    return keccak256(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(data.data()),
        data.size()));
}

// ---------------------------------------------------------------------------
// keccak256d  --  double Keccak-256 (span overload)
// ---------------------------------------------------------------------------
//
//  Keccak-256d(m) = Keccak-256( Keccak-256(m) )
//
//  Prevents length-extension attacks.  Used for block hashes, txids,
//  Merkle tree nodes, and PoT difficulty target comparisons.
//

rnet::uint256 keccak256d(std::span<const uint8_t> data)
{
    rnet::uint256 first = keccak256(data);
    return keccak256(first.span());
}

// ---------------------------------------------------------------------------
// keccak256d  --  double Keccak-256 (string_view overload)
// ---------------------------------------------------------------------------

rnet::uint256 keccak256d(std::string_view data)
{
    return keccak256d(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(data.data()),
        data.size()));
}

// ---------------------------------------------------------------------------
// KeccakHasher::KeccakHasher  --  construct and zero the sponge state
// ---------------------------------------------------------------------------

KeccakHasher::KeccakHasher()
{
    reset();
}

// ---------------------------------------------------------------------------
// KeccakHasher::reset  --  reinitialise to a clean sponge
// ---------------------------------------------------------------------------

void KeccakHasher::reset()
{
    std::memset(state_, 0, sizeof(state_));
    std::memset(buf_, 0, sizeof(buf_));
    buf_len_ = 0;
}

// ---------------------------------------------------------------------------
// KeccakHasher::write  --  absorb data incrementally (span overload)
// ---------------------------------------------------------------------------
//
//  Buffers partial blocks internally.  Full r-byte blocks are absorbed
//  into the sponge state immediately.
//

void KeccakHasher::write(std::span<const uint8_t> data)
{
    const uint8_t* ptr = data.data();
    size_t remaining = data.size();

    // 1. If we have buffered bytes, try to complete a full block.
    if (buf_len_ > 0) {
        size_t space = KECCAK256_RATE - buf_len_;
        size_t copy = (remaining < space) ? remaining : space;
        std::memcpy(buf_ + buf_len_, ptr, copy);
        buf_len_ += copy;
        ptr += copy;
        remaining -= copy;

        if (buf_len_ == KECCAK256_RATE) {
            // Full block ready -- absorb it.
            for (size_t i = 0; i < KECCAK256_RATE / 8; ++i) {
                uint64_t lane;
                std::memcpy(&lane, buf_ + i * 8, 8);
                state_[i] ^= lane;
            }
            keccak_f1600(state_);
            buf_len_ = 0;
        }
    }

    // 2. Absorb full blocks directly from the input pointer.
    while (remaining >= KECCAK256_RATE) {
        for (size_t i = 0; i < KECCAK256_RATE / 8; ++i) {
            uint64_t lane;
            std::memcpy(&lane, ptr + i * 8, 8);
            state_[i] ^= lane;
        }
        keccak_f1600(state_);
        ptr += KECCAK256_RATE;
        remaining -= KECCAK256_RATE;
    }

    // 3. Buffer any remaining bytes for the next call.
    if (remaining > 0) {
        std::memcpy(buf_, ptr, remaining);
        buf_len_ = remaining;
    }
}

// ---------------------------------------------------------------------------
// KeccakHasher::write  --  absorb data incrementally (string_view overload)
// ---------------------------------------------------------------------------

void KeccakHasher::write(std::string_view data)
{
    write(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(data.data()),
        data.size()));
}

// ---------------------------------------------------------------------------
// KeccakHasher::finalize  --  pad, absorb final block, squeeze digest
// ---------------------------------------------------------------------------
//
//  After this call the hasher is in an undefined state.
//  Call reset() before reusing.
//

rnet::uint256 KeccakHasher::finalize()
{
    // 1. Zero-fill the remainder of the buffer.
    std::memset(buf_ + buf_len_, 0, KECCAK256_RATE - buf_len_);

    // 2. Apply pad10*1 with Keccak domain byte (0x01).
    buf_[buf_len_] = KECCAK_DOMAIN_SEP;
    buf_[KECCAK256_RATE - 1] |= 0x80;

    // 3. Absorb the final padded block.
    for (size_t i = 0; i < KECCAK256_RATE / 8; ++i) {
        uint64_t lane;
        std::memcpy(&lane, buf_ + i * 8, 8);
        state_[i] ^= lane;
    }
    keccak_f1600(state_);

    // 4. Squeeze -- copy first 32 bytes from the state.
    rnet::uint256 result;
    std::memcpy(result.data(), state_, KECCAK256_OUTPUT);
    return result;
}

// ---------------------------------------------------------------------------
// KeccakHasher::finalize_double  --  produce Keccak-256d digest
// ---------------------------------------------------------------------------
//
//  finalize_double() = Keccak-256( finalize() )
//

rnet::uint256 KeccakHasher::finalize_double()
{
    rnet::uint256 first = finalize();
    return keccak256(first.span());
}

// ---------------------------------------------------------------------------
// keccak256_file  --  stream-hash a file with Keccak-256
// ---------------------------------------------------------------------------

rnet::Result<rnet::uint256> keccak256_file(
    const std::filesystem::path& path)
{
    // 1. Open the file in binary mode.
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return rnet::Result<rnet::uint256>::err(
            "keccak256_file: cannot open file: " + path.string());
    }

    // 2. Feed the file through the incremental hasher in 64 KiB chunks.
    KeccakHasher hasher;
    constexpr size_t CHUNK_SIZE = 65536;
    uint8_t chunk[CHUNK_SIZE];

    while (file.good()) {
        file.read(reinterpret_cast<char*>(chunk), CHUNK_SIZE);
        auto bytes_read = static_cast<size_t>(file.gcount());
        if (bytes_read > 0) {
            hasher.write(std::span<const uint8_t>(chunk, bytes_read));
        }
    }

    // 3. Check for I/O errors.
    if (file.bad()) {
        return rnet::Result<rnet::uint256>::err(
            "keccak256_file: read error on file: " + path.string());
    }

    return rnet::Result<rnet::uint256>::ok(hasher.finalize());
}

// ---------------------------------------------------------------------------
// keccak256d_file  --  stream-hash a file with Keccak-256d
// ---------------------------------------------------------------------------
//
//  keccak256d_file(path) = Keccak-256( keccak256_file(path) )
//

rnet::Result<rnet::uint256> keccak256d_file(
    const std::filesystem::path& path)
{
    // 1. Compute single Keccak-256 of the file contents.
    auto result = keccak256_file(path);
    if (result.is_err()) {
        return result;
    }

    // 2. Hash the result again to get Keccak-256d.
    rnet::uint256 first = result.value();
    return rnet::Result<rnet::uint256>::ok(keccak256(first.span()));
}

} // namespace rnet::crypto

// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "lightning/onion.h"
#include "lightning/channel_state.h"

#include "core/serialize.h"
#include "core/stream.h"
#include "crypto/chacha20.h"
#include "crypto/keccak.h"

#include <algorithm>
#include <cstring>

namespace rnet::lightning {

// ===========================================================================
//  HopData
// ===========================================================================

// ---------------------------------------------------------------------------
// HopData::serialize
// ---------------------------------------------------------------------------
// Wire format: [8-byte scid][8-byte amount][4-byte cltv][12-byte padding]
// = 32 bytes total.  Fields are written in little-endian via the DataStream
// serialisation helpers.  The 12-byte padding is reserved for future TLV
// extensions and is currently zero-filled.
// ---------------------------------------------------------------------------
std::vector<uint8_t> HopData::serialize() const {
    // 1. Open a DataStream for binary output
    core::DataStream ss;

    // 2. Write the three routing fields in fixed order
    core::ser_write_u64(ss, short_channel_id);
    core::ser_write_i64(ss, amount);
    core::ser_write_u32(ss, cltv_expiry);

    // 3. Append the 12-byte padding block
    ss.write(padding, sizeof(padding));

    // 4. Return the completed 32-byte buffer
    return ss.vch();
}

// ---------------------------------------------------------------------------
// HopData::deserialize
// ---------------------------------------------------------------------------
// Wire format: [8-byte scid][8-byte amount][4-byte cltv][12-byte padding]
// = 32 bytes total.  Rejects any buffer shorter than 32 bytes.
// ---------------------------------------------------------------------------
Result<HopData> HopData::deserialize(const std::vector<uint8_t>& data) {
    // 1. Validate minimum length
    if (data.size() < 32) {
        return Result<HopData>::err("HopData too short");
    }

    // 2. Wrap the raw bytes in a readable stream
    core::DataStream ss(data);

    // 3. Read fields in the same order they were written
    HopData hd;
    hd.short_channel_id = core::ser_read_u64(ss);
    hd.amount = core::ser_read_i64(ss);
    hd.cltv_expiry = core::ser_read_u32(ss);
    ss.read(hd.padding, sizeof(hd.padding));

    return Result<HopData>::ok(std::move(hd));
}

// ===========================================================================
//  OnionPacket
// ===========================================================================

// ---------------------------------------------------------------------------
// OnionPacket::serialized_size
// ---------------------------------------------------------------------------
// Wire format: [1-byte version][32-byte ephemeral_key]
//              [N-byte encrypted_data][32-byte hmac]
// ---------------------------------------------------------------------------
size_t OnionPacket::serialized_size() const {
    return 1 + 32 + encrypted_data.size() + 32;  // version + key + data + hmac
}

// ---------------------------------------------------------------------------
// OnionPacket::serialize
// ---------------------------------------------------------------------------
// Wire format: [1-byte version][32-byte ephemeral_key]
//              [N-byte encrypted_data][32-byte hmac]
// Concatenates the four fields into a single byte vector.
// ---------------------------------------------------------------------------
std::vector<uint8_t> OnionPacket::serialize() const {
    // 1. Pre-allocate to avoid re-allocation
    std::vector<uint8_t> result;
    result.reserve(serialized_size());

    // 2. Version byte
    result.push_back(version);

    // 3. 32-byte ephemeral public key
    result.insert(result.end(), ephemeral_key.data.begin(),
                  ephemeral_key.data.end());

    // 4. Variable-length encrypted payload
    result.insert(result.end(), encrypted_data.begin(), encrypted_data.end());

    // 5. 32-byte HMAC
    result.insert(result.end(), hmac.begin(), hmac.end());

    return result;
}

// ---------------------------------------------------------------------------
// OnionPacket::deserialize
// ---------------------------------------------------------------------------
// Wire format: [1-byte version][32-byte ephemeral_key]
//              [N-byte encrypted_data][32-byte hmac]
// Minimum 65 bytes (version + key + hmac with zero-length payload).
// ---------------------------------------------------------------------------
Result<OnionPacket> OnionPacket::deserialize(const std::vector<uint8_t>& data) {
    // 1. Enforce minimum length: version(1) + key(32) + hmac(32)
    if (data.size() < 65) {
        return Result<OnionPacket>::err("OnionPacket too short");
    }

    // 2. Read and validate version byte
    OnionPacket pkt;
    pkt.version = data[0];
    if (pkt.version != ONION_VERSION) {
        return Result<OnionPacket>::err("Unknown onion version: " +
                                         std::to_string(pkt.version));
    }

    // 3. Extract the 32-byte ephemeral public key
    std::memcpy(pkt.ephemeral_key.data.data(), data.data() + 1, 32);

    // 4. Extract the variable-length encrypted data (everything between key and hmac)
    size_t encrypted_len = data.size() - 65;
    pkt.encrypted_data.assign(data.begin() + 33,
                               data.begin() + 33 + static_cast<int64_t>(encrypted_len));

    // 5. Extract the trailing 32-byte HMAC
    std::memcpy(pkt.hmac.data(), data.data() + data.size() - 32, 32);

    return Result<OnionPacket>::ok(std::move(pkt));
}

// ===========================================================================
//  Shared Secret Derivation
// ===========================================================================

// ---------------------------------------------------------------------------
// compute_shared_secret
// ---------------------------------------------------------------------------
// shared = Keccak256d(our_secret || their_pubkey)
// Simplified ECDH stand-in: a real implementation would use X25519 key
// exchange.  The tagged double-hash is deterministic and sufficient for
// the protocol structure while the full ECDH primitive is not yet wired up.
// ---------------------------------------------------------------------------
static uint256 compute_shared_secret(
    const crypto::Ed25519PublicKey& their_pubkey,
    const crypto::Ed25519SecretKey& our_secret) {
    // 1. Initialize a Keccak hasher
    crypto::KeccakHasher hasher;

    // 2. Feed our secret key seed followed by their public key
    hasher.write(std::span<const uint8_t>(our_secret.seed()));
    hasher.write(std::span<const uint8_t>(their_pubkey.data));

    // 3. Finalize with Keccak256d (hash-of-hash)
    return hasher.finalize_double();
}

// ---------------------------------------------------------------------------
// derive_stream_key
// ---------------------------------------------------------------------------
// key = Keccak256d(shared_secret || info_string)
// The info string selects the derived key's purpose:
//   "rho"   — stream cipher key for payload encryption
//   "mu"    — HMAC key for integrity verification
//   "um"    — HMAC key for error packet verification
//   "ammag" — stream cipher key for error packet encryption
// ---------------------------------------------------------------------------
static uint256 derive_stream_key(const uint256& shared_secret,
                                  const char* info) {
    // 1. Hash the shared secret concatenated with the info tag
    crypto::KeccakHasher hasher;
    hasher.write(shared_secret.span());
    hasher.write(std::string_view(info));

    // 2. Double-hash to produce the derived key
    return hasher.finalize_double();
}

// ===========================================================================
//  Onion Construction
// ===========================================================================

// ---------------------------------------------------------------------------
// create_onion_packet
// ---------------------------------------------------------------------------
// Sphinx construction — build from last hop to first:
//   1. Serialize hop data
//   2. Shift payload right, insert hop + HMAC
//   3. Encrypt with ChaCha20(stream_key=derive("rho", shared_secret))
//   4. Compute HMAC with mu_key over encrypted payload + associated_data
// Each iteration wraps another layer of encryption around the packet, so
// only the intended recipient of each layer can peel it off.
// ---------------------------------------------------------------------------
Result<OnionPacket> create_onion_packet(
    const crypto::Ed25519KeyPair& session_key,
    const std::vector<crypto::Ed25519PublicKey>& hop_pubkeys,
    const std::vector<HopData>& hops,
    const uint256& associated_data) {

    // 1. Validate inputs
    if (hop_pubkeys.size() != hops.size()) {
        return Result<OnionPacket>::err(
            "hop_pubkeys and hops must have the same size");
    }
    if (hops.empty()) {
        return Result<OnionPacket>::err("Route must have at least one hop");
    }
    if (hops.size() > MAX_ROUTE_HOPS) {
        return Result<OnionPacket>::err("Route exceeds maximum hops: " +
                                         std::to_string(hops.size()));
    }

    // 2. Compute payload size for all hops (hop data + per-hop HMAC each)
    size_t num_hops = hops.size();
    size_t payload_size = num_hops * (ONION_HOP_DATA_SIZE + ONION_HMAC_SIZE);

    // 3. Initialize filler with zeros
    std::vector<uint8_t> packet_data(payload_size, 0);

    // 4. Build from last hop to first, wrapping one layer per iteration
    uint256 current_hmac;
    for (int i = static_cast<int>(num_hops) - 1; i >= 0; --i) {
        // 5. Compute shared secret with this hop
        uint256 ss = compute_shared_secret(hop_pubkeys[static_cast<size_t>(i)],
                                            session_key.secret);

        // 6. Serialize this hop's routing data
        auto hop_bytes = hops[static_cast<size_t>(i)].serialize();

        // 7. Shift existing data right, insert hop data + HMAC at front
        std::vector<uint8_t> new_data(payload_size, 0);
        std::memcpy(new_data.data(), hop_bytes.data(),
                     std::min(hop_bytes.size(), ONION_HOP_DATA_SIZE));
        std::memcpy(new_data.data() + ONION_HOP_DATA_SIZE,
                     current_hmac.data(), 32);
        if (payload_size > ONION_HOP_DATA_SIZE + 32) {
            size_t remaining = payload_size - ONION_HOP_DATA_SIZE - 32;
            std::memcpy(new_data.data() + ONION_HOP_DATA_SIZE + 32,
                         packet_data.data(), remaining);
        }
        packet_data = std::move(new_data);

        // 8. Encrypt with ChaCha20 keyed by derive("rho", shared_secret)
        uint256 stream_key = derive_stream_key(ss, "rho");
        std::array<uint8_t, crypto::ChaCha20::NONCE_SIZE> nonce{};
        crypto::ChaCha20 cipher(
            std::span<const uint8_t, 32>(stream_key.data(), 32),
            std::span<const uint8_t, 12>(nonce.data(), 12));
        cipher.crypt(std::span<uint8_t>(packet_data));

        // 9. Compute HMAC with mu_key over encrypted payload + associated_data
        uint256 mu_key = derive_stream_key(ss, "mu");
        crypto::KeccakHasher hmac_hasher;
        hmac_hasher.write(mu_key.span());
        hmac_hasher.write(std::span<const uint8_t>(packet_data));
        hmac_hasher.write(associated_data.span());
        current_hmac = hmac_hasher.finalize_double();
    }

    // 10. Assemble the final OnionPacket
    OnionPacket pkt;
    pkt.version = ONION_VERSION;
    pkt.ephemeral_key = session_key.public_key;
    pkt.encrypted_data = std::move(packet_data);
    pkt.hmac = current_hmac;

    return Result<OnionPacket>::ok(std::move(pkt));
}

// ===========================================================================
//  Onion Peeling
// ===========================================================================

// ---------------------------------------------------------------------------
// peel_onion_packet
// ---------------------------------------------------------------------------
// Reverse of construction — verify HMAC, decrypt, extract hop, build next
// packet.  Each intermediate node peels exactly one layer and forwards the
// remaining packet to the next hop.  The final recipient sees is_final=true
// when the inner HMAC is all-zeros.
// ---------------------------------------------------------------------------
Result<OnionPeelResult> peel_onion_packet(
    const crypto::Ed25519SecretKey& our_secret,
    const OnionPacket& packet,
    const uint256& associated_data) {

    // 1. Reject unknown versions
    if (packet.version != ONION_VERSION) {
        return Result<OnionPeelResult>::err("Unknown onion version");
    }

    // 2. Compute shared secret with the sender
    uint256 ss = compute_shared_secret(packet.ephemeral_key, our_secret);

    // 3. Verify HMAC: mu_key over encrypted_data + associated_data
    uint256 mu_key = derive_stream_key(ss, "mu");
    crypto::KeccakHasher hmac_hasher;
    hmac_hasher.write(mu_key.span());
    hmac_hasher.write(std::span<const uint8_t>(packet.encrypted_data));
    hmac_hasher.write(associated_data.span());
    uint256 expected_hmac = hmac_hasher.finalize_double();

    if (expected_hmac != packet.hmac) {
        return Result<OnionPeelResult>::err("Onion HMAC verification failed");
    }

    // 4. Decrypt the payload with ChaCha20 keyed by derive("rho", ss)
    std::vector<uint8_t> decrypted = packet.encrypted_data;
    uint256 stream_key = derive_stream_key(ss, "rho");
    std::array<uint8_t, crypto::ChaCha20::NONCE_SIZE> nonce{};
    crypto::ChaCha20 cipher(
        std::span<const uint8_t, 32>(stream_key.data(), 32),
        std::span<const uint8_t, 12>(nonce.data(), 12));
    cipher.crypt(std::span<uint8_t>(decrypted));

    // 5. Extract hop data from the first ONION_HOP_DATA_SIZE bytes
    std::vector<uint8_t> hop_bytes(decrypted.begin(),
                                     decrypted.begin() + ONION_HOP_DATA_SIZE);
    auto hop_result = HopData::deserialize(hop_bytes);
    if (!hop_result) {
        return Result<OnionPeelResult>::err(hop_result.error());
    }

    // 6. Extract the next-hop HMAC (immediately after hop data)
    uint256 next_hmac;
    if (decrypted.size() >= ONION_HOP_DATA_SIZE + 32) {
        std::memcpy(next_hmac.data(),
                     decrypted.data() + ONION_HOP_DATA_SIZE, 32);
    }

    // 7. Assemble the result
    OnionPeelResult result;
    result.hop_data = hop_result.value();
    result.is_final = next_hmac.is_zero();

    // 8. Build next packet — shift data left, dropping our hop
    result.next_packet.version = ONION_VERSION;
    // In a real implementation, the ephemeral key would be blinded
    result.next_packet.ephemeral_key = packet.ephemeral_key;
    result.next_packet.hmac = next_hmac;

    if (decrypted.size() > ONION_HOP_DATA_SIZE + 32) {
        result.next_packet.encrypted_data.assign(
            decrypted.begin() + ONION_HOP_DATA_SIZE + 32,
            decrypted.end());
        // Pad to maintain constant size
        result.next_packet.encrypted_data.resize(
            packet.encrypted_data.size(), 0);
    }

    return Result<OnionPeelResult>::ok(std::move(result));
}

// ===========================================================================
//  Error Handling
// ===========================================================================

// ---------------------------------------------------------------------------
// create_onion_error
// ---------------------------------------------------------------------------
// Constructs an error packet: [32-byte HMAC][256-byte padded payload].
// The payload contains the failure code and optional data, zero-padded to
// 256 bytes.  The HMAC is computed with the "um" derived key, then the
// entire packet is encrypted with ChaCha20 using the "ammag" derived key.
// ---------------------------------------------------------------------------
std::vector<uint8_t> create_onion_error(
    const uint256& shared_secret,
    uint16_t failure_code,
    const std::vector<uint8_t>& failure_data) {

    // 1. Serialize the failure code and data into a padded payload
    core::DataStream ss;
    core::ser_write_u16(ss, failure_code);
    core::ser_write_u16(ss, static_cast<uint16_t>(failure_data.size()));
    if (!failure_data.empty()) {
        ss.write(failure_data.data(), failure_data.size());
    }

    // 2. Pad to fixed 256-byte size
    auto data = ss.vch();
    data.resize(256, 0);

    // 3. Compute HMAC over the payload using the "um" derived key
    uint256 um_key = derive_stream_key(shared_secret, "um");
    crypto::KeccakHasher hasher;
    hasher.write(um_key.span());
    hasher.write(std::span<const uint8_t>(data));
    uint256 mac = hasher.finalize_double();

    // 4. Concatenate HMAC + payload
    std::vector<uint8_t> result;
    result.insert(result.end(), mac.begin(), mac.end());
    result.insert(result.end(), data.begin(), data.end());

    // 5. Encrypt the entire packet with ChaCha20 using the "ammag" derived key
    uint256 ammag_key = derive_stream_key(shared_secret, "ammag");
    std::array<uint8_t, crypto::ChaCha20::NONCE_SIZE> nonce{};
    crypto::ChaCha20 cipher(
        std::span<const uint8_t, 32>(ammag_key.data(), 32),
        std::span<const uint8_t, 12>(nonce.data(), 12));
    cipher.crypt(std::span<uint8_t>(result));

    return result;
}

// ---------------------------------------------------------------------------
// decode_onion_error
// ---------------------------------------------------------------------------
// Try each shared secret in order: decrypt with the "ammag" key, then
// verify the HMAC with the "um" key.  The first secret whose HMAC matches
// identifies the failing hop.  The decrypted payload contains the failure
// code and optional failure data.
// ---------------------------------------------------------------------------
Result<OnionError> decode_onion_error(
    const std::vector<uint256>& shared_secrets,
    const std::vector<uint8_t>& error_packet) {

    // 1. Copy the packet so we can decrypt in-place iteratively
    std::vector<uint8_t> packet = error_packet;

    for (uint32_t i = 0; i < shared_secrets.size(); ++i) {
        // 2. Decrypt with ChaCha20 using the "ammag" derived key
        uint256 ammag_key = derive_stream_key(shared_secrets[i], "ammag");
        std::array<uint8_t, crypto::ChaCha20::NONCE_SIZE> nonce{};
        crypto::ChaCha20 cipher(
            std::span<const uint8_t, 32>(ammag_key.data(), 32),
            std::span<const uint8_t, 12>(nonce.data(), 12));
        cipher.crypt(std::span<uint8_t>(packet));

        // 3. Need at least 32-byte HMAC + 4-byte header (code + length)
        if (packet.size() < 32 + 4) continue;

        // 4. Extract the HMAC from the first 32 bytes
        uint256 mac;
        std::memcpy(mac.data(), packet.data(), 32);

        // 5. Verify HMAC over the payload using the "um" derived key
        std::span<const uint8_t> payload(packet.data() + 32,
                                          packet.size() - 32);
        uint256 um_key = derive_stream_key(shared_secrets[i], "um");
        crypto::KeccakHasher hasher;
        hasher.write(um_key.span());
        hasher.write(payload);
        uint256 expected = hasher.finalize_double();

        if (mac == expected) {
            // 6. Found the originating hop — decode the failure payload
            core::SpanReader reader(payload);
            uint16_t code = core::ser_read_u16(reader);
            uint16_t data_len = core::ser_read_u16(reader);

            OnionError err;
            err.failure_code = code;
            err.failing_hop = i;
            if (data_len > 0 && reader.remaining() >= data_len) {
                err.failure_data.resize(data_len);
                reader.read(err.failure_data.data(), data_len);
            }
            return Result<OnionError>::ok(std::move(err));
        }
    }

    return Result<OnionError>::err("Could not decode onion error");
}

} // namespace rnet::lightning

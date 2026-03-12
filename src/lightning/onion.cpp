#include "lightning/onion.h"
#include "lightning/channel_state.h"

#include <algorithm>
#include <cstring>

#include "core/stream.h"
#include "core/serialize.h"
#include "crypto/keccak.h"
#include "crypto/chacha20.h"

namespace rnet::lightning {

// ── HopData ─────────────────────────────────────────────────────────

std::vector<uint8_t> HopData::serialize() const {
    core::DataStream ss;
    core::ser_write_u64(ss, short_channel_id);
    core::ser_write_i64(ss, amount);
    core::ser_write_u32(ss, cltv_expiry);
    ss.write(padding, sizeof(padding));
    return ss.vch();
}

Result<HopData> HopData::deserialize(const std::vector<uint8_t>& data) {
    if (data.size() < 32) {
        return Result<HopData>::err("HopData too short");
    }
    core::DataStream ss(data);
    HopData hd;
    hd.short_channel_id = core::ser_read_u64(ss);
    hd.amount = core::ser_read_i64(ss);
    hd.cltv_expiry = core::ser_read_u32(ss);
    ss.read(hd.padding, sizeof(hd.padding));
    return Result<HopData>::ok(std::move(hd));
}

// ── OnionPacket ─────────────────────────────────────────────────────

size_t OnionPacket::serialized_size() const {
    return 1 + 32 + encrypted_data.size() + 32;  // version + key + data + hmac
}

std::vector<uint8_t> OnionPacket::serialize() const {
    std::vector<uint8_t> result;
    result.reserve(serialized_size());
    result.push_back(version);
    result.insert(result.end(), ephemeral_key.data.begin(),
                  ephemeral_key.data.end());
    result.insert(result.end(), encrypted_data.begin(), encrypted_data.end());
    result.insert(result.end(), hmac.begin(), hmac.end());
    return result;
}

Result<OnionPacket> OnionPacket::deserialize(const std::vector<uint8_t>& data) {
    if (data.size() < 65) {  // version(1) + key(32) + hmac(32)
        return Result<OnionPacket>::err("OnionPacket too short");
    }

    OnionPacket pkt;
    pkt.version = data[0];
    if (pkt.version != ONION_VERSION) {
        return Result<OnionPacket>::err("Unknown onion version: " +
                                         std::to_string(pkt.version));
    }

    std::memcpy(pkt.ephemeral_key.data.data(), data.data() + 1, 32);

    size_t encrypted_len = data.size() - 65;
    pkt.encrypted_data.assign(data.begin() + 33,
                               data.begin() + 33 + static_cast<int64_t>(encrypted_len));

    std::memcpy(pkt.hmac.data(), data.data() + data.size() - 32, 32);

    return Result<OnionPacket>::ok(std::move(pkt));
}

// ── Shared secret computation ───────────────────────────────────────

static uint256 compute_shared_secret(
    const crypto::Ed25519PublicKey& their_pubkey,
    const crypto::Ed25519SecretKey& our_secret) {
    // Derive shared secret via tagged hash of combined keys
    // In a real implementation this would use ECDH (X25519)
    // Here we use a deterministic derivation for the protocol structure
    crypto::KeccakHasher hasher;
    hasher.write(std::span<const uint8_t>(our_secret.seed()));
    hasher.write(std::span<const uint8_t>(their_pubkey.data));
    return hasher.finalize_double();
}

static uint256 derive_stream_key(const uint256& shared_secret,
                                  const char* info) {
    crypto::KeccakHasher hasher;
    hasher.write(shared_secret.span());
    hasher.write(std::string_view(info));
    return hasher.finalize_double();
}

// ── Onion construction ──────────────────────────────────────────────

Result<OnionPacket> create_onion_packet(
    const crypto::Ed25519KeyPair& session_key,
    const std::vector<crypto::Ed25519PublicKey>& hop_pubkeys,
    const std::vector<HopData>& hops,
    const uint256& associated_data) {

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

    size_t num_hops = hops.size();
    size_t payload_size = num_hops * (ONION_HOP_DATA_SIZE + ONION_HMAC_SIZE);

    // Initialize filler with zeros
    std::vector<uint8_t> packet_data(payload_size, 0);

    // Build from last hop to first
    uint256 current_hmac;
    for (int i = static_cast<int>(num_hops) - 1; i >= 0; --i) {
        // Compute shared secret with this hop
        uint256 ss = compute_shared_secret(hop_pubkeys[static_cast<size_t>(i)],
                                            session_key.secret);

        // Serialize hop data
        auto hop_bytes = hops[static_cast<size_t>(i)].serialize();

        // Shift existing data right
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

        // Encrypt with stream cipher derived from shared secret
        uint256 stream_key = derive_stream_key(ss, "rho");
        std::array<uint8_t, crypto::ChaCha20::NONCE_SIZE> nonce{};
        crypto::ChaCha20 cipher(
            std::span<const uint8_t, 32>(stream_key.data(), 32),
            std::span<const uint8_t, 12>(nonce.data(), 12));
        cipher.crypt(std::span<uint8_t>(packet_data));

        // Compute HMAC for this layer
        uint256 mu_key = derive_stream_key(ss, "mu");
        crypto::KeccakHasher hmac_hasher;
        hmac_hasher.write(mu_key.span());
        hmac_hasher.write(std::span<const uint8_t>(packet_data));
        hmac_hasher.write(associated_data.span());
        current_hmac = hmac_hasher.finalize_double();
    }

    OnionPacket pkt;
    pkt.version = ONION_VERSION;
    pkt.ephemeral_key = session_key.public_key;
    pkt.encrypted_data = std::move(packet_data);
    pkt.hmac = current_hmac;

    return Result<OnionPacket>::ok(std::move(pkt));
}

Result<OnionPeelResult> peel_onion_packet(
    const crypto::Ed25519SecretKey& our_secret,
    const OnionPacket& packet,
    const uint256& associated_data) {

    if (packet.version != ONION_VERSION) {
        return Result<OnionPeelResult>::err("Unknown onion version");
    }

    // Compute shared secret
    uint256 ss = compute_shared_secret(packet.ephemeral_key, our_secret);

    // Verify HMAC
    uint256 mu_key = derive_stream_key(ss, "mu");
    crypto::KeccakHasher hmac_hasher;
    hmac_hasher.write(mu_key.span());
    hmac_hasher.write(std::span<const uint8_t>(packet.encrypted_data));
    hmac_hasher.write(associated_data.span());
    uint256 expected_hmac = hmac_hasher.finalize_double();

    if (expected_hmac != packet.hmac) {
        return Result<OnionPeelResult>::err("Onion HMAC verification failed");
    }

    // Decrypt the payload
    std::vector<uint8_t> decrypted = packet.encrypted_data;
    uint256 stream_key = derive_stream_key(ss, "rho");
    std::array<uint8_t, crypto::ChaCha20::NONCE_SIZE> nonce{};
    crypto::ChaCha20 cipher(
        std::span<const uint8_t, 32>(stream_key.data(), 32),
        std::span<const uint8_t, 12>(nonce.data(), 12));
    cipher.crypt(std::span<uint8_t>(decrypted));

    // Extract hop data (first ONION_HOP_DATA_SIZE bytes)
    std::vector<uint8_t> hop_bytes(decrypted.begin(),
                                     decrypted.begin() + ONION_HOP_DATA_SIZE);
    auto hop_result = HopData::deserialize(hop_bytes);
    if (!hop_result) {
        return Result<OnionPeelResult>::err(hop_result.error());
    }

    // Extract next HMAC
    uint256 next_hmac;
    if (decrypted.size() >= ONION_HOP_DATA_SIZE + 32) {
        std::memcpy(next_hmac.data(),
                     decrypted.data() + ONION_HOP_DATA_SIZE, 32);
    }

    OnionPeelResult result;
    result.hop_data = hop_result.value();
    result.is_final = next_hmac.is_zero();

    // Build next packet (shift data left)
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

std::vector<uint8_t> create_onion_error(
    const uint256& shared_secret,
    uint16_t failure_code,
    const std::vector<uint8_t>& failure_data) {

    core::DataStream ss;
    core::ser_write_u16(ss, failure_code);
    core::ser_write_u16(ss, static_cast<uint16_t>(failure_data.size()));
    if (!failure_data.empty()) {
        ss.write(failure_data.data(), failure_data.size());
    }

    // Pad to fixed size
    auto data = ss.vch();
    data.resize(256, 0);

    // Add HMAC
    uint256 um_key = derive_stream_key(shared_secret, "um");
    crypto::KeccakHasher hasher;
    hasher.write(um_key.span());
    hasher.write(std::span<const uint8_t>(data));
    uint256 mac = hasher.finalize_double();

    std::vector<uint8_t> result;
    result.insert(result.end(), mac.begin(), mac.end());
    result.insert(result.end(), data.begin(), data.end());

    // Encrypt with ammag key
    uint256 ammag_key = derive_stream_key(shared_secret, "ammag");
    std::array<uint8_t, crypto::ChaCha20::NONCE_SIZE> nonce{};
    crypto::ChaCha20 cipher(
        std::span<const uint8_t, 32>(ammag_key.data(), 32),
        std::span<const uint8_t, 12>(nonce.data(), 12));
    cipher.crypt(std::span<uint8_t>(result));

    return result;
}

Result<OnionError> decode_onion_error(
    const std::vector<uint256>& shared_secrets,
    const std::vector<uint8_t>& error_packet) {

    std::vector<uint8_t> packet = error_packet;

    for (uint32_t i = 0; i < shared_secrets.size(); ++i) {
        // Decrypt with ammag key
        uint256 ammag_key = derive_stream_key(shared_secrets[i], "ammag");
        std::array<uint8_t, crypto::ChaCha20::NONCE_SIZE> nonce{};
        crypto::ChaCha20 cipher(
            std::span<const uint8_t, 32>(ammag_key.data(), 32),
            std::span<const uint8_t, 12>(nonce.data(), 12));
        cipher.crypt(std::span<uint8_t>(packet));

        if (packet.size() < 32 + 4) continue;

        // Verify HMAC
        uint256 mac;
        std::memcpy(mac.data(), packet.data(), 32);

        std::span<const uint8_t> payload(packet.data() + 32,
                                          packet.size() - 32);
        uint256 um_key = derive_stream_key(shared_secrets[i], "um");
        crypto::KeccakHasher hasher;
        hasher.write(um_key.span());
        hasher.write(payload);
        uint256 expected = hasher.finalize_double();

        if (mac == expected) {
            // Found the originating hop
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

}  // namespace rnet::lightning

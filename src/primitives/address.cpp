#include "primitives/address.h"

#include <cstring>

#include "core/base58.h"
#include "core/bech32.h"
#include "crypto/hash.h"

namespace rnet::primitives {

// ---------------------------------------------------------------------------
// Bech32 addresses (P2WPKH, P2WSH)
// ---------------------------------------------------------------------------

std::string encode_p2wpkh_address(const uint8_t* hash160, NetworkType net) {
    auto hrp = get_bech32_hrp(net);
    return core::encode_segwit_addr(hrp, 0,
        std::span<const uint8_t>(hash160, 20));
}

std::string encode_p2wsh_address(const uint8_t* hash256, NetworkType net) {
    auto hrp = get_bech32_hrp(net);
    return core::encode_segwit_addr(hrp, 0,
        std::span<const uint8_t>(hash256, 32));
}

// ---------------------------------------------------------------------------
// Base58check addresses (P2PKH, P2SH)
// ---------------------------------------------------------------------------

/// Checksum function for base58check using keccak256d
static void keccak_checksum(const uint8_t* data, size_t len,
                            uint8_t out[32]) {
    auto h = crypto::hash256(std::span<const uint8_t>(data, len));
    std::memcpy(out, h.data(), 32);
}

std::string encode_p2pkh_address(const uint8_t* hash160, NetworkType net) {
    std::vector<uint8_t> payload(21);
    payload[0] = get_p2pkh_version(net);
    std::memcpy(payload.data() + 1, hash160, 20);
    return core::base58check_encode(
        std::span<const uint8_t>(payload), keccak_checksum);
}

std::string encode_p2sh_address(const uint8_t* hash160, NetworkType net) {
    std::vector<uint8_t> payload(21);
    payload[0] = get_p2sh_version(net);
    std::memcpy(payload.data() + 1, hash160, 20);
    return core::base58check_encode(
        std::span<const uint8_t>(payload), keccak_checksum);
}

// ---------------------------------------------------------------------------
// Public key -> address
// ---------------------------------------------------------------------------

std::string encode_pubkey_address(const uint8_t* pubkey_32,
                                  AddressType type, NetworkType net) {
    // Compute Hash160 of the public key
    auto h160 = crypto::hash160(
        std::span<const uint8_t>(pubkey_32, 32));

    switch (type) {
        case AddressType::P2WPKH:
            return encode_p2wpkh_address(h160.data(), net);
        case AddressType::P2PKH:
            return encode_p2pkh_address(h160.data(), net);
        default:
            return {};  // Other types need more than just a pubkey
    }
}

// ---------------------------------------------------------------------------
// Decode address
// ---------------------------------------------------------------------------

rnet::Result<DecodedAddress> decode_address(std::string_view addr,
                                            NetworkType net) {
    DecodedAddress result;
    result.network = net;

    // Try bech32/segwit first
    auto hrp = get_bech32_hrp(net);
    auto segwit = core::decode_segwit_addr(hrp, addr);
    if (segwit.valid) {
        result.witness_version = segwit.witness_version;
        result.hash = std::move(segwit.witness_program);

        if (segwit.witness_version == 0) {
            if (result.hash.size() == 20) {
                result.type = AddressType::P2WPKH;
            } else if (result.hash.size() == 32) {
                result.type = AddressType::P2WSH;
            } else {
                return rnet::Result<DecodedAddress>::err(
                    "invalid witness program length");
            }
        } else {
            // Future witness versions
            result.type = AddressType::UNKNOWN;
        }

        return rnet::Result<DecodedAddress>::ok(std::move(result));
    }

    // Try base58check
    auto decoded = core::base58check_decode(addr, keccak_checksum);
    if (decoded.has_value() && decoded->size() >= 21) {
        uint8_t version = (*decoded)[0];
        result.witness_version = -1;
        result.hash.assign(decoded->begin() + 1, decoded->end());

        if (version == get_p2pkh_version(net) && result.hash.size() == 20) {
            result.type = AddressType::P2PKH;
            return rnet::Result<DecodedAddress>::ok(std::move(result));
        }
        if (version == get_p2sh_version(net) && result.hash.size() == 20) {
            result.type = AddressType::P2SH;
            return rnet::Result<DecodedAddress>::ok(std::move(result));
        }

        return rnet::Result<DecodedAddress>::err(
            "unknown address version byte");
    }

    return rnet::Result<DecodedAddress>::err(
        "failed to decode address: invalid format");
}

bool is_valid_address(std::string_view addr, NetworkType net) {
    auto result = decode_address(addr, net);
    return result.is_ok();
}

// ---------------------------------------------------------------------------
// Script <-> Address type
// ---------------------------------------------------------------------------

AddressType address_type_from_script(const std::vector<uint8_t>& script) {
    // P2WPKH: OP_0 OP_PUSHBYTES_20 <20 bytes> (size = 22)
    if (script.size() == 22 && script[0] == 0x00 && script[1] == 0x14) {
        return AddressType::P2WPKH;
    }

    // P2WSH: OP_0 OP_PUSHBYTES_32 <32 bytes> (size = 34)
    if (script.size() == 34 && script[0] == 0x00 && script[1] == 0x20) {
        return AddressType::P2WSH;
    }

    // P2PKH: OP_DUP OP_HASH160 OP_PUSHBYTES_20 <20 bytes> OP_EQUALVERIFY OP_CHECKSIG
    // size = 25: [0x76][0xa9][0x14][20 bytes][0x88][0xac]
    if (script.size() == 25 && script[0] == 0x76 && script[1] == 0xa9 &&
        script[2] == 0x14 && script[23] == 0x88 && script[24] == 0xac) {
        return AddressType::P2PKH;
    }

    // P2SH: OP_HASH160 OP_PUSHBYTES_20 <20 bytes> OP_EQUAL
    // size = 23: [0xa9][0x14][20 bytes][0x87]
    if (script.size() == 23 && script[0] == 0xa9 && script[1] == 0x14 &&
        script[22] == 0x87) {
        return AddressType::P2SH;
    }

    return AddressType::UNKNOWN;
}

std::vector<uint8_t> script_from_address(const DecodedAddress& addr) {
    switch (addr.type) {
        case AddressType::P2WPKH:
            if (addr.hash.size() == 20) {
                std::vector<uint8_t> script(22);
                script[0] = 0x00;
                script[1] = 0x14;
                std::memcpy(script.data() + 2, addr.hash.data(), 20);
                return script;
            }
            break;

        case AddressType::P2WSH:
            if (addr.hash.size() == 32) {
                std::vector<uint8_t> script(34);
                script[0] = 0x00;
                script[1] = 0x20;
                std::memcpy(script.data() + 2, addr.hash.data(), 32);
                return script;
            }
            break;

        case AddressType::P2PKH:
            if (addr.hash.size() == 20) {
                std::vector<uint8_t> script(25);
                script[0] = 0x76;   // OP_DUP
                script[1] = 0xa9;   // OP_HASH160
                script[2] = 0x14;   // Push 20 bytes
                std::memcpy(script.data() + 3, addr.hash.data(), 20);
                script[23] = 0x88;  // OP_EQUALVERIFY
                script[24] = 0xac;  // OP_CHECKSIG
                return script;
            }
            break;

        case AddressType::P2SH:
            if (addr.hash.size() == 20) {
                std::vector<uint8_t> script(23);
                script[0] = 0xa9;   // OP_HASH160
                script[1] = 0x14;   // Push 20 bytes
                std::memcpy(script.data() + 2, addr.hash.data(), 20);
                script[22] = 0x87;  // OP_EQUAL
                return script;
            }
            break;

        default:
            break;
    }

    return {};
}

std::string_view address_type_name(AddressType type) {
    switch (type) {
        case AddressType::P2PKH:  return "P2PKH";
        case AddressType::P2SH:   return "P2SH";
        case AddressType::P2WPKH: return "P2WPKH";
        case AddressType::P2WSH:  return "P2WSH";
        default:                  return "UNKNOWN";
    }
}

}  // namespace rnet::primitives

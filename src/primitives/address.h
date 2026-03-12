#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"
#include "core/error.h"

namespace rnet::primitives {

/// Address types supported by ResonanceNet.
enum class AddressType : uint8_t {
    P2PKH  = 0,    ///< Pay-to-PubKey-Hash (legacy, base58check)
    P2SH   = 1,    ///< Pay-to-Script-Hash (legacy, base58check)
    P2WPKH = 2,    ///< Pay-to-Witness-PubKey-Hash (bech32, segwit v0)
    P2WSH  = 3,    ///< Pay-to-Witness-Script-Hash (bech32, segwit v0)
    UNKNOWN = 0xFF,
};

/// Network type for address encoding.
enum class NetworkType : uint8_t {
    MAINNET = 0,
    TESTNET = 1,
    REGTEST = 2,
};

/// Bech32 human-readable prefix by network.
inline std::string_view get_bech32_hrp(NetworkType net) {
    switch (net) {
        case NetworkType::MAINNET: return "rn";
        case NetworkType::TESTNET: return "trn";
        case NetworkType::REGTEST: return "rnrt";
    }
    return "rn";
}

/// Base58 version byte for P2PKH addresses by network.
inline uint8_t get_p2pkh_version(NetworkType net) {
    switch (net) {
        case NetworkType::MAINNET: return 0x3C;  // Starts with 'R'
        case NetworkType::TESTNET: return 0x6F;  // Starts with 'm' or 'n'
        case NetworkType::REGTEST: return 0x6F;
    }
    return 0x3C;
}

/// Base58 version byte for P2SH addresses by network.
inline uint8_t get_p2sh_version(NetworkType net) {
    switch (net) {
        case NetworkType::MAINNET: return 0x3E;  // Starts with 'S'
        case NetworkType::TESTNET: return 0xC4;
        case NetworkType::REGTEST: return 0xC4;
    }
    return 0x3E;
}

/// Decoded address information.
struct DecodedAddress {
    AddressType type = AddressType::UNKNOWN;
    NetworkType network = NetworkType::MAINNET;
    std::vector<uint8_t> hash;     ///< 20 bytes for P2PKH/P2WPKH, 32 for P2WSH
    int witness_version = -1;      ///< -1 for legacy, 0+ for segwit
};

/// Encode a P2WPKH address from a 20-byte Hash160.
/// Uses bech32 encoding with HRP "rn" (mainnet) or "trn" (testnet).
std::string encode_p2wpkh_address(const uint8_t* hash160,
                                  NetworkType net = NetworkType::MAINNET);

/// Encode a P2WSH address from a 32-byte script hash.
std::string encode_p2wsh_address(const uint8_t* hash256,
                                 NetworkType net = NetworkType::MAINNET);

/// Encode a P2PKH address from a 20-byte Hash160 (base58check).
std::string encode_p2pkh_address(const uint8_t* hash160,
                                 NetworkType net = NetworkType::MAINNET);

/// Encode a P2SH address from a 20-byte script hash (base58check).
std::string encode_p2sh_address(const uint8_t* hash160,
                                NetworkType net = NetworkType::MAINNET);

/// Encode an address from a public key (defaults to P2WPKH).
/// Computes Hash160 of the Ed25519 public key.
std::string encode_pubkey_address(const uint8_t* pubkey_32,
                                  AddressType type = AddressType::P2WPKH,
                                  NetworkType net = NetworkType::MAINNET);

/// Decode any ResonanceNet address string.
/// Tries bech32 first, then base58check.
rnet::Result<DecodedAddress> decode_address(std::string_view addr,
                                            NetworkType net = NetworkType::MAINNET);

/// Validate an address string without fully decoding.
bool is_valid_address(std::string_view addr,
                      NetworkType net = NetworkType::MAINNET);

/// Get the address type from a scriptPubKey.
AddressType address_type_from_script(const std::vector<uint8_t>& script);

/// Create a scriptPubKey from a decoded address.
std::vector<uint8_t> script_from_address(const DecodedAddress& addr);

/// Human-readable name for an address type.
std::string_view address_type_name(AddressType type);

}  // namespace rnet::primitives

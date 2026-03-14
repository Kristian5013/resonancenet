// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "rpc/misc.h"

#include "core/hex.h"
#include "core/logging.h"
#include "node/context.h"
#include "primitives/address.h"

namespace rnet::rpc {

// ===========================================================================
//  validateaddress
// ===========================================================================

// ---------------------------------------------------------------------------
// Validates a ResonanceNet address and returns metadata (type, witness flag).
// Supports bech32 prefixes (rn1, trn1, rnrt1) and legacy base58.
// ---------------------------------------------------------------------------
static JsonValue rpc_validateaddress(const RPCRequest& req,
                                     node::NodeContext& ctx) {
    // 1. Extract the address parameter.
    const auto& addr_param = get_param(req, 0);
    if (!addr_param.is_string()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "address required (string)");
    }

    std::string address = addr_param.as_string();

    JsonValue result = JsonValue::object();
    result.set("address", JsonValue(address));

    // 2. Determine address type by prefix.
    bool is_valid = false;
    std::string address_type = "unknown";

    // 3. Check bech32 addresses (rn1..., trn1..., rnrt1...).
    if (address.substr(0, 3) == "rn1" ||
        address.substr(0, 4) == "trn1" ||
        address.substr(0, 5) == "rnrt1") {
        is_valid = address.size() >= 14 && address.size() <= 90;
        if (is_valid) {
            address_type = "witness_v0_keyhash";  // P2WPKH most common
        }
    } else if (address.size() >= 25 && address.size() <= 36) {
        // 4. Could be base58 (P2PKH or P2SH) -- basic length check only.
        is_valid = true;
        address_type = "pubkeyhash";
    }

    // 5. Populate the result fields.
    result.set("isvalid", JsonValue(is_valid));
    if (is_valid) {
        result.set("scriptPubKey", JsonValue(std::string("")));
        result.set("isscript", JsonValue(false));
        result.set("iswitness",
                    JsonValue(address_type.find("witness") != std::string::npos));
        result.set("address_type", JsonValue(address_type));
    }

    return result;
}

// ===========================================================================
//  createmultisig
// ===========================================================================

// ---------------------------------------------------------------------------
// Creates a multi-signature redeem script from N-of-M Ed25519 public keys.
// Returns the redeem script hex and a placeholder address.
// ---------------------------------------------------------------------------
static JsonValue rpc_createmultisig(const RPCRequest& req,
                                    node::NodeContext& ctx) {
    // 1. Validate required parameters.
    const auto& nrequired_param = get_param(req, 0);
    const auto& keys_param = get_param(req, 1);

    if (!nrequired_param.is_int()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "nrequired (int) is required");
    }
    if (!keys_param.is_array()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "keys (array of hex pubkeys) is required");
    }

    // 2. Range-check nrequired vs number of keys.
    int nrequired = static_cast<int>(nrequired_param.as_int());
    int nkeys = static_cast<int>(keys_param.size());

    if (nrequired < 1 || nrequired > nkeys) {
        return make_rpc_error(RPC_INVALID_PARAMETER,
                              "nrequired must be between 1 and number of keys");
    }
    if (nkeys > 16) {
        return make_rpc_error(RPC_INVALID_PARAMETER,
                              "too many keys (max 16)");
    }

    // 3. Collect and validate each public key (32-byte Ed25519 = 64 hex).
    std::vector<std::string> pubkeys;
    for (size_t i = 0; i < keys_param.size(); ++i) {
        const auto& key = keys_param[i];
        if (!key.is_string()) {
            return make_rpc_error(RPC_INVALID_PARAMETER,
                                  "each key must be a hex string");
        }
        std::string hex_key = key.as_string();
        if (hex_key.size() != 64) {
            return make_rpc_error(RPC_INVALID_ADDRESS_OR_KEY,
                                  "invalid public key length (expected 32 bytes / 64 hex)");
        }
        if (!is_valid_hex(hex_key)) {
            return make_rpc_error(RPC_INVALID_ADDRESS_OR_KEY,
                                  "invalid hex in public key");
        }
        pubkeys.push_back(hex_key);
    }

    // 4. Build multisig script: OP_N <key1> ... <keyN> OP_M OP_CHECKMULTISIG.
    std::string redeem_script_hex;
    redeem_script_hex += bytes_to_hex(
        reinterpret_cast<const uint8_t*>(&nrequired), 1);
    for (const auto& pk : pubkeys) {
        uint8_t push_len = 32;  // Ed25519 pubkey is 32 bytes
        redeem_script_hex += bytes_to_hex(&push_len, 1);
        redeem_script_hex += pk;
    }
    uint8_t m = static_cast<uint8_t>(nkeys);
    redeem_script_hex += bytes_to_hex(&m, 1);
    // OP_CHECKMULTISIG = 0xAE
    uint8_t op_cms = 0xAE;
    redeem_script_hex += bytes_to_hex(&op_cms, 1);

    // 5. Return the result.
    JsonValue result = JsonValue::object();
    result.set("address", JsonValue(std::string("multisig_placeholder")));
    result.set("redeemScript", JsonValue(redeem_script_hex));
    result.set("descriptor",
               JsonValue(std::string("multi(" + std::to_string(nrequired) +
                                     ",...)")));

    return result;
}

// ===========================================================================
//  signmessage
// ===========================================================================

// ---------------------------------------------------------------------------
// Signs a message with the private key belonging to the given address.
// Requires wallet access (placeholder implementation).
// ---------------------------------------------------------------------------
static JsonValue rpc_signmessage(const RPCRequest& req,
                                 node::NodeContext& ctx) {
    // 1. Validate parameters.
    const auto& addr_param = get_param(req, 0);
    const auto& msg_param = get_param(req, 1);

    if (!addr_param.is_string()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "address required (string)");
    }
    if (!msg_param.is_string()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "message required (string)");
    }

    std::string address = addr_param.as_string();
    std::string message = msg_param.as_string();

    // 2. In a full implementation, we would:
    //    a. Find the private key for this address in the wallet.
    //    b. Sign the message with Ed25519.
    //    c. Return the base64-encoded signature.

    LogPrint(RPC, "signmessage: addr=%s msg_len=%zu",
             address.c_str(), message.size());

    // 3. Placeholder signature.
    return JsonValue(std::string(
        "placeholder_signature_base64_would_go_here"));
}

// ===========================================================================
//  verifymessage
// ===========================================================================

// ---------------------------------------------------------------------------
// Verifies an Ed25519 signature against an address and message.
// Placeholder -- always returns false until crypto integration is complete.
// ---------------------------------------------------------------------------
static JsonValue rpc_verifymessage(const RPCRequest& req,
                                   node::NodeContext& ctx) {
    // 1. Validate all three parameters.
    const auto& addr_param = get_param(req, 0);
    const auto& sig_param = get_param(req, 1);
    const auto& msg_param = get_param(req, 2);

    if (!addr_param.is_string()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "address required (string)");
    }
    if (!sig_param.is_string()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "signature required (base64 string)");
    }
    if (!msg_param.is_string()) {
        return make_rpc_error(RPC_INVALID_PARAMS,
                              "message required (string)");
    }

    std::string address = addr_param.as_string();
    std::string signature = sig_param.as_string();
    std::string message = msg_param.as_string();

    // 2. In a full implementation:
    //    a. Decode the base64 signature.
    //    b. Recover/verify the Ed25519 public key from address.
    //    c. Verify the signature against the message.
    //    d. Check that the recovered pubkey matches the address.

    LogPrint(RPC, "verifymessage: addr=%s sig_len=%zu msg_len=%zu",
             address.c_str(), signature.size(), message.size());

    // 3. Placeholder: return false (unverified).
    return JsonValue(false);
}

// ===========================================================================
//  Registration
// ===========================================================================

// ---------------------------------------------------------------------------
// register_misc_rpcs -- add all misc/utility RPCs to the command table.
// ---------------------------------------------------------------------------
void register_misc_rpcs(RPCTable& table) {
    table.register_command({
        "validateaddress",
        rpc_validateaddress,
        "Return information about the given ResonanceNet address.\n"
        "Arguments: address (string)",
        "Util"
    });

    table.register_command({
        "createmultisig",
        rpc_createmultisig,
        "Creates a multi-signature address.\n"
        "Arguments: nrequired (int), keys (array of hex pubkeys)",
        "Util"
    });

    table.register_command({
        "signmessage",
        rpc_signmessage,
        "Sign a message with the private key of an address.\n"
        "Arguments: address (string), message (string)",
        "Util"
    });

    table.register_command({
        "verifymessage",
        rpc_verifymessage,
        "Verify a signed message.\n"
        "Arguments: address (string), signature (base64), message (string)",
        "Util"
    });
}

} // namespace rnet::rpc

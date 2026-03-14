// Copyright (c) 2025-2026 The ResonanceNet Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "wallet/sign_tx.h"

#include "core/logging.h"
#include "script/script.h"
#include "script/standard.h"

namespace rnet::wallet {

// ===========================================================================
//  Transaction signing -- Ed25519 signatures via the script engine
// ===========================================================================
//
//  WalletSigningProvider adapts the KeyStore into the interface expected
//  by script::sign_transaction.  For each input the signer extracts the
//  20-byte pubkey hash from the previous output's P2WPKH script, looks
//  up the Ed25519 secret in the keystore, and produces a witness with
//  [signature, pubkey].

// ---------------------------------------------------------------------------
// WalletSigningProvider -- bridge between KeyStore and script signer
// ---------------------------------------------------------------------------

WalletSigningProvider::WalletSigningProvider(const KeyStore& keys)
    : keys_(keys) {}

bool WalletSigningProvider::get_ed25519_key(
    const std::vector<uint8_t>& pubkey_hash,
    std::vector<uint8_t>& secret_out,
    std::vector<uint8_t>& pubkey_out) const {

    if (pubkey_hash.size() != 20) return false;

    uint160 hash(std::span<const uint8_t>(pubkey_hash.data(), pubkey_hash.size()));
    return keys_.get_signing_key(hash, secret_out, pubkey_out);
}

// ---------------------------------------------------------------------------
// sign_wallet_transaction -- sign all inputs, return count of signed
// ---------------------------------------------------------------------------

int sign_wallet_transaction(
    const KeyStore& keys,
    primitives::CMutableTransaction& mtx,
    const std::vector<WalletCoin>& spent_coins) {

    // 1. Validate input count matches spent-coin metadata.
    if (mtx.vin.size() != spent_coins.size()) {
        LogPrint(WALLET, "sign_wallet_transaction: input count mismatch");
        return 0;
    }

    // 2. Build the signing provider and previous-output vectors.
    WalletSigningProvider provider(keys);

    std::vector<script::CScript> prev_scripts;
    std::vector<int64_t> prev_amounts;

    prev_scripts.reserve(spent_coins.size());
    prev_amounts.reserve(spent_coins.size());

    for (const auto& coin : spent_coins) {
        prev_scripts.emplace_back(
            coin.txout.script_pub_key.begin(),
            coin.txout.script_pub_key.end());
        prev_amounts.push_back(coin.txout.value);
    }

    // 3. Delegate to the script engine.
    return script::sign_transaction(provider, mtx, prev_scripts, prev_amounts);
}

// ---------------------------------------------------------------------------
// is_fully_signed -- check that every input carries a valid witness/scriptsig
// ---------------------------------------------------------------------------

bool is_fully_signed(const primitives::CMutableTransaction& tx) {
    for (const auto& txin : tx.vin) {
        // 1. For P2WPKH, witness must have 2 items (sig + pubkey).
        if (txin.witness.stack.size() < 2) {
            // 2. Check if it has a non-empty scriptSig (legacy fallback).
            if (txin.script_sig.empty()) {
                return false;
            }
        }
    }
    return true;
}

} // namespace rnet::wallet

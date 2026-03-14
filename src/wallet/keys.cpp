// Copyright (c) 2025-2026 The ResonanceNet Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "wallet/keys.h"

#include "core/logging.h"
#include "core/random.h"
#include "crypto/hash.h"

#include <chrono>

namespace rnet::wallet {

// ===========================================================================
//  KeyStore -- in-memory Ed25519 key pool indexed by Hash160(pubkey)
// ===========================================================================
//
//  Keys are stored in a map<uint160, WalletKey>.  The uint160 is the
//  first 20 bytes of Keccak256d(pubkey), matching the P2WPKH witness
//  program used on-chain.
//
//  HD-derived keys carry is_hd=true and record their BIP32 index/chain
//  so the wallet can re-derive them from the mnemonic during recovery.

// ---------------------------------------------------------------------------
// WalletKey::wipe -- zero the secret key material
// ---------------------------------------------------------------------------

void WalletKey::wipe() {
    secret.wipe();
}

// ---------------------------------------------------------------------------
// Destructor -- wipe all keys on teardown
// ---------------------------------------------------------------------------

KeyStore::~KeyStore() {
    clear();
}

// ---------------------------------------------------------------------------
// add_key -- insert a pre-built key (rejects duplicates)
// ---------------------------------------------------------------------------

Result<void> KeyStore::add_key(const WalletKey& key) {
    LOCK(mutex_);
    if (keys_.count(key.pubkey_hash)) {
        return Result<void>::err("key already exists");
    }
    keys_[key.pubkey_hash] = key;
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// generate_key -- create a random (non-HD) Ed25519 keypair
// ---------------------------------------------------------------------------

Result<WalletKey> KeyStore::generate_key(const std::string& label) {
    // 1. Generate a fresh Ed25519 keypair.
    auto kp_result = crypto::ed25519_generate();
    if (!kp_result) {
        return Result<WalletKey>::err("ed25519 key generation failed");
    }
    auto& kp = kp_result.value();

    // 2. Populate WalletKey fields.
    WalletKey wk;
    wk.secret = kp.secret;
    wk.pubkey = kp.public_key;
    wk.pubkey_hash = compute_pubkey_hash(kp.public_key);
    wk.creation_time = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    wk.label = label;
    wk.is_hd = false;

    // 3. Add to the store.
    auto add_result = add_key(wk);
    if (!add_result) {
        return Result<WalletKey>::err(add_result.error());
    }
    return Result<WalletKey>::ok(std::move(wk));
}

// ---------------------------------------------------------------------------
// Lookups
// ---------------------------------------------------------------------------

bool KeyStore::have_key(const uint160& pubkey_hash) const {
    LOCK(mutex_);
    return keys_.count(pubkey_hash) > 0;
}

Result<WalletKey> KeyStore::get_key(const uint160& pubkey_hash) const {
    LOCK(mutex_);
    auto it = keys_.find(pubkey_hash);
    if (it == keys_.end()) {
        return Result<WalletKey>::err("key not found");
    }
    return Result<WalletKey>::ok(it->second);
}

Result<crypto::Ed25519PublicKey> KeyStore::get_pubkey(const uint160& pubkey_hash) const {
    LOCK(mutex_);
    auto it = keys_.find(pubkey_hash);
    if (it == keys_.end()) {
        return Result<crypto::Ed25519PublicKey>::err("key not found");
    }
    return Result<crypto::Ed25519PublicKey>::ok(it->second.pubkey);
}

// ---------------------------------------------------------------------------
// get_signing_key -- export raw secret + pubkey bytes for the signer
// ---------------------------------------------------------------------------

bool KeyStore::get_signing_key(const uint160& pubkey_hash,
                               std::vector<uint8_t>& secret_out,
                               std::vector<uint8_t>& pubkey_out) const {
    LOCK(mutex_);
    auto it = keys_.find(pubkey_hash);
    if (it == keys_.end()) {
        return false;
    }
    secret_out.assign(it->second.secret.data.begin(), it->second.secret.data.end());
    pubkey_out.assign(it->second.pubkey.data.begin(), it->second.pubkey.data.end());
    return true;
}

// ---------------------------------------------------------------------------
// get_all_pubkey_hashes -- enumerate every key in the store
// ---------------------------------------------------------------------------

std::vector<uint160> KeyStore::get_all_pubkey_hashes() const {
    LOCK(mutex_);
    std::vector<uint160> result;
    result.reserve(keys_.size());
    for (const auto& [hash, _] : keys_) {
        result.push_back(hash);
    }
    return result;
}

// ---------------------------------------------------------------------------
// size / clear
// ---------------------------------------------------------------------------

size_t KeyStore::size() const {
    LOCK(mutex_);
    return keys_.size();
}

void KeyStore::clear() {
    LOCK(mutex_);
    for (auto& [_, key] : keys_) {
        key.wipe();
    }
    keys_.clear();
}

// ---------------------------------------------------------------------------
// compute_pubkey_hash -- Hash160 = first 20 bytes of Keccak256d(pubkey)
// ---------------------------------------------------------------------------

uint160 compute_pubkey_hash(const crypto::Ed25519PublicKey& pubkey) {
    return crypto::hash160(std::span<const uint8_t>(pubkey.data.data(), pubkey.data.size()));
}

} // namespace rnet::wallet

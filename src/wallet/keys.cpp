#include "wallet/keys.h"

#include "core/logging.h"
#include "core/random.h"
#include "crypto/hash.h"

#include <chrono>

namespace rnet::wallet {

void WalletKey::wipe() {
    secret.wipe();
}

KeyStore::~KeyStore() {
    clear();
}

Result<void> KeyStore::add_key(const WalletKey& key) {
    LOCK(mutex_);
    if (keys_.count(key.pubkey_hash)) {
        return Result<void>::err("key already exists");
    }
    keys_[key.pubkey_hash] = key;
    return Result<void>::ok();
}

Result<WalletKey> KeyStore::generate_key(const std::string& label) {
    auto kp_result = crypto::ed25519_generate();
    if (!kp_result) {
        return Result<WalletKey>::err("ed25519 key generation failed");
    }
    auto& kp = kp_result.value();

    WalletKey wk;
    wk.secret = kp.secret;
    wk.pubkey = kp.public_key;
    wk.pubkey_hash = compute_pubkey_hash(kp.public_key);
    wk.creation_time = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    wk.label = label;
    wk.is_hd = false;

    auto add_result = add_key(wk);
    if (!add_result) {
        return Result<WalletKey>::err(add_result.error());
    }
    return Result<WalletKey>::ok(std::move(wk));
}

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

std::vector<uint160> KeyStore::get_all_pubkey_hashes() const {
    LOCK(mutex_);
    std::vector<uint160> result;
    result.reserve(keys_.size());
    for (const auto& [hash, _] : keys_) {
        result.push_back(hash);
    }
    return result;
}

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

uint160 compute_pubkey_hash(const crypto::Ed25519PublicKey& pubkey) {
    return crypto::hash160(std::span<const uint8_t>(pubkey.data.data(), pubkey.data.size()));
}

}  // namespace rnet::wallet

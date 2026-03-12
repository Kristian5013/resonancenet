#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "core/error.h"
#include "core/sync.h"
#include "core/types.h"
#include "crypto/ed25519.h"

namespace rnet::wallet {

/// A wallet key entry: Ed25519 keypair + metadata.
struct WalletKey {
    crypto::Ed25519SecretKey secret;
    crypto::Ed25519PublicKey pubkey;
    uint160 pubkey_hash;          ///< Hash160(pubkey)
    int64_t creation_time = 0;    ///< Unix timestamp
    std::string label;
    bool is_hd = false;
    uint32_t hd_index = 0;
    uint32_t hd_change = 0;

    /// Securely wipe secret key material.
    void wipe();
};

/// KeyStore: manages Ed25519 keys for the wallet.
/// Thread-safe via internal mutex.
class KeyStore {
public:
    KeyStore() = default;
    ~KeyStore();

    KeyStore(const KeyStore&) = delete;
    KeyStore& operator=(const KeyStore&) = delete;

    /// Add a key to the store.
    Result<void> add_key(const WalletKey& key);

    /// Generate a new random key (non-HD).
    Result<WalletKey> generate_key(const std::string& label = "");

    /// Check if we have a key for the given pubkey hash.
    bool have_key(const uint160& pubkey_hash) const;

    /// Get the key for a given pubkey hash.
    Result<WalletKey> get_key(const uint160& pubkey_hash) const;

    /// Get the public key for a given pubkey hash.
    Result<crypto::Ed25519PublicKey> get_pubkey(const uint160& pubkey_hash) const;

    /// Get the secret key bytes (seed + pubkey) for signing.
    bool get_signing_key(const uint160& pubkey_hash,
                         std::vector<uint8_t>& secret_out,
                         std::vector<uint8_t>& pubkey_out) const;

    /// Get all pubkey hashes in the store.
    std::vector<uint160> get_all_pubkey_hashes() const;

    /// Number of keys stored.
    size_t size() const;

    /// Remove all keys (with secure wipe).
    void clear();

private:
    mutable core::Mutex mutex_;
    std::map<uint160, WalletKey> keys_;  ///< pubkey_hash -> key
};

/// Compute Hash160 of an Ed25519 public key.
uint160 compute_pubkey_hash(const crypto::Ed25519PublicKey& pubkey);

}  // namespace rnet::wallet

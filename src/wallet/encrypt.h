#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/error.h"
#include "core/sync.h"
#include "core/types.h"

namespace rnet::wallet {

/// Wallet encryption state.
enum class EncryptionState {
    UNENCRYPTED,
    LOCKED,
    UNLOCKED,
};

/// WalletEncryptor: manages wallet encryption/decryption.
/// Uses passphrase -> Argon2id -> AES-256-CBC key derivation.
class WalletEncryptor {
public:
    WalletEncryptor() = default;

    /// Encrypt the wallet with a passphrase.
    /// Derives an encryption key via Argon2id, encrypts all private keys.
    Result<void> encrypt(const std::string& passphrase);

    /// Unlock an encrypted wallet (makes keys available for signing).
    Result<void> unlock(const std::string& passphrase);

    /// Lock the wallet (wipe decrypted keys from memory).
    void lock();

    /// Change the encryption passphrase.
    Result<void> change_passphrase(const std::string& old_passphrase,
                                   const std::string& new_passphrase);

    /// Get the current encryption state.
    EncryptionState state() const;

    /// Check if wallet is encrypted.
    bool is_encrypted() const;

    /// Check if wallet is unlocked.
    bool is_unlocked() const;

    /// Encrypt raw data with the current encryption key.
    Result<std::vector<uint8_t>> encrypt_data(
        std::span<const uint8_t> plaintext) const;

    /// Decrypt raw data with the current encryption key.
    Result<std::vector<uint8_t>> decrypt_data(
        std::span<const uint8_t> ciphertext) const;

    /// Get the encryption salt (for serialization).
    const std::array<uint8_t, 32>& salt() const { return salt_; }

    /// Set the encryption salt (when loading from DB).
    void set_salt(const std::array<uint8_t, 32>& s) { salt_ = s; }

    /// Set the encrypted master key (when loading from DB).
    void set_encrypted_key(const std::vector<uint8_t>& enc_key);

    /// Get the encrypted master key (for serialization).
    const std::vector<uint8_t>& get_encrypted_key() const { return encrypted_master_key_; }

private:
    mutable core::Mutex mutex_;
    EncryptionState state_ = EncryptionState::UNENCRYPTED;
    std::array<uint8_t, 32> salt_{};
    std::vector<uint8_t> encrypted_master_key_;
    std::array<uint8_t, 32> decrypted_key_{};  ///< Active AES key (zeroed when locked)

    /// Derive the AES key from passphrase + salt.
    Result<std::array<uint8_t, 32>> derive_key(const std::string& passphrase) const;

    /// Wipe the decrypted key from memory.
    void wipe_key();
};

}  // namespace rnet::wallet

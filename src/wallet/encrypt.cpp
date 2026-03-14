// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "wallet/encrypt.h"

#include "core/logging.h"
#include "core/random.h"
#include "crypto/aes.h"
#include "crypto/argon2.h"
#include "crypto/hash.h"

namespace rnet::wallet {

// ---------------------------------------------------------------------------
// Design note — wallet encryption at rest
//
// Private keys are protected by a two-layer encryption scheme:
//
//   1. A random 256-bit *master key* is generated once when the user first
//      encrypts the wallet.  All private-key material is encrypted with this
//      master key via AES-256-CBC.
//   2. The master key itself is encrypted with a *derived key* produced by
//      Argon2id(passphrase, salt).  The encrypted master key and salt are
//      stored in the wallet database.
//
// Unlocking the wallet re-derives the key from the passphrase, decrypts the
// master key into memory, and keeps it available until `lock()` is called,
// at which point the plaintext master key is securely wiped.
//
// Changing the passphrase only re-encrypts the master key — all data
// encrypted with the master key remains valid without re-encryption.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// WalletEncryptor::encrypt
// ---------------------------------------------------------------------------
Result<void> WalletEncryptor::encrypt(const std::string& passphrase) {
    LOCK(mutex_);

    // 1. Guard: only an unencrypted wallet can be encrypted.
    if (state_ != EncryptionState::UNENCRYPTED) {
        return Result<void>::err("wallet already encrypted");
    }
    if (passphrase.empty()) {
        return Result<void>::err("passphrase cannot be empty");
    }

    // 2. Generate a random salt for Argon2id.
    auto salt = crypto::generate_argon2_salt();
    salt_ = salt;

    // 3. Derive the wrapping key from the passphrase.
    auto key_result = derive_key(passphrase);
    if (!key_result) {
        return Result<void>::err("key derivation failed: " + key_result.error());
    }
    decrypted_key_ = key_result.value();

    // 4. Generate a random 256-bit master key.
    std::array<uint8_t, 32> master_key{};
    core::get_rand_bytes(std::span<uint8_t>(master_key.data(), master_key.size()));

    // 5. Encrypt the master key with the derived key.
    auto enc_result = crypto::AES256CBC::encrypt_with_random_iv(
        std::span<const uint8_t>(decrypted_key_.data(), decrypted_key_.size()),
        std::span<const uint8_t>(master_key.data(), master_key.size()));
    if (!enc_result) {
        return Result<void>::err("encryption failed: " + enc_result.error());
    }
    encrypted_master_key_ = std::move(enc_result.value());

    // 6. The active key for data encryption is the master key itself.
    decrypted_key_ = master_key;

    state_ = EncryptionState::UNLOCKED;
    LogPrint(WALLET, "wallet encrypted successfully");
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// WalletEncryptor::unlock
// ---------------------------------------------------------------------------
Result<void> WalletEncryptor::unlock(const std::string& passphrase) {
    LOCK(mutex_);

    // 1. Guard: wallet must be encrypted.
    if (state_ == EncryptionState::UNENCRYPTED) {
        return Result<void>::err("wallet is not encrypted");
    }
    if (state_ == EncryptionState::UNLOCKED) {
        return Result<void>::ok();  // Already unlocked.
    }

    // 2. Derive the wrapping key from the passphrase.
    auto key_result = derive_key(passphrase);
    if (!key_result) {
        return Result<void>::err("key derivation failed: " + key_result.error());
    }
    auto derived = key_result.value();

    // 3. Decrypt the master key.
    auto dec_result = crypto::AES256CBC::decrypt_with_iv_prefix(
        std::span<const uint8_t>(derived.data(), derived.size()),
        std::span<const uint8_t>(encrypted_master_key_.data(),
                                  encrypted_master_key_.size()));
    if (!dec_result) {
        return Result<void>::err("wrong passphrase");
    }

    // 4. Validate the decrypted master key length.
    auto& master = dec_result.value();
    if (master.size() != 32) {
        return Result<void>::err("decrypted key has wrong size");
    }

    // 5. Store the master key and transition to unlocked state.
    std::copy(master.begin(), master.end(), decrypted_key_.begin());
    state_ = EncryptionState::UNLOCKED;
    LogPrint(WALLET, "wallet unlocked");
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// WalletEncryptor::lock
// ---------------------------------------------------------------------------
void WalletEncryptor::lock() {
    LOCK(mutex_);

    // 1. Wipe the plaintext key and transition to locked state.
    if (state_ == EncryptionState::UNLOCKED) {
        wipe_key();
        state_ = EncryptionState::LOCKED;
        LogPrint(WALLET, "wallet locked");
    }
}

// ---------------------------------------------------------------------------
// WalletEncryptor::change_passphrase
// ---------------------------------------------------------------------------
Result<void> WalletEncryptor::change_passphrase(
    const std::string& old_passphrase,
    const std::string& new_passphrase) {
    LOCK(mutex_);

    // 1. Guard: wallet must be encrypted.
    if (state_ == EncryptionState::UNENCRYPTED) {
        return Result<void>::err("wallet is not encrypted");
    }
    if (new_passphrase.empty()) {
        return Result<void>::err("new passphrase cannot be empty");
    }

    // 2. Derive the old wrapping key and decrypt the master key.
    auto old_key_result = derive_key(old_passphrase);
    if (!old_key_result) {
        return Result<void>::err("old key derivation failed");
    }
    auto old_derived = old_key_result.value();

    auto dec_result = crypto::AES256CBC::decrypt_with_iv_prefix(
        std::span<const uint8_t>(old_derived.data(), old_derived.size()),
        std::span<const uint8_t>(encrypted_master_key_.data(),
                                  encrypted_master_key_.size()));
    if (!dec_result) {
        return Result<void>::err("wrong old passphrase");
    }
    auto& master = dec_result.value();

    // 3. Generate a new salt for the new passphrase.
    salt_ = crypto::generate_argon2_salt();

    // 4. Derive the new wrapping key.
    auto new_key_result = derive_key(new_passphrase);
    if (!new_key_result) {
        return Result<void>::err("new key derivation failed");
    }
    auto new_derived = new_key_result.value();

    // 5. Re-encrypt the master key with the new wrapping key.
    auto enc_result = crypto::AES256CBC::encrypt_with_random_iv(
        std::span<const uint8_t>(new_derived.data(), new_derived.size()),
        std::span<const uint8_t>(master.data(), master.size()));
    if (!enc_result) {
        return Result<void>::err("re-encryption failed");
    }
    encrypted_master_key_ = std::move(enc_result.value());

    LogPrint(WALLET, "wallet passphrase changed");
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// WalletEncryptor::state
// ---------------------------------------------------------------------------
EncryptionState WalletEncryptor::state() const {
    LOCK(mutex_);
    return state_;
}

// ---------------------------------------------------------------------------
// WalletEncryptor::is_encrypted
// ---------------------------------------------------------------------------
bool WalletEncryptor::is_encrypted() const {
    LOCK(mutex_);
    return state_ != EncryptionState::UNENCRYPTED;
}

// ---------------------------------------------------------------------------
// WalletEncryptor::is_unlocked
// ---------------------------------------------------------------------------
bool WalletEncryptor::is_unlocked() const {
    LOCK(mutex_);
    return state_ == EncryptionState::UNLOCKED ||
           state_ == EncryptionState::UNENCRYPTED;
}

// ---------------------------------------------------------------------------
// WalletEncryptor::encrypt_data
// ---------------------------------------------------------------------------
Result<std::vector<uint8_t>> WalletEncryptor::encrypt_data(
    std::span<const uint8_t> plaintext) const {
    LOCK(mutex_);

    // 1. Guard: wallet must be unlocked (or unencrypted) to encrypt data.
    if (state_ != EncryptionState::UNLOCKED &&
        state_ != EncryptionState::UNENCRYPTED) {
        return Result<std::vector<uint8_t>>::err("wallet is locked");
    }

    // 2. Encrypt with the active master key.
    return crypto::AES256CBC::encrypt_with_random_iv(
        std::span<const uint8_t>(decrypted_key_.data(), decrypted_key_.size()),
        plaintext);
}

// ---------------------------------------------------------------------------
// WalletEncryptor::decrypt_data
// ---------------------------------------------------------------------------
Result<std::vector<uint8_t>> WalletEncryptor::decrypt_data(
    std::span<const uint8_t> ciphertext) const {
    LOCK(mutex_);

    // 1. Guard: wallet must be unlocked (or unencrypted) to decrypt data.
    if (state_ != EncryptionState::UNLOCKED &&
        state_ != EncryptionState::UNENCRYPTED) {
        return Result<std::vector<uint8_t>>::err("wallet is locked");
    }

    // 2. Decrypt with the active master key.
    return crypto::AES256CBC::decrypt_with_iv_prefix(
        std::span<const uint8_t>(decrypted_key_.data(), decrypted_key_.size()),
        ciphertext);
}

// ---------------------------------------------------------------------------
// WalletEncryptor::set_encrypted_key
// ---------------------------------------------------------------------------
void WalletEncryptor::set_encrypted_key(const std::vector<uint8_t>& enc_key) {
    LOCK(mutex_);

    // 1. Store the encrypted master key blob.
    encrypted_master_key_ = enc_key;

    // 2. If a key was provided the wallet is encrypted but locked.
    if (!enc_key.empty()) {
        state_ = EncryptionState::LOCKED;
    }
}

// ---------------------------------------------------------------------------
// WalletEncryptor::derive_key
// ---------------------------------------------------------------------------
Result<std::array<uint8_t, 32>> WalletEncryptor::derive_key(
    const std::string& passphrase) const {
    // 1. Run Argon2id with the stored salt.
    auto result = crypto::argon2id_derive(passphrase,
        std::span<const uint8_t>(salt_.data(), salt_.size()));
    if (!result) {
        return Result<std::array<uint8_t, 32>>::err(
            "argon2id failed: " + result.error());
    }

    // 2. Truncate to 32 bytes for AES-256.
    auto& derived = result.value();
    if (derived.size() < 32) {
        return Result<std::array<uint8_t, 32>>::err("derived key too short");
    }
    std::array<uint8_t, 32> key{};
    std::copy_n(derived.begin(), 32, key.begin());
    return Result<std::array<uint8_t, 32>>::ok(key);
}

// ---------------------------------------------------------------------------
// WalletEncryptor::wipe_key
// ---------------------------------------------------------------------------
void WalletEncryptor::wipe_key() {
    crypto::secure_wipe(decrypted_key_.data(), decrypted_key_.size());
}

} // namespace rnet::wallet

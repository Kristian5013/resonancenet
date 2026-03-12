#include "wallet/encrypt.h"

#include "core/logging.h"
#include "core/random.h"
#include "crypto/aes.h"
#include "crypto/argon2.h"
#include "crypto/hash.h"

namespace rnet::wallet {

Result<void> WalletEncryptor::encrypt(const std::string& passphrase) {
    LOCK(mutex_);
    if (state_ != EncryptionState::UNENCRYPTED) {
        return Result<void>::err("wallet already encrypted");
    }
    if (passphrase.empty()) {
        return Result<void>::err("passphrase cannot be empty");
    }

    // Generate random salt
    auto salt = crypto::generate_argon2_salt();
    salt_ = salt;

    // Derive encryption key
    auto key_result = derive_key(passphrase);
    if (!key_result) {
        return Result<void>::err("key derivation failed: " + key_result.error());
    }
    decrypted_key_ = key_result.value();

    // Generate a random master key
    std::array<uint8_t, 32> master_key{};
    core::get_rand_bytes(std::span<uint8_t>(master_key.data(), master_key.size()));

    // Encrypt the master key with the derived key
    auto enc_result = crypto::AES256CBC::encrypt_with_random_iv(
        std::span<const uint8_t>(decrypted_key_.data(), decrypted_key_.size()),
        std::span<const uint8_t>(master_key.data(), master_key.size()));
    if (!enc_result) {
        return Result<void>::err("encryption failed: " + enc_result.error());
    }
    encrypted_master_key_ = std::move(enc_result.value());

    // The actual key used for data encryption is the master key
    decrypted_key_ = master_key;

    state_ = EncryptionState::UNLOCKED;
    LogPrint(WALLET, "wallet encrypted successfully");
    return Result<void>::ok();
}

Result<void> WalletEncryptor::unlock(const std::string& passphrase) {
    LOCK(mutex_);
    if (state_ == EncryptionState::UNENCRYPTED) {
        return Result<void>::err("wallet is not encrypted");
    }
    if (state_ == EncryptionState::UNLOCKED) {
        return Result<void>::ok();  // Already unlocked
    }

    // Derive key from passphrase
    auto key_result = derive_key(passphrase);
    if (!key_result) {
        return Result<void>::err("key derivation failed: " + key_result.error());
    }
    auto derived = key_result.value();

    // Decrypt the master key
    auto dec_result = crypto::AES256CBC::decrypt_with_iv_prefix(
        std::span<const uint8_t>(derived.data(), derived.size()),
        std::span<const uint8_t>(encrypted_master_key_.data(),
                                  encrypted_master_key_.size()));
    if (!dec_result) {
        return Result<void>::err("wrong passphrase");
    }

    auto& master = dec_result.value();
    if (master.size() != 32) {
        return Result<void>::err("decrypted key has wrong size");
    }

    std::copy(master.begin(), master.end(), decrypted_key_.begin());
    state_ = EncryptionState::UNLOCKED;
    LogPrint(WALLET, "wallet unlocked");
    return Result<void>::ok();
}

void WalletEncryptor::lock() {
    LOCK(mutex_);
    if (state_ == EncryptionState::UNLOCKED) {
        wipe_key();
        state_ = EncryptionState::LOCKED;
        LogPrint(WALLET, "wallet locked");
    }
}

Result<void> WalletEncryptor::change_passphrase(
    const std::string& old_passphrase,
    const std::string& new_passphrase) {
    LOCK(mutex_);
    if (state_ == EncryptionState::UNENCRYPTED) {
        return Result<void>::err("wallet is not encrypted");
    }
    if (new_passphrase.empty()) {
        return Result<void>::err("new passphrase cannot be empty");
    }

    // Decrypt with old passphrase
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

    // Generate new salt
    salt_ = crypto::generate_argon2_salt();

    // Derive new key
    auto new_key_result = derive_key(new_passphrase);
    if (!new_key_result) {
        return Result<void>::err("new key derivation failed");
    }
    auto new_derived = new_key_result.value();

    // Re-encrypt master key
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

EncryptionState WalletEncryptor::state() const {
    LOCK(mutex_);
    return state_;
}

bool WalletEncryptor::is_encrypted() const {
    LOCK(mutex_);
    return state_ != EncryptionState::UNENCRYPTED;
}

bool WalletEncryptor::is_unlocked() const {
    LOCK(mutex_);
    return state_ == EncryptionState::UNLOCKED ||
           state_ == EncryptionState::UNENCRYPTED;
}

Result<std::vector<uint8_t>> WalletEncryptor::encrypt_data(
    std::span<const uint8_t> plaintext) const {
    LOCK(mutex_);
    if (state_ != EncryptionState::UNLOCKED &&
        state_ != EncryptionState::UNENCRYPTED) {
        return Result<std::vector<uint8_t>>::err("wallet is locked");
    }
    return crypto::AES256CBC::encrypt_with_random_iv(
        std::span<const uint8_t>(decrypted_key_.data(), decrypted_key_.size()),
        plaintext);
}

Result<std::vector<uint8_t>> WalletEncryptor::decrypt_data(
    std::span<const uint8_t> ciphertext) const {
    LOCK(mutex_);
    if (state_ != EncryptionState::UNLOCKED &&
        state_ != EncryptionState::UNENCRYPTED) {
        return Result<std::vector<uint8_t>>::err("wallet is locked");
    }
    return crypto::AES256CBC::decrypt_with_iv_prefix(
        std::span<const uint8_t>(decrypted_key_.data(), decrypted_key_.size()),
        ciphertext);
}

void WalletEncryptor::set_encrypted_key(const std::vector<uint8_t>& enc_key) {
    LOCK(mutex_);
    encrypted_master_key_ = enc_key;
    if (!enc_key.empty()) {
        state_ = EncryptionState::LOCKED;
    }
}

Result<std::array<uint8_t, 32>> WalletEncryptor::derive_key(
    const std::string& passphrase) const {
    auto result = crypto::argon2id_derive(passphrase,
        std::span<const uint8_t>(salt_.data(), salt_.size()));
    if (!result) {
        return Result<std::array<uint8_t, 32>>::err(
            "argon2id failed: " + result.error());
    }
    auto& derived = result.value();
    if (derived.size() < 32) {
        return Result<std::array<uint8_t, 32>>::err("derived key too short");
    }
    std::array<uint8_t, 32> key{};
    std::copy_n(derived.begin(), 32, key.begin());
    return Result<std::array<uint8_t, 32>>::ok(key);
}

void WalletEncryptor::wipe_key() {
    crypto::secure_wipe(decrypted_key_.data(), decrypted_key_.size());
}

}  // namespace rnet::wallet

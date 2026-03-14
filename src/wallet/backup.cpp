// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "wallet/backup.h"

#include "core/logging.h"
#include "core/random.h"
#include "crypto/aes.h"
#include "crypto/argon2.h"
#include "crypto/keccak.h"

#include <cstring>
#include <fstream>

namespace rnet::wallet {

// ---------------------------------------------------------------------------
// Design note — Argon2id key derivation + AES-256-CBC encryption
//
// Wallet backups are encrypted at rest so that an attacker who obtains the
// file cannot recover private keys without the passphrase.  The scheme:
//
//   1. A 32-byte random salt is generated per backup.
//   2. The passphrase is fed through Argon2id(salt) to produce a 32-byte
//      AES-256 key.  Argon2id is memory-hard, protecting against GPU /
//      ASIC brute-force attacks.
//   3. The wallet payload is encrypted with AES-256-CBC using a random IV
//      (prepended to the ciphertext).
//   4. A Keccak-256d checksum of the *encrypted* payload is stored in the
//      header so integrity can be verified before attempting decryption.
//
// File layout (all little-endian):
//   [4B magic "RNBK"] [4B version] [32B salt] [32B checksum] [ciphertext...]
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// derive_backup_key
// ---------------------------------------------------------------------------
static Result<std::vector<uint8_t>> derive_backup_key(
    const std::string& passphrase,
    std::span<const uint8_t> salt) {

    return crypto::argon2id_derive(passphrase, salt);
}

// ---------------------------------------------------------------------------
// create_backup
// ---------------------------------------------------------------------------
Result<void> create_backup(const std::filesystem::path& path,
                           std::span<const uint8_t> payload,
                           const std::string& passphrase) {
    // 1. Validate inputs.
    if (passphrase.empty()) {
        return Result<void>::err("passphrase cannot be empty");
    }
    if (payload.empty()) {
        return Result<void>::err("payload is empty");
    }

    // 2. Generate a fresh random salt.
    auto salt = crypto::generate_argon2_salt();

    // 3. Derive the AES-256 key via Argon2id.
    auto key_result = derive_backup_key(passphrase,
        std::span<const uint8_t>(salt.data(), salt.size()));
    if (!key_result) {
        return Result<void>::err("key derivation failed: " + key_result.error());
    }
    auto& key = key_result.value();
    if (key.size() < 32) {
        return Result<void>::err("derived key too short");
    }

    // 4. Encrypt the payload with AES-256-CBC (random IV prepended).
    auto enc_result = crypto::AES256CBC::encrypt_with_random_iv(
        std::span<const uint8_t>(key.data(), 32),
        payload);
    if (!enc_result) {
        return Result<void>::err("encryption failed: " + enc_result.error());
    }
    auto& encrypted = enc_result.value();

    // 5. Compute a Keccak-256d checksum over the ciphertext.
    auto checksum = crypto::keccak256d(
        std::span<const uint8_t>(encrypted.data(), encrypted.size()));

    // 6. Write the backup file.
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return Result<void>::err("cannot open file for writing: " + path.string());
    }

    // 6a. Magic bytes.
    file.write(reinterpret_cast<const char*>(BACKUP_MAGIC), 4);

    // 6b. Version (little-endian uint32).
    uint32_t ver = BACKUP_VERSION;
    file.write(reinterpret_cast<const char*>(&ver), 4);

    // 6c. Salt.
    file.write(reinterpret_cast<const char*>(salt.data()), 32);

    // 6d. Checksum.
    file.write(reinterpret_cast<const char*>(checksum.data()), 32);

    // 6e. Encrypted payload.
    file.write(reinterpret_cast<const char*>(encrypted.data()),
               static_cast<std::streamsize>(encrypted.size()));

    if (!file.good()) {
        return Result<void>::err("write error");
    }

    LogPrint(WALLET, "backup created: %s", path.string().c_str());
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// restore_backup
// ---------------------------------------------------------------------------
Result<std::vector<uint8_t>> restore_backup(
    const std::filesystem::path& path,
    const std::string& passphrase) {

    // 1. Validate the passphrase.
    if (passphrase.empty()) {
        return Result<std::vector<uint8_t>>::err("passphrase cannot be empty");
    }

    // 2. Open the backup file and determine its size.
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return Result<std::vector<uint8_t>>::err(
            "cannot open backup file: " + path.string());
    }

    auto file_size = file.tellg();
    if (file_size < 72) {  // 4 + 4 + 32 + 32 = 72 minimum header
        return Result<std::vector<uint8_t>>::err("backup file too small");
    }
    file.seekg(0);

    // 3. Read and verify the magic bytes.
    uint8_t magic[4];
    file.read(reinterpret_cast<char*>(magic), 4);
    if (std::memcmp(magic, BACKUP_MAGIC, 4) != 0) {
        return Result<std::vector<uint8_t>>::err("invalid backup magic");
    }

    // 4. Read and verify the version.
    uint32_t version = 0;
    file.read(reinterpret_cast<char*>(&version), 4);
    if (version != BACKUP_VERSION) {
        return Result<std::vector<uint8_t>>::err("unsupported backup version");
    }

    // 5. Read the salt.
    std::array<uint8_t, 32> salt{};
    file.read(reinterpret_cast<char*>(salt.data()), 32);

    // 6. Read the stored checksum.
    uint256 stored_checksum;
    file.read(reinterpret_cast<char*>(stored_checksum.data()), 32);

    // 7. Read the encrypted payload.
    auto payload_size = static_cast<size_t>(file_size) - 72;
    std::vector<uint8_t> encrypted(payload_size);
    file.read(reinterpret_cast<char*>(encrypted.data()),
              static_cast<std::streamsize>(payload_size));

    // 8. Verify the checksum before attempting decryption.
    auto computed_checksum = crypto::keccak256d(
        std::span<const uint8_t>(encrypted.data(), encrypted.size()));
    if (computed_checksum != stored_checksum) {
        return Result<std::vector<uint8_t>>::err("backup checksum mismatch");
    }

    // 9. Derive the AES key from passphrase + salt.
    auto key_result = derive_backup_key(passphrase,
        std::span<const uint8_t>(salt.data(), salt.size()));
    if (!key_result) {
        return Result<std::vector<uint8_t>>::err("key derivation failed");
    }
    auto& key = key_result.value();

    // 10. Decrypt the payload.
    auto dec_result = crypto::AES256CBC::decrypt_with_iv_prefix(
        std::span<const uint8_t>(key.data(), 32),
        std::span<const uint8_t>(encrypted.data(), encrypted.size()));
    if (!dec_result) {
        return Result<std::vector<uint8_t>>::err("decryption failed (wrong passphrase?)");
    }

    LogPrint(WALLET, "backup restored from: %s", path.string().c_str());
    return Result<std::vector<uint8_t>>::ok(std::move(dec_result.value()));
}

// ---------------------------------------------------------------------------
// verify_backup
// ---------------------------------------------------------------------------
Result<void> verify_backup(const std::filesystem::path& path,
                           const std::string& passphrase) {
    // 1. Attempt a full restore; discard the payload on success.
    auto result = restore_backup(path, passphrase);
    if (!result) {
        return Result<void>::err(result.error());
    }
    return Result<void>::ok();
}

} // namespace rnet::wallet

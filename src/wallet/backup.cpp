#include "wallet/backup.h"

#include "core/logging.h"
#include "core/random.h"
#include "crypto/aes.h"
#include "crypto/argon2.h"
#include "crypto/keccak.h"

#include <cstring>
#include <fstream>

namespace rnet::wallet {

static Result<std::vector<uint8_t>> derive_backup_key(
    const std::string& passphrase,
    std::span<const uint8_t> salt) {

    return crypto::argon2id_derive(passphrase, salt);
}

Result<void> create_backup(const std::filesystem::path& path,
                           std::span<const uint8_t> payload,
                           const std::string& passphrase) {
    if (passphrase.empty()) {
        return Result<void>::err("passphrase cannot be empty");
    }
    if (payload.empty()) {
        return Result<void>::err("payload is empty");
    }

    // Generate salt
    auto salt = crypto::generate_argon2_salt();

    // Derive key
    auto key_result = derive_backup_key(passphrase,
        std::span<const uint8_t>(salt.data(), salt.size()));
    if (!key_result) {
        return Result<void>::err("key derivation failed: " + key_result.error());
    }
    auto& key = key_result.value();
    if (key.size() < 32) {
        return Result<void>::err("derived key too short");
    }

    // Encrypt payload
    auto enc_result = crypto::AES256CBC::encrypt_with_random_iv(
        std::span<const uint8_t>(key.data(), 32),
        payload);
    if (!enc_result) {
        return Result<void>::err("encryption failed: " + enc_result.error());
    }
    auto& encrypted = enc_result.value();

    // Compute checksum of encrypted payload
    auto checksum = crypto::keccak256d(
        std::span<const uint8_t>(encrypted.data(), encrypted.size()));

    // Write file
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return Result<void>::err("cannot open file for writing: " + path.string());
    }

    // Magic
    file.write(reinterpret_cast<const char*>(BACKUP_MAGIC), 4);

    // Version (little-endian)
    uint32_t ver = BACKUP_VERSION;
    file.write(reinterpret_cast<const char*>(&ver), 4);

    // Salt
    file.write(reinterpret_cast<const char*>(salt.data()), 32);

    // Checksum
    file.write(reinterpret_cast<const char*>(checksum.data()), 32);

    // Encrypted payload
    file.write(reinterpret_cast<const char*>(encrypted.data()),
               static_cast<std::streamsize>(encrypted.size()));

    if (!file.good()) {
        return Result<void>::err("write error");
    }

    LogPrint(WALLET, "backup created: %s", path.string().c_str());
    return Result<void>::ok();
}

Result<std::vector<uint8_t>> restore_backup(
    const std::filesystem::path& path,
    const std::string& passphrase) {

    if (passphrase.empty()) {
        return Result<std::vector<uint8_t>>::err("passphrase cannot be empty");
    }

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

    // Read magic
    uint8_t magic[4];
    file.read(reinterpret_cast<char*>(magic), 4);
    if (std::memcmp(magic, BACKUP_MAGIC, 4) != 0) {
        return Result<std::vector<uint8_t>>::err("invalid backup magic");
    }

    // Read version
    uint32_t version = 0;
    file.read(reinterpret_cast<char*>(&version), 4);
    if (version != BACKUP_VERSION) {
        return Result<std::vector<uint8_t>>::err("unsupported backup version");
    }

    // Read salt
    std::array<uint8_t, 32> salt{};
    file.read(reinterpret_cast<char*>(salt.data()), 32);

    // Read checksum
    uint256 stored_checksum;
    file.read(reinterpret_cast<char*>(stored_checksum.data()), 32);

    // Read encrypted payload
    auto payload_size = static_cast<size_t>(file_size) - 72;
    std::vector<uint8_t> encrypted(payload_size);
    file.read(reinterpret_cast<char*>(encrypted.data()),
              static_cast<std::streamsize>(payload_size));

    // Verify checksum
    auto computed_checksum = crypto::keccak256d(
        std::span<const uint8_t>(encrypted.data(), encrypted.size()));
    if (computed_checksum != stored_checksum) {
        return Result<std::vector<uint8_t>>::err("backup checksum mismatch");
    }

    // Derive key
    auto key_result = derive_backup_key(passphrase,
        std::span<const uint8_t>(salt.data(), salt.size()));
    if (!key_result) {
        return Result<std::vector<uint8_t>>::err("key derivation failed");
    }
    auto& key = key_result.value();

    // Decrypt
    auto dec_result = crypto::AES256CBC::decrypt_with_iv_prefix(
        std::span<const uint8_t>(key.data(), 32),
        std::span<const uint8_t>(encrypted.data(), encrypted.size()));
    if (!dec_result) {
        return Result<std::vector<uint8_t>>::err("decryption failed (wrong passphrase?)");
    }

    LogPrint(WALLET, "backup restored from: %s", path.string().c_str());
    return Result<std::vector<uint8_t>>::ok(std::move(dec_result.value()));
}

Result<void> verify_backup(const std::filesystem::path& path,
                           const std::string& passphrase) {
    auto result = restore_backup(path, passphrase);
    if (!result) {
        return Result<void>::err(result.error());
    }
    return Result<void>::ok();
}

}  // namespace rnet::wallet

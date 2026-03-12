#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "core/error.h"

namespace rnet::wallet {

/// Backup file format:
///   [4B magic "RNBK"]
///   [4B version]
///   [32B salt]
///   [32B checksum (keccak256d of encrypted payload)]
///   [AES-256-CBC encrypted payload]
///
/// Key derivation: passphrase -> Argon2id(salt) -> 32-byte AES key.

inline constexpr uint8_t BACKUP_MAGIC[4] = {'R', 'N', 'B', 'K'};
inline constexpr uint32_t BACKUP_VERSION = 1;

/// Create an encrypted backup file from wallet data.
/// @param path        Output file path.
/// @param payload     Raw wallet data to encrypt.
/// @param passphrase  Encryption passphrase.
/// @return Ok on success, error on failure.
Result<void> create_backup(const std::filesystem::path& path,
                           std::span<const uint8_t> payload,
                           const std::string& passphrase);

/// Restore wallet data from an encrypted backup file.
/// @param path        Input file path.
/// @param passphrase  Decryption passphrase.
/// @return Decrypted payload, or error.
Result<std::vector<uint8_t>> restore_backup(
    const std::filesystem::path& path,
    const std::string& passphrase);

/// Verify a backup file's integrity without fully decrypting.
/// Checks magic, version, and checksum.
Result<void> verify_backup(const std::filesystem::path& path,
                           const std::string& passphrase);

}  // namespace rnet::wallet

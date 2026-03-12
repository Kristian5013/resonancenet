#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "core/error.h"
#include "core/stream.h"
#include "core/sync.h"
#include "core/types.h"
#include "wallet/keys.h"
#include "wallet/hd.h"
#include "wallet/addresses.h"
#include "wallet/coins.h"

namespace rnet::wallet {

/// Record type tags for the flat-file wallet database.
enum class WalletDBRecordType : uint8_t {
    KEY         = 0x01,   ///< Private key
    HD_STATE    = 0x02,   ///< HD wallet state
    ADDRESS     = 0x03,   ///< Address entry
    COIN        = 0x04,   ///< UTXO entry
    TX          = 0x05,   ///< Transaction record
    ENCRYPT     = 0x06,   ///< Encryption metadata
    RECOVERY    = 0x07,   ///< Recovery policy
    METADATA    = 0x08,   ///< Wallet metadata (name, creation time, etc.)
    END_MARKER  = 0xFF,   ///< End of file marker
};

/// Wallet metadata persisted in DB.
struct WalletMetadata {
    std::string name;
    int64_t creation_time = 0;
    uint32_t version = 1;
    std::string network;
};

/// WalletDB: flat-file wallet database.
///
/// File format:
///   [4B magic "RNWL"]
///   [4B version]
///   Repeated records:
///     [1B type tag]
///     [4B payload length]
///     [payload bytes]
///   [1B END_MARKER]
class WalletDB {
public:
    explicit WalletDB(const std::filesystem::path& path);
    ~WalletDB() = default;

    WalletDB(const WalletDB&) = delete;
    WalletDB& operator=(const WalletDB&) = delete;

    /// Open or create the wallet database file.
    Result<void> open();

    /// Close the database.
    void close();

    /// Check if database is open.
    bool is_open() const;

    /// Write all wallet data to the file (full rewrite).
    Result<void> write(const WalletMetadata& metadata,
                       const KeyStore& keys,
                       const HDState& hd_state,
                       const std::vector<AddressEntry>& addresses,
                       const std::vector<WalletCoin>& coins,
                       const std::vector<uint8_t>& encrypt_data,
                       const std::vector<uint8_t>& recovery_data);

    /// Read all wallet data from the file.
    Result<void> read(WalletMetadata& metadata,
                      std::vector<WalletKey>& keys,
                      HDState& hd_state,
                      std::vector<AddressEntry>& addresses,
                      std::vector<WalletCoin>& coins,
                      std::vector<uint8_t>& encrypt_data,
                      std::vector<uint8_t>& recovery_data);

    /// Flush pending writes to disk.
    Result<void> flush();

    /// Get the database file path.
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
    mutable core::Mutex mutex_;
    bool open_ = false;

    static constexpr uint8_t DB_MAGIC[4] = {'R', 'N', 'W', 'L'};
    static constexpr uint32_t DB_VERSION = 1;

    /// Serialize a WalletKey to bytes.
    static std::vector<uint8_t> serialize_key(const WalletKey& key);
    static Result<WalletKey> deserialize_key(std::span<const uint8_t> data);

    /// Serialize an AddressEntry to bytes.
    static std::vector<uint8_t> serialize_address(const AddressEntry& entry);
    static Result<AddressEntry> deserialize_address(std::span<const uint8_t> data);

    /// Serialize a WalletCoin to bytes.
    static std::vector<uint8_t> serialize_coin(const WalletCoin& coin);
    static Result<WalletCoin> deserialize_coin(std::span<const uint8_t> data);

    /// Serialize HDState to bytes.
    static std::vector<uint8_t> serialize_hd_state(const HDState& state);
    static Result<HDState> deserialize_hd_state(std::span<const uint8_t> data);

    /// Serialize WalletMetadata to bytes.
    static std::vector<uint8_t> serialize_metadata(const WalletMetadata& meta);
    static Result<WalletMetadata> deserialize_metadata(std::span<const uint8_t> data);
};

}  // namespace rnet::wallet

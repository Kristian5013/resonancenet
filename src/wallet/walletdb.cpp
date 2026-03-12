#include "wallet/walletdb.h"

#include "core/logging.h"
#include "core/serialize.h"

#include <cstring>
#include <fstream>

namespace rnet::wallet {

WalletDB::WalletDB(const std::filesystem::path& path)
    : path_(path) {}

Result<void> WalletDB::open() {
    LOCK(mutex_);

    // Create parent directory if needed
    auto parent = path_.parent_path();
    if (!parent.empty() && !std::filesystem::exists(parent)) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            return Result<void>::err("cannot create directory: " + parent.string());
        }
    }

    open_ = true;
    LogPrint(WALLET, "wallet db opened: %s", path_.string().c_str());
    return Result<void>::ok();
}

void WalletDB::close() {
    LOCK(mutex_);
    open_ = false;
}

bool WalletDB::is_open() const {
    LOCK(mutex_);
    return open_;
}

Result<void> WalletDB::write(const WalletMetadata& metadata,
                              const KeyStore& keys,
                              const HDState& hd_state,
                              const std::vector<AddressEntry>& addresses,
                              const std::vector<WalletCoin>& coins,
                              const std::vector<uint8_t>& encrypt_data,
                              const std::vector<uint8_t>& recovery_data) {
    LOCK(mutex_);
    if (!open_) {
        return Result<void>::err("database not open");
    }

    // Write to temp file first, then rename (atomic)
    auto tmp_path = path_;
    tmp_path += ".tmp";

    std::ofstream file(tmp_path, std::ios::binary);
    if (!file.is_open()) {
        return Result<void>::err("cannot open temp file for writing");
    }

    // Magic
    file.write(reinterpret_cast<const char*>(DB_MAGIC), 4);

    // Version
    uint32_t ver = DB_VERSION;
    file.write(reinterpret_cast<const char*>(&ver), 4);

    // Helper to write a record
    auto write_record = [&](WalletDBRecordType type,
                            const std::vector<uint8_t>& payload) {
        auto t = static_cast<uint8_t>(type);
        file.write(reinterpret_cast<const char*>(&t), 1);
        uint32_t len = static_cast<uint32_t>(payload.size());
        file.write(reinterpret_cast<const char*>(&len), 4);
        if (!payload.empty()) {
            file.write(reinterpret_cast<const char*>(payload.data()),
                       static_cast<std::streamsize>(len));
        }
    };

    // Write metadata
    write_record(WalletDBRecordType::METADATA, serialize_metadata(metadata));

    // Write HD state
    write_record(WalletDBRecordType::HD_STATE, serialize_hd_state(hd_state));

    // Write keys
    auto all_hashes = keys.get_all_pubkey_hashes();
    for (const auto& hash : all_hashes) {
        auto key_result = keys.get_key(hash);
        if (key_result) {
            write_record(WalletDBRecordType::KEY,
                         serialize_key(key_result.value()));
        }
    }

    // Write addresses
    for (const auto& addr : addresses) {
        write_record(WalletDBRecordType::ADDRESS, serialize_address(addr));
    }

    // Write coins
    for (const auto& coin : coins) {
        write_record(WalletDBRecordType::COIN, serialize_coin(coin));
    }

    // Write encryption data
    if (!encrypt_data.empty()) {
        write_record(WalletDBRecordType::ENCRYPT, encrypt_data);
    }

    // Write recovery data
    if (!recovery_data.empty()) {
        write_record(WalletDBRecordType::RECOVERY, recovery_data);
    }

    // End marker
    auto end = static_cast<uint8_t>(WalletDBRecordType::END_MARKER);
    file.write(reinterpret_cast<const char*>(&end), 1);

    file.close();

    if (!file.good()) {
        return Result<void>::err("write error");
    }

    // Atomic rename
    std::error_code ec;
    std::filesystem::rename(tmp_path, path_, ec);
    if (ec) {
        return Result<void>::err("rename failed: " + ec.message());
    }

    return Result<void>::ok();
}

Result<void> WalletDB::read(WalletMetadata& metadata,
                             std::vector<WalletKey>& keys,
                             HDState& hd_state,
                             std::vector<AddressEntry>& addresses,
                             std::vector<WalletCoin>& coins,
                             std::vector<uint8_t>& encrypt_data,
                             std::vector<uint8_t>& recovery_data) {
    LOCK(mutex_);
    if (!open_) {
        return Result<void>::err("database not open");
    }

    if (!std::filesystem::exists(path_)) {
        return Result<void>::ok();  // Empty wallet, nothing to read
    }

    std::ifstream file(path_, std::ios::binary);
    if (!file.is_open()) {
        return Result<void>::err("cannot open database file");
    }

    // Read magic
    uint8_t magic[4];
    file.read(reinterpret_cast<char*>(magic), 4);
    if (std::memcmp(magic, DB_MAGIC, 4) != 0) {
        return Result<void>::err("invalid wallet database magic");
    }

    // Read version
    uint32_t version = 0;
    file.read(reinterpret_cast<char*>(&version), 4);
    if (version != DB_VERSION) {
        return Result<void>::err("unsupported wallet database version");
    }

    // Read records
    while (file.good()) {
        uint8_t type_byte = 0;
        file.read(reinterpret_cast<char*>(&type_byte), 1);
        if (!file.good()) break;

        auto record_type = static_cast<WalletDBRecordType>(type_byte);
        if (record_type == WalletDBRecordType::END_MARKER) {
            break;
        }

        uint32_t payload_len = 0;
        file.read(reinterpret_cast<char*>(&payload_len), 4);
        if (!file.good() || payload_len > 100 * 1024 * 1024) {  // 100 MB limit
            return Result<void>::err("invalid record length");
        }

        std::vector<uint8_t> payload(payload_len);
        if (payload_len > 0) {
            file.read(reinterpret_cast<char*>(payload.data()),
                      static_cast<std::streamsize>(payload_len));
            if (!file.good()) {
                return Result<void>::err("truncated record");
            }
        }

        switch (record_type) {
            case WalletDBRecordType::METADATA: {
                auto r = deserialize_metadata(payload);
                if (r) metadata = std::move(r.value());
                break;
            }
            case WalletDBRecordType::KEY: {
                auto r = deserialize_key(payload);
                if (r) keys.push_back(std::move(r.value()));
                break;
            }
            case WalletDBRecordType::HD_STATE: {
                auto r = deserialize_hd_state(payload);
                if (r) hd_state = std::move(r.value());
                break;
            }
            case WalletDBRecordType::ADDRESS: {
                auto r = deserialize_address(payload);
                if (r) addresses.push_back(std::move(r.value()));
                break;
            }
            case WalletDBRecordType::COIN: {
                auto r = deserialize_coin(payload);
                if (r) coins.push_back(std::move(r.value()));
                break;
            }
            case WalletDBRecordType::ENCRYPT:
                encrypt_data = std::move(payload);
                break;
            case WalletDBRecordType::RECOVERY:
                recovery_data = std::move(payload);
                break;
            default:
                // Skip unknown record types (forward compatibility)
                break;
        }
    }

    LogPrint(WALLET, "wallet db loaded: %zu keys, %zu addresses, %zu coins",
             keys.size(), addresses.size(), coins.size());
    return Result<void>::ok();
}

Result<void> WalletDB::flush() {
    // For flat file, write() already flushes
    return Result<void>::ok();
}

// ─── Serialization helpers ──────────────────────────────────────────

std::vector<uint8_t> WalletDB::serialize_key(const WalletKey& key) {
    core::DataStream s;
    s.write(key.secret.data.data(), 64);
    s.write(key.pubkey.data.data(), 32);
    s.write(key.pubkey_hash.data(), 20);
    core::Serialize(s, key.creation_time);
    core::Serialize(s, key.label);
    core::Serialize(s, key.is_hd);
    core::Serialize(s, key.hd_index);
    core::Serialize(s, key.hd_change);
    return std::vector<uint8_t>(s.data(), s.data() + s.size());
}

Result<WalletKey> WalletDB::deserialize_key(std::span<const uint8_t> data) {
    if (data.size() < 116) {  // 64 + 32 + 20 minimum
        return Result<WalletKey>::err("key record too short");
    }
    core::DataStream s(std::vector<uint8_t>(data.begin(), data.end()));
    WalletKey key;
    s.read(key.secret.data.data(), 64);
    s.read(key.pubkey.data.data(), 32);
    s.read(key.pubkey_hash.data(), 20);
    core::Unserialize(s, key.creation_time);
    core::Unserialize(s, key.label);
    core::Unserialize(s, key.is_hd);
    core::Unserialize(s, key.hd_index);
    core::Unserialize(s, key.hd_change);
    return Result<WalletKey>::ok(std::move(key));
}

std::vector<uint8_t> WalletDB::serialize_address(const AddressEntry& entry) {
    core::DataStream s;
    core::Serialize(s, entry.address);
    core::Serialize(s, static_cast<uint8_t>(entry.type));
    s.write(entry.pubkey_hash.data(), 20);
    core::Serialize(s, entry.label);
    core::Serialize(s, entry.creation_time);
    core::Serialize(s, entry.is_change);
    core::Serialize(s, entry.is_used);
    return std::vector<uint8_t>(s.data(), s.data() + s.size());
}

Result<AddressEntry> WalletDB::deserialize_address(std::span<const uint8_t> data) {
    core::DataStream s(std::vector<uint8_t>(data.begin(), data.end()));
    AddressEntry entry;
    core::Unserialize(s, entry.address);
    uint8_t type_byte = 0;
    core::Unserialize(s, type_byte);
    entry.type = static_cast<primitives::AddressType>(type_byte);
    s.read(entry.pubkey_hash.data(), 20);
    core::Unserialize(s, entry.label);
    core::Unserialize(s, entry.creation_time);
    core::Unserialize(s, entry.is_change);
    core::Unserialize(s, entry.is_used);
    return Result<AddressEntry>::ok(std::move(entry));
}

std::vector<uint8_t> WalletDB::serialize_coin(const WalletCoin& coin) {
    core::DataStream s;
    coin.outpoint.serialize(s);
    coin.txout.serialize(s);
    core::Serialize(s, coin.height);
    s.write(coin.pubkey_hash.data(), 20);
    core::Serialize(s, coin.is_spent);
    core::Serialize(s, coin.is_change);
    core::Serialize(s, coin.creation_time);
    core::Serialize(s, coin.val_loss_at_creation);
    return std::vector<uint8_t>(s.data(), s.data() + s.size());
}

Result<WalletCoin> WalletDB::deserialize_coin(std::span<const uint8_t> data) {
    core::DataStream s(std::vector<uint8_t>(data.begin(), data.end()));
    WalletCoin coin;
    coin.outpoint.unserialize(s);
    coin.txout.unserialize(s);
    core::Unserialize(s, coin.height);
    s.read(coin.pubkey_hash.data(), 20);
    core::Unserialize(s, coin.is_spent);
    core::Unserialize(s, coin.is_change);
    core::Unserialize(s, coin.creation_time);
    core::Unserialize(s, coin.val_loss_at_creation);
    return Result<WalletCoin>::ok(std::move(coin));
}

std::vector<uint8_t> WalletDB::serialize_hd_state(const HDState& state) {
    core::DataStream s;
    core::Serialize(s, state.mnemonic);
    s.write(state.seed.data(), 64);
    // Serialize ExtKey fields
    s.write(state.master_key.key.data(), 32);
    s.write(state.master_key.chaincode.data(), 32);
    core::Serialize(s, state.master_key.depth);
    core::Serialize(s, state.master_key.fingerprint);
    core::Serialize(s, state.master_key.child_num);
    core::Serialize(s, state.master_key.is_private);
    core::Serialize(s, state.account);
    core::Serialize(s, state.next_external_index);
    core::Serialize(s, state.next_internal_index);
    return std::vector<uint8_t>(s.data(), s.data() + s.size());
}

Result<HDState> WalletDB::deserialize_hd_state(std::span<const uint8_t> data) {
    core::DataStream s(std::vector<uint8_t>(data.begin(), data.end()));
    HDState state;
    core::Unserialize(s, state.mnemonic);
    s.read(state.seed.data(), 64);
    s.read(state.master_key.key.data(), 32);
    s.read(state.master_key.chaincode.data(), 32);
    core::Unserialize(s, state.master_key.depth);
    core::Unserialize(s, state.master_key.fingerprint);
    core::Unserialize(s, state.master_key.child_num);
    core::Unserialize(s, state.master_key.is_private);
    core::Unserialize(s, state.account);
    core::Unserialize(s, state.next_external_index);
    core::Unserialize(s, state.next_internal_index);
    return Result<HDState>::ok(std::move(state));
}

std::vector<uint8_t> WalletDB::serialize_metadata(const WalletMetadata& meta) {
    core::DataStream s;
    core::Serialize(s, meta.name);
    core::Serialize(s, meta.creation_time);
    core::Serialize(s, meta.version);
    core::Serialize(s, meta.network);
    return std::vector<uint8_t>(s.data(), s.data() + s.size());
}

Result<WalletMetadata> WalletDB::deserialize_metadata(std::span<const uint8_t> data) {
    core::DataStream s(std::vector<uint8_t>(data.begin(), data.end()));
    WalletMetadata meta;
    core::Unserialize(s, meta.name);
    core::Unserialize(s, meta.creation_time);
    core::Unserialize(s, meta.version);
    core::Unserialize(s, meta.network);
    return Result<WalletMetadata>::ok(std::move(meta));
}

}  // namespace rnet::wallet

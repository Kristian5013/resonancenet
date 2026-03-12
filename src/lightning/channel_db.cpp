#include "lightning/channel_db.h"

#include <cstdio>
#include <filesystem>

#include "core/logging.h"
#include "core/serialize.h"

namespace rnet::lightning {

// ── Database file format ────────────────────────────────────────────
// Header:  "RNLT" (4 bytes) + version(u32) + channel_count(u32)
// Records: serialized ChannelRecord entries

static constexpr uint32_t DB_MAGIC = 0x544C4E52;  // "RNLT"
static constexpr uint32_t DB_VERSION = 1;

Result<void> ChannelDB::open(const std::string& db_path) {
    LOCK(mutex_);

    if (is_open_) {
        return Result<void>::err("Database already open");
    }

    db_path_ = db_path;

    // Try to read existing data
    if (std::filesystem::exists(db_path_)) {
        auto r = read_from_disk();
        if (!r) {
            LogPrint(LIGHTNING, "ChannelDB: failed to read %s: %s",
                     db_path_.c_str(), r.error().c_str());
            // Start fresh
            channels_.clear();
            htlcs_.clear();
            revocations_.clear();
        }
    }

    is_open_ = true;
    LogPrint(LIGHTNING, "ChannelDB: opened %s with %zu channels",
             db_path_.c_str(), channels_.size());
    return Result<void>::ok();
}

void ChannelDB::close() {
    LOCK(mutex_);
    if (!is_open_) return;

    if (dirty_) {
        auto r = write_to_disk();
        if (!r) {
            LogPrint(LIGHTNING, "ChannelDB: failed to flush on close: %s",
                     r.error().c_str());
        }
    }

    is_open_ = false;
    channels_.clear();
    htlcs_.clear();
    revocations_.clear();
}

Result<void> ChannelDB::put_channel(const ChannelRecord& record) {
    LOCK(mutex_);
    if (!is_open_) return Result<void>::err("Database not open");

    channels_[record.channel_id] = record;
    dirty_ = true;
    return Result<void>::ok();
}

Result<ChannelRecord> ChannelDB::get_channel(const ChannelId& channel_id) const {
    LOCK(mutex_);
    if (!is_open_) return Result<ChannelRecord>::err("Database not open");

    auto it = channels_.find(channel_id);
    if (it == channels_.end()) {
        return Result<ChannelRecord>::err("Channel not found: " +
                                           channel_id.to_hex());
    }
    return Result<ChannelRecord>::ok(it->second);
}

Result<void> ChannelDB::delete_channel(const ChannelId& channel_id) {
    LOCK(mutex_);
    if (!is_open_) return Result<void>::err("Database not open");

    auto it = channels_.find(channel_id);
    if (it == channels_.end()) {
        return Result<void>::err("Channel not found");
    }

    channels_.erase(it);
    htlcs_.erase(channel_id);
    dirty_ = true;
    return Result<void>::ok();
}

std::vector<ChannelRecord> ChannelDB::get_all_channels() const {
    LOCK(mutex_);
    std::vector<ChannelRecord> result;
    result.reserve(channels_.size());
    for (const auto& [_, rec] : channels_) {
        result.push_back(rec);
    }
    return result;
}

std::vector<ChannelRecord> ChannelDB::get_open_channels() const {
    LOCK(mutex_);
    std::vector<ChannelRecord> result;
    for (const auto& [_, rec] : channels_) {
        if (rec.state != ChannelState::CLOSED) {
            result.push_back(rec);
        }
    }
    return result;
}

std::vector<ChannelRecord> ChannelDB::get_channels_with_node(
    const crypto::Ed25519PublicKey& node_id) const {
    LOCK(mutex_);
    std::vector<ChannelRecord> result;
    for (const auto& [_, rec] : channels_) {
        if (rec.remote_node_id == node_id) {
            result.push_back(rec);
        }
    }
    return result;
}

Result<void> ChannelDB::flush() {
    LOCK(mutex_);
    if (!is_open_) return Result<void>::err("Database not open");
    if (!dirty_) return Result<void>::ok();

    auto r = write_to_disk();
    if (r) dirty_ = false;
    return r;
}

bool ChannelDB::is_open() const {
    LOCK(mutex_);
    return is_open_;
}

size_t ChannelDB::channel_count() const {
    LOCK(mutex_);
    return channels_.size();
}

Result<void> ChannelDB::put_htlc(const ChannelId& channel_id, const Htlc& htlc) {
    LOCK(mutex_);
    if (!is_open_) return Result<void>::err("Database not open");

    auto& vec = htlcs_[channel_id];
    // Update existing or add new
    for (auto& existing : vec) {
        if (existing.id == htlc.id) {
            existing = htlc;
            dirty_ = true;
            return Result<void>::ok();
        }
    }
    vec.push_back(htlc);
    dirty_ = true;
    return Result<void>::ok();
}

std::vector<Htlc> ChannelDB::get_htlcs(const ChannelId& channel_id) const {
    LOCK(mutex_);
    auto it = htlcs_.find(channel_id);
    if (it == htlcs_.end()) return {};
    return it->second;
}

Result<void> ChannelDB::put_revocation(const ChannelId& channel_id,
                                        uint64_t commitment_number,
                                        const uint256& secret) {
    LOCK(mutex_);
    if (!is_open_) return Result<void>::err("Database not open");

    RevocationKey key{channel_id, commitment_number};
    revocations_[key] = secret;
    dirty_ = true;
    return Result<void>::ok();
}

Result<uint256> ChannelDB::get_revocation(const ChannelId& channel_id,
                                           uint64_t commitment_number) const {
    LOCK(mutex_);
    RevocationKey key{channel_id, commitment_number};
    auto it = revocations_.find(key);
    if (it == revocations_.end()) {
        return Result<uint256>::err("Revocation not found");
    }
    return Result<uint256>::ok(it->second);
}

Result<void> ChannelDB::write_to_disk() const {
    // Write to a temp file, then rename
    std::string tmp_path = db_path_ + ".tmp";

    FILE* f = std::fopen(tmp_path.c_str(), "wb");
    if (!f) {
        return Result<void>::err("Failed to open " + tmp_path + " for writing");
    }

    core::AutoFile file(f);
    core::DataStream ss;

    // Header
    core::ser_write_u32(ss, DB_MAGIC);
    core::ser_write_u32(ss, DB_VERSION);
    core::ser_write_u32(ss, static_cast<uint32_t>(channels_.size()));

    // Channel records
    for (const auto& [_, rec] : channels_) {
        rec.serialize(ss);
    }

    // Write to file
    auto& data = ss.vch();
    file.write(data.data(), data.size());
    file.flush();
    file.close();

    // Rename
    std::error_code ec;
    std::filesystem::rename(tmp_path, db_path_, ec);
    if (ec) {
        return Result<void>::err("Failed to rename database file: " +
                                  ec.message());
    }

    return Result<void>::ok();
}

Result<void> ChannelDB::read_from_disk() {
    FILE* f = std::fopen(db_path_.c_str(), "rb");
    if (!f) {
        return Result<void>::err("Failed to open " + db_path_);
    }

    // Read entire file
    std::fseek(f, 0, SEEK_END);
    long file_size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);

    if (file_size <= 0) {
        std::fclose(f);
        return Result<void>::ok();  // Empty file is OK
    }

    std::vector<uint8_t> data(static_cast<size_t>(file_size));
    size_t read = std::fread(data.data(), 1, data.size(), f);
    std::fclose(f);

    if (read != data.size()) {
        return Result<void>::err("Short read from database file");
    }

    core::DataStream ss(std::move(data));

    // Read header
    uint32_t magic = core::ser_read_u32(ss);
    if (magic != DB_MAGIC) {
        return Result<void>::err("Invalid database magic");
    }

    uint32_t version = core::ser_read_u32(ss);
    if (version != DB_VERSION) {
        return Result<void>::err("Unsupported database version: " +
                                  std::to_string(version));
    }

    uint32_t count = core::ser_read_u32(ss);

    // Read channel records
    channels_.clear();
    for (uint32_t i = 0; i < count; ++i) {
        ChannelRecord rec;
        rec.unserialize(ss);
        channels_[rec.channel_id] = std::move(rec);
    }

    return Result<void>::ok();
}

}  // namespace rnet::lightning

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/types.h"
#include "core/error.h"
#include "core/sync.h"
#include "core/stream.h"
#include "lightning/channel_state.h"
#include "lightning/htlc.h"

namespace rnet::lightning {

// ── Persistent channel record ───────────────────────────────────────

struct ChannelRecord {
    ChannelId                        channel_id;
    TempChannelId                    temp_channel_id;
    ChannelState                     state = ChannelState::PREOPENING;
    bool                             is_funder = false;
    int64_t                          capacity = 0;
    ChannelBalance                   balance;
    ChannelSideState                 local_state;
    ChannelSideState                 remote_state;
    crypto::Ed25519PublicKey         remote_node_id;
    primitives::COutPoint            funding_outpoint;
    uint32_t                         confirmation_height = 0;
    uint64_t                         created_at = 0;   // Unix timestamp
    uint64_t                         closed_at = 0;    // Unix timestamp (0 if open)

    template<typename Stream>
    void serialize(Stream& s) const {
        core::Serialize(s, channel_id);
        core::Serialize(s, temp_channel_id);
        core::ser_write_u8(s, static_cast<uint8_t>(state));
        core::ser_write_bool(s, is_funder);
        core::Serialize(s, capacity);
        core::Serialize(s, balance);
        core::Serialize(s, local_state);
        core::Serialize(s, remote_state);
        s.write(remote_node_id.data.data(), 32);
        core::Serialize(s, funding_outpoint);
        core::Serialize(s, confirmation_height);
        core::Serialize(s, created_at);
        core::Serialize(s, closed_at);
    }

    template<typename Stream>
    void unserialize(Stream& s) {
        core::Unserialize(s, channel_id);
        core::Unserialize(s, temp_channel_id);
        state = static_cast<ChannelState>(core::ser_read_u8(s));
        is_funder = core::ser_read_bool(s);
        core::Unserialize(s, capacity);
        core::Unserialize(s, balance);
        core::Unserialize(s, local_state);
        core::Unserialize(s, remote_state);
        s.read(remote_node_id.data.data(), 32);
        core::Unserialize(s, funding_outpoint);
        core::Unserialize(s, confirmation_height);
        core::Unserialize(s, created_at);
        core::Unserialize(s, closed_at);
    }
};

// ── Channel database ────────────────────────────────────────────────

/// Persistent storage for channel state.
/// Uses flat file storage with periodic flushing.
class ChannelDB {
public:
    ChannelDB() = default;

    /// Open/create the database at the given path
    Result<void> open(const std::string& db_path);

    /// Close the database (flushes pending writes)
    void close();

    /// Write a channel record
    Result<void> put_channel(const ChannelRecord& record);

    /// Read a channel record
    Result<ChannelRecord> get_channel(const ChannelId& channel_id) const;

    /// Delete a channel record
    Result<void> delete_channel(const ChannelId& channel_id);

    /// Get all channel records
    std::vector<ChannelRecord> get_all_channels() const;

    /// Get all open (non-closed) channels
    std::vector<ChannelRecord> get_open_channels() const;

    /// Get all channels with a specific remote node
    std::vector<ChannelRecord> get_channels_with_node(
        const crypto::Ed25519PublicKey& node_id) const;

    /// Flush pending writes to disk
    Result<void> flush();

    /// Check if database is open
    bool is_open() const;

    /// Get the number of stored channels
    size_t channel_count() const;

    /// Store an HTLC state update
    Result<void> put_htlc(const ChannelId& channel_id, const Htlc& htlc);

    /// Get all HTLCs for a channel
    std::vector<Htlc> get_htlcs(const ChannelId& channel_id) const;

    /// Store a revocation secret
    Result<void> put_revocation(const ChannelId& channel_id,
                                 uint64_t commitment_number,
                                 const uint256& secret);

    /// Get a revocation secret
    Result<uint256> get_revocation(const ChannelId& channel_id,
                                    uint64_t commitment_number) const;

private:
    /// Serialize all data to disk
    Result<void> write_to_disk() const;

    /// Read all data from disk
    Result<void> read_from_disk();

    mutable core::Mutex mutex_;
    std::string db_path_;
    bool is_open_ = false;
    bool dirty_ = false;

    std::unordered_map<ChannelId, ChannelRecord> channels_;
    std::unordered_map<ChannelId, std::vector<Htlc>> htlcs_;

    // Revocation storage: channel_id -> (commitment_number -> secret)
    struct RevocationKey {
        ChannelId channel_id;
        uint64_t commitment_number;
        bool operator==(const RevocationKey& other) const {
            return channel_id == other.channel_id &&
                   commitment_number == other.commitment_number;
        }
    };

    struct RevocationKeyHash {
        size_t operator()(const RevocationKey& k) const {
            size_t h = std::hash<ChannelId>{}(k.channel_id);
            h ^= std::hash<uint64_t>{}(k.commitment_number) +
                 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    std::unordered_map<RevocationKey, uint256, RevocationKeyHash> revocations_;
};

}  // namespace rnet::lightning

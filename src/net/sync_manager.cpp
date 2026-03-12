#include "net/sync_manager.h"

#include <algorithm>
#include <cstring>

#include "core/logging.h"
#include "core/serialize.h"
#include "core/stream.h"
#include "core/time.h"
#include "net/conn_manager.h"

namespace rnet::net {

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

SyncManager::SyncManager(ConnManager& connman)
    : connman_(connman)
{}

SyncManager::~SyncManager() {
    stop();
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void SyncManager::start_sync(int32_t tip_height,
                             const rnet::uint256& tip_hash) {
    LOCK(cs_sync_);

    tip_height_ = tip_height;
    tip_hash_ = tip_hash;
    best_header_height_ = tip_height;
    best_header_hash_ = tip_hash;

    // Find the best peer and start header sync
    uint64_t best_peer = select_header_sync_peer();
    if (best_peer != 0) {
        header_sync_peer_ = best_peer;
        header_sync_start_time_ = core::get_time();
        stage_.store(SyncStage::HEADERS_SYNC);

        send_getheaders(best_peer, tip_hash, rnet::uint256{});

        LogPrintf("Starting header sync from peer %llu (our height=%d)",
                  static_cast<unsigned long long>(best_peer),
                  tip_height);
    } else {
        // No peers available yet; wait for connections
        stage_.store(SyncStage::IDLE);
        LogPrintf("No peers available for sync (our height=%d)", tip_height);
    }
}

void SyncManager::stop() {
    LOCK(cs_sync_);
    stage_.store(SyncStage::IDLE);
    peer_states_.clear();
    all_blocks_in_flight_.clear();
    download_queue_.clear();
    header_sync_peer_ = 0;
}

// ---------------------------------------------------------------------------
// Event handlers
// ---------------------------------------------------------------------------

void SyncManager::on_headers(uint64_t peer_id,
                             const std::vector<uint8_t>& header_data) {
    LOCK(cs_sync_);

    if (stage_.load() != SyncStage::HEADERS_SYNC) {
        // Accept headers even when not actively syncing (for new block
        // announcements), but don't change stage.
        if (on_validate_header_) {
            on_validate_header_(header_data);
        }
        return;
    }

    // Validate headers through callback
    if (on_validate_header_) {
        if (!on_validate_header_(header_data)) {
            // Bad headers from sync peer
            auto it = peer_states_.find(peer_id);
            if (it != peer_states_.end()) {
                LogPrint(NET, "Bad headers from peer %llu — disconnecting",
                         static_cast<unsigned long long>(peer_id));
            }

            // Try another peer
            if (peer_id == header_sync_peer_) {
                header_sync_peer_ = 0;
                uint64_t new_peer = select_header_sync_peer();
                if (new_peer != 0) {
                    header_sync_peer_ = new_peer;
                    send_getheaders(new_peer, best_header_hash_,
                                    rnet::uint256{});
                }
            }
            return;
        }
    }

    // Count headers received (approximate: each header is ~80 bytes)
    size_t approx_header_count = 0;
    if (!header_data.empty()) {
        // Read count from compact size at start
        core::DataStream ds(std::span<const uint8_t>(
            header_data.data(), header_data.size()));
        if (ds.remaining() > 0) {
            approx_header_count =
                static_cast<size_t>(core::unserialize_compact_size(ds));
        }
    }

    LogDebug(NET, "Received ~%zu headers from peer %llu",
             approx_header_count,
             static_cast<unsigned long long>(peer_id));

    // If we received a full batch, request more
    if (approx_header_count >= static_cast<size_t>(MAX_HEADERS_RESULTS)) {
        send_getheaders(peer_id, best_header_hash_, rnet::uint256{});
    } else {
        // Header sync complete — move to block download
        LogPrintf("Header sync complete at height %d", best_header_height_);
        start_block_download();
    }
}

void SyncManager::on_block(uint64_t peer_id,
                           const std::vector<uint8_t>& block_data) {
    LOCK(cs_sync_);

    // Find the block hash in flight for this peer
    auto peer_it = peer_states_.find(peer_id);
    if (peer_it != peer_states_.end()) {
        // We can't easily determine the hash without deserializing.
        // For now, remove one block from this peer's in-flight set.
        // In a real implementation, we'd extract the block hash from
        // the deserialized header.
        auto& pstate = peer_it->second;
        if (!pstate.blocks_in_flight.empty()) {
            auto block_hash = *pstate.blocks_in_flight.begin();
            pstate.blocks_in_flight.erase(pstate.blocks_in_flight.begin());
            all_blocks_in_flight_.erase(block_hash);
        }
    }

    ++blocks_received_;

    // Pass to chain for validation and storage
    if (on_block_ready_) {
        bool accepted = on_block_ready_(block_data);
        if (accepted) {
            ++tip_height_;
        } else {
            LogPrint(NET, "Block from peer %llu rejected by chain",
                     static_cast<unsigned long long>(peer_id));
        }
    }

    // Request more blocks if needed
    assign_blocks_to_peers();

    // Check if sync is complete
    update_stage();
}

void SyncManager::on_new_peer(uint64_t peer_id, int32_t peer_height) {
    LOCK(cs_sync_);

    PeerSyncState pstate;
    pstate.peer_id = peer_id;
    pstate.best_height = peer_height;
    peer_states_[peer_id] = pstate;

    // Update target height
    if (peer_height > target_height_) {
        target_height_ = peer_height;
    }

    // If we're idle and this peer is ahead, start syncing
    if (stage_.load() == SyncStage::IDLE && peer_height > tip_height_) {
        start_sync(tip_height_, tip_hash_);
    }

    // If we're downloading blocks and have room, assign to this peer
    if (stage_.load() == SyncStage::BLOCK_SYNC) {
        assign_blocks_to_peers();
    }
}

void SyncManager::on_peer_disconnected(uint64_t peer_id) {
    LOCK(cs_sync_);

    auto it = peer_states_.find(peer_id);
    if (it == peer_states_.end()) return;

    auto& pstate = it->second;

    // Return in-flight blocks to the download queue
    for (const auto& hash : pstate.blocks_in_flight) {
        all_blocks_in_flight_.erase(hash);
        download_queue_.push_back(hash);
    }

    peer_states_.erase(it);

    // If this was our header sync peer, find another
    if (peer_id == header_sync_peer_) {
        header_sync_peer_ = 0;
        if (stage_.load() == SyncStage::HEADERS_SYNC) {
            uint64_t new_peer = select_header_sync_peer();
            if (new_peer != 0) {
                header_sync_peer_ = new_peer;
                send_getheaders(new_peer, best_header_hash_,
                                rnet::uint256{});
                LogPrint(NET, "Switched header sync to peer %llu",
                         static_cast<unsigned long long>(new_peer));
            } else {
                LogPrint(NET, "No peers available for header sync");
                stage_.store(SyncStage::IDLE);
            }
        }
    }

    // Reassign blocks
    if (stage_.load() == SyncStage::BLOCK_SYNC && !download_queue_.empty()) {
        assign_blocks_to_peers();
    }
}

void SyncManager::tick() {
    LOCK(cs_sync_);

    int64_t now = core::get_time();

    // Check header sync timeout
    if (stage_.load() == SyncStage::HEADERS_SYNC &&
        header_sync_peer_ != 0) {
        if ((now - header_sync_start_time_) > HEADER_SYNC_TIMEOUT) {
            LogPrint(NET, "Header sync timed out from peer %llu",
                     static_cast<unsigned long long>(header_sync_peer_));
            handle_stalled_peer(header_sync_peer_);
        }
    }

    // Check for stalled block downloads
    if (stage_.load() == SyncStage::BLOCK_SYNC) {
        std::vector<uint64_t> stalled;
        for (auto& [pid, pstate] : peer_states_) {
            if (!pstate.blocks_in_flight.empty() &&
                pstate.last_request_time > 0 &&
                (now - pstate.last_request_time) > STALL_TIMEOUT) {
                stalled.push_back(pid);
            }
        }

        for (auto pid : stalled) {
            handle_stalled_peer(pid);
        }

        // Try to assign more blocks
        assign_blocks_to_peers();
    }
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

float SyncManager::sync_progress() const {
    LOCK(cs_sync_);

    if (target_height_ <= 0) return 0.0f;
    if (tip_height_ >= target_height_) return 1.0f;

    return static_cast<float>(tip_height_) /
           static_cast<float>(target_height_);
}

int32_t SyncManager::blocks_remaining() const {
    LOCK(cs_sync_);
    int32_t remaining = target_height_ - tip_height_;
    return remaining > 0 ? remaining : 0;
}

bool SyncManager::is_initial_block_download() const {
    auto s = stage_.load();
    return s == SyncStage::HEADERS_SYNC ||
           s == SyncStage::BLOCK_SYNC ||
           s == SyncStage::CHECKPOINT_SYNC;
}

size_t SyncManager::blocks_in_flight() const {
    LOCK(cs_sync_);
    return all_blocks_in_flight_.size();
}

// ---------------------------------------------------------------------------
// Block request interface
// ---------------------------------------------------------------------------

void SyncManager::request_blocks(uint64_t peer_id,
                                 const std::vector<rnet::uint256>& hashes) {
    LOCK(cs_sync_);

    auto it = peer_states_.find(peer_id);
    if (it == peer_states_.end()) return;

    std::vector<rnet::uint256> to_request;
    for (const auto& hash : hashes) {
        if (all_blocks_in_flight_.count(hash) == 0) {
            to_request.push_back(hash);
            all_blocks_in_flight_.insert(hash);
            it->second.blocks_in_flight.insert(hash);
        }
    }

    if (!to_request.empty()) {
        it->second.last_request_time = core::get_time();
        send_getblocks_data(peer_id, to_request);
    }
}

bool SyncManager::is_block_in_flight(const rnet::uint256& hash) const {
    LOCK(cs_sync_);
    return all_blocks_in_flight_.count(hash) > 0;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

uint64_t SyncManager::select_header_sync_peer() {
    // cs_sync_ must be held
    uint64_t best_peer = 0;
    int32_t best_height = tip_height_;

    for (const auto& [pid, pstate] : peer_states_) {
        if (pstate.best_height > best_height) {
            best_height = pstate.best_height;
            best_peer = pid;
        }
    }

    if (best_peer != 0) {
        target_height_ = best_height;
    }
    return best_peer;
}

void SyncManager::send_getheaders(uint64_t peer_id,
                                  const rnet::uint256& locator_hash,
                                  const rnet::uint256& stop_hash) {
    core::DataStream payload;

    // Version
    core::ser_write_i32(payload, PROTOCOL_VERSION);

    // Locator: one hash
    core::serialize_compact_size(payload, 1);
    locator_hash.serialize(payload);

    // Stop hash
    stop_hash.serialize(payload);

    connman_.send_to(peer_id, msg::GETHEADERS, payload.span());
}

void SyncManager::send_getblocks_data(
    uint64_t peer_id,
    const std::vector<rnet::uint256>& hashes) {
    core::DataStream payload;

    core::serialize_compact_size(payload, hashes.size());
    for (const auto& hash : hashes) {
        // Write as inv entries (type=BLOCK)
        core::ser_write_u32(payload,
                            static_cast<uint32_t>(InvType::INV_BLOCK));
        hash.serialize(payload);
    }

    connman_.send_to(peer_id, msg::GETDATA, payload.span());

    LogDebug(NET, "Requested %zu blocks from peer %llu",
             hashes.size(),
             static_cast<unsigned long long>(peer_id));
}

void SyncManager::assign_blocks_to_peers() {
    // cs_sync_ must be held

    if (download_queue_.empty()) return;

    for (auto& [pid, pstate] : peer_states_) {
        // Check how many more blocks we can request from this peer
        int available = MAX_BLOCKS_PER_PEER -
                        static_cast<int>(pstate.blocks_in_flight.size());
        if (available <= 0) continue;

        // Check global limit
        int global_available = MAX_BLOCKS_TOTAL -
                               static_cast<int>(all_blocks_in_flight_.size());
        if (global_available <= 0) break;

        int to_assign = (std::min)(available, global_available);
        to_assign = (std::min)(to_assign,
                               static_cast<int>(download_queue_.size()));
        if (to_assign <= 0) continue;

        std::vector<rnet::uint256> batch;
        batch.reserve(static_cast<size_t>(to_assign));

        for (int i = 0; i < to_assign && !download_queue_.empty(); ++i) {
            auto hash = download_queue_.front();
            download_queue_.erase(download_queue_.begin());

            batch.push_back(hash);
            all_blocks_in_flight_.insert(hash);
            pstate.blocks_in_flight.insert(hash);
        }

        if (!batch.empty()) {
            pstate.last_request_time = core::get_time();
            send_getblocks_data(pid, batch);
        }

        if (download_queue_.empty()) break;
    }
}

void SyncManager::handle_stalled_peer(uint64_t peer_id) {
    auto it = peer_states_.find(peer_id);
    if (it == peer_states_.end()) return;

    LogPrint(NET, "Peer %llu stalled with %zu blocks in flight",
             static_cast<unsigned long long>(peer_id),
             it->second.blocks_in_flight.size());

    // Return blocks to the download queue
    for (const auto& hash : it->second.blocks_in_flight) {
        all_blocks_in_flight_.erase(hash);
        download_queue_.push_back(hash);
    }
    it->second.blocks_in_flight.clear();
    it->second.last_request_time = 0;

    // If this was the header sync peer, switch
    if (peer_id == header_sync_peer_) {
        header_sync_peer_ = 0;
        uint64_t new_peer = select_header_sync_peer();
        if (new_peer != 0) {
            header_sync_peer_ = new_peer;
            header_sync_start_time_ = core::get_time();
            send_getheaders(new_peer, best_header_hash_, rnet::uint256{});
        }
    }

    // Try to reassign blocks to other peers
    assign_blocks_to_peers();
}

void SyncManager::update_stage() {
    // cs_sync_ must be held

    auto current = stage_.load();

    if (current == SyncStage::BLOCK_SYNC) {
        // Check if all blocks are downloaded
        if (download_queue_.empty() && all_blocks_in_flight_.empty()) {
            if (tip_height_ >= target_height_) {
                stage_.store(SyncStage::DONE);
                LogPrintf("Block sync complete at height %d", tip_height_);
            }
        }
    }
}

void SyncManager::start_block_download() {
    // cs_sync_ must be held

    if (download_queue_.empty() && tip_height_ >= best_header_height_) {
        stage_.store(SyncStage::DONE);
        LogPrintf("Already synced to header tip (%d)", tip_height_);
        return;
    }

    stage_.store(SyncStage::BLOCK_SYNC);
    LogPrintf("Starting block download (%zu blocks queued, target=%d)",
              download_queue_.size(), target_height_);

    assign_blocks_to_peers();
}

}  // namespace rnet::net

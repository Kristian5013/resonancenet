#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include "core/error.h"
#include "core/sync.h"
#include "core/types.h"
#include "net/connection.h"
#include "net/protocol.h"

namespace rnet::net {

class ConnManager;

/// Synchronization stage
enum class SyncStage : uint8_t {
    IDLE,               ///< Not syncing (pre-start or error)
    HEADERS_SYNC,       ///< Downloading block headers
    BLOCK_SYNC,         ///< Downloading block bodies
    CHECKPOINT_SYNC,    ///< Downloading model checkpoints
    DONE,               ///< Fully synced
};

/// Per-peer sync state
struct PeerSyncState {
    uint64_t peer_id = 0;
    int32_t best_height = 0;
    rnet::uint256 best_hash;
    bool headers_syncing = false;  ///< Are we downloading headers from this peer?

    /// Blocks in flight from this peer
    std::set<rnet::uint256> blocks_in_flight;

    /// Time of last block request to this peer
    int64_t last_request_time = 0;
};

/// SyncManager — manages initial block download (IBD) and ongoing sync.
///
/// Implements headers-first synchronization:
///   1. Download all block headers from the best peer
///   2. Validate the header chain
///   3. Download block bodies in parallel from multiple peers
///   4. Optionally download model checkpoints from NODE_CHECKPOINT peers
///
/// After IBD, the SyncManager handles:
///   - Requesting new blocks announced via inv/headers
///   - Managing blocks-in-flight to avoid duplicate requests
///   - Timeout and retry for stalled downloads
class SyncManager {
public:
    /// Maximum blocks in flight per peer
    static constexpr int MAX_BLOCKS_PER_PEER = 16;

    /// Maximum total blocks in flight
    static constexpr int MAX_BLOCKS_TOTAL = 1024;

    /// Maximum headers per getheaders request
    static constexpr int MAX_HEADERS_RESULTS = 2000;

    /// Stall timeout (seconds) — retry if a peer hasn't delivered
    static constexpr int64_t STALL_TIMEOUT = 30;

    /// Header sync timeout (seconds)
    static constexpr int64_t HEADER_SYNC_TIMEOUT = 120;

    SyncManager(ConnManager& connman);
    ~SyncManager();

    // Non-copyable
    SyncManager(const SyncManager&) = delete;
    SyncManager& operator=(const SyncManager&) = delete;

    // ── Lifecycle ───────────────────────────────────────────────────

    /// Start the sync process with our current tip
    void start_sync(int32_t tip_height, const rnet::uint256& tip_hash);

    /// Stop the sync process
    void stop();

    // ── Event handlers ──────────────────────────────────────────────

    /// Called when headers are received from a peer.
    /// header_data is raw serialized headers (the handler will deserialize).
    void on_headers(uint64_t peer_id,
                    const std::vector<uint8_t>& header_data);

    /// Called when a block is received from a peer.
    /// block_data is raw serialized block (the handler will deserialize).
    void on_block(uint64_t peer_id,
                  const std::vector<uint8_t>& block_data);

    /// Called when a new peer connects (possibly start syncing from them)
    void on_new_peer(uint64_t peer_id, int32_t peer_height);

    /// Called when a peer disconnects (reassign their in-flight blocks)
    void on_peer_disconnected(uint64_t peer_id);

    /// Periodic maintenance (check for stalled downloads, etc.)
    void tick();

    // ── Queries ─────────────────────────────────────────────────────

    /// Get current sync stage
    SyncStage stage() const { return stage_.load(); }

    /// Get sync progress as a float in [0.0, 1.0]
    float sync_progress() const;

    /// Get the number of blocks remaining to download
    int32_t blocks_remaining() const;

    /// Check if we are still in initial block download
    bool is_initial_block_download() const;

    /// Total blocks in flight across all peers
    size_t blocks_in_flight() const;

    // ── Block request interface ─────────────────────────────────────

    /// Request specific blocks from a peer
    void request_blocks(uint64_t peer_id,
                        const std::vector<rnet::uint256>& hashes);

    /// Check if a block is currently in flight
    bool is_block_in_flight(const rnet::uint256& hash) const;

    // ── Callbacks ───────────────────────────────────────────────────

    /// Called when a validated block is ready for the chain
    using BlockReadyCallback = std::function<bool(
        const std::vector<uint8_t>& block_data)>;
    void set_on_block_ready(BlockReadyCallback cb) {
        on_block_ready_ = std::move(cb);
    }

    /// Called when header validation is needed
    using ValidateHeaderCallback = std::function<bool(
        const std::vector<uint8_t>& header_data)>;
    void set_on_validate_header(ValidateHeaderCallback cb) {
        on_validate_header_ = std::move(cb);
    }

private:
    ConnManager& connman_;

    std::atomic<SyncStage> stage_{SyncStage::IDLE};

    mutable core::Mutex cs_sync_;

    // Our current tip
    int32_t tip_height_ = 0;
    rnet::uint256 tip_hash_;

    // Best header height we know about (from any peer)
    int32_t best_header_height_ = 0;
    rnet::uint256 best_header_hash_;

    // Target height for IBD (best peer's height)
    int32_t target_height_ = 0;

    // The peer we are syncing headers from
    uint64_t header_sync_peer_ = 0;
    int64_t header_sync_start_time_ = 0;

    // Per-peer sync state
    std::unordered_map<uint64_t, PeerSyncState> peer_states_;

    // Global set of blocks in flight (for dedup)
    std::set<rnet::uint256> all_blocks_in_flight_;

    // Queue of block hashes to download (in chain order)
    std::vector<rnet::uint256> download_queue_;

    // Blocks received but not yet in chain order
    int32_t blocks_received_ = 0;

    // Callbacks
    BlockReadyCallback on_block_ready_;
    ValidateHeaderCallback on_validate_header_;

    // ── Internal helpers ────────────────────────────────────────────

    /// Select the best peer for header synchronization
    uint64_t select_header_sync_peer();

    /// Send a getheaders request to a peer
    void send_getheaders(uint64_t peer_id,
                         const rnet::uint256& locator_hash,
                         const rnet::uint256& stop_hash);

    /// Send a getdata request for blocks to a peer
    void send_getblocks_data(uint64_t peer_id,
                             const std::vector<rnet::uint256>& hashes);

    /// Assign blocks from the download queue to available peers
    void assign_blocks_to_peers();

    /// Handle a stalled peer (reassign their blocks)
    void handle_stalled_peer(uint64_t peer_id);

    /// Update sync stage based on current state
    void update_stage();

    /// Advance to block download after headers are complete
    void start_block_download();
};

}  // namespace rnet::net

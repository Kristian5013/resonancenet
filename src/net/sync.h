#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include "chain/block_index.h"
#include "core/error.h"
#include "core/sync.h"
#include "core/types.h"
#include "primitives/block.h"
#include "net/peer.h"

namespace rnet::chain { class CChainState; }

namespace rnet::net {

class ConnManager;

/// SyncState — tracks the synchronization state of the node
enum class SyncStage {
    NOT_STARTED,
    DOWNLOADING_HEADERS,
    DOWNLOADING_BLOCKS,
    DOWNLOADING_CHECKPOINTS,
    SYNCED,
};

/// BlockSync — manages block synchronization with peers.
/// Implements headers-first download: get all headers, then
/// download blocks in parallel from multiple peers.
class BlockSync {
public:
    BlockSync(chain::CChainState& chainstate,
              ConnManager& connman);
    ~BlockSync();

    // Non-copyable
    BlockSync(const BlockSync&) = delete;
    BlockSync& operator=(const BlockSync&) = delete;

    /// Start the sync process
    void start();

    /// Stop the sync process
    void stop();

    /// Handle a received headers message
    void on_headers(CPeer& peer,
                    const std::vector<primitives::CBlockHeader>& headers);

    /// Handle a received block
    void on_block(CPeer& peer, const primitives::CBlock& block);

    /// Handle a new peer connection (potentially start syncing from them)
    void on_new_peer(CPeer& peer);

    /// Handle a peer disconnect (reassign their work)
    void on_peer_disconnected(uint64_t peer_id);

    /// Get current sync stage
    SyncStage stage() const { return stage_.load(); }

    /// Get sync progress (0.0 to 1.0)
    float progress() const;

    /// Get the number of blocks remaining to download
    int blocks_remaining() const;

    /// Check if initial block download is complete
    bool is_initial_block_download() const;

private:
    chain::CChainState& chainstate_;
    ConnManager& connman_;

    std::atomic<SyncStage> stage_{SyncStage::NOT_STARTED};

    mutable core::Mutex cs_sync_;

    /// Best header height we know about (from any peer)
    int best_header_height_ = 0;

    /// Peer we are syncing headers from
    uint64_t header_sync_peer_ = 0;

    /// Last received header hash (for getheaders locator)
    rnet::uint256 last_header_hash_;

    /// Blocks we have requested but not yet received
    std::set<rnet::uint256> blocks_in_flight_;

    /// Map from peer_id -> set of blocks requested from that peer
    std::unordered_map<uint64_t, std::set<rnet::uint256>> peer_blocks_;

    /// Maximum blocks in flight per peer
    static constexpr int MAX_BLOCKS_IN_FLIGHT_PER_PEER = 16;

    /// Request headers from a peer
    void request_headers(CPeer& peer);

    /// Request blocks from peers
    void request_blocks();

    /// Select the best peer for header sync
    uint64_t select_header_sync_peer();

    /// Check if we need more headers
    bool needs_more_headers() const;
};

}  // namespace rnet::net

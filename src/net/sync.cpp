#include "net/sync.h"

#include "chain/chainstate.h"
#include "core/logging.h"
#include "core/stream.h"
#include "net/conn_manager.h"

namespace rnet::net {

BlockSync::BlockSync(chain::CChainState& chainstate,
                     ConnManager& connman)
    : chainstate_(chainstate)
    , connman_(connman)
{}

BlockSync::~BlockSync() {
    stop();
}

void BlockSync::start() {
    stage_.store(SyncStage::DOWNLOADING_HEADERS);
    LogPrintf("Block sync started");
}

void BlockSync::stop() {
    stage_.store(SyncStage::NOT_STARTED);
}

void BlockSync::on_headers(CPeer& peer,
                           const std::vector<primitives::CBlockHeader>& headers) {
    LOCK(cs_sync_);

    if (headers.empty()) {
        // No more headers — switch to block download
        if (stage_.load() == SyncStage::DOWNLOADING_HEADERS) {
            stage_.store(SyncStage::DOWNLOADING_BLOCKS);
            LogPrintf("Header sync complete, switching to block download");
            request_blocks();
        }
        return;
    }

    // Accept each header into the chain state
    for (const auto& header : headers) {
        auto result = chainstate_.accept_block_header(header);
        if (result.is_err()) {
            LogPrint(NET,"Peer %llu sent invalid header: %s",
                     static_cast<unsigned long long>(peer.id),
                     result.error().c_str());
            peer.misbehaving(20, "invalid header");
            return;
        }

        auto* idx = result.value();
        if (idx->height > best_header_height_) {
            best_header_height_ = idx->height;
        }
    }

    // Request more headers
    request_headers(peer);
}

void BlockSync::on_block(CPeer& peer, const primitives::CBlock& block) {
    LOCK(cs_sync_);

    auto hash = block.hash();

    // Remove from in-flight
    blocks_in_flight_.erase(hash);
    auto pit = peer_blocks_.find(peer.id);
    if (pit != peer_blocks_.end()) {
        pit->second.erase(hash);
    }

    // Accept the block
    auto result = chainstate_.accept_block(block);
    if (result.is_err()) {
        LogPrint(NET,"Peer %llu sent invalid block: %s",
                 static_cast<unsigned long long>(peer.id),
                 result.error().c_str());
        peer.misbehaving(100, "invalid block");
        return;
    }

    LogPrintf("Accepted block height=%d from peer %llu",
             result.value()->height,
             static_cast<unsigned long long>(peer.id));

    // Request more blocks if needed
    if (blocks_in_flight_.empty()) {
        if (chainstate_.height() >= best_header_height_) {
            stage_.store(SyncStage::SYNCED);
            LogPrintf("Initial block download complete at height %d",
                      chainstate_.height());
        } else {
            request_blocks();
        }
    }
}

void BlockSync::on_new_peer(CPeer& peer) {
    LOCK(cs_sync_);

    // Update best known height
    if (peer.start_height > best_header_height_) {
        best_header_height_ = peer.start_height;
    }

    // If we need headers and don't have a sync peer, use this one
    if (stage_.load() == SyncStage::DOWNLOADING_HEADERS &&
        header_sync_peer_ == 0) {
        header_sync_peer_ = peer.id;
        request_headers(peer);
    }
}

void BlockSync::on_peer_disconnected(uint64_t peer_id) {
    LOCK(cs_sync_);

    // Put their in-flight blocks back for re-request
    auto pit = peer_blocks_.find(peer_id);
    if (pit != peer_blocks_.end()) {
        for (const auto& hash : pit->second) {
            blocks_in_flight_.erase(hash);
        }
        peer_blocks_.erase(pit);
    }

    // If this was our header sync peer, pick a new one
    if (header_sync_peer_ == peer_id) {
        header_sync_peer_ = select_header_sync_peer();
    }
}

float BlockSync::progress() const {
    if (best_header_height_ <= 0) return 0.0f;
    int current = chainstate_.height();
    if (current >= best_header_height_) return 1.0f;
    return static_cast<float>(current) /
           static_cast<float>(best_header_height_);
}

int BlockSync::blocks_remaining() const {
    int remaining = best_header_height_ - chainstate_.height();
    return remaining > 0 ? remaining : 0;
}

bool BlockSync::is_initial_block_download() const {
    return stage_.load() != SyncStage::SYNCED;
}

void BlockSync::request_headers(CPeer& peer) {
    // Build a getHeaders message with our chain tip hash
    auto* tip = chainstate_.tip();
    if (!tip) return;

    // Send getheaders with the tip hash as the locator
    core::DataStream ss;
    core::Serialize(ss, static_cast<uint32_t>(PROTOCOL_VERSION));
    core::Serialize(ss, static_cast<uint32_t>(1));  // count = 1
    tip->block_hash.serialize(ss);
    rnet::uint256 stop_hash;
    stop_hash.serialize(ss);

    connman_.send_to(peer.id, msg::GETHEADERS, ss.span());
}

void BlockSync::request_blocks() {
    // Walk the block index for blocks we need to download
    // For simplicity, request blocks by height from tip+1
    int start_height = chainstate_.height() + 1;

    for (int h = start_height; h <= best_header_height_; ++h) {
        if (blocks_in_flight_.size() >=
            static_cast<size_t>(MAX_BLOCKS_IN_FLIGHT_PER_PEER * 4)) {
            break;
        }

        auto* idx = chainstate_.get_block_by_height(h);
        if (!idx) continue;

        if (blocks_in_flight_.count(idx->block_hash) > 0) continue;

        blocks_in_flight_.insert(idx->block_hash);

        // Send getdata for this block to any connected peer
        CInv inv(InvType::INV_BLOCK, idx->block_hash);
        core::DataStream ss;
        core::Serialize(ss, static_cast<uint32_t>(1));  // count
        inv.serialize(ss);

        // Broadcast getdata to first available peer
        connman_.broadcast(msg::GETDATA, ss.span());
    }
}

uint64_t BlockSync::select_header_sync_peer() {
    // Pick peer with highest start_height
    // For now, just return 0 (will be set on next on_new_peer)
    return 0;
}

bool BlockSync::needs_more_headers() const {
    return chainstate_.height() < best_header_height_;
}

}  // namespace rnet::net

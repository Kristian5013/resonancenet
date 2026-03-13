#include "net/sync.h"

#include <algorithm>

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
        last_header_hash_ = idx->block_hash;
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
    // Build a getHeaders message with our best known header hash as locator
    // Use last_header_hash_ if we've received headers, otherwise chain tip
    rnet::uint256 locator_hash;
    if (!last_header_hash_.is_zero()) {
        locator_hash = last_header_hash_;
    } else {
        auto* tip = chainstate_.tip();
        if (!tip) return;
        locator_hash = tip->block_hash;
    }

    core::DataStream ss;
    core::ser_write_i32(ss, static_cast<int32_t>(PROTOCOL_VERSION));
    core::serialize_compact_size(ss, 1);  // count = 1
    locator_hash.serialize(ss);
    rnet::uint256 stop_hash;
    stop_hash.serialize(ss);

    connman_.send_to(peer.id, msg::GETHEADERS, ss.span());
}

void BlockSync::request_blocks() {
    // Collect header-only block index entries that need downloading
    // Walk from best header back to our tip via prev pointers
    int start_height = chainstate_.height() + 1;

    // Build list of hashes we need, by walking block_index
    std::vector<rnet::uint256> needed;
    for (const auto& [hash, idx] : chainstate_.block_index()) {
        if (idx->height >= start_height &&
            idx->height <= best_header_height_ &&
            idx->status < chain::CBlockIndex::FULLY_VALIDATED) {
            needed.push_back(hash);
        }
    }

    // Sort by height
    std::sort(needed.begin(), needed.end(),
        [this](const rnet::uint256& a, const rnet::uint256& b) {
            auto* ia = chainstate_.lookup_block_index(a);
            auto* ib = chainstate_.lookup_block_index(b);
            return ia->height < ib->height;
        });

    // Request blocks in batches
    std::vector<CInv> to_request;
    for (const auto& hash : needed) {
        if (blocks_in_flight_.size() + to_request.size() >=
            static_cast<size_t>(MAX_BLOCKS_IN_FLIGHT_PER_PEER * 4)) {
            break;
        }
        if (blocks_in_flight_.count(hash) > 0) continue;

        blocks_in_flight_.insert(hash);
        to_request.emplace_back(InvType::INV_BLOCK, hash);
    }

    if (to_request.empty()) return;

    // Send single getdata with all requested blocks
    core::DataStream ss;
    core::serialize_compact_size(ss, to_request.size());
    for (const auto& inv : to_request) {
        inv.serialize(ss);
    }
    connman_.broadcast(msg::GETDATA, ss.span());

    LogPrintf("Requested %zu blocks for IBD (height %d..%d)",
             to_request.size(), start_height, best_header_height_);
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

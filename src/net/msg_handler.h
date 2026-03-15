#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/stream.h"
#include "core/sync.h"
#include "core/types.h"
#include "net/connection.h"
#include "net/protocol.h"

namespace rnet::net {

class AddrManager;
class CheckpointStore;
class ConnManager;

/// Callback types for chain/mempool integration
using BlockCallback = std::function<bool(
    uint64_t peer_id,
    const std::vector<uint8_t>& block_data)>;

using TxCallback = std::function<bool(
    uint64_t peer_id,
    const std::vector<uint8_t>& tx_data)>;

using HeadersCallback = std::function<bool(
    uint64_t peer_id,
    const std::vector<uint8_t>& headers_data)>;

using InvCallback = std::function<void(
    uint64_t peer_id,
    const std::vector<CInv>& inv_list)>;

/// MsgHandler — processes incoming P2P protocol messages.
///
/// Routes messages by command name to specific handler functions.
/// Each handler deserializes the payload and performs the appropriate
/// action (update peer state, relay data, request more data, etc.).
///
/// Callbacks are used to notify the chain/mempool/sync layers about
/// new blocks, transactions, and headers without creating circular
/// dependencies.
class MsgHandler {
public:
    explicit MsgHandler(ConnManager& connman, AddrManager* addrman = nullptr);
    ~MsgHandler();

    // Non-copyable
    MsgHandler(const MsgHandler&) = delete;
    MsgHandler& operator=(const MsgHandler&) = delete;

    /// Register all message handlers with the ConnManager
    void register_handlers();

    // ── Callbacks for chain/mempool integration ─────────────────────

    /// Called when a new block is received
    void set_on_new_block(BlockCallback cb) { on_new_block_ = std::move(cb); }

    /// Called when a new transaction is received
    void set_on_new_tx(TxCallback cb) { on_new_tx_ = std::move(cb); }

    /// Called when headers are received
    void set_on_headers(HeadersCallback cb) { on_headers_ = std::move(cb); }

    /// Called when inventory is received
    void set_on_inv(InvCallback cb) { on_inv_ = std::move(cb); }

    /// Set our local inventory check function (do we have this item?)
    using HaveItemFn = std::function<bool(const CInv&)>;
    void set_have_item(HaveItemFn fn) { have_item_ = std::move(fn); }

    /// Set the function to get block data for sending
    using GetBlockDataFn = std::function<
        std::vector<uint8_t>(const rnet::uint256& hash)>;
    void set_get_block_data(GetBlockDataFn fn) {
        get_block_data_ = std::move(fn);
    }

    /// Set the function to get tx data for sending
    using GetTxDataFn = std::function<
        std::vector<uint8_t>(const rnet::uint256& hash)>;
    void set_get_tx_data(GetTxDataFn fn) {
        get_tx_data_ = std::move(fn);
    }

    /// Set the function to find fork point from locator and return block hashes
    /// Returns up to 500 block hashes starting after the fork point
    using GetBlockHashesFn = std::function<
        std::vector<rnet::uint256>(
            const std::vector<rnet::uint256>& locator,
            const rnet::uint256& stop_hash,
            int max_count)>;
    void set_get_block_hashes(GetBlockHashesFn fn) {
        get_block_hashes_ = std::move(fn);
    }

    /// Set the function to get headers from locator
    using GetHeadersFn = std::function<
        std::vector<std::vector<uint8_t>>(
            const std::vector<rnet::uint256>& locator,
            const rnet::uint256& stop_hash,
            int max_count)>;
    void set_get_headers(GetHeadersFn fn) {
        get_headers_ = std::move(fn);
    }

    /// Set the function to get all mempool tx hashes
    using GetMempoolTxIdsFn = std::function<std::vector<rnet::uint256>()>;
    void set_get_mempool_txids(GetMempoolTxIdsFn fn) {
        get_mempool_txids_ = std::move(fn);
    }

    /// Set the checkpoint store for serving/receiving checkpoint files
    void set_checkpoint_store(CheckpointStore* store) {
        checkpoint_store_ = store;
    }

    // ── Addr propagation ────────────────────────────────────────────

    /// Set our discovered external address (learned from version messages).
    /// Once set, this address is advertised to newly-connected peers and
    /// periodically re-broadcast to keep it fresh in the network.
    void set_local_addr(const CNetAddr& addr);

    /// Send our local address to a specific peer as an addr message.
    /// Called after handshake completion to self-advertise.
    void push_local_addr(CConnection& conn);

    /// Send our local address to a random connected peer.
    /// Called periodically (~30 min) by the maintenance loop.
    void advertise_local_addr();

private:
    ConnManager& connman_;
    AddrManager* addrman_ = nullptr;
    CheckpointStore* checkpoint_store_ = nullptr;

    // Addr propagation state
    mutable core::Mutex cs_local_addr_;
    CNetAddr local_addr_;                   ///< Our discovered external address
    std::atomic<bool> local_addr_set_{false}; ///< True once local_addr_ is valid

    // Callbacks
    BlockCallback on_new_block_;
    TxCallback on_new_tx_;
    HeadersCallback on_headers_;
    InvCallback on_inv_;
    HaveItemFn have_item_;
    GetBlockDataFn get_block_data_;
    GetTxDataFn get_tx_data_;
    GetBlockHashesFn get_block_hashes_;
    GetHeadersFn get_headers_;
    GetMempoolTxIdsFn get_mempool_txids_;

    // ── Handler functions ───────────────────────────────────────────

    void process_ping(CConnection& conn, const std::string& cmd,
                      core::DataStream& payload);
    void process_pong(CConnection& conn, const std::string& cmd,
                      core::DataStream& payload);
    void process_addr(CConnection& conn, const std::string& cmd,
                      core::DataStream& payload);
    void process_getaddr(CConnection& conn, const std::string& cmd,
                         core::DataStream& payload);
    void process_inv(CConnection& conn, const std::string& cmd,
                     core::DataStream& payload);
    void process_getdata(CConnection& conn, const std::string& cmd,
                         core::DataStream& payload);
    void process_getblocks(CConnection& conn, const std::string& cmd,
                           core::DataStream& payload);
    void process_getheaders(CConnection& conn, const std::string& cmd,
                            core::DataStream& payload);
    void process_tx(CConnection& conn, const std::string& cmd,
                    core::DataStream& payload);
    void process_block(CConnection& conn, const std::string& cmd,
                       core::DataStream& payload);
    void process_headers(CConnection& conn, const std::string& cmd,
                         core::DataStream& payload);
    void process_sendheaders(CConnection& conn, const std::string& cmd,
                             core::DataStream& payload);
    void process_mempool(CConnection& conn, const std::string& cmd,
                         core::DataStream& payload);
    void process_notfound(CConnection& conn, const std::string& cmd,
                          core::DataStream& payload);
    void process_reject(CConnection& conn, const std::string& cmd,
                        core::DataStream& payload);
    void process_checkpoint(CConnection& conn, const std::string& cmd,
                            core::DataStream& payload);
    void process_getchkpt(CConnection& conn, const std::string& cmd,
                          core::DataStream& payload);
    void process_trainstatus(CConnection& conn, const std::string& cmd,
                             core::DataStream& payload);
    void process_growthinfo(CConnection& conn, const std::string& cmd,
                            core::DataStream& payload);

    /// Send a reject message to a peer
    void send_reject(CConnection& conn, const std::string& rejected_cmd,
                     uint8_t code, const std::string& reason);

    /// Send notfound for items we don't have
    void send_notfound(CConnection& conn, const std::vector<CInv>& items);

    /// Relay addresses to other connected peers (probabilistic, ~50%).
    /// Excludes the originating peer to prevent echo loops.
    void relay_addr(uint64_t from_id, const std::vector<CNetAddr>& addrs);

    /// Build a serialized addr message payload from an address list.
    static core::DataStream build_addr_payload(const std::vector<CNetAddr>& addrs);
};

}  // namespace rnet::net

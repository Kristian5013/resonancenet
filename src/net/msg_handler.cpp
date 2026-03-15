// Copyright (c) 2024-2026 The ResonanceNet Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "net/msg_handler.h"

#include "core/logging.h"
#include "core/random.h"
#include "core/serialize.h"
#include "core/time.h"
#include "net/addr_man.h"
#include "net/conn_manager.h"

#include <algorithm>
#include <cstring>

namespace rnet::net {

// ---------------------------------------------------------------------------
// MsgHandler (constructor)
// ---------------------------------------------------------------------------
MsgHandler::MsgHandler(ConnManager& connman, AddrManager* addrman)
    : connman_(connman), addrman_(addrman)
{}

// ---------------------------------------------------------------------------
// ~MsgHandler (destructor)
// ---------------------------------------------------------------------------
MsgHandler::~MsgHandler() = default;

// ---------------------------------------------------------------------------
// register_handlers
// ---------------------------------------------------------------------------
// Wires every recognised P2P command string to its handler lambda.
// Called once during node startup after ConnManager is initialised.
// ---------------------------------------------------------------------------
void MsgHandler::register_handlers() {
    // 1. Core protocol messages
    connman_.register_handler(msg::PING,
        [this](CConnection& c, const std::string& cmd, core::DataStream& p) {
            process_ping(c, cmd, p);
        });
    connman_.register_handler(msg::PONG,
        [this](CConnection& c, const std::string& cmd, core::DataStream& p) {
            process_pong(c, cmd, p);
        });

    // 2. Address gossip
    connman_.register_handler(msg::ADDR,
        [this](CConnection& c, const std::string& cmd, core::DataStream& p) {
            process_addr(c, cmd, p);
        });
    connman_.register_handler(msg::GETADDR,
        [this](CConnection& c, const std::string& cmd, core::DataStream& p) {
            process_getaddr(c, cmd, p);
        });

    // 3. Inventory relay
    connman_.register_handler(msg::INV,
        [this](CConnection& c, const std::string& cmd, core::DataStream& p) {
            process_inv(c, cmd, p);
        });
    connman_.register_handler(msg::GETDATA,
        [this](CConnection& c, const std::string& cmd, core::DataStream& p) {
            process_getdata(c, cmd, p);
        });

    // 4. Block/header sync
    connman_.register_handler(msg::GETBLOCKS,
        [this](CConnection& c, const std::string& cmd, core::DataStream& p) {
            process_getblocks(c, cmd, p);
        });
    connman_.register_handler(msg::GETHEADERS,
        [this](CConnection& c, const std::string& cmd, core::DataStream& p) {
            process_getheaders(c, cmd, p);
        });

    // 5. Data relay
    connman_.register_handler(msg::TX,
        [this](CConnection& c, const std::string& cmd, core::DataStream& p) {
            process_tx(c, cmd, p);
        });
    connman_.register_handler(msg::BLOCK,
        [this](CConnection& c, const std::string& cmd, core::DataStream& p) {
            process_block(c, cmd, p);
        });
    connman_.register_handler(msg::HEADERS,
        [this](CConnection& c, const std::string& cmd, core::DataStream& p) {
            process_headers(c, cmd, p);
        });

    // 6. Miscellaneous
    connman_.register_handler(msg::SENDHEADERS,
        [this](CConnection& c, const std::string& cmd, core::DataStream& p) {
            process_sendheaders(c, cmd, p);
        });
    connman_.register_handler(msg::MEMPOOL,
        [this](CConnection& c, const std::string& cmd, core::DataStream& p) {
            process_mempool(c, cmd, p);
        });
    connman_.register_handler(msg::NOTFOUND,
        [this](CConnection& c, const std::string& cmd, core::DataStream& p) {
            process_notfound(c, cmd, p);
        });
    connman_.register_handler(msg::REJECT,
        [this](CConnection& c, const std::string& cmd, core::DataStream& p) {
            process_reject(c, cmd, p);
        });

    // 7. ResonanceNet PoT-specific messages
    connman_.register_handler(msg::CHECKPOINT,
        [this](CConnection& c, const std::string& cmd, core::DataStream& p) {
            process_checkpoint(c, cmd, p);
        });
    connman_.register_handler(msg::GETCHECKPOINT,
        [this](CConnection& c, const std::string& cmd, core::DataStream& p) {
            process_getchkpt(c, cmd, p);
        });
    connman_.register_handler(msg::TRAININGSTATUS,
        [this](CConnection& c, const std::string& cmd, core::DataStream& p) {
            process_trainstatus(c, cmd, p);
        });
    connman_.register_handler(msg::GROWTHINFO,
        [this](CConnection& c, const std::string& cmd, core::DataStream& p) {
            process_growthinfo(c, cmd, p);
        });

    LogPrint(NET, "Registered %d P2P message handlers", 20);
}

// ---------------------------------------------------------------------------
// process_ping
// ---------------------------------------------------------------------------
// Handles an incoming "ping" message.
//
// If the payload contains an 8-byte nonce, echoes it back as a "pong".
// Legacy peers may send a zero-length ping; reply with an empty pong.
// ---------------------------------------------------------------------------
void MsgHandler::process_ping(CConnection& conn, const std::string& /*cmd*/,
                              core::DataStream& payload) {
    if (payload.remaining() >= 8) {
        // 1. Read the 8-byte nonce from the ping payload.
        uint64_t nonce = core::ser_read_u64(payload);

        // 2. Serialize the same nonce into a pong reply.
        core::DataStream pong_payload;
        core::ser_write_u64(pong_payload, nonce);
        conn.send_message(msg::PONG, pong_payload.span());
    } else {
        // 3. Old-style ping with no nonce -- reply with empty pong.
        conn.send_message(msg::PONG);
    }
}

// ---------------------------------------------------------------------------
// process_pong
// ---------------------------------------------------------------------------
// Handles an incoming "pong" response.
//
// Matches the returned nonce against the outstanding ping to compute
// round-trip latency.  A mismatched nonce is logged but not penalised.
// ---------------------------------------------------------------------------
void MsgHandler::process_pong(CConnection& conn, const std::string& /*cmd*/,
                              core::DataStream& payload) {
    // 1. Validate minimum payload length.
    if (payload.remaining() < 8) {
        conn.misbehaving(1, "Short pong payload");
        return;
    }

    // 2. Read the nonce and record latency.
    uint64_t nonce = core::ser_read_u64(payload);
    int64_t latency = conn.record_pong(nonce);

    // 3. Log the result.
    if (latency >= 0) {
        LogDebug(NET, "Pong from peer %llu: %lldms",
                 static_cast<unsigned long long>(conn.id()),
                 static_cast<long long>(latency));
    } else {
        LogDebug(NET, "Unexpected pong from peer %llu (nonce mismatch)",
                 static_cast<unsigned long long>(conn.id()));
    }
}

// ---------------------------------------------------------------------------
// process_addr
// ---------------------------------------------------------------------------
// Handles an incoming "addr" message containing peer network addresses.
//
// Reads up to 1000 serialised CNetAddr entries and forwards them to the
// AddrManager for persistent storage and future connection attempts.
// ---------------------------------------------------------------------------
void MsgHandler::process_addr(CConnection& conn, const std::string& /*cmd*/,
                              core::DataStream& payload) {
    // 1. Read the address count (varint).
    if (payload.remaining() < 1) return;
    uint64_t count = core::unserialize_compact_size(payload);

    // 2. Enforce the 1000-address-per-message limit.
    if (count > 1000) {
        conn.misbehaving(20, "Oversized addr message: " +
                             std::to_string(count));
        return;
    }

    // 3. Deserialize each address.
    std::vector<CNetAddr> addrs;
    addrs.reserve(static_cast<size_t>(count));
    for (uint64_t i = 0; i < count; ++i) {
        CNetAddr addr;
        addr.unserialize(payload);
        addrs.push_back(addr);
    }

    LogDebug(NET, "Received %llu addresses from peer %llu",
             static_cast<unsigned long long>(count),
             static_cast<unsigned long long>(conn.id()));

    // 4. Forward to AddrManager for storage.
    if (addrman_ && !addrs.empty()) {
        std::string source = conn.addr().to_string();
        addrman_->add(addrs, source);

        LogDebug(NET, "Added %llu addresses from peer %llu to addrman",
                 static_cast<unsigned long long>(addrs.size()),
                 static_cast<unsigned long long>(conn.id()));
    }
}

// ---------------------------------------------------------------------------
// process_getaddr
// ---------------------------------------------------------------------------
// Handles an incoming "getaddr" request.
//
// Responds with up to 1000 known addresses from AddrManager, serialised
// as an "addr" message.
// ---------------------------------------------------------------------------
void MsgHandler::process_getaddr(CConnection& conn,
                                 const std::string& /*cmd*/,
                                 core::DataStream& /*payload*/) {
    LogDebug(NET, "Received getaddr from peer %llu",
             static_cast<unsigned long long>(conn.id()));

    // 1. Retrieve known addresses from AddrManager.
    std::vector<CNetAddr> addrs;
    if (addrman_) {
        addrs = addrman_->get_addr(1000);
    }

    // 2. Serialize and send the addr response.
    core::DataStream addr_payload;
    core::serialize_compact_size(addr_payload, addrs.size());
    for (const auto& addr : addrs) {
        addr.serialize(addr_payload);
    }
    conn.send_message(msg::ADDR, addr_payload.span());

    LogDebug(NET, "Sent %zu addresses to peer %llu",
             addrs.size(),
             static_cast<unsigned long long>(conn.id()));
}

// ---------------------------------------------------------------------------
// process_inv
// ---------------------------------------------------------------------------
// Handles an incoming "inv" (inventory announcement) message.
//
// Deserializes the list of CInv entries, checks each against the local
// inventory via have_item_(), and sends a "getdata" request for any
// items we do not yet possess.
// ---------------------------------------------------------------------------
void MsgHandler::process_inv(CConnection& conn, const std::string& /*cmd*/,
                             core::DataStream& payload) {
    // 1. Read and validate the inventory count.
    uint64_t count = core::unserialize_compact_size(payload);
    if (count > 50000) {
        conn.misbehaving(20, "Oversized inv message: " +
                             std::to_string(count));
        return;
    }

    // 2. Deserialize each CInv and partition into known/unknown.
    std::vector<CInv> inv_list;
    inv_list.reserve(static_cast<size_t>(count));
    std::vector<CInv> to_request;

    for (uint64_t i = 0; i < count; ++i) {
        CInv inv;
        inv.unserialize(payload);
        inv_list.push_back(inv);

        // 3. Check local inventory.
        bool have = false;
        if (have_item_) {
            have = have_item_(inv);
        }
        if (!have) {
            to_request.push_back(inv);
        }
    }

    // 4. Notify the inventory callback.
    if (on_inv_ && !inv_list.empty()) {
        on_inv_(conn.id(), inv_list);
    }

    // 5. Request items we don't have.
    if (!to_request.empty()) {
        core::DataStream getdata_payload;
        core::serialize_compact_size(getdata_payload, to_request.size());
        for (const auto& inv : to_request) {
            inv.serialize(getdata_payload);
        }
        conn.send_message(msg::GETDATA, getdata_payload.span());

        LogDebug(NET, "Requesting %zu items from peer %llu",
                 to_request.size(),
                 static_cast<unsigned long long>(conn.id()));
    }
}

// ---------------------------------------------------------------------------
// process_getdata
// ---------------------------------------------------------------------------
// Handles an incoming "getdata" request.
//
// For each requested CInv, looks up block or transaction data via the
// registered callbacks and sends it back.  Items that cannot be served
// are collected and returned in a single "notfound" message.
// ---------------------------------------------------------------------------
void MsgHandler::process_getdata(CConnection& conn,
                                 const std::string& /*cmd*/,
                                 core::DataStream& payload) {
    // 1. Read and validate the request count.
    uint64_t count = core::unserialize_compact_size(payload);
    if (count > 50000) {
        conn.misbehaving(20, "Oversized getdata message: " +
                             std::to_string(count));
        return;
    }

    std::vector<CInv> not_found;

    // 2. Process each requested inventory item.
    for (uint64_t i = 0; i < count; ++i) {
        CInv inv;
        inv.unserialize(payload);

        bool sent = false;

        // 3a. Serve block data if available.
        if (inv.type == InvType::INV_BLOCK ||
            inv.type == InvType::INV_WITNESS_BLOCK) {
            if (get_block_data_) {
                auto data = get_block_data_(inv.hash);
                if (!data.empty()) {
                    conn.send_message(msg::BLOCK,
                        std::span<const uint8_t>(data.data(), data.size()));
                    sent = true;
                }
            }
        // 3b. Serve transaction data if available.
        } else if (inv.type == InvType::INV_TX ||
                   inv.type == InvType::INV_WITNESS_TX) {
            if (get_tx_data_) {
                auto data = get_tx_data_(inv.hash);
                if (!data.empty()) {
                    conn.send_message(msg::TX,
                        std::span<const uint8_t>(data.data(), data.size()));
                    sent = true;
                }
            }
        }

        // 4. Track items we could not serve.
        if (!sent) {
            not_found.push_back(inv);
        }
    }

    // 5. Send a single notfound for all missing items.
    if (!not_found.empty()) {
        send_notfound(conn, not_found);
    }
}

// ---------------------------------------------------------------------------
// process_notfound
// ---------------------------------------------------------------------------
// Handles an incoming "notfound" message.
//
// Logs each item the peer could not serve.  No further action is taken;
// the sync layer will retry with a different peer if needed.
// ---------------------------------------------------------------------------
void MsgHandler::process_notfound(CConnection& conn,
                                  const std::string& /*cmd*/,
                                  core::DataStream& payload) {
    uint64_t count = core::unserialize_compact_size(payload);

    for (uint64_t i = 0; i < count; ++i) {
        CInv inv;
        inv.unserialize(payload);

        LogDebug(NET, "Peer %llu notfound: %s",
                 static_cast<unsigned long long>(conn.id()),
                 inv.to_string().c_str());
    }
}

// ---------------------------------------------------------------------------
// process_getblocks
// ---------------------------------------------------------------------------
// Handles an incoming "getblocks" request.
//
// Reads a block locator (up to 101 hashes) and a stop-hash, resolves
// them via get_block_hashes_(), and sends the actual block data directly
// as sequential "block" messages.  This enables simplified Initial Block
// Download (IBD) without requiring the inv -> getdata round-trip.
//
// Steps:
//   1. Read and discard the protocol version field
//   2. Read and validate the locator hash count
//   3. Deserialize the locator hashes
//   4. Read the stop hash
//   5. Resolve the locator to a list of block hashes (up to 500)
//   6. For each hash, look up the serialized block data and send it
// ---------------------------------------------------------------------------
void MsgHandler::process_getblocks(CConnection& conn,
                                   const std::string& /*cmd*/,
                                   core::DataStream& payload) {
    // 1. Read protocol version (unused but must be consumed).
    int32_t version = core::ser_read_i32(payload);
    (void)version;

    // 2. Read and validate the locator hash count.
    uint64_t locator_count = core::unserialize_compact_size(payload);
    if (locator_count > 101) {
        conn.misbehaving(20, "Oversized getblocks locator");
        return;
    }

    // 3. Deserialize the locator hashes.
    std::vector<rnet::uint256> locator;
    locator.reserve(static_cast<size_t>(locator_count));
    for (uint64_t i = 0; i < locator_count; ++i) {
        rnet::uint256 hash;
        hash.unserialize(payload);
        locator.push_back(hash);
    }

    // 4. Read the stop hash.
    rnet::uint256 stop_hash;
    stop_hash.unserialize(payload);

    LogDebug(NET, "Received getblocks from peer %llu (locator size=%llu)",
             static_cast<unsigned long long>(conn.id()),
             static_cast<unsigned long long>(locator_count));

    if (!get_block_hashes_) return;

    // 5. Resolve the locator to a list of block hashes.
    auto hashes = get_block_hashes_(locator, stop_hash, 500);
    if (hashes.empty()) return;

    // 6. Send each block directly as a "block" message for IBD.
    size_t blocks_sent = 0;
    for (const auto& h : hashes) {
        if (!get_block_data_) break;
        auto data = get_block_data_(h);
        if (!data.empty()) {
            conn.send_message(msg::BLOCK,
                std::span<const uint8_t>(data.data(), data.size()));
            ++blocks_sent;
        }
    }

    LogPrint(NET, "Sent %zu blocks (of %zu requested) to peer %llu",
             blocks_sent, hashes.size(),
             static_cast<unsigned long long>(conn.id()));
}

// ---------------------------------------------------------------------------
// process_getheaders
// ---------------------------------------------------------------------------
// Handles an incoming "getheaders" request.
//
// Same locator mechanism as getblocks, but responds with serialised
// block headers (up to 2000) rather than inventory entries.  Each
// header is followed by a zero transaction count per the Bitcoin wire
// protocol convention.
// ---------------------------------------------------------------------------
void MsgHandler::process_getheaders(CConnection& conn,
                                    const std::string& /*cmd*/,
                                    core::DataStream& payload) {
    // 1. Read protocol version (unused but must be consumed).
    int32_t version = core::ser_read_i32(payload);
    (void)version;

    // 2. Read and validate the locator hash count.
    uint64_t locator_count = core::unserialize_compact_size(payload);
    if (locator_count > 101) {
        conn.misbehaving(20, "Oversized getheaders locator");
        return;
    }

    // 3. Deserialize the locator hashes.
    std::vector<rnet::uint256> locator;
    locator.reserve(static_cast<size_t>(locator_count));
    for (uint64_t i = 0; i < locator_count; ++i) {
        rnet::uint256 hash;
        hash.unserialize(payload);
        locator.push_back(hash);
    }

    // 4. Read the stop hash.
    rnet::uint256 stop_hash;
    stop_hash.unserialize(payload);

    LogDebug(NET, "Received getheaders from peer %llu (locator size=%llu)",
             static_cast<unsigned long long>(conn.id()),
             static_cast<unsigned long long>(locator_count));

    if (!get_headers_) return;

    // 5. Fetch matching headers from the chain index.
    auto header_blobs = get_headers_(locator, stop_hash, 2000);

    // 6. Serialize and send the headers response.
    core::DataStream headers_payload;
    core::serialize_compact_size(headers_payload, header_blobs.size());
    for (const auto& hdr_data : header_blobs) {
        headers_payload.write(
            reinterpret_cast<const char*>(hdr_data.data()), hdr_data.size());
        // Bitcoin sends tx_count=0 after each header
        core::serialize_compact_size(headers_payload, 0);
    }
    conn.send_message(msg::HEADERS, headers_payload.span());

    LogDebug(NET, "Sent %zu headers to peer %llu",
             header_blobs.size(),
             static_cast<unsigned long long>(conn.id()));
}

// ---------------------------------------------------------------------------
// process_tx
// ---------------------------------------------------------------------------
// Handles an incoming "tx" (transaction relay) message.
//
// Copies the raw payload and forwards it to the on_new_tx_ callback
// for mempool validation and acceptance.
// ---------------------------------------------------------------------------
void MsgHandler::process_tx(CConnection& conn, const std::string& /*cmd*/,
                            core::DataStream& payload) {
    LogDebug(NET, "Received tx (%zu bytes) from peer %llu",
             payload.remaining(),
             static_cast<unsigned long long>(conn.id()));

    if (on_new_tx_) {
        // 1. Copy the raw transaction bytes.
        std::vector<uint8_t> tx_data(payload.unread_span().begin(),
                                     payload.unread_span().end());

        // 2. Submit to the mempool via callback.
        bool accepted = on_new_tx_(conn.id(), tx_data);
        if (!accepted) {
            LogDebug(NET, "Tx from peer %llu rejected",
                     static_cast<unsigned long long>(conn.id()));
        }
    }
}

// ---------------------------------------------------------------------------
// process_block
// ---------------------------------------------------------------------------
// Handles an incoming "block" relay message.
//
// Copies the raw payload and forwards it to the on_new_block_ callback
// for chain validation and acceptance.
// ---------------------------------------------------------------------------
void MsgHandler::process_block(CConnection& conn,
                               const std::string& /*cmd*/,
                               core::DataStream& payload) {
    LogDebug(NET, "Received block (%zu bytes) from peer %llu",
             payload.remaining(),
             static_cast<unsigned long long>(conn.id()));

    if (on_new_block_) {
        // 1. Copy the raw block bytes.
        std::vector<uint8_t> block_data(payload.unread_span().begin(),
                                        payload.unread_span().end());

        // 2. Submit to the chain via callback.
        bool accepted = on_new_block_(conn.id(), block_data);
        if (!accepted) {
            LogDebug(NET, "Block from peer %llu rejected",
                     static_cast<unsigned long long>(conn.id()));
        }
    }
}

// ---------------------------------------------------------------------------
// process_headers
// ---------------------------------------------------------------------------
// Handles an incoming "headers" message during header-first sync.
//
// Forwards the raw header data to the on_headers_ callback for chain
// index processing.
// ---------------------------------------------------------------------------
void MsgHandler::process_headers(CConnection& conn,
                                 const std::string& /*cmd*/,
                                 core::DataStream& payload) {
    LogDebug(NET, "Received headers (%zu bytes) from peer %llu",
             payload.remaining(),
             static_cast<unsigned long long>(conn.id()));

    if (on_headers_) {
        // 1. Copy the raw headers payload.
        std::vector<uint8_t> headers_data(payload.unread_span().begin(),
                                          payload.unread_span().end());

        // 2. Forward to the sync layer.
        on_headers_(conn.id(), headers_data);
    }
}

// ---------------------------------------------------------------------------
// process_sendheaders
// ---------------------------------------------------------------------------
// Handles the "sendheaders" control message.
//
// The peer indicates it prefers to receive new-block announcements as
// "headers" messages rather than "inv" messages.
// ---------------------------------------------------------------------------
void MsgHandler::process_sendheaders(CConnection& conn,
                                     const std::string& /*cmd*/,
                                     core::DataStream& /*payload*/) {
    // 1. Record the peer's preference.
    conn.set_prefer_headers(true);

    LogDebug(NET, "Peer %llu prefers header announcements",
             static_cast<unsigned long long>(conn.id()));
}

// ---------------------------------------------------------------------------
// process_mempool
// ---------------------------------------------------------------------------
// Handles an incoming "mempool" request.
//
// Responds with an "inv" message listing all transaction IDs currently
// in the local mempool.
// ---------------------------------------------------------------------------
void MsgHandler::process_mempool(CConnection& conn,
                                 const std::string& /*cmd*/,
                                 core::DataStream& /*payload*/) {
    LogDebug(NET, "Received mempool request from peer %llu",
             static_cast<unsigned long long>(conn.id()));

    // 1. Retrieve all mempool transaction IDs.
    if (!get_mempool_txids_) return;
    auto txids = get_mempool_txids_();
    if (txids.empty()) return;

    // 2. Serialize as an inv message and send.
    core::DataStream inv_payload;
    core::serialize_compact_size(inv_payload, txids.size());
    for (const auto& txid : txids) {
        CInv inv(InvType::INV_TX, txid);
        inv.serialize(inv_payload);
    }
    conn.send_message(msg::INV, inv_payload.span());
}

// ---------------------------------------------------------------------------
// process_reject
// ---------------------------------------------------------------------------
// Handles an incoming "reject" message (BIP 61, deprecated but still
// sent by some peers).
//
// Parses the rejected command name, numeric code, and human-readable
// reason string, then logs it.
// ---------------------------------------------------------------------------
void MsgHandler::process_reject(CConnection& conn,
                                const std::string& /*cmd*/,
                                core::DataStream& payload) {
    if (payload.remaining() < 3) return;

    // 1. Read the rejected message type (compact-size prefixed string).
    std::string rejected_msg;
    uint64_t msg_len = core::unserialize_compact_size(payload);
    if (msg_len > 0 && msg_len <= 12) {
        rejected_msg.resize(static_cast<size_t>(msg_len));
        payload.read(rejected_msg.data(), rejected_msg.size());
    }

    // 2. Read the single-byte reject code.
    uint8_t code = core::ser_read_u8(payload);

    // 3. Read the reason string.
    std::string reason;
    if (payload.remaining() > 0) {
        uint64_t reason_len = core::unserialize_compact_size(payload);
        if (reason_len > 0 && reason_len <= 256) {
            reason.resize(static_cast<size_t>(reason_len));
            payload.read(reason.data(), reason.size());
        }
    }

    // 4. Log the rejection.
    LogPrint(NET, "Peer %llu rejected '%s' (code=%d): %s",
             static_cast<unsigned long long>(conn.id()),
             rejected_msg.c_str(),
             static_cast<int>(code),
             reason.c_str());
}

// ---------------------------------------------------------------------------
// process_checkpoint
// ---------------------------------------------------------------------------
// Handles an incoming "checkpoint" message (PoT-specific).
//
// Receives a model checkpoint blob from a peer.  The checkpoint hash
// must be verified and the data stored to disk before the local
// training state is updated.
// ---------------------------------------------------------------------------
void MsgHandler::process_checkpoint(CConnection& conn,
                                    const std::string& /*cmd*/,
                                    core::DataStream& payload) {
    LogPrint(NET, "Received checkpoint (%zu bytes) from peer %llu",
             payload.remaining(),
             static_cast<unsigned long long>(conn.id()));

    // TODO: Process model checkpoint data
    // Verify checkpoint hash, store to disk, update training state
}

// ---------------------------------------------------------------------------
// process_getchkpt
// ---------------------------------------------------------------------------
// Handles an incoming "getcheckpoint" request (PoT-specific).
//
// If this node advertises NODE_CHECKPOINT services, reads the requested
// checkpoint hash and (once implemented) replies with the checkpoint data.
// ---------------------------------------------------------------------------
void MsgHandler::process_getchkpt(CConnection& conn,
                                  const std::string& /*cmd*/,
                                  core::DataStream& payload) {
    LogDebug(NET, "Received getchkpt from peer %llu",
             static_cast<unsigned long long>(conn.id()));

    // 1. Check if we serve checkpoints.
    if (!(connman_.local_services() & NODE_CHECKPOINT)) {
        return;
    }

    // 2. Read the requested checkpoint hash.
    if (payload.remaining() >= 32) {
        rnet::uint256 checkpoint_hash;
        checkpoint_hash.unserialize(payload);

        LogDebug(NET, "Peer %llu requesting checkpoint %s",
                 static_cast<unsigned long long>(conn.id()),
                 checkpoint_hash.to_hex().substr(0, 16).c_str());

        // TODO: Look up checkpoint data and send it
    }
}

// ---------------------------------------------------------------------------
// process_trainstatus
// ---------------------------------------------------------------------------
// Handles an incoming "trainingstatus" message (PoT-specific).
//
// Reads the peer's current training metrics (epoch, validation loss,
// training loss, batch number) for peer-state tracking and consensus.
// ---------------------------------------------------------------------------
void MsgHandler::process_trainstatus(CConnection& conn,
                                     const std::string& /*cmd*/,
                                     core::DataStream& payload) {
    // 1. Validate minimum payload size (4 + 4 + 4 + 4 = 16 bytes).
    if (payload.remaining() < 16) {
        conn.misbehaving(1, "Short trainstatus message");
        return;
    }

    // 2. Read training status fields.
    int32_t epoch      = core::ser_read_i32(payload);
    float   val_loss   = core::ser_read_float(payload);
    float   train_loss = core::ser_read_float(payload);
    int32_t batch      = core::ser_read_i32(payload);

    // 3. Log the peer's training state.
    LogDebug(NET, "Peer %llu training status: epoch=%d val_loss=%.6f "
             "train_loss=%.6f batch=%d",
             static_cast<unsigned long long>(conn.id()),
             epoch, val_loss, train_loss, batch);

    // TODO: Store in peer training state tracker
}

// ---------------------------------------------------------------------------
// process_growthinfo
// ---------------------------------------------------------------------------
// Handles an incoming "growthinfo" message (PoT-specific).
//
// Reads the peer's current network architecture size (layer count and
// total parameter count) for progressive-growth consensus tracking.
// ---------------------------------------------------------------------------
void MsgHandler::process_growthinfo(CConnection& conn,
                                    const std::string& /*cmd*/,
                                    core::DataStream& payload) {
    // 1. Validate minimum payload size (4 + 8 = 12 bytes).
    if (payload.remaining() < 12) {
        conn.misbehaving(1, "Short growthinfo message");
        return;
    }

    // 2. Read growth information fields.
    int32_t layer_count = core::ser_read_i32(payload);
    int64_t param_count = core::ser_read_i64(payload);

    // 3. Log the peer's growth state.
    LogDebug(NET, "Peer %llu growth info: layers=%d params=%lld",
             static_cast<unsigned long long>(conn.id()),
             layer_count,
             static_cast<long long>(param_count));

    // TODO: Process network growth information
}

// ---------------------------------------------------------------------------
// send_reject
// ---------------------------------------------------------------------------
// Sends a "reject" message to a peer, indicating that a previously
// received message was invalid or unacceptable.
// ---------------------------------------------------------------------------
void MsgHandler::send_reject(CConnection& conn,
                             const std::string& rejected_cmd,
                             uint8_t code,
                             const std::string& reason) {
    core::DataStream payload;

    // 1. Write the rejected command as a compact-size prefixed string.
    core::serialize_compact_size(payload, rejected_cmd.size());
    payload.write(rejected_cmd.data(), rejected_cmd.size());

    // 2. Write the reject code.
    core::ser_write_u8(payload, code);

    // 3. Write the reason string.
    core::serialize_compact_size(payload, reason.size());
    payload.write(reason.data(), reason.size());

    conn.send_message(msg::REJECT, payload.span());
}

// ---------------------------------------------------------------------------
// send_notfound
// ---------------------------------------------------------------------------
// Sends a "notfound" message listing inventory items this node could
// not serve in response to a prior "getdata" request.
// ---------------------------------------------------------------------------
void MsgHandler::send_notfound(CConnection& conn,
                               const std::vector<CInv>& items) {
    core::DataStream payload;

    // 1. Serialize the item count and each CInv entry.
    core::serialize_compact_size(payload, items.size());
    for (const auto& inv : items) {
        inv.serialize(payload);
    }

    // 2. Send the notfound message.
    conn.send_message(msg::NOTFOUND, payload.span());
}

} // namespace rnet::net

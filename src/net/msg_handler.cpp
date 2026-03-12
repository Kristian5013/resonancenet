#include "net/msg_handler.h"

#include <algorithm>
#include <cstring>

#include "core/logging.h"
#include "core/random.h"
#include "core/serialize.h"
#include "core/time.h"
#include "net/conn_manager.h"

namespace rnet::net {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

MsgHandler::MsgHandler(ConnManager& connman)
    : connman_(connman)
{}

MsgHandler::~MsgHandler() = default;

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void MsgHandler::register_handlers() {
    connman_.register_handler(msg::PING,
        [this](CConnection& c, const std::string& cmd, core::DataStream& p) {
            process_ping(c, cmd, p);
        });
    connman_.register_handler(msg::PONG,
        [this](CConnection& c, const std::string& cmd, core::DataStream& p) {
            process_pong(c, cmd, p);
        });
    connman_.register_handler(msg::ADDR,
        [this](CConnection& c, const std::string& cmd, core::DataStream& p) {
            process_addr(c, cmd, p);
        });
    connman_.register_handler(msg::GETADDR,
        [this](CConnection& c, const std::string& cmd, core::DataStream& p) {
            process_getaddr(c, cmd, p);
        });
    connman_.register_handler(msg::INV,
        [this](CConnection& c, const std::string& cmd, core::DataStream& p) {
            process_inv(c, cmd, p);
        });
    connman_.register_handler(msg::GETDATA,
        [this](CConnection& c, const std::string& cmd, core::DataStream& p) {
            process_getdata(c, cmd, p);
        });
    connman_.register_handler(msg::GETBLOCKS,
        [this](CConnection& c, const std::string& cmd, core::DataStream& p) {
            process_getblocks(c, cmd, p);
        });
    connman_.register_handler(msg::GETHEADERS,
        [this](CConnection& c, const std::string& cmd, core::DataStream& p) {
            process_getheaders(c, cmd, p);
        });
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
// ping / pong
// ---------------------------------------------------------------------------

void MsgHandler::process_ping(CConnection& conn, const std::string& /*cmd*/,
                              core::DataStream& payload) {
    // Read the ping nonce and echo it back as pong
    if (payload.remaining() >= 8) {
        uint64_t nonce = core::ser_read_u64(payload);

        core::DataStream pong_payload;
        core::ser_write_u64(pong_payload, nonce);
        conn.send_message(msg::PONG, pong_payload.span());
    } else {
        // Old-style ping with no nonce; just send pong with empty payload
        conn.send_message(msg::PONG);
    }
}

void MsgHandler::process_pong(CConnection& conn, const std::string& /*cmd*/,
                              core::DataStream& payload) {
    if (payload.remaining() < 8) {
        conn.misbehaving(1, "Short pong payload");
        return;
    }

    uint64_t nonce = core::ser_read_u64(payload);
    int64_t latency = conn.record_pong(nonce);

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
// addr / getaddr
// ---------------------------------------------------------------------------

void MsgHandler::process_addr(CConnection& conn, const std::string& /*cmd*/,
                              core::DataStream& payload) {
    // Read count (varint)
    if (payload.remaining() < 1) return;

    uint64_t count = core::unserialize_compact_size(payload);

    // Limit to 1000 addresses per message
    if (count > 1000) {
        conn.misbehaving(20, "Oversized addr message: " +
                             std::to_string(count));
        return;
    }

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

    // TODO: Forward to AddrManager for storage
}

void MsgHandler::process_getaddr(CConnection& conn,
                                 const std::string& /*cmd*/,
                                 core::DataStream& /*payload*/) {
    LogDebug(NET, "Received getaddr from peer %llu",
             static_cast<unsigned long long>(conn.id()));

    // TODO: Respond with known addresses from AddrManager
    // For now, send empty addr message
    core::DataStream addr_payload;
    core::serialize_compact_size(addr_payload, 0);
    conn.send_message(msg::ADDR, addr_payload.span());
}

// ---------------------------------------------------------------------------
// inv / getdata / notfound
// ---------------------------------------------------------------------------

void MsgHandler::process_inv(CConnection& conn, const std::string& /*cmd*/,
                             core::DataStream& payload) {
    uint64_t count = core::unserialize_compact_size(payload);

    if (count > 50000) {
        conn.misbehaving(20, "Oversized inv message: " +
                             std::to_string(count));
        return;
    }

    std::vector<CInv> inv_list;
    inv_list.reserve(static_cast<size_t>(count));
    std::vector<CInv> to_request;

    for (uint64_t i = 0; i < count; ++i) {
        CInv inv;
        inv.unserialize(payload);
        inv_list.push_back(inv);

        // Check if we already have this item
        bool have = false;
        if (have_item_) {
            have = have_item_(inv);
        }

        if (!have) {
            to_request.push_back(inv);
        }
    }

    // Notify callback
    if (on_inv_ && !inv_list.empty()) {
        on_inv_(conn.id(), inv_list);
    }

    // Request items we don't have
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

void MsgHandler::process_getdata(CConnection& conn,
                                 const std::string& /*cmd*/,
                                 core::DataStream& payload) {
    uint64_t count = core::unserialize_compact_size(payload);

    if (count > 50000) {
        conn.misbehaving(20, "Oversized getdata message: " +
                             std::to_string(count));
        return;
    }

    std::vector<CInv> not_found;

    for (uint64_t i = 0; i < count; ++i) {
        CInv inv;
        inv.unserialize(payload);

        bool sent = false;

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

        if (!sent) {
            not_found.push_back(inv);
        }
    }

    // Send notfound for items we couldn't serve
    if (!not_found.empty()) {
        send_notfound(conn, not_found);
    }
}

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
// getblocks / getheaders
// ---------------------------------------------------------------------------

void MsgHandler::process_getblocks(CConnection& conn,
                                   const std::string& /*cmd*/,
                                   core::DataStream& payload) {
    // Read protocol version
    int32_t version = core::ser_read_i32(payload);
    (void)version;

    // Read locator hashes
    uint64_t locator_count = core::unserialize_compact_size(payload);
    if (locator_count > 101) {
        conn.misbehaving(20, "Oversized getblocks locator");
        return;
    }

    std::vector<rnet::uint256> locator;
    locator.reserve(static_cast<size_t>(locator_count));
    for (uint64_t i = 0; i < locator_count; ++i) {
        rnet::uint256 hash;
        hash.unserialize(payload);
        locator.push_back(hash);
    }

    // Read stop hash
    rnet::uint256 stop_hash;
    stop_hash.unserialize(payload);

    LogDebug(NET, "Received getblocks from peer %llu (locator size=%llu)",
             static_cast<unsigned long long>(conn.id()),
             static_cast<unsigned long long>(locator_count));

    // TODO: Build inv response from chain
}

void MsgHandler::process_getheaders(CConnection& conn,
                                    const std::string& /*cmd*/,
                                    core::DataStream& payload) {
    int32_t version = core::ser_read_i32(payload);
    (void)version;

    uint64_t locator_count = core::unserialize_compact_size(payload);
    if (locator_count > 101) {
        conn.misbehaving(20, "Oversized getheaders locator");
        return;
    }

    std::vector<rnet::uint256> locator;
    locator.reserve(static_cast<size_t>(locator_count));
    for (uint64_t i = 0; i < locator_count; ++i) {
        rnet::uint256 hash;
        hash.unserialize(payload);
        locator.push_back(hash);
    }

    rnet::uint256 stop_hash;
    stop_hash.unserialize(payload);

    LogDebug(NET, "Received getheaders from peer %llu (locator size=%llu)",
             static_cast<unsigned long long>(conn.id()),
             static_cast<unsigned long long>(locator_count));

    // TODO: Build headers response from chain
}

// ---------------------------------------------------------------------------
// tx / block / headers
// ---------------------------------------------------------------------------

void MsgHandler::process_tx(CConnection& conn, const std::string& /*cmd*/,
                            core::DataStream& payload) {
    LogDebug(NET, "Received tx (%zu bytes) from peer %llu",
             payload.remaining(),
             static_cast<unsigned long long>(conn.id()));

    if (on_new_tx_) {
        std::vector<uint8_t> tx_data(payload.unread_span().begin(),
                                     payload.unread_span().end());
        bool accepted = on_new_tx_(conn.id(), tx_data);

        if (!accepted) {
            LogDebug(NET, "Tx from peer %llu rejected",
                     static_cast<unsigned long long>(conn.id()));
        }
    }
}

void MsgHandler::process_block(CConnection& conn,
                               const std::string& /*cmd*/,
                               core::DataStream& payload) {
    LogDebug(NET, "Received block (%zu bytes) from peer %llu",
             payload.remaining(),
             static_cast<unsigned long long>(conn.id()));

    if (on_new_block_) {
        std::vector<uint8_t> block_data(payload.unread_span().begin(),
                                        payload.unread_span().end());
        bool accepted = on_new_block_(conn.id(), block_data);

        if (!accepted) {
            LogDebug(NET, "Block from peer %llu rejected",
                     static_cast<unsigned long long>(conn.id()));
        }
    }
}

void MsgHandler::process_headers(CConnection& conn,
                                 const std::string& /*cmd*/,
                                 core::DataStream& payload) {
    LogDebug(NET, "Received headers (%zu bytes) from peer %llu",
             payload.remaining(),
             static_cast<unsigned long long>(conn.id()));

    if (on_headers_) {
        std::vector<uint8_t> headers_data(payload.unread_span().begin(),
                                          payload.unread_span().end());
        on_headers_(conn.id(), headers_data);
    }
}

// ---------------------------------------------------------------------------
// sendheaders / mempool
// ---------------------------------------------------------------------------

void MsgHandler::process_sendheaders(CConnection& conn,
                                     const std::string& /*cmd*/,
                                     core::DataStream& /*payload*/) {
    conn.set_prefer_headers(true);
    LogDebug(NET, "Peer %llu prefers header announcements",
             static_cast<unsigned long long>(conn.id()));
}

void MsgHandler::process_mempool(CConnection& conn,
                                 const std::string& /*cmd*/,
                                 core::DataStream& /*payload*/) {
    LogDebug(NET, "Received mempool request from peer %llu",
             static_cast<unsigned long long>(conn.id()));

    // TODO: Send inv messages for all mempool transactions
}

// ---------------------------------------------------------------------------
// reject
// ---------------------------------------------------------------------------

void MsgHandler::process_reject(CConnection& conn,
                                const std::string& /*cmd*/,
                                core::DataStream& payload) {
    if (payload.remaining() < 3) return;

    // Read rejected message type
    std::string rejected_msg;
    uint64_t msg_len = core::unserialize_compact_size(payload);
    if (msg_len > 0 && msg_len <= 12) {
        rejected_msg.resize(static_cast<size_t>(msg_len));
        payload.read(rejected_msg.data(), rejected_msg.size());
    }

    uint8_t code = core::ser_read_u8(payload);

    // Read reason
    std::string reason;
    if (payload.remaining() > 0) {
        uint64_t reason_len = core::unserialize_compact_size(payload);
        if (reason_len > 0 && reason_len <= 256) {
            reason.resize(static_cast<size_t>(reason_len));
            payload.read(reason.data(), reason.size());
        }
    }

    LogPrint(NET, "Peer %llu rejected '%s' (code=%d): %s",
             static_cast<unsigned long long>(conn.id()),
             rejected_msg.c_str(),
             static_cast<int>(code),
             reason.c_str());
}

// ---------------------------------------------------------------------------
// ResonanceNet-specific: checkpoint, trainstatus, growthinfo
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

void MsgHandler::process_getchkpt(CConnection& conn,
                                  const std::string& /*cmd*/,
                                  core::DataStream& payload) {
    LogDebug(NET, "Received getchkpt from peer %llu",
             static_cast<unsigned long long>(conn.id()));

    // Check if we serve checkpoints
    if (!(connman_.local_services() & NODE_CHECKPOINT)) {
        // We don't serve checkpoints — ignore
        return;
    }

    // Read requested checkpoint hash
    if (payload.remaining() >= 32) {
        rnet::uint256 checkpoint_hash;
        checkpoint_hash.unserialize(payload);

        LogDebug(NET, "Peer %llu requesting checkpoint %s",
                 static_cast<unsigned long long>(conn.id()),
                 checkpoint_hash.to_hex().substr(0, 16).c_str());

        // TODO: Look up checkpoint data and send it
    }
}

void MsgHandler::process_trainstatus(CConnection& conn,
                                     const std::string& /*cmd*/,
                                     core::DataStream& payload) {
    if (payload.remaining() < 16) {
        conn.misbehaving(1, "Short trainstatus message");
        return;
    }

    // Read training status fields
    int32_t epoch = core::ser_read_i32(payload);
    float val_loss = core::ser_read_float(payload);
    float train_loss = core::ser_read_float(payload);
    int32_t batch = core::ser_read_i32(payload);

    LogDebug(NET, "Peer %llu training status: epoch=%d val_loss=%.6f "
             "train_loss=%.6f batch=%d",
             static_cast<unsigned long long>(conn.id()),
             epoch, val_loss, train_loss, batch);

    // Update peer's known training state
    // TODO: Store in peer training state tracker
}

void MsgHandler::process_growthinfo(CConnection& conn,
                                    const std::string& /*cmd*/,
                                    core::DataStream& payload) {
    if (payload.remaining() < 12) {
        conn.misbehaving(1, "Short growthinfo message");
        return;
    }

    int32_t layer_count = core::ser_read_i32(payload);
    int64_t param_count = core::ser_read_i64(payload);

    LogDebug(NET, "Peer %llu growth info: layers=%d params=%lld",
             static_cast<unsigned long long>(conn.id()),
             layer_count,
             static_cast<long long>(param_count));

    // TODO: Process network growth information
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

void MsgHandler::send_reject(CConnection& conn,
                             const std::string& rejected_cmd,
                             uint8_t code,
                             const std::string& reason) {
    core::DataStream payload;

    // Write rejected command as compact-size prefixed string
    core::serialize_compact_size(payload, rejected_cmd.size());
    payload.write(rejected_cmd.data(), rejected_cmd.size());

    // Write reject code
    core::ser_write_u8(payload, code);

    // Write reason
    core::serialize_compact_size(payload, reason.size());
    payload.write(reason.data(), reason.size());

    conn.send_message(msg::REJECT, payload.span());
}

void MsgHandler::send_notfound(CConnection& conn,
                               const std::vector<CInv>& items) {
    core::DataStream payload;
    core::serialize_compact_size(payload, items.size());
    for (const auto& inv : items) {
        inv.serialize(payload);
    }
    conn.send_message(msg::NOTFOUND, payload.span());
}

}  // namespace rnet::net

// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "net/message.h"

#include "core/serialize.h"

namespace rnet::net::message {

// ===========================================================================
//  Inventory messages
// ===========================================================================

// ---------------------------------------------------------------------------
// make_inv / parse_inv
//
// Design: compact-size prefixed vector of CInv items.  parse_inv caps
// at 50 000 entries as a sanity limit.
// ---------------------------------------------------------------------------

std::vector<uint8_t> make_inv(const std::vector<CInv>& inv) {
    core::DataStream ss;
    // 1. Write count
    core::serialize_compact_size(ss, inv.size());
    // 2. Write each inventory item
    for (const auto& item : inv) {
        item.serialize(ss);
    }
    return ss.vch();
}

std::vector<CInv> parse_inv(core::DataStream& stream) {
    // 1. Read count with sanity cap
    auto count = core::unserialize_compact_size(stream);
    if (count > 50000) count = 50000;
    // 2. Deserialize each item
    std::vector<CInv> result;
    result.reserve(static_cast<size_t>(count));
    for (uint64_t i = 0; i < count; ++i) {
        CInv item;
        item.unserialize(stream);
        result.push_back(item);
    }
    return result;
}

// ---------------------------------------------------------------------------
// make_getdata
//
// Design: same wire format as inv.
// ---------------------------------------------------------------------------

std::vector<uint8_t> make_getdata(const std::vector<CInv>& inv) {
    return make_inv(inv);
}

// ===========================================================================
//  Header messages
// ===========================================================================

// ---------------------------------------------------------------------------
// make_headers / parse_headers
//
// Design: each header is followed by a zero tx-count (per Bitcoin
// protocol).  parse_headers caps at 2000 entries per message.
// ---------------------------------------------------------------------------

std::vector<uint8_t> make_headers(
    const std::vector<primitives::CBlockHeader>& headers)
{
    core::DataStream ss;
    core::serialize_compact_size(ss, headers.size());
    for (const auto& hdr : headers) {
        // 1. Serialize header
        hdr.serialize(ss);
        // 2. Append zero tx count (per Bitcoin protocol)
        core::serialize_compact_size(ss, 0);
    }
    return ss.vch();
}

std::vector<primitives::CBlockHeader> parse_headers(
    core::DataStream& stream)
{
    // 1. Read count with sanity cap
    auto count = core::unserialize_compact_size(stream);
    if (count > 2000) count = 2000;
    // 2. Deserialize each header
    std::vector<primitives::CBlockHeader> result;
    result.reserve(static_cast<size_t>(count));
    for (uint64_t i = 0; i < count; ++i) {
        primitives::CBlockHeader hdr;
        hdr.unserialize(stream);
        // 3. Read and discard tx count
        core::unserialize_compact_size(stream);
        result.push_back(hdr);
    }
    return result;
}

// ===========================================================================
//  Block messages
// ===========================================================================

// ---------------------------------------------------------------------------
// make_block / parse_block
// ---------------------------------------------------------------------------

std::vector<uint8_t> make_block(const primitives::CBlock& block) {
    core::DataStream ss;
    block.serialize(ss);
    return ss.vch();
}

primitives::CBlock parse_block(core::DataStream& stream) {
    primitives::CBlock block;
    block.unserialize(stream);
    return block;
}

// ===========================================================================
//  Transaction messages
// ===========================================================================

// ---------------------------------------------------------------------------
// make_tx / parse_tx
//
// Design: parse_tx returns a shared_ptr via const_cast to populate the
// immutable CTransaction fields during deserialization.
// ---------------------------------------------------------------------------

std::vector<uint8_t> make_tx(const primitives::CTransaction& tx) {
    core::DataStream ss;
    tx.serialize(ss);
    return ss.vch();
}

primitives::CTransactionRef parse_tx(core::DataStream& stream) {
    auto tx = std::make_shared<primitives::CTransaction>();
    auto& mtx = const_cast<primitives::CTransaction&>(*tx);
    mtx.unserialize(stream);
    return tx;
}

// ===========================================================================
//  Address messages
// ===========================================================================

// ---------------------------------------------------------------------------
// make_addr / parse_addr
//
// Design: compact-size prefixed vector of CNetAddr.  parse_addr caps
// at 1000 entries as a sanity limit.
// ---------------------------------------------------------------------------

std::vector<uint8_t> make_addr(const std::vector<CNetAddr>& addrs) {
    core::DataStream ss;
    core::serialize_compact_size(ss, addrs.size());
    for (const auto& addr : addrs) {
        addr.serialize(ss);
    }
    return ss.vch();
}

std::vector<CNetAddr> parse_addr(core::DataStream& stream) {
    // 1. Read count with sanity cap
    auto count = core::unserialize_compact_size(stream);
    if (count > 1000) count = 1000;
    // 2. Deserialize each address
    std::vector<CNetAddr> result;
    result.reserve(static_cast<size_t>(count));
    for (uint64_t i = 0; i < count; ++i) {
        CNetAddr addr;
        addr.unserialize(stream);
        result.push_back(addr);
    }
    return result;
}

// ===========================================================================
//  Ping messages
// ===========================================================================

// ---------------------------------------------------------------------------
// make_ping / parse_ping
// ---------------------------------------------------------------------------

std::vector<uint8_t> make_ping(uint64_t nonce) {
    core::DataStream ss;
    core::ser_write_u64(ss, nonce);
    return ss.vch();
}

uint64_t parse_ping(core::DataStream& stream) {
    return core::ser_read_u64(stream);
}

// ===========================================================================
//  Locator messages
// ===========================================================================

// ---------------------------------------------------------------------------
// make_getblocks / make_getheaders
//
// Design: protocol version + locator vector + stop hash.
// getheaders uses the same wire format as getblocks.
// ---------------------------------------------------------------------------

std::vector<uint8_t> make_getblocks(
    const std::vector<rnet::uint256>& locator,
    const rnet::uint256& stop_hash)
{
    core::DataStream ss;
    core::ser_write_u32(ss, PROTOCOL_VERSION);
    core::Serialize(ss, locator);
    core::Serialize(ss, stop_hash);
    return ss.vch();
}

std::vector<uint8_t> make_getheaders(
    const std::vector<rnet::uint256>& locator,
    const rnet::uint256& stop_hash)
{
    return make_getblocks(locator, stop_hash);
}

} // namespace rnet::net::message

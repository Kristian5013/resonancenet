#pragma once

#include <cstdint>
#include <vector>

#include "core/stream.h"
#include "core/types.h"
#include "net/protocol.h"
#include "primitives/block.h"
#include "primitives/block_header.h"
#include "primitives/transaction.h"

namespace rnet::net {

/// Message serialization/deserialization helpers.
/// These functions build wire-format payloads for specific message types.
namespace message {

/// Build a serialized inv message payload
std::vector<uint8_t> make_inv(const std::vector<CInv>& inv);

/// Parse an inv message payload
std::vector<CInv> parse_inv(core::DataStream& stream);

/// Build a serialized getdata message payload
std::vector<uint8_t> make_getdata(const std::vector<CInv>& inv);

/// Build a serialized headers message payload
std::vector<uint8_t> make_headers(
    const std::vector<primitives::CBlockHeader>& headers);

/// Parse a headers message payload
std::vector<primitives::CBlockHeader> parse_headers(
    core::DataStream& stream);

/// Build a serialized block message payload
std::vector<uint8_t> make_block(const primitives::CBlock& block);

/// Parse a block message payload
primitives::CBlock parse_block(core::DataStream& stream);

/// Build a serialized tx message payload
std::vector<uint8_t> make_tx(const primitives::CTransaction& tx);

/// Parse a tx message payload
primitives::CTransactionRef parse_tx(core::DataStream& stream);

/// Build a serialized addr message payload
std::vector<uint8_t> make_addr(const std::vector<CNetAddr>& addrs);

/// Parse an addr message payload
std::vector<CNetAddr> parse_addr(core::DataStream& stream);

/// Build a serialized ping/pong message payload
std::vector<uint8_t> make_ping(uint64_t nonce);

/// Parse a ping/pong nonce
uint64_t parse_ping(core::DataStream& stream);

/// Build a serialized getblocks message payload
std::vector<uint8_t> make_getblocks(
    const std::vector<rnet::uint256>& locator,
    const rnet::uint256& stop_hash);

/// Build a serialized getheaders message payload
std::vector<uint8_t> make_getheaders(
    const std::vector<rnet::uint256>& locator,
    const rnet::uint256& stop_hash);

}  // namespace message

}  // namespace rnet::net

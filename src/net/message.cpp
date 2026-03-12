#include "net/message.h"

#include "core/serialize.h"

namespace rnet::net::message {

std::vector<uint8_t> make_inv(const std::vector<CInv>& inv) {
    core::DataStream ss;
    core::serialize_compact_size(ss, inv.size());
    for (const auto& item : inv) {
        item.serialize(ss);
    }
    return ss.vch();
}

std::vector<CInv> parse_inv(core::DataStream& stream) {
    auto count = core::unserialize_compact_size(stream);
    if (count > 50000) count = 50000;  // Sanity limit
    std::vector<CInv> result;
    result.reserve(static_cast<size_t>(count));
    for (uint64_t i = 0; i < count; ++i) {
        CInv item;
        item.unserialize(stream);
        result.push_back(item);
    }
    return result;
}

std::vector<uint8_t> make_getdata(const std::vector<CInv>& inv) {
    return make_inv(inv);  // Same format as inv
}

std::vector<uint8_t> make_headers(
    const std::vector<primitives::CBlockHeader>& headers)
{
    core::DataStream ss;
    core::serialize_compact_size(ss, headers.size());
    for (const auto& hdr : headers) {
        hdr.serialize(ss);
        // Append zero tx count (per Bitcoin protocol)
        core::serialize_compact_size(ss, 0);
    }
    return ss.vch();
}

std::vector<primitives::CBlockHeader> parse_headers(
    core::DataStream& stream)
{
    auto count = core::unserialize_compact_size(stream);
    if (count > 2000) count = 2000;  // Max 2000 headers per message
    std::vector<primitives::CBlockHeader> result;
    result.reserve(static_cast<size_t>(count));
    for (uint64_t i = 0; i < count; ++i) {
        primitives::CBlockHeader hdr;
        hdr.unserialize(stream);
        // Read and discard tx count
        core::unserialize_compact_size(stream);
        result.push_back(hdr);
    }
    return result;
}

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

std::vector<uint8_t> make_addr(const std::vector<CNetAddr>& addrs) {
    core::DataStream ss;
    core::serialize_compact_size(ss, addrs.size());
    for (const auto& addr : addrs) {
        addr.serialize(ss);
    }
    return ss.vch();
}

std::vector<CNetAddr> parse_addr(core::DataStream& stream) {
    auto count = core::unserialize_compact_size(stream);
    if (count > 1000) count = 1000;  // Sanity limit
    std::vector<CNetAddr> result;
    result.reserve(static_cast<size_t>(count));
    for (uint64_t i = 0; i < count; ++i) {
        CNetAddr addr;
        addr.unserialize(stream);
        result.push_back(addr);
    }
    return result;
}

std::vector<uint8_t> make_ping(uint64_t nonce) {
    core::DataStream ss;
    core::ser_write_u64(ss, nonce);
    return ss.vch();
}

uint64_t parse_ping(core::DataStream& stream) {
    return core::ser_read_u64(stream);
}

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
    return make_getblocks(locator, stop_hash);  // Same format
}

}  // namespace rnet::net::message

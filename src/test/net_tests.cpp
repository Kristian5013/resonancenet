#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <set>
#include <filesystem>
#include "util/fs.h"
#include "canon/canon.h"
#include "consensus/genesis.h"
#include "consensus/params.h"
#include "net/message.h"
#include "net/protocol.h"
#include "test/framework.h"
#include "util/hex.h"

using namespace rnet;
using namespace rnet::net;

namespace {

const std::array<uint8_t, 4> kMagic = consensus::Params(consensus::Network::Regtest).message_magic;

VersionMessage MakeVersion() {
    VersionMessage v;
    v.services = kServiceRelay | kServiceVerifier;
    v.timestamp = 1'700'000'000;
    v.nonce = 0x1234'5678'9ABC'DEF0ull;
    v.user_agent = "rnet/0.1.0";
    v.worker_id = 42;
    v.determinism_class = 1;
    v.genesis_hash = crypto::Sha3_256(std::string_view("genesis"));
    v.policy_hash = crypto::Sha3_256(std::string_view("policy"));
    return v;
}

}  // namespace

// ---------------------------------------------------------------------------
// Framing
// ---------------------------------------------------------------------------
RNET_TEST(Message, RoundTrips) {
    const std::vector<uint8_t> payload{1, 2, 3, 4, 5};
    auto framed = SerializeMessage(kMagic, cmd::kPing, payload);
    RNET_CHECK_OK(framed);
    RNET_CHECK_EQ(framed.value().size(), kHeaderSize + payload.size());

    auto header = ParseHeader(framed.value(), kMagic);
    RNET_CHECK_OK(header);
    RNET_CHECK_STREQ(header.value().command, cmd::kPing);
    RNET_CHECK_EQ(header.value().payload_size, 5u);
    RNET_CHECK_OK(VerifyPayload(header.value(),
                                std::span<const uint8_t>(framed.value()).subspan(kHeaderSize)));
}

// A peer on another network is rejected on the first four bytes, before anything
// that follows is interpreted.
RNET_TEST(Message, ForeignMagicIsRejected) {
    auto framed = SerializeMessage(kMagic, cmd::kPing, {});
    RNET_CHECK_OK(framed);
    const auto other = consensus::Params(consensus::Network::Main).message_magic;
    RNET_CHECK_ERR(ParseHeader(framed.value(), other));
}

RNET_TEST(Message, CorruptedPayloadIsDetected) {
    const std::vector<uint8_t> payload{9, 8, 7, 6};
    auto framed = SerializeMessage(kMagic, cmd::kInv, payload);
    RNET_CHECK_OK(framed);

    auto corrupted = framed.value();
    corrupted.back() ^= 0x01;
    auto header = ParseHeader(corrupted, kMagic);
    RNET_CHECK_OK(header);
    auto st = VerifyPayload(header.value(),
                            std::span<const uint8_t>(corrupted).subspan(kHeaderSize));
    RNET_CHECK(!st);
}

// Peer-supplied text reaches logs; control bytes and non-NUL padding must not.
RNET_TEST(Message, HostileCommandFieldsAreRejected) {
    auto framed = SerializeMessage(kMagic, cmd::kPing, {});
    RNET_CHECK_OK(framed);

    auto control = framed.value();
    control[4] = 0x07;                       // bell character in the command
    RNET_CHECK_ERR(ParseHeader(control, kMagic));

    auto smuggled = framed.value();
    smuggled[4 + kCommandSize - 1] = 'x';    // junk after the NUL terminator
    RNET_CHECK_ERR(ParseHeader(smuggled, kMagic));

    auto empty = framed.value();
    empty[4] = 0;                            // no command at all
    RNET_CHECK_ERR(ParseHeader(empty, kMagic));
}

// The declared length is refused before a byte is allocated for it.
RNET_TEST(Message, OversizedDeclaredLengthIsRefused) {
    auto framed = SerializeMessage(kMagic, cmd::kInv, {});
    RNET_CHECK_OK(framed);
    auto oversized = framed.value();
    const size_t offset = 4 + kCommandSize;
    oversized[offset] = 0xFF;
    oversized[offset + 1] = 0xFF;
    oversized[offset + 2] = 0xFF;
    oversized[offset + 3] = 0xFF;
    RNET_CHECK_ERR(ParseHeader(oversized, kMagic));
}

RNET_TEST(Message, OversizedPayloadCannotBeSent) {
    const std::vector<uint8_t> huge(kMaxMessageSize + 1, 0);
    RNET_CHECK_ERR(SerializeMessage(kMagic, cmd::kObject, huge));
}

// The invariant that the Lattica audit exists to prevent: there must be no
// object size the rules allow and the transport cannot carry. Bulk data is
// chunked, and a chunk always fits in a message.
RNET_TEST(Message, EveryBulkObjectIsTransmittable) {
    static_assert(kBulkChunkSize + kChunkMetadataAllowance < kMaxMessageSize);

    for (uint64_t size : {uint64_t{1}, uint64_t{kBulkChunkSize - 1}, uint64_t{kBulkChunkSize},
                          uint64_t{kBulkChunkSize + 1}, uint64_t{2ull << 30}}) {
        const uint64_t chunks = ChunkCount(size);
        RNET_CHECK(chunks > 0);
        // Every chunk, including the last, is within one message.
        const uint64_t last = size - (chunks - 1) * kBulkChunkSize;
        RNET_CHECK(last > 0 && last <= kBulkChunkSize);
    }
    RNET_CHECK_EQ(ChunkCount(0), 0ull);
}

// ---------------------------------------------------------------------------
// Handshake
// ---------------------------------------------------------------------------
RNET_TEST(Version, RoundTrips) {
    const auto v = MakeVersion();
    auto decoded = VersionMessage::Deserialize(v.Serialize());
    RNET_CHECK_OK(decoded);
    RNET_CHECK_EQ(decoded.value().worker_id, 42ull);
    RNET_CHECK_EQ(decoded.value().determinism_class, uint16_t{1});
    RNET_CHECK_EQ(decoded.value().genesis_hash, v.genesis_hash);
    RNET_CHECK_EQ(decoded.value().policy_hash, v.policy_hash);
    RNET_CHECK_STREQ(decoded.value().user_agent, "rnet/0.1.0");
}

// A peer training a different round, or running under different rules, is not a
// peer — both anchors are part of the handshake.
RNET_TEST(Version, RequiresBothAnchors) {
    auto v = MakeVersion();
    v.genesis_hash = crypto::Hash256{};
    RNET_CHECK_ERR(VersionMessage::Deserialize(v.Serialize()));

    v = MakeVersion();
    v.policy_hash = crypto::Hash256{};
    RNET_CHECK_ERR(VersionMessage::Deserialize(v.Serialize()));
}

RNET_TEST(Version, ObsoleteProtocolIsRejected) {
    auto v = MakeVersion();
    v.protocol_version = 0;
    RNET_CHECK_ERR(VersionMessage::Deserialize(v.Serialize()));
}

RNET_TEST(Version, UserAgentIsBoundedAndSanitised) {
    auto v = MakeVersion();
    v.user_agent = std::string(200, 'x');           // beyond the limit
    RNET_CHECK_ERR(VersionMessage::Deserialize(v.Serialize()));

    v = MakeVersion();
    v.user_agent = "rnet\n\rinjected";               // control characters
    RNET_CHECK_ERR(VersionMessage::Deserialize(v.Serialize()));
}

// ---------------------------------------------------------------------------
// Addresses
// ---------------------------------------------------------------------------
RNET_TEST(Address, ParsesAndFormatsIPv4) {
    auto addr = NetAddress::FromString("203.0.113.7:9444", 0);
    RNET_CHECK_OK(addr);
    RNET_CHECK(addr.value().IsIPv4());
    RNET_CHECK_EQ(addr.value().port, uint16_t{9444});
    RNET_CHECK_STREQ(addr.value().ToString(), "203.0.113.7:9444");

    auto defaulted = NetAddress::FromString("203.0.113.7", 9444);
    RNET_CHECK_OK(defaulted);
    RNET_CHECK_EQ(defaulted.value().port, uint16_t{9444});
}

RNET_TEST(Address, RejectsMalformedLiterals) {
    RNET_CHECK_ERR(NetAddress::FromString("203.0.113", 9444));
    RNET_CHECK_ERR(NetAddress::FromString("203.0.113.999", 9444));
    RNET_CHECK_ERR(NetAddress::FromString("203.0.113.7:notaport", 0));
}

// Gossiping unroutable addresses would poison every peer's database with entries
// that can never connect.
RNET_TEST(Address, UnroutableAddressesAreRecognised) {
    for (const char* text : {"127.0.0.1:9444", "10.0.0.5:9444", "192.168.1.10:9444",
                             "172.16.4.1:9444", "169.254.1.1:9444", "0.0.0.0:9444",
                             "224.0.0.1:9444"}) {
        auto addr = NetAddress::FromString(text, 9444);
        RNET_CHECK_OK(addr);
        if (addr.value().IsRoutable()) RNET_FAIL("{} should not be routable", text);
    }

    auto public_addr = NetAddress::FromString("203.0.113.7:9444", 9444);
    RNET_CHECK_OK(public_addr);
    RNET_CHECK(public_addr.value().IsRoutable());

    auto no_port = NetAddress::FromString("203.0.113.7:0", 0);
    RNET_CHECK_OK(no_port);
    RNET_CHECK(!no_port.value().IsRoutable());
}

RNET_TEST(Addr, RoundTripsAndIsBounded) {
    AddrMessage m;
    for (int i = 1; i <= 3; ++i) {
        auto addr = NetAddress::FromString(std::format("203.0.113.{}:9444", i), 9444);
        RNET_CHECK_OK(addr);
        m.addresses.push_back(addr.value());
    }
    auto decoded = AddrMessage::Deserialize(m.Serialize());
    RNET_CHECK_OK(decoded);
    RNET_CHECK_EQ(decoded.value().addresses.size(), size_t{3});

    // A peer must not be able to flood the address database in one message.
    canon::Writer w;
    w.U32(static_cast<uint32_t>(kMaxAddrPerMessage + 1));
    RNET_CHECK_ERR(AddrMessage::Deserialize(w.data()));
}

// ---------------------------------------------------------------------------
// Inventory
// ---------------------------------------------------------------------------
RNET_TEST(Inv, RoundTripsAndIsBounded) {
    InvMessage m;
    m.entries.push_back({InvType::Checkpoint, crypto::Sha3_256(std::string_view("ckpt"))});
    m.entries.push_back({InvType::Contribution, crypto::Sha3_256(std::string_view("contrib"))});

    auto decoded = InvMessage::Deserialize(m.Serialize());
    RNET_CHECK_OK(decoded);
    RNET_CHECK_EQ(decoded.value().entries.size(), size_t{2});
    RNET_CHECK(decoded.value().entries[0] == m.entries[0]);

    canon::Writer w;
    w.U32(static_cast<uint32_t>(kMaxInvPerMessage + 1));
    RNET_CHECK_ERR(InvMessage::Deserialize(w.data()));
}

RNET_TEST(Inv, UnknownObjectTypeIsRejected) {
    canon::Writer w;
    w.U32(1);
    w.U16(99);   // not a type this build knows
    w.Hash(crypto::Sha3_256(std::string_view("x")));
    RNET_CHECK_ERR(InvMessage::Deserialize(w.data()));
}

// ---------------------------------------------------------------------------
// Bulk transfer
// ---------------------------------------------------------------------------
RNET_TEST(GetChunks, BoundsTheRequest) {
    GetChunksMessage m;
    m.type = InvType::Checkpoint;
    m.hash = crypto::Sha3_256(std::string_view("ckpt"));
    m.first_chunk = 0;
    m.chunk_count = 16;
    RNET_CHECK_OK(GetChunksMessage::Deserialize(m.Serialize()));

    m.chunk_count = 0;
    RNET_CHECK_ERR(GetChunksMessage::Deserialize(m.Serialize()));

    // One request must not be able to demand an unbounded stream.
    m.chunk_count = 1'000'000;
    RNET_CHECK_ERR(GetChunksMessage::Deserialize(m.Serialize()));
}

RNET_TEST(Chunk, OnlyTheLastMayBePartial) {
    ChunkMessage c;
    c.type = InvType::Checkpoint;
    c.hash = crypto::Sha3_256(std::string_view("ckpt"));
    c.total_chunks = 3;

    c.index = 0;
    c.data.assign(kBulkChunkSize, 0xAB);
    RNET_CHECK_OK(c.Validate());

    // A short interior chunk would leave a hole the assembler could not see.
    c.data.assign(100, 0xAB);
    RNET_CHECK_ERR(c.Validate());

    c.index = 2;                                   // the last one may be short
    RNET_CHECK_OK(c.Validate());

    c.index = 3;                                   // past the end
    RNET_CHECK_ERR(c.Validate());
}

RNET_TEST(Assembler, ReassemblesAndVerifies) {
    std::vector<uint8_t> object(kBulkChunkSize * 2 + 1234);
    for (size_t i = 0; i < object.size(); ++i) object[i] = static_cast<uint8_t>(i * 31);
    const crypto::Hash256 hash = crypto::Sha3_256(object);

    ChunkAssembler assembler(InvType::Checkpoint, hash, object.size());
    RNET_CHECK_EQ(assembler.total_chunks(), 3ull);

    for (uint64_t i = 0; i < assembler.total_chunks(); ++i) {
        ChunkMessage c;
        c.type = InvType::Checkpoint;
        c.hash = hash;
        c.index = i;
        c.total_chunks = assembler.total_chunks();
        const size_t offset = static_cast<size_t>(i) * kBulkChunkSize;
        const size_t len = std::min<size_t>(kBulkChunkSize, object.size() - offset);
        c.data.assign(object.begin() + static_cast<ptrdiff_t>(offset),
                      object.begin() + static_cast<ptrdiff_t>(offset + len));
        RNET_CHECK_OK(assembler.Accept(c));
    }

    RNET_CHECK(assembler.Complete());
    auto finished = assembler.Finish();
    RNET_CHECK_OK(finished);
    RNET_CHECK_EQ(finished.value(), object);
}

// An incomplete transfer must never be handed to a caller as if it were the
// object — that is how zero-padded data reaches consensus code.
RNET_TEST(Assembler, IncompleteTransferIsRefused) {
    std::vector<uint8_t> object(kBulkChunkSize * 2, 0x5A);
    const crypto::Hash256 hash = crypto::Sha3_256(object);

    ChunkAssembler assembler(InvType::Checkpoint, hash, object.size());
    ChunkMessage c;
    c.type = InvType::Checkpoint;
    c.hash = hash;
    c.index = 0;
    c.total_chunks = 2;
    c.data.assign(kBulkChunkSize, 0x5A);
    RNET_CHECK_OK(assembler.Accept(c));

    RNET_CHECK(!assembler.Complete());
    RNET_CHECK_ERR(assembler.Finish());
}

RNET_TEST(Assembler, TamperedChunkFailsTheFinalHash) {
    std::vector<uint8_t> object(kBulkChunkSize + 10, 0x11);
    const crypto::Hash256 hash = crypto::Sha3_256(object);

    ChunkAssembler assembler(InvType::Checkpoint, hash, object.size());
    for (uint64_t i = 0; i < 2; ++i) {
        ChunkMessage c;
        c.type = InvType::Checkpoint;
        c.hash = hash;
        c.index = i;
        c.total_chunks = 2;
        const size_t offset = static_cast<size_t>(i) * kBulkChunkSize;
        const size_t len = std::min<size_t>(kBulkChunkSize, object.size() - offset);
        c.data.assign(len, i == 1 ? 0x22 : 0x11);   // second chunk altered
        RNET_CHECK_OK(assembler.Accept(c));
    }
    RNET_CHECK(assembler.Complete());
    // Chunks carry no proof individually; only the whole object is checked.
    RNET_CHECK_ERR(assembler.Finish());
}

RNET_TEST(Assembler, RejectsChunksForOtherObjects) {
    std::vector<uint8_t> object(1000, 0x33);
    const crypto::Hash256 hash = crypto::Sha3_256(object);
    ChunkAssembler assembler(InvType::Checkpoint, hash, object.size());

    ChunkMessage wrong_object;
    wrong_object.type = InvType::Checkpoint;
    wrong_object.hash = crypto::Sha3_256(std::string_view("something else"));
    wrong_object.index = 0;
    wrong_object.total_chunks = 1;
    wrong_object.data.assign(1000, 0x33);
    RNET_CHECK_ERR(assembler.Accept(wrong_object));

    ChunkMessage wrong_type = wrong_object;
    wrong_type.hash = hash;
    wrong_type.type = InvType::Contribution;
    RNET_CHECK_ERR(assembler.Accept(wrong_type));

    ChunkMessage wrong_count = wrong_object;
    wrong_count.hash = hash;
    wrong_count.total_chunks = 7;
    RNET_CHECK_ERR(assembler.Accept(wrong_count));
}

RNET_TEST(Assembler, DuplicateChunkIsHarmless) {
    std::vector<uint8_t> object(500, 0x44);
    const crypto::Hash256 hash = crypto::Sha3_256(object);
    ChunkAssembler assembler(InvType::Contribution, hash, object.size());

    ChunkMessage c;
    c.type = InvType::Contribution;
    c.hash = hash;
    c.index = 0;
    c.total_chunks = 1;
    c.data = object;

    RNET_CHECK_OK(assembler.Accept(c));
    RNET_CHECK_OK(assembler.Accept(c));       // retransmission
    RNET_CHECK_EQ(assembler.received_chunks(), 1ull);
    RNET_CHECK_OK(assembler.Finish());
}

// ---------------------------------------------------------------------------
// Reject
// ---------------------------------------------------------------------------
RNET_TEST(Reject, RoundTripsAndBoundsItsText) {
    RejectMessage m;
    m.command = cmd::kInv;
    m.code = RejectCode::RateLimited;
    m.reason = "too many inventory messages";

    auto decoded = RejectMessage::Deserialize(m.Serialize());
    RNET_CHECK_OK(decoded);
    RNET_CHECK(decoded.value().code == RejectCode::RateLimited);

    m.reason = std::string(500, 'x');
    RNET_CHECK_ERR(RejectMessage::Deserialize(m.Serialize()));

    m.reason = "line\nbreak";
    RNET_CHECK_ERR(RejectMessage::Deserialize(m.Serialize()));
}

// ---------------------------------------------------------------------------
// Address manager — the eclipse defence
// ---------------------------------------------------------------------------
#include "net/addrman.h"

namespace {

NetAddress Addr(const std::string& text) {
    auto a = NetAddress::FromString(text, 9444);
    return a.ok() ? a.value() : NetAddress{};
}

AddressManager MakeAddrMan() {
    return AddressManager(crypto::Sha3_256(std::string_view("addrman-key")));
}

}  // namespace

RNET_TEST(AddrMan, StoresAndPromotes) {
    auto man = MakeAddrMan();
    const auto source = Addr("203.0.113.1:9444");
    const auto peer = Addr("198.51.100.5:9444");

    RNET_CHECK(man.Add(peer, source, 1000));
    RNET_CHECK(man.Contains(peer));
    RNET_CHECK_EQ(man.NewCount(), size_t{1});
    RNET_CHECK_EQ(man.TriedCount(), size_t{0});

    // Promotion is earned by a real connection, not by gossip.
    man.MarkGood(peer, 2000);
    RNET_CHECK_EQ(man.TriedCount(), size_t{1});
    RNET_CHECK_EQ(man.NewCount(), size_t{0});
}

RNET_TEST(AddrMan, IgnoresUnroutableAddresses) {
    auto man = MakeAddrMan();
    const auto source = Addr("203.0.113.1:9444");
    // Storing these would waste slots and, later, connection attempts.
    RNET_CHECK(!man.Add(Addr("127.0.0.1:9444"), source, 1000));
    RNET_CHECK(!man.Add(Addr("192.168.1.1:9444"), source, 1000));
    RNET_CHECK(!man.Add(Addr("10.1.2.3:9444"), source, 1000));
    RNET_CHECK_EQ(man.Size(), size_t{0});
}

RNET_TEST(AddrMan, DuplicatesRefreshRatherThanGrow) {
    auto man = MakeAddrMan();
    const auto source = Addr("203.0.113.1:9444");
    const auto peer = Addr("198.51.100.5:9444");

    RNET_CHECK(man.Add(peer, source, 1000));
    RNET_CHECK(!man.Add(peer, source, 2000));
    RNET_CHECK_EQ(man.Size(), size_t{1});
}

// Grouping is what makes buying addresses stop buying influence.
RNET_TEST(AddrMan, GroupsByNetworkNotByAddress) {
    RNET_CHECK_EQ(NetworkGroup(Addr("203.0.113.1:9444")), NetworkGroup(Addr("203.0.113.99:9444")));
    RNET_CHECK(NetworkGroup(Addr("203.0.113.1:9444")) != NetworkGroup(Addr("203.1.113.1:9444")));
}

// The eclipse scenario: one attacker floods thousands of addresses from its own
// network. It must not be able to fill the table and crowd out everyone else.
RNET_TEST(AddrMan, FloodFromOneNetworkIsConfined) {
    auto man = MakeAddrMan();
    const auto attacker = Addr("198.51.100.1:9444");

    for (int a = 0; a < 20; ++a) {
        for (int b = 1; b < 250; ++b) {
            man.Add(Addr(std::format("198.51.{}.{}:9444", a, b)), attacker, 1000);
        }
    }
    const size_t attacker_entries = man.Size();

    // ~5000 addresses offered. They all share one /16 group and one source group,
    // so they compete for a SINGLE bucket: the attacker's entire flood is worth
    // one bucket's capacity, no matter how many addresses it invents.
    if (attacker_entries > kEntriesPerBucket) {
        RNET_FAIL("flood of ~5000 addresses kept {} entries, expected at most {}",
                  attacker_entries, kEntriesPerBucket);
    }

    // An honest peer from elsewhere still gets in afterwards.
    const auto honest_source = Addr("203.0.113.1:9444");
    RNET_CHECK(man.Add(Addr("192.0.2.50:9444"), honest_source, 2000));
    RNET_CHECK(man.Contains(Addr("192.0.2.50:9444")));
}

// A peer that has actually worked must not be displaced by gossip.
RNET_TEST(AddrMan, GossipCannotEvictProvenPeers) {
    auto man = MakeAddrMan();
    const auto honest = Addr("192.0.2.7:9444");
    man.Add(honest, Addr("203.0.113.1:9444"), 1000);
    man.MarkGood(honest, 1000);

    const auto attacker = Addr("198.51.100.1:9444");
    for (int a = 0; a < 20; ++a) {
        for (int b = 1; b < 250; ++b) {
            man.Add(Addr(std::format("198.51.{}.{}:9444", a, b)), attacker, 2000);
        }
    }
    RNET_CHECK(man.Contains(honest));
    RNET_CHECK_EQ(man.TriedCount(), size_t{1});
}

RNET_TEST(AddrMan, DeadAddressesAreReleased) {
    auto man = MakeAddrMan();
    const auto peer = Addr("198.51.100.5:9444");
    man.Add(peer, Addr("203.0.113.1:9444"), 1000);

    for (uint32_t i = 0; i < kMaxFailures - 1; ++i) man.MarkFailed(peer, 1000 + i);
    RNET_CHECK(man.Contains(peer));      // still worth retrying

    man.MarkFailed(peer, 2000);
    RNET_CHECK(!man.Contains(peer));     // clearly dead: release the slot
}

RNET_TEST(AddrMan, SelectReturnsSomethingWhenAnythingIsKnown) {
    auto man = MakeAddrMan();
    RNET_CHECK(!man.Select(12345).has_value());   // nothing known yet

    man.Add(Addr("192.0.2.7:9444"), Addr("203.0.113.1:9444"), 1000);
    // Even with a strong bias toward tried, a node holding only new addresses
    // must still make progress rather than stalling.
    RNET_CHECK(man.Select(12345, /*bias_tried=*/100).has_value());
}

RNET_TEST(AddrMan, GetAddressesIsCappedAndRotates) {
    auto man = MakeAddrMan();
    const auto source = Addr("203.0.113.1:9444");
    for (int i = 1; i < 60; ++i) man.Add(Addr(std::format("192.0.2.{}:9444", i)), source, 1000);

    // A getaddr must not pull a node's whole view in one request.
    RNET_CHECK_EQ(man.GetAddresses(10, 0).size(), size_t{10});

    // Nor return the same prefix every time, which would fingerprint the node.
    RNET_CHECK(man.GetAddresses(10, 0) != man.GetAddresses(10, 7));
}

// Two nodes with different keys must bucket differently, or an attacker could
// compute the layout once and craft a set that fills the buckets everyone selects
// from.
// Without a per-node key an attacker could compute the bucket layout once and
// craft a set that fills exactly the buckets every victim selects from. The input
// here spans many /16 groups so that bucket assignment — and therefore which
// entries survive contention — actually depends on the key.
RNET_TEST(AddrMan, BucketingIsKeyed) {
    AddressManager a(crypto::Sha3_256(std::string_view("key-a")));
    AddressManager b(crypto::Sha3_256(std::string_view("key-b")));

    for (int group = 0; group < 200; ++group) {
        const auto source = Addr(std::format("203.{}.0.1:9444", group % 128));
        for (int host = 1; host <= 60; ++host) {
            const auto addr = Addr(std::format("198.{}.{}.{}:9444", group / 32, group % 32, host));
            a.Add(addr, source, 1000);
            b.Add(addr, source, 1000);
        }
    }

    // Same offered set, different keys: the retained sets differ, so the layout
    // cannot be precomputed against a victim.
    const auto kept_a = a.GetAddresses(100000, 0);
    const auto kept_b = b.GetAddresses(100000, 0);
    RNET_CHECK(kept_a != kept_b);
    RNET_CHECK(!kept_a.empty() && !kept_b.empty());
}

// ---------------------------------------------------------------------------
// Peer: framing a byte stream, and refusing what should not be framed
// ---------------------------------------------------------------------------
#include "net/peer.h"

namespace {

Peer MakePeer(bool inbound = true) {
    return Peer(1, Socket(), Addr("203.0.113.9:9444"), inbound, kMagic);
}

std::vector<uint8_t> Framed(std::string_view command, std::span<const uint8_t> payload = {}) {
    auto framed = SerializeMessage(kMagic, command, payload);
    return framed.ok() ? framed.value() : std::vector<uint8_t>{};
}

}  // namespace

// TCP delivers bytes, not messages: a header split across reads is normal, not an
// error, and must simply leave the parser waiting.
RNET_TEST(Peer, ReassemblesMessagesSplitAcrossReads) {
    auto peer = MakePeer();
    auto version = MakeVersion();
    const auto bytes = Framed(cmd::kVersion, version.Serialize());

    // One byte at a time: the worst case a real network can produce.
    for (size_t i = 0; i + 1 < bytes.size(); ++i) {
        auto out = peer.ReceiveBytes(std::span<const uint8_t>(&bytes[i], 1));
        RNET_CHECK_OK(out);
        RNET_CHECK(out.value().empty());        // nothing complete yet
    }
    auto out = peer.ReceiveBytes(std::span<const uint8_t>(&bytes.back(), 1));
    RNET_CHECK_OK(out);
    RNET_CHECK_EQ(out.value().size(), size_t{1});
    RNET_CHECK_STREQ(out.value()[0].command, cmd::kVersion);
}

RNET_TEST(Peer, ExtractsSeveralMessagesFromOneRead) {
    auto peer = MakePeer();
    auto version = MakeVersion();

    std::vector<uint8_t> stream = Framed(cmd::kVersion, version.Serialize());
    auto first = peer.ReceiveBytes(stream);
    RNET_CHECK_OK(first);
    RNET_CHECK_OK(peer.HandleVersion(version, version, /*local_nonce=*/1));
    RNET_CHECK_OK(peer.HandleVerack());          // now Ready

    std::vector<uint8_t> batch;
    for (const char* c : {cmd::kPing, cmd::kPong, cmd::kGetAddr}) {
        const auto framed = Framed(c);
        batch.insert(batch.end(), framed.begin(), framed.end());
    }
    auto out = peer.ReceiveBytes(batch);
    RNET_CHECK_OK(out);
    RNET_CHECK_EQ(out.value().size(), size_t{3});
    RNET_CHECK_STREQ(out.value()[2].command, cmd::kGetAddr);
}

// Bad magic leaves no way to know where the next message begins, so the stream is
// unrecoverable and the connection ends.
RNET_TEST(Peer, GarbageStreamEndsTheConnection) {
    auto peer = MakePeer();
    std::vector<uint8_t> garbage(kHeaderSize, 0xAB);
    RNET_CHECK_ERR(peer.ReceiveBytes(garbage));
    RNET_CHECK(peer.should_disconnect());
}

// An unbounded receive buffer is how a peer exhausts a node's memory.
RNET_TEST(Peer, ReceiveBufferIsBounded) {
    auto peer = MakePeer();
    // A valid header promising a payload, followed by an endless dribble that
    // never completes it.
    std::vector<uint8_t> partial = Framed(cmd::kVersion, std::vector<uint8_t>(1000, 0));
    partial.resize(kHeaderSize + 500);           // deliberately incomplete
    RNET_CHECK_OK(peer.ReceiveBytes(partial));

    const std::vector<uint8_t> filler(kMaxReceiveBuffer, 0x00);
    RNET_CHECK_ERR(peer.ReceiveBytes(filler));
    RNET_CHECK(peer.should_disconnect());
}

// A peer that stops reading must not be able to make the node buffer on its
// behalf until it falls over.
RNET_TEST(Peer, SendQueueIsBounded) {
    auto peer = MakePeer();
    const std::vector<uint8_t> payload(kMaxMessageSize - 1000, 0x5A);

    bool refused = false;
    for (int i = 0; i < 100; ++i) {
        if (!peer.Send(cmd::kObject, payload)) { refused = true; break; }
    }
    RNET_CHECK(refused);
    RNET_CHECK(peer.should_disconnect());
}

// Driving the node's object logic before identifying yourself is misbehaviour,
// not merely early.
RNET_TEST(Peer, MessagesBeforeHandshakeAreRefused) {
    RNET_CHECK(CommandAllowedInState(cmd::kVersion, PeerState::AwaitingVersion));
    RNET_CHECK(!CommandAllowedInState(cmd::kInv, PeerState::AwaitingVersion));
    RNET_CHECK(!CommandAllowedInState(cmd::kGetData, PeerState::AwaitingVersion));
    RNET_CHECK(CommandAllowedInState(cmd::kVerack, PeerState::AwaitingVerack));
    RNET_CHECK(!CommandAllowedInState(cmd::kInv, PeerState::AwaitingVerack));
    RNET_CHECK(CommandAllowedInState(cmd::kInv, PeerState::Ready));

    // Framing accepts it; the refusal happens when it is handled, which is the
    // only point where the peer's state is current.
    auto peer = MakePeer();
    auto out = peer.ReceiveBytes(Framed(cmd::kInv));
    RNET_CHECK_OK(out);
    RNET_CHECK_EQ(out.value().size(), size_t{1});
    RNET_CHECK_ERR(peer.CheckCommandAllowed(out.value()[0].command));
    RNET_CHECK(peer.misbehaviour() > 0);
}

RNET_TEST(Peer, HandshakeCompletes) {
    auto peer = MakePeer();
    RNET_CHECK(peer.state() == PeerState::AwaitingVersion);

    auto remote = MakeVersion();
    remote.nonce = 999;
    auto local = MakeVersion();
    local.nonce = 111;

    RNET_CHECK_OK(peer.HandleVersion(remote, local, local.nonce));
    RNET_CHECK(peer.state() == PeerState::AwaitingVerack);
    RNET_CHECK_OK(peer.HandleVerack());
    RNET_CHECK(peer.state() == PeerState::Ready);
    RNET_CHECK_EQ(peer.version().worker_id, remote.worker_id);
}

// A peer on another round or under other rules is not a peer at all — its work
// would be incompatible with everything this node does.
RNET_TEST(Peer, HandshakeRejectsForeignConsensus) {
    auto local = MakeVersion();
    local.nonce = 111;

    auto wrong_genesis = MakeVersion();
    wrong_genesis.nonce = 222;
    wrong_genesis.genesis_hash = crypto::Sha3_256(std::string_view("other-round"));
    auto a = MakePeer();
    RNET_CHECK_ERR(a.HandleVersion(wrong_genesis, local, local.nonce));
    RNET_CHECK(a.should_disconnect());

    auto wrong_policy = MakeVersion();
    wrong_policy.nonce = 333;
    wrong_policy.policy_hash = crypto::Sha3_256(std::string_view("other-policy"));
    auto b = MakePeer();
    RNET_CHECK_ERR(b.HandleVersion(wrong_policy, local, local.nonce));
    RNET_CHECK(b.should_disconnect());
}

// Connecting to yourself wastes a slot and pollutes the address database.
RNET_TEST(Peer, SelfConnectionIsDetected) {
    auto local = MakeVersion();
    local.nonce = 0xABCDEF;
    auto peer = MakePeer();
    RNET_CHECK_ERR(peer.HandleVersion(local, local, local.nonce));
    RNET_CHECK(peer.should_disconnect());
}

RNET_TEST(Peer, DuplicateVersionIsMisbehaviour) {
    auto peer = MakePeer();
    auto remote = MakeVersion();
    remote.nonce = 999;
    auto local = MakeVersion();
    local.nonce = 111;

    RNET_CHECK_OK(peer.HandleVersion(remote, local, local.nonce));
    RNET_CHECK_ERR(peer.HandleVersion(remote, local, local.nonce));
    RNET_CHECK(peer.misbehaviour() > 0);
}

// One slip is forgiven; a pattern is not.
RNET_TEST(Peer, MisbehaviourAccumulatesToDisconnect) {
    auto peer = MakePeer();
    peer.Misbehaving(10, "minor");
    RNET_CHECK(!peer.should_disconnect());
    peer.Misbehaving(kBanThreshold, "serious");
    RNET_CHECK(peer.should_disconnect());
}

RNET_TEST(Peer, SendQueueDrainsInOrder) {
    auto peer = MakePeer();
    RNET_CHECK_OK(peer.Send(cmd::kPing, {}));
    RNET_CHECK(peer.HasPendingSend());

    const auto pending = peer.PendingSend();
    const auto expected = Framed(cmd::kPing);
    RNET_CHECK_EQ(std::vector<uint8_t>(pending.begin(), pending.end()), expected);

    peer.ConsumeSent(pending.size());
    RNET_CHECK(!peer.HasPendingSend());
}

// ---------------------------------------------------------------------------
// Node: two real nodes over loopback
//
// Everything above tests logic without sockets. These tests use real TCP on the
// loopback interface, because the parts that only appear with a real kernel —
// partial writes, connect completing asynchronously, EAGAIN — are exactly where
// networking code goes wrong.
// ---------------------------------------------------------------------------
#include "net/node.h"

namespace {

NodeConfig TestNodeConfig(uint16_t port) {
    NodeConfig config;
    config.network = consensus::Network::Regtest;
    auto bind = NetAddress::FromString(std::format("127.0.0.1:{}", port), port);
    config.bind_address = bind.ok() ? bind.value() : NetAddress{};
    config.max_inbound = 8;
    config.max_outbound = 8;
    config.handshake_timeout_ms = 5000;
    return config;
}

// Drives both nodes until `done` or the budget runs out. Time is supplied, not
// read from a clock, so the test is deterministic.
bool Pump(Node& a, Node& b, const std::function<bool()>& done, int iterations = 200) {
    for (int i = 0; i < iterations; ++i) {
        const int64_t now = 1000 + i * 10;
        (void)a.Tick(now, 5);
        (void)b.Tick(now, 5);
        if (done()) return true;
    }
    return done();
}

}  // namespace

RNET_TEST(Node, TwoNodesCompleteAHandshake) {
    Node listener(TestNodeConfig(19801), crypto::Sha3_256(std::string_view("a")), 1111);
    Node dialer(TestNodeConfig(0), crypto::Sha3_256(std::string_view("b")), 2222);

    RNET_CHECK_OK(listener.Start());
    RNET_CHECK_OK(dialer.Start());

    auto target = NetAddress::FromString("127.0.0.1:19801", 19801);
    RNET_CHECK_OK(target);
    RNET_CHECK_OK(dialer.ConnectTo(target.value(), 1000));

    const bool established = Pump(listener, dialer, [&] {
        return dialer.ReadyCount() == 1 && listener.ReadyCount() == 1;
    });
    if (!established) {
        RNET_FAIL("handshake did not complete: dialer ready={} listener ready={}",
                  dialer.ReadyCount(), listener.ReadyCount());
    }
    RNET_CHECK_EQ(listener.InboundCount(), size_t{1});
    RNET_CHECK_EQ(dialer.OutboundCount(), size_t{1});
}

// A node's own version must carry the anchors it actually runs on, or the
// handshake check is decorative.
RNET_TEST(Node, LocalVersionCarriesBothAnchors) {
    Node node(TestNodeConfig(0), crypto::Sha3_256(std::string_view("v")), 42);
    const auto version = node.LocalVersion(1'700'000'000'000);
    const auto& params = consensus::Params(consensus::Network::Regtest);

    RNET_CHECK_STREQ(util::ToHex(version.genesis_hash), params.genesis_hash_hex);
    RNET_CHECK_STREQ(util::ToHex(version.policy_hash), params.policy_hash_hex);
    RNET_CHECK_OK(version.Validate());
}

// Peers on different networks must not federate: the magic differs, so the very
// first bytes are rejected.
RNET_TEST(Node, NodesOnDifferentNetworksDoNotPeer) {
    auto listener_config = TestNodeConfig(19802);
    auto dialer_config = TestNodeConfig(0);
    dialer_config.network = consensus::Network::Test;   // different magic entirely

    Node listener(listener_config, crypto::Sha3_256(std::string_view("a")), 1111);
    Node dialer(dialer_config, crypto::Sha3_256(std::string_view("b")), 2222);
    RNET_CHECK_OK(listener.Start());
    RNET_CHECK_OK(dialer.Start());

    auto target = NetAddress::FromString("127.0.0.1:19802", 19802);
    RNET_CHECK_OK(target);
    RNET_CHECK_OK(dialer.ConnectTo(target.value(), 1000));

    Pump(listener, dialer, [&] { return false; }, 60);
    RNET_CHECK_EQ(dialer.ReadyCount(), size_t{0});
    RNET_CHECK_EQ(listener.ReadyCount(), size_t{0});
}

// Message exchange after the handshake, through a registered handler.
RNET_TEST(Node, EstablishedPeersExchangeMessages) {
    Node listener(TestNodeConfig(19803), crypto::Sha3_256(std::string_view("a")), 1111);
    Node dialer(TestNodeConfig(0), crypto::Sha3_256(std::string_view("b")), 2222);
    RNET_CHECK_OK(listener.Start());
    RNET_CHECK_OK(dialer.Start());

    int received = 0;
    listener.SetHandler(cmd::kInv, [&](Peer&, const ReceivedMessage& message, int64_t) -> Status {
        auto inv = InvMessage::Deserialize(message.payload);
        if (!inv) return Err(inv.error());
        received += static_cast<int>(inv.value().entries.size());
        return Status::Ok();
    });

    auto target = NetAddress::FromString("127.0.0.1:19803", 19803);
    RNET_CHECK_OK(target);
    RNET_CHECK_OK(dialer.ConnectTo(target.value(), 1000));
    Pump(listener, dialer, [&] { return dialer.ReadyCount() == 1 && listener.ReadyCount() == 1; });

    InvMessage inv;
    inv.entries.push_back({InvType::Checkpoint, crypto::Sha3_256(std::string_view("ckpt"))});
    inv.entries.push_back({InvType::Contribution, crypto::Sha3_256(std::string_view("contrib"))});
    RNET_CHECK_EQ(dialer.Broadcast(cmd::kInv, inv.Serialize()), size_t{1});

    const bool delivered = Pump(listener, dialer, [&] { return received == 2; });
    if (!delivered) RNET_FAIL("handler saw {} entries, expected 2", received);
}

// Peer exchange: a node that completes a handshake asks for addresses, and what
// it learns is remembered.
RNET_TEST(Node, PeersExchangeAddresses) {
    Node listener(TestNodeConfig(19804), crypto::Sha3_256(std::string_view("a")), 1111);
    Node dialer(TestNodeConfig(0), crypto::Sha3_256(std::string_view("b")), 2222);
    RNET_CHECK_OK(listener.Start());
    RNET_CHECK_OK(dialer.Start());

    // Give the listener something worth gossiping.
    auto known = NetAddress::FromString("203.0.113.50:9444", 9444);
    RNET_CHECK_OK(known);
    listener.addresses().Add(known.value(), Addr("203.0.113.1:9444"), 1);

    auto target = NetAddress::FromString("127.0.0.1:19804", 19804);
    RNET_CHECK_OK(target);
    RNET_CHECK_OK(dialer.ConnectTo(target.value(), 1000));

    const bool learned = Pump(listener, dialer,
                              [&] { return dialer.addresses().Contains(known.value()); });
    if (!learned) RNET_FAIL("dialer did not learn the gossiped address");
}

// Inbound slots are budgeted separately: filling them must not stop a node from
// reaching out to peers of its own choosing.
RNET_TEST(Node, InboundAndOutboundSlotsAreSeparate) {
    auto config = TestNodeConfig(19805);
    config.max_inbound = 1;
    config.max_outbound = 4;

    Node node(config, crypto::Sha3_256(std::string_view("n")), 3333);
    RNET_CHECK_OK(node.Start());
    RNET_CHECK_EQ(node.InboundCount(), size_t{0});
    RNET_CHECK_EQ(node.OutboundCount(), size_t{0});

    // Outbound attempts remain possible regardless of inbound pressure; the two
    // budgets never draw from one pool.
    for (int i = 1; i <= 4; ++i) {
        auto address = NetAddress::FromString(std::format("203.0.113.{}:9444", i), 9444);
        RNET_CHECK_OK(address);
        RNET_CHECK_OK(node.ConnectTo(address.value(), 1000));
    }
    RNET_CHECK_EQ(node.OutboundCount(), size_t{4});

    auto extra = NetAddress::FromString("203.0.113.99:9444", 9444);
    RNET_CHECK_OK(extra);
    RNET_CHECK_ERR(node.ConnectTo(extra.value(), 1000));   // outbound budget exhausted
}

RNET_TEST(Node, BansExpire) {
    Node node(TestNodeConfig(0), crypto::Sha3_256(std::string_view("n")), 4444);
    RNET_CHECK_OK(node.Start());

    auto address = NetAddress::FromString("203.0.113.77:9444", 9444);
    RNET_CHECK_OK(address);

    node.Ban(address.value(), 1000);
    RNET_CHECK(node.IsBanned(address.value(), 1000));
    RNET_CHECK_ERR(node.ConnectTo(address.value(), 1000));

    // A permanent ban list is itself a resource an attacker can grow, and
    // addresses change hands.
    RNET_CHECK(!node.IsBanned(address.value(), 1000 + node.config().ban_duration_ms + 1));
}

// A peer that connects and says nothing costs it nothing and costs the node a
// slot, so the slot is reclaimed.
RNET_TEST(Node, SilentPeersLoseTheirSlot) {
    auto config = TestNodeConfig(19806);
    config.handshake_timeout_ms = 100;
    Node listener(config, crypto::Sha3_256(std::string_view("a")), 1111);
    RNET_CHECK_OK(listener.Start());

    auto target = NetAddress::FromString("127.0.0.1:19806", 19806);
    RNET_CHECK_OK(target);

    // A raw socket that connects and then stays silent.
    bool in_progress = false;
    auto rude = Socket::StartConnect(target.value(), in_progress);
    RNET_CHECK_OK(rude);

    for (int i = 0; i < 20; ++i) (void)listener.Tick(1000 + i * 5, 5);
    RNET_CHECK_EQ(listener.PeerCount(), size_t{1});      // accepted, still handshaking

    (void)listener.Tick(2000, 5);                        // well past the timeout
    RNET_CHECK_EQ(listener.PeerCount(), size_t{0});
}

RNET_TEST(Node, RefusesDuplicateAndBannedConnections) {
    Node node(TestNodeConfig(0), crypto::Sha3_256(std::string_view("n")), 5555);
    RNET_CHECK_OK(node.Start());

    auto address = NetAddress::FromString("203.0.113.11:9444", 9444);
    RNET_CHECK_OK(address);
    RNET_CHECK_OK(node.ConnectTo(address.value(), 1000));
    RNET_CHECK_ERR(node.ConnectTo(address.value(), 1000));   // already connected
}

// ---------------------------------------------------------------------------
// Object exchange. Announce → request → deliver, with every step being a place a
// peer can misbehave, so each is tested for what it refuses as much as for what
// it does.
// ---------------------------------------------------------------------------
#include "net/objectstore.h"

namespace {

std::vector<uint8_t> Bytes(std::string_view s) { return {s.begin(), s.end()}; }

InvEntry EntryFor(std::span<const uint8_t> data, InvType type = InvType::Contribution) {
    return InvEntry{type, crypto::Sha3_256(data)};
}

}  // namespace

// The identity of an object is the hash of its bytes. A peer that answers with
// something else is substituting data, not making a mistake, so the store must
// never take its word for it.
RNET_TEST(ObjectStore, AdmitsOnlyBytesThatHashToTheirIdentity) {
    ObjectStore store;
    const auto honest = Bytes("a contribution header");
    const auto hash = crypto::Sha3_256(honest);

    auto substituted = store.Put(InvType::Contribution, hash, Bytes("something else"), 1000);
    RNET_CHECK_ERR(substituted);
    RNET_CHECK_EQ(store.Size(), size_t{0});

    auto accepted = store.Put(InvType::Contribution, hash, honest, 1000);
    RNET_CHECK_OK(accepted);
    RNET_CHECK(accepted.value());              // newly stored
    RNET_CHECK(store.Has(hash));

    auto again = store.Put(InvType::Contribution, hash, honest, 2000);
    RNET_CHECK_OK(again);
    RNET_CHECK(!again.value());                // already held, and not an error
    RNET_CHECK_EQ(store.Size(), size_t{1});
}

// Announcing is free, so it must not cause storage. Only what the node lacks and
// is not already fetching comes back as worth asking for.
RNET_TEST(ObjectStore, AnnouncementsAskOnlyForWhatIsMissing) {
    ObjectStore store;
    const auto held = Bytes("already have this");
    RNET_CHECK_OK(store.Put(InvType::Contribution, crypto::Sha3_256(held), held, 1000));

    std::vector<InvEntry> announced{EntryFor(held), EntryFor(Bytes("new one")),
                                    EntryFor(Bytes("another"))};
    const auto wanted = store.NoteAnnouncement(7, announced, 1000);
    RNET_CHECK_EQ(wanted.size(), size_t{2});
    RNET_CHECK_EQ(store.Size(), size_t{1});    // announcing stored nothing

    // Once requested, a second announcement of the same object adds nothing:
    // asking every peer for one object multiplies the node's own bandwidth cost
    // by its peer count.
    RNET_CHECK_OK(store.NoteRequested(7, wanted[0], 1000));
    const auto second = store.NoteAnnouncement(8, announced, 1000);
    RNET_CHECK_EQ(second.size(), size_t{1});
    RNET_CHECK_ERR(store.NoteRequested(8, wanted[0], 1000));
}

RNET_TEST(ObjectStore, OutstandingRequestsPerPeerAreBounded) {
    ObjectStore store;
    for (size_t i = 0; i < kMaxRequestsPerPeer; ++i) {
        const auto entry = EntryFor(Bytes(std::format("object-{}", i)));
        RNET_CHECK_OK(store.NoteRequested(1, entry, 1000));
    }
    RNET_CHECK_EQ(store.PendingFor(1), kMaxRequestsPerPeer);
    RNET_CHECK_ERR(store.NoteRequested(1, EntryFor(Bytes("one too many")), 1000));

    // The bound is per peer, not global: a well-behaved peer is not punished for
    // a greedy one.
    RNET_CHECK_OK(store.NoteRequested(2, EntryFor(Bytes("from another peer")), 1000));
}

RNET_TEST(ObjectStore, AnnouncementsRememberedPerPeerAreBounded) {
    ObjectStore store;
    std::vector<InvEntry> flood;
    for (size_t i = 0; i < kMaxAnnouncementsPerPeer + 500; ++i) {
        flood.push_back(EntryFor(Bytes(std::format("flood-{}", i))));
    }
    (void)store.NoteAnnouncement(1, flood, 1000);

    // Memory occupied by one peer's talking is capped; the entries past the cap
    // are simply not remembered as its offers.
    size_t remembered = 0;
    for (const auto& entry : flood) {
        if (!store.PeersAnnouncing(entry.hash).empty()) ++remembered;
    }
    RNET_CHECK_EQ(remembered, kMaxAnnouncementsPerPeer);
}

// A peer that announces and never delivers stalls the node silently. The request
// must expire and be retried elsewhere.
RNET_TEST(ObjectStore, UndeliveredRequestsExpireAndCanGoElsewhere) {
    ObjectStore store;
    const auto entry = EntryFor(Bytes("promised but never sent"));
    (void)store.NoteAnnouncement(1, {entry}, 1000);
    (void)store.NoteAnnouncement(2, {entry}, 1000);
    RNET_CHECK_OK(store.NoteRequested(1, entry, 1000));

    RNET_CHECK(store.ExpiredRequests(1000 + kRequestTimeoutMs).empty());   // not yet
    const auto expired = store.ExpiredRequests(1000 + kRequestTimeoutMs + 1);
    RNET_CHECK_EQ(expired.size(), size_t{1});
    RNET_CHECK_EQ(expired[0].peer_id, uint64_t{1});

    const auto announcers = store.PeersAnnouncing(entry.hash);
    RNET_CHECK_EQ(announcers.size(), size_t{2});

    store.DropRequest(entry.hash);
    RNET_CHECK_OK(store.NoteRequested(2, entry, 70'000));
}

// A departed peer cannot deliver what it owed, and its offers are no longer
// somewhere a retry can go.
RNET_TEST(ObjectStore, ForgettingAPeerReleasesWhatItOwed) {
    ObjectStore store;
    const auto entry = EntryFor(Bytes("owed by a peer that left"));
    (void)store.NoteAnnouncement(1, {entry}, 1000);
    RNET_CHECK_OK(store.NoteRequested(1, entry, 1000));
    RNET_CHECK_EQ(store.PendingCount(), size_t{1});

    store.ForgetPeer(1);
    RNET_CHECK_EQ(store.PendingCount(), size_t{0});
    RNET_CHECK(store.PeersAnnouncing(entry.hash).empty());
    RNET_CHECK_OK(store.NoteRequested(2, entry, 1000));   // free to fetch elsewhere
}

RNET_TEST(ObjectStore, EvictsTheOldestWhenFull) {
    ObjectStore store(3);
    for (int i = 0; i < 3; ++i) {
        const auto data = Bytes(std::format("object-{}", i));
        RNET_CHECK_OK(store.Put(InvType::Contribution, crypto::Sha3_256(data), data,
                                1000 + i * 10));
    }
    const auto oldest = Bytes("object-0");
    RNET_CHECK(store.Has(crypto::Sha3_256(oldest)));

    const auto newcomer = Bytes("object-3");
    RNET_CHECK_OK(store.Put(InvType::Contribution, crypto::Sha3_256(newcomer), newcomer, 1040));
    RNET_CHECK_EQ(store.Size(), size_t{3});
    RNET_CHECK(!store.Has(crypto::Sha3_256(oldest)));
    RNET_CHECK(store.Has(crypto::Sha3_256(newcomer)));
}

// The whole exchange over a real socket: one node publishes, the other learns of
// it from an announcement, asks for it, and ends up holding the same bytes.
RNET_TEST(Node, ObjectsPropagateFromAnnouncementToStorage) {
    Node listener(TestNodeConfig(19810), crypto::Sha3_256(std::string_view("a")), 1111);
    Node dialer(TestNodeConfig(0), crypto::Sha3_256(std::string_view("b")), 2222);
    RNET_CHECK_OK(listener.Start());
    RNET_CHECK_OK(dialer.Start());

    auto target = NetAddress::FromString("127.0.0.1:19810", 19810);
    RNET_CHECK_OK(target);
    RNET_CHECK_OK(dialer.ConnectTo(target.value(), 1000));
    Pump(listener, dialer, [&] { return dialer.ReadyCount() == 1 && listener.ReadyCount() == 1; });

    const auto payload = Bytes("a verification verdict worth propagating");
    auto published = listener.Publish(InvType::Verdict, payload, 2000);
    RNET_CHECK_OK(published);

    const bool arrived = Pump(listener, dialer,
                              [&] { return dialer.objects().Has(published.value()); });
    if (!arrived) RNET_FAIL("object did not propagate to the peer");

    auto held = dialer.objects().Get(published.value());
    RNET_CHECK_OK(held);
    RNET_CHECK(held.value().data == payload);
    RNET_CHECK_EQ(static_cast<int>(held.value().type), static_cast<int>(InvType::Verdict));
    RNET_CHECK_EQ(dialer.objects().PendingCount(), size_t{0});   // request was cleared
}

// A peer that answers a request with different bytes is trying to substitute
// content. The node must not store it, and must score the peer for it.
RNET_TEST(Node, SubstitutedObjectBytesAreRejectedAndScored) {
    Node listener(TestNodeConfig(19811), crypto::Sha3_256(std::string_view("a")), 1111);
    Node dialer(TestNodeConfig(0), crypto::Sha3_256(std::string_view("b")), 2222);
    RNET_CHECK_OK(listener.Start());
    RNET_CHECK_OK(dialer.Start());

    auto target = NetAddress::FromString("127.0.0.1:19811", 19811);
    RNET_CHECK_OK(target);
    RNET_CHECK_OK(dialer.ConnectTo(target.value(), 1000));
    Pump(listener, dialer, [&] { return dialer.ReadyCount() == 1 && listener.ReadyCount() == 1; });

    // The listener announces one hash and hands over bytes that do not match it.
    const auto claimed = crypto::Sha3_256(std::string_view("what was promised"));
    InvMessage inv;
    inv.entries.push_back(InvEntry{InvType::Contribution, claimed});
    RNET_CHECK_EQ(listener.Broadcast(cmd::kInv, inv.Serialize()), size_t{1});

    bool asked = false;
    listener.SetHandler(cmd::kGetData, [&](Peer& peer, const ReceivedMessage&, int64_t) -> Status {
        if (asked) return Status::Ok();
        asked = true;
        canon::Writer w;
        w.U16(static_cast<uint16_t>(InvType::Contribution));
        w.Hash(claimed);
        w.VarBytes(Bytes("not what was promised"));
        return peer.Send(cmd::kObject, w.data());
    });

    Pump(listener, dialer, [&] { return false; }, 80);
    RNET_CHECK(asked);
    RNET_CHECK(!dialer.objects().Has(claimed));
    RNET_CHECK_EQ(dialer.objects().Size(), size_t{0});
}

// A bulk object does not fit in one message. Refusing it outright would be a
// lie — it exists — so the requester is told, in so many words, to use chunks.
RNET_TEST(Node, OversizedObjectsAreRefusedWithAReasonNotSilence) {
    Node listener(TestNodeConfig(19812), crypto::Sha3_256(std::string_view("a")), 1111);
    Node dialer(TestNodeConfig(0), crypto::Sha3_256(std::string_view("b")), 2222);
    RNET_CHECK_OK(listener.Start());
    RNET_CHECK_OK(dialer.Start());

    // Just past what a single message can carry.
    std::vector<uint8_t> weights(kMaxMessageSize - kChunkMetadataAllowance + 1, 0xAB);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = static_cast<uint8_t>(i * 7 + 3);
    const auto hash = crypto::Sha3_256(weights);
    RNET_CHECK_OK(listener.objects().Put(InvType::Checkpoint, hash, weights, 1000));

    auto target = NetAddress::FromString("127.0.0.1:19812", 19812);
    RNET_CHECK_OK(target);
    RNET_CHECK_OK(dialer.ConnectTo(target.value(), 1000));
    Pump(listener, dialer, [&] { return dialer.ReadyCount() == 1 && listener.ReadyCount() == 1; });

    std::string refusal;
    dialer.SetHandler(cmd::kReject, [&](Peer&, const ReceivedMessage& message, int64_t) -> Status {
        auto reject = RejectMessage::Deserialize(message.payload);
        if (!reject) return Err(reject.error());
        refusal = reject.value().reason;
        return Status::Ok();
    });

    InvMessage inv;
    inv.entries.push_back(InvEntry{InvType::Checkpoint, hash});
    RNET_CHECK_EQ(listener.Broadcast(cmd::kInv, inv.Serialize()), size_t{1});
    Pump(listener, dialer, [&] { return !refusal.empty(); });

    if (refusal.find("getchunks") == std::string::npos) {
        RNET_FAIL("expected a refusal naming the bulk path, got '{}'", refusal);
    }
    if (refusal.find(util::ToHex(hash)) == std::string::npos) {
        RNET_FAIL("refusal must name which object it refers to, got '{}'", refusal);
    }
}

// The bulk path end to end: a multi-chunk object crosses the wire and is admitted
// only after the assembled whole hashes correctly.
RNET_TEST(Node, BulkObjectsTransferInChunks) {
    Node listener(TestNodeConfig(19813), crypto::Sha3_256(std::string_view("a")), 1111);
    Node dialer(TestNodeConfig(0), crypto::Sha3_256(std::string_view("b")), 2222);
    RNET_CHECK_OK(listener.Start());
    RNET_CHECK_OK(dialer.Start());

    // Two and a bit chunks, so the short final chunk is exercised too.
    std::vector<uint8_t> weights(kBulkChunkSize * 2 + 7777);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = static_cast<uint8_t>(i * 31 + 11);
    const auto hash = crypto::Sha3_256(weights);
    RNET_CHECK_OK(listener.objects().Put(InvType::Checkpoint, hash, weights, 1000));

    auto target = NetAddress::FromString("127.0.0.1:19813", 19813);
    RNET_CHECK_OK(target);
    RNET_CHECK_OK(dialer.ConnectTo(target.value(), 1000));
    Pump(listener, dialer, [&] { return dialer.ReadyCount() == 1 && listener.ReadyCount() == 1; });

    uint64_t peer_id = 0;
    for (uint64_t candidate = 1; candidate <= 8; ++candidate) {
        if (dialer.objects().PendingFor(candidate) == 0) { peer_id = candidate; break; }
    }
    RNET_CHECK_EQ(peer_id, uint64_t{1});

    // The size is not asked of the sender: it comes from the header that named
    // this payload, so a peer cannot make the node allocate by claiming a size.
    RNET_CHECK_OK(dialer.RequestBulk(1, InvType::Checkpoint, hash, weights.size(), 2000));
    RNET_CHECK(dialer.BulkInFlight(hash));

    const bool assembled =
        Pump(listener, dialer, [&] { return dialer.objects().Has(hash); }, 400);
    if (!assembled) RNET_FAIL("bulk transfer did not complete");

    auto held = dialer.objects().Get(hash);
    RNET_CHECK_OK(held);
    RNET_CHECK(held.value().data == weights);
    RNET_CHECK(!dialer.BulkInFlight(hash));
}

// A chunk carries no proof of its own. Nothing may be admitted on the strength of
// having arrived — only the finished object is verified, and a peer that corrupts
// one slice loses the whole transfer.
RNET_TEST(Node, CorruptedChunksLoseTheWholeTransfer) {
    Node listener(TestNodeConfig(19814), crypto::Sha3_256(std::string_view("a")), 1111);
    Node dialer(TestNodeConfig(0), crypto::Sha3_256(std::string_view("b")), 2222);
    RNET_CHECK_OK(listener.Start());
    RNET_CHECK_OK(dialer.Start());

    std::vector<uint8_t> weights(kBulkChunkSize + 1024);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = static_cast<uint8_t>(i * 13 + 5);
    const auto hash = crypto::Sha3_256(weights);

    auto target = NetAddress::FromString("127.0.0.1:19814", 19814);
    RNET_CHECK_OK(target);
    RNET_CHECK_OK(dialer.ConnectTo(target.value(), 1000));
    Pump(listener, dialer, [&] { return dialer.ReadyCount() == 1 && listener.ReadyCount() == 1; });

    // The listener serves the chunks itself, flipping one byte in the second.
    listener.SetHandler(cmd::kGetChunks, [&](Peer& peer, const ReceivedMessage&, int64_t) -> Status {
        const uint64_t total = 2;
        for (uint64_t index = 0; index < total; ++index) {
            const size_t offset = static_cast<size_t>(index) * kBulkChunkSize;
            const size_t length = std::min<size_t>(kBulkChunkSize, weights.size() - offset);
            ChunkMessage chunk;
            chunk.type = InvType::Checkpoint;
            chunk.hash = hash;
            chunk.index = index;
            chunk.total_chunks = total;
            chunk.data.assign(weights.begin() + static_cast<ptrdiff_t>(offset),
                              weights.begin() + static_cast<ptrdiff_t>(offset + length));
            if (index == 1) chunk.data[0] ^= 0xFF;
            if (auto st = peer.Send(cmd::kChunk, chunk.Serialize()); !st) return st;
        }
        return Status::Ok();
    });

    RNET_CHECK_OK(dialer.RequestBulk(1, InvType::Checkpoint, hash, weights.size(), 2000));
    Pump(listener, dialer, [&] { return !dialer.BulkInFlight(hash); }, 400);

    RNET_CHECK(!dialer.objects().Has(hash));   // never admitted
    RNET_CHECK(!dialer.BulkInFlight(hash));    // and the transfer was abandoned
}

// Chunks nobody asked for are how a peer makes a node allocate on demand.
RNET_TEST(Node, UnsolicitedChunksAreRefused) {
    Node listener(TestNodeConfig(19815), crypto::Sha3_256(std::string_view("a")), 1111);
    Node dialer(TestNodeConfig(0), crypto::Sha3_256(std::string_view("b")), 2222);
    RNET_CHECK_OK(listener.Start());
    RNET_CHECK_OK(dialer.Start());

    auto target = NetAddress::FromString("127.0.0.1:19815", 19815);
    RNET_CHECK_OK(target);
    RNET_CHECK_OK(dialer.ConnectTo(target.value(), 1000));
    Pump(listener, dialer, [&] { return dialer.ReadyCount() == 1 && listener.ReadyCount() == 1; });

    ChunkMessage chunk;
    chunk.type = InvType::Checkpoint;
    chunk.hash = crypto::Sha3_256(std::string_view("never requested"));
    chunk.index = 0;
    chunk.total_chunks = 1;
    chunk.data = Bytes("unsolicited payload");
    RNET_CHECK_EQ(listener.Broadcast(cmd::kChunk, chunk.Serialize()), size_t{1});

    Pump(listener, dialer, [&] { return false; }, 60);
    RNET_CHECK_EQ(dialer.objects().Size(), size_t{0});
    RNET_CHECK(!dialer.BulkInFlight(chunk.hash));
}

// A node with three peers must not fetch the same object three times: that turns
// being well connected into being the easiest to exhaust.
RNET_TEST(Node, AnObjectIsFetchedFromOnePeerOnly) {
    ObjectStore store;
    const auto entry = EntryFor(Bytes("announced by everyone"));
    for (uint64_t peer = 1; peer <= 3; ++peer) {
        const auto wanted = store.NoteAnnouncement(peer, {entry}, 1000);
        if (peer == 1) {
            RNET_CHECK_EQ(wanted.size(), size_t{1});
            RNET_CHECK_OK(store.NoteRequested(peer, entry, 1000));
        } else {
            RNET_CHECK_EQ(wanted.size(), size_t{0});   // already in flight
        }
    }
    RNET_CHECK_EQ(store.PendingCount(), size_t{1});
    RNET_CHECK_EQ(store.PeersAnnouncing(entry.hash).size(), size_t{3});
}

// ---------------------------------------------------------------------------
// Persistence and bootstrap. A node that forgets everything when it restarts has
// to re-bootstrap from seeds each time, which hands an attacker a fresh eclipse
// opportunity on every restart — so the address database survives, and what it
// restores is treated as untrusted input rather than as instructions.
// ---------------------------------------------------------------------------
#include "net/bootstrap.h"

namespace {

std::filesystem::path TempPath(std::string_view name) {
    return std::filesystem::temp_directory_path() / std::format("rnet-test-{}", name);
}

NetAddress Routable(int a, int b, int c, int d, uint16_t port = 9444) {
    auto address = NetAddress::FromString(std::format("{}.{}.{}.{}:{}", a, b, c, d, port), port);
    return address.ok() ? address.value() : NetAddress{};
}

}  // namespace

RNET_TEST(AddressManager, SurvivesARestart) {
    const auto path = TempPath("peers-roundtrip.dat");
    std::filesystem::remove(path);

    const auto key = crypto::Sha3_256(std::string_view("node key"));
    const auto source = Routable(198, 51, 100, 1);

    AddressManager saved(key);
    for (int i = 1; i <= 20; ++i) saved.Add(Routable(203, 0, i, 7), source, 1'700'000'000);
    // Only addresses this node actually reached belong in `tried`.
    saved.MarkGood(Routable(203, 0, 5, 7), 1'700'000'100);
    saved.MarkGood(Routable(203, 0, 9, 7), 1'700'000'200);
    RNET_CHECK_EQ(saved.TriedCount(), size_t{2});
    RNET_CHECK_OK(saved.Save(path.string()));

    AddressManager restored(key);
    RNET_CHECK_OK(restored.Load(path.string()));
    RNET_CHECK_EQ(restored.Size(), saved.Size());
    RNET_CHECK_EQ(restored.TriedCount(), size_t{2});
    RNET_CHECK(restored.Contains(Routable(203, 0, 5, 7)));
    RNET_CHECK(restored.Contains(Routable(203, 0, 20, 7)));

    std::filesystem::remove(path);
}

// The file is data, not authority. A tampered peers file may cost a node its
// memory of the network; it must not be able to place addresses where an attacker
// wants them, or to assert that an address was ever reached.
RNET_TEST(AddressManager, ATamperedFileIsRefusedNotObeyed) {
    const auto path = TempPath("peers-tampered.dat");
    std::filesystem::remove(path);

    AddressManager saved(crypto::Sha3_256(std::string_view("k")));
    saved.Add(Routable(203, 0, 113, 4), Routable(198, 51, 100, 1), 1'700'000'000);
    RNET_CHECK_OK(saved.Save(path.string()));

    auto raw = util::ReadFile(path);
    RNET_CHECK_OK(raw);
    auto bytes = raw.value();
    bytes[bytes.size() / 2] ^= 0x01;          // one bit, somewhere in the body
    RNET_CHECK_OK(util::WriteFileAtomic(path, bytes));

    AddressManager loaded(crypto::Sha3_256(std::string_view("k")));
    RNET_CHECK_ERR(loaded.Load(path.string()));
    RNET_CHECK_EQ(loaded.Size(), size_t{0});   // nothing half-restored

    std::filesystem::remove(path);
}

// Bucket placement is keyed per node. Restoring the same file into a node with a
// different key must not reproduce the same layout — otherwise an attacker who
// works out one node's bucketing knows every node's, and can craft an address set
// that fills exactly the buckets any victim draws from.
//
// The observable consequence is which peer gets DIALLED: Select walks the buckets
// in order, so a different layout means a different choice from the same draw.
// (Gossip is deliberately not key-dependent — GetAddresses hands out entries, not
// buckets — so it is not the place to look for this property.)
RNET_TEST(AddressManager, RestoredEntriesAreRebucketedWithThisNodesKey) {
    const auto path = TempPath("peers-rebucket.dat");
    std::filesystem::remove(path);

    AddressManager saved(crypto::Sha3_256(std::string_view("key one")));
    const auto source = Routable(198, 51, 100, 1);
    // Distinct /16 groups, so each lands in a bucket of its own rather than
    // colliding for reasons that have nothing to do with the key.
    for (int i = 1; i <= 60; ++i) saved.Add(Routable(203, i, 1, 7), source, 1'700'000'000);
    RNET_CHECK_OK(saved.Save(path.string()));

    AddressManager other(crypto::Sha3_256(std::string_view("key two")));
    RNET_CHECK_OK(other.Load(path.string()));
    RNET_CHECK_EQ(other.Size(), saved.Size());
    RNET_CHECK(other.Contains(Routable(203, 42, 1, 7)));   // same contents

    size_t differing = 0;
    for (uint64_t draw = 0; draw < 32; ++draw) {
        const uint64_t randomness = draw * 977 + 13;
        const auto a = saved.Select(randomness, /*bias_tried=*/0);
        const auto b = other.Select(randomness, /*bias_tried=*/0);
        RNET_CHECK(a.has_value());
        RNET_CHECK(b.has_value());
        if (!(a->address == b->address)) ++differing;
    }
    // Identical choices on every one of 32 draws would mean the key changed
    // nothing about placement.
    if (differing == 0) {
        RNET_FAIL("two nodes with different keys dialled identically on all 32 draws");
    }

    std::filesystem::remove(path);
}

// A peers file claiming more entries than the tables can hold is refused before
// anything is allocated on the strength of the claim.
RNET_TEST(AddressManager, ImpossibleEntryCountsAreRefusedBeforeAllocation) {
    const auto path = TempPath("peers-huge.dat");
    std::filesystem::remove(path);

    canon::Writer body;
    body.U32(0xFFFFFFFFu);   // "four billion addresses follow"

    canon::Writer file;
    const uint8_t magic[4] = {'R', 'N', 'P', 'D'};
    file.Bytes(std::span<const uint8_t>(magic, 4));
    file.U32(1);
    file.VarBytes(body.data());
    file.Hash(crypto::Sha3_256(body.data()));   // checksum is honest; the count is not
    RNET_CHECK_OK(util::WriteFileAtomic(path, file.data()));

    AddressManager loaded(crypto::Sha3_256(std::string_view("k")));
    RNET_CHECK_ERR(loaded.Load(path.string()));
    RNET_CHECK_EQ(loaded.Size(), size_t{0});

    std::filesystem::remove(path);
}

// A seed that answers with loopback or private addresses is not offering peers,
// whether through misconfiguration or on purpose.
RNET_TEST(Bootstrap, SeedsCannotPointANodeAtItself) {
    auto resolved = ResolveSeed("localhost", 9444);
    // Either the name does not resolve, or every answer was dropped as
    // unroutable. Both are the same outcome: nothing usable came back.
    if (resolved) {
        for (const auto& address : resolved.value()) {
            if (!address.IsRoutable()) RNET_FAIL("an unroutable seed address was returned");
        }
        RNET_FAIL("localhost yielded {} routable addresses", resolved.value().size());
    }
}

// A network with no seed infrastructure yet must say so rather than appear to
// have bootstrapped.
RNET_TEST(Bootstrap, NoSeedsMeansNoAddressesAndNoPretence) {
    AddressManager addresses(crypto::Sha3_256(std::string_view("k")));
    const auto& params = consensus::Params(consensus::Network::Regtest);

    const auto result = BootstrapFromSeeds(params, addresses, 1'700'000'000);
    RNET_CHECK_EQ(result.seeds_queried, params.dns_seeds.size());
    RNET_CHECK_EQ(result.addresses_added, size_t{0});
    RNET_CHECK_EQ(addresses.Size(), size_t{0});
}

// A contribution payload for the launch model is 983,635,968 bytes — 1,877 chunks.
// The send queue holds 8 MiB, which is 16 chunks. Serving all of them into it in
// one Tick overflows the queue, and Peer::Send responds to an overflow by
// DISCONNECTING — so the node hangs up on the peer it was in the middle of
// answering, and the larger the object the sooner it happens.
//
// 40 chunks is past the cliff and small enough to run in a test.
RNET_TEST(Node, LargeObjectsSurviveTheSendQueue) {
    Node listener(TestNodeConfig(19816), crypto::Sha3_256(std::string_view("a")), 1111);
    Node dialer(TestNodeConfig(0), crypto::Sha3_256(std::string_view("b")), 2222);
    RNET_CHECK_OK(listener.Start());
    RNET_CHECK_OK(dialer.Start());

    std::vector<uint8_t> weights(kBulkChunkSize * 40 + 12345);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = static_cast<uint8_t>(i * 17 + 3);
    const auto hash = crypto::Sha3_256(weights);
    RNET_CHECK_OK(listener.objects().Put(InvType::Checkpoint, hash, weights, 1000));

    auto target = NetAddress::FromString("127.0.0.1:19816", 19816);
    RNET_CHECK_OK(target);
    RNET_CHECK_OK(dialer.ConnectTo(target.value(), 1000));
    Pump(listener, dialer, [&] { return dialer.ReadyCount() == 1 && listener.ReadyCount() == 1; });

    RNET_CHECK_OK(dialer.RequestBulk(1, InvType::Checkpoint, hash, weights.size(), 2000));

    const bool assembled =
        Pump(listener, dialer, [&] { return dialer.objects().Has(hash); }, 4000);
    if (!assembled) {
        RNET_FAIL("a {}-chunk object did not transfer (listener peers: {}, dialer peers: {})",
                  (weights.size() + kBulkChunkSize - 1) / kBulkChunkSize, listener.ReadyCount(),
                  dialer.ReadyCount());
    }
    auto held = dialer.objects().Get(hash);
    RNET_CHECK_OK(held);
    RNET_CHECK(held.value().data == weights);
}

// ---------------------------------------------------------------------------
// Checkpoint weights.
//
// A CheckpointHeader carries a hash and no bytes. Until these existed, a network
// of one node worked and a second node could never join: it could derive the
// genesis weights but had no way to reach the state everyone else was at.
// ---------------------------------------------------------------------------
#include "net/weightsstore.h"

namespace {

std::filesystem::path TestWeightsDir(std::string_view name) {
    return std::filesystem::temp_directory_path() / std::format("rnet-weights-{}", name);
}

// A model-sized-ish blob: several chunks, with a short final one.
std::vector<uint8_t> FakeWeights(size_t size, uint8_t salt) {
    std::vector<uint8_t> bytes(size);
    for (size_t i = 0; i < size; ++i) bytes[i] = static_cast<uint8_t>(i * 31 + salt);
    return bytes;
}

int SealedFdOf(const std::vector<uint8_t>& bytes) {
    const int fd = ::memfd_create("rnet-weights-test", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (fd < 0) return -1;
    size_t written = 0;
    while (written < bytes.size()) {
        const ssize_t n = ::write(fd, bytes.data() + written, bytes.size() - written);
        if (n <= 0) { ::close(fd); return -1; }
        written += static_cast<size_t>(n);
    }
    return fd;
}

}  // namespace

// The bytes are named by their hash and checked before they are given that name.
// A node that served weights it had not verified would spread a model nobody
// agreed on, under a name everybody trusts.
RNET_TEST(WeightsStore, AdmitsOnlyBytesThatHashToTheirName) {
    const auto dir = TestWeightsDir("admit");
    std::filesystem::remove_all(dir);
    WeightsStore store(dir);

    const auto bytes = FakeWeights(300'000, 3);
    const auto hash = crypto::Sha3_256(bytes);

    // Offered under the wrong name: refused, and nothing is left behind.
    const auto wrong = crypto::Sha3_256(std::string_view("some other checkpoint"));
    const int bad_fd = SealedFdOf(bytes);
    RNET_CHECK(bad_fd >= 0);
    RNET_CHECK_ERR(store.PutFromFd(wrong, bad_fd, bytes.size()));
    ::close(bad_fd);
    RNET_CHECK(!store.Has(wrong));
    RNET_CHECK_EQ(store.List().value().size(), size_t{0});

    const int good_fd = SealedFdOf(bytes);
    RNET_CHECK(good_fd >= 0);
    RNET_CHECK_OK(store.PutFromFd(hash, good_fd, bytes.size()));
    ::close(good_fd);
    RNET_CHECK(store.Has(hash));
    RNET_CHECK_EQ(store.SizeOf(hash).value(), uint64_t{300'000});

    // And the bytes come back exactly.
    std::vector<uint8_t> read_back(bytes.size());
    auto got = store.ReadRange(hash, 0, read_back);
    RNET_CHECK_OK(got);
    RNET_CHECK_EQ(got.value(), bytes.size());
    RNET_CHECK(read_back == bytes);

    std::filesystem::remove_all(dir);
}

// A source that ends early must not leave a file under a name that implies it is
// complete: a truncated file keeping its final name is a checkpoint whose bytes
// silently disagree with its hash.
RNET_TEST(WeightsStore, ATruncatedWriteLeavesNothingBehind) {
    const auto dir = TestWeightsDir("truncated");
    std::filesystem::remove_all(dir);
    WeightsStore store(dir);

    const auto bytes = FakeWeights(100'000, 7);
    const auto hash = crypto::Sha3_256(bytes);
    const int fd = SealedFdOf(bytes);
    RNET_CHECK(fd >= 0);

    // Claims twice as many bytes as the source holds.
    RNET_CHECK_ERR(store.PutFromFd(hash, fd, bytes.size() * 2));
    ::close(fd);
    RNET_CHECK(!store.Has(hash));
    RNET_CHECK_EQ(store.List().value().size(), size_t{0});

    std::filesystem::remove_all(dir);
}

// Retention belongs to the policy, so the store prunes to whatever set it is
// told to keep rather than inventing a rule of its own.
RNET_TEST(WeightsStore, PrunesToTheRetentionSet) {
    const auto dir = TestWeightsDir("prune");
    std::filesystem::remove_all(dir);
    WeightsStore store(dir);

    std::vector<crypto::Hash256> hashes;
    for (uint8_t i = 0; i < 5; ++i) {
        const auto bytes = FakeWeights(50'000, i);
        const auto hash = crypto::Sha3_256(bytes);
        const int fd = SealedFdOf(bytes);
        RNET_CHECK(fd >= 0);
        RNET_CHECK_OK(store.PutFromFd(hash, fd, bytes.size()));
        ::close(fd);
        hashes.push_back(hash);
    }
    RNET_CHECK_EQ(store.List().value().size(), size_t{5});

    const std::set<crypto::Hash256> keep{hashes[3], hashes[4]};
    auto removed = store.RetainOnly(keep);
    RNET_CHECK_OK(removed);
    RNET_CHECK_EQ(removed.value(), size_t{3});
    RNET_CHECK_EQ(store.List().value().size(), size_t{2});
    RNET_CHECK(store.Has(hashes[3]));
    RNET_CHECK(store.Has(hashes[4]));
    RNET_CHECK(!store.Has(hashes[0]));

    std::filesystem::remove_all(dir);
}

// The point of the whole thing: a node that holds weights can hand them to a node
// that does not, so a second node can join a network that has moved past genesis.
RNET_TEST(Node, WeightsTransferFromANodeThatHasThemToOneThatDoesNot) {
    const auto server_dir = TestWeightsDir("serve");
    const auto client_dir = TestWeightsDir("fetch");
    std::filesystem::remove_all(server_dir);
    std::filesystem::remove_all(client_dir);

    WeightsStore server_store(server_dir);
    WeightsStore client_store(client_dir);

    // Several chunks, with a short final one — the shape a real model has.
    const auto weights = FakeWeights(kBulkChunkSize * 5 + 4321, 11);
    const auto hash = crypto::Sha3_256(weights);
    const int fd = SealedFdOf(weights);
    RNET_CHECK(fd >= 0);
    RNET_CHECK_OK(server_store.PutFromFd(hash, fd, weights.size()));
    ::close(fd);

    Node listener(TestNodeConfig(19817), crypto::Sha3_256(std::string_view("a")), 1111);
    Node dialer(TestNodeConfig(0), crypto::Sha3_256(std::string_view("b")), 2222);
    listener.SetWeightsStore(&server_store);
    dialer.SetWeightsStore(&client_store);
    RNET_CHECK_OK(listener.Start());
    RNET_CHECK_OK(dialer.Start());

    auto target = NetAddress::FromString("127.0.0.1:19817", 19817);
    RNET_CHECK_OK(target);
    RNET_CHECK_OK(dialer.ConnectTo(target.value(), 1000));
    Pump(listener, dialer, [&] { return dialer.ReadyCount() == 1 && listener.ReadyCount() == 1; });

    // The size is derived from the checkpoint header the joining node already
    // holds — n_params times the dtype width — never asked of the peer.
    RNET_CHECK_OK(dialer.RequestBulk(1, InvType::Checkpoint, hash, weights.size(), 2000));

    const bool arrived =
        Pump(listener, dialer, [&] { return client_store.Has(hash); }, 4000);
    if (!arrived) RNET_FAIL("the weights never arrived");

    // Landed on disk, not in the in-memory object store: a two-gigabyte model in
    // RAM would defeat the reason the weights store exists.
    RNET_CHECK(!dialer.objects().Has(hash));
    RNET_CHECK_EQ(client_store.SizeOf(hash).value(), weights.size());

    std::vector<uint8_t> read_back(weights.size());
    RNET_CHECK_OK(client_store.ReadRange(hash, 0, read_back));
    RNET_CHECK(read_back == weights);

    std::filesystem::remove_all(server_dir);
    std::filesystem::remove_all(client_dir);
}

// ---------------------------------------------------------------------------
// Serving the corpus.
//
// The deterministic batch derivation is what stops a worker choosing its own
// data — but only if it reads THE corpus, the one the round descriptor pinned. A
// worker with a different file computes the same offsets into different tokens
// and nothing on the wire can tell. So chunks travel with an inclusion proof, and
// the receiver checks them against a consensus value rather than against whoever
// answered.
// ---------------------------------------------------------------------------
#include "dataset/index.h"
#include "dataset/manifest.h"
#include "net/corpus.h"

namespace {

// A small corpus with an uneven tail, because the last chunk being short is the
// normal case rather than an edge case.
struct TestCorpus {
    std::filesystem::path path;
    canon::DatasetManifest manifest;
};

// A corpus of text documents, since that is what a corpus is now. `n_docs`
// documents of varying length, separated by a blank line, so chunk boundaries
// fall where the builder would put them.
Result<TestCorpus> MakeCorpus(std::string_view name, uint32_t target_chunk_bytes, uint64_t n_docs,
                              uint32_t salt = 0) {
    TestCorpus corpus;
    corpus.path = std::filesystem::temp_directory_path() / std::format("rnet-corpus-{}.txt", name);
    std::string text;
    for (uint64_t i = 0; i < n_docs; ++i) {
        const size_t len = 300 + static_cast<size_t>((i * 137 + salt) % 900);
        text.append("Document ").append(std::to_string(i + salt)).append(". ");
        for (size_t j = 0; j < len; ++j) {
            text.push_back(static_cast<char>('a' + ((i * 31 + j * 7 + salt) % 26)));
            if (j % 9 == 8) text.push_back(' ');
        }
        text.append("\n\n");
    }
    if (auto st = util::WriteTextFileAtomic(corpus.path, text); !st) return Err(st.error());

    auto index = dataset::DatasetIndex::Build(corpus.path, target_chunk_bytes);
    if (!index) return Err(index.error());
    corpus.manifest = index.value().ToManifest(crypto::Sha3_256(std::string_view("tok")));
    return corpus;
}

}  // namespace

// A node serving a file that is not the corpus is refused when it opens it, not
// discovered by the first worker whose proof fails — a failure there would look
// like the network accusing an honest seed.
RNET_TEST(Corpus, RefusesAFileThatIsNotTheCorpusItClaims) {
    auto real = MakeCorpus("real", 64, 500);
    RNET_CHECK_OK(real);
    // A genuinely different corpus, or the comparison below would prove nothing.
    auto other = MakeCorpus("other", 64, 500, /*salt=*/12345);
    RNET_CHECK_OK(other);
    if (other.value().manifest.dataset_root == real.value().manifest.dataset_root) {
        RNET_FAIL("the two test corpora hash the same, so this test proves nothing");
    }

    // The right file under the right manifest opens.
    RNET_CHECK_OK(CorpusSource::Open(real.value().path, real.value().manifest));

    // A different file under that manifest does not, and the refusal says why.
    auto mismatched = CorpusSource::Open(other.value().path, real.value().manifest);
    RNET_CHECK_ERR(mismatched);
    if (mismatched.error().find("not that corpus") == std::string::npos) {
        RNET_FAIL("the refusal should name the mismatch, got '{}'", mismatched.error());
    }

    std::filesystem::remove(real.value().path);
    std::filesystem::remove(other.value().path);
}

// A chunk is checked against the pinned root, so an altered one is caught by
// arithmetic rather than by trusting whoever sent it.
RNET_TEST(Corpus, AnAlteredChunkFailsItsProof) {
    auto corpus = MakeCorpus("altered", 64, 500);
    RNET_CHECK_OK(corpus);
    auto source = CorpusSource::Open(corpus.value().path, corpus.value().manifest);
    RNET_CHECK_OK(source);

    const uint64_t index = 3;
    auto size = source.value().ChunkBytes(index);
    RNET_CHECK_OK(size);
    auto proof = source.value().ProofForChunk(index);
    RNET_CHECK_OK(proof);

    std::vector<uint8_t> bytes(static_cast<size_t>(size.value()));
    auto got = source.value().ReadChunkRange(index, 0, bytes);
    RNET_CHECK_OK(got);
    RNET_CHECK_EQ(got.value(), bytes.size());

    // Honest bytes pass.
    {
        CorpusAssembler assembler(index, size.value(), proof.value());
        RNET_CHECK_OK(assembler.Accept(0, bytes));
        RNET_CHECK(assembler.Complete());
        RNET_CHECK_OK(assembler.Finish(corpus.value().manifest.dataset_root));
    }
    // One flipped token does not.
    {
        auto tampered = bytes;
        tampered[7] ^= 0x01;
        CorpusAssembler assembler(index, size.value(), proof.value());
        RNET_CHECK_OK(assembler.Accept(0, tampered));
        auto finished = assembler.Finish(corpus.value().manifest.dataset_root);
        RNET_CHECK_ERR(finished);
        if (finished.error().find("pinned dataset root") == std::string::npos) {
            RNET_FAIL("the refusal should name the root, got '{}'", finished.error());
        }
    }
    std::filesystem::remove(corpus.value().path);
}

// The last chunk is short whenever the corpus does not divide evenly, which is
// the normal case.
RNET_TEST(Corpus, TheFinalChunkIsShortAndStillVerifies) {
    auto corpus = MakeCorpus("tail", 4096, 40);
    RNET_CHECK_OK(corpus);
    auto source = CorpusSource::Open(corpus.value().path, corpus.value().manifest);
    RNET_CHECK_OK(source);
    RNET_CHECK(source.value().n_chunks() > 1);

    // Chunks end at the first document separator past the target, so every chunk
    // but the last is at least the target long and the last one is whatever
    // remains. Asserted as a property rather than as arithmetic: the exact size
    // depends on where the documents fall, which is the point of cutting there.
    const uint64_t last = source.value().n_chunks() - 1;
    auto size = source.value().ChunkBytes(last);
    RNET_CHECK_OK(size);
    RNET_CHECK(size.value() > 0);
    auto first = source.value().ChunkBytes(0);
    RNET_CHECK_OK(first);
    RNET_CHECK(first.value() >= 4096);

    std::vector<uint8_t> bytes(static_cast<size_t>(size.value()));
    RNET_CHECK_OK(source.value().ReadChunkRange(last, 0, bytes));
    auto proof = source.value().ProofForChunk(last);
    RNET_CHECK_OK(proof);

    CorpusAssembler assembler(last, size.value(), proof.value());
    RNET_CHECK_OK(assembler.Accept(0, bytes));
    RNET_CHECK_OK(assembler.Finish(corpus.value().manifest.dataset_root));

    std::filesystem::remove(corpus.value().path);
}

// The point of the whole thing: a worker with no corpus can get the chunks it
// needs from a node that has one, and know they are the right tokens.
RNET_TEST(Node, CorpusChunksTravelWithProofsAndVerifyAgainstTheRoot) {
    // Chunks large enough that one spans several parts.
    auto corpus = MakeCorpus("wire", 200'000, 700'000);
    RNET_CHECK_OK(corpus);
    auto source = CorpusSource::Open(corpus.value().path, corpus.value().manifest);
    RNET_CHECK_OK(source);

    Node seed(TestNodeConfig(19818), crypto::Sha3_256(std::string_view("a")), 1111);
    Node worker(TestNodeConfig(0), crypto::Sha3_256(std::string_view("b")), 2222);
    seed.SetCorpus(&source.value());
    RNET_CHECK_OK(seed.Start());
    RNET_CHECK_OK(worker.Start());

    // Everything the worker receives, assembled and checked against the root it
    // already trusts from the round descriptor.
    std::map<uint64_t, std::unique_ptr<CorpusAssembler>> assembling;
    std::map<uint64_t, std::vector<uint8_t>> verified;
    worker.SetHandler(cmd::kCorpus,
                      [&](Peer&, const ReceivedMessage& message, int64_t) -> Status {
                          auto part = CorpusMessage::Deserialize(message.payload);
                          if (!part) return Err(part.error());
                          const uint64_t index = part.value().chunk_index;
                          if (part.value().part == 0 && assembling.find(index) == assembling.end()) {
                              assembling[index] = std::make_unique<CorpusAssembler>(
                                  index, part.value().chunk_bytes, part.value().proof);
                          }
                          const auto it = assembling.find(index);
                          if (it == assembling.end()) return Status::Ok();
                          if (auto st = it->second->Accept(part.value().part, part.value().data);
                              !st) {
                              return st;
                          }
                          if (!it->second->Complete()) return Status::Ok();
                          auto finished = it->second->Finish(corpus.value().manifest.dataset_root);
                          if (!finished) return Err(finished.error());
                          verified[index] = std::move(finished.value());
                          assembling.erase(it);
                          return Status::Ok();
                      });

    auto target = NetAddress::FromString("127.0.0.1:19818", 19818);
    RNET_CHECK_OK(target);
    RNET_CHECK_OK(worker.ConnectTo(target.value(), 1000));
    Pump(seed, worker, [&] { return seed.ReadyCount() == 1 && worker.ReadyCount() == 1; });

    GetCorpusMessage request;
    request.first_chunk = 1;
    request.chunk_count = 2;
    RNET_CHECK_EQ(worker.Broadcast(cmd::kGetCorpus, request.Serialize()), size_t{1});

    const bool arrived = Pump(seed, worker, [&] { return verified.size() == 2; }, 2000);
    if (!arrived) {
        RNET_FAIL("only {} of 2 corpus chunks verified", verified.size());
    }

    // And the bytes are exactly what the seed holds — not merely something that
    // passed a proof.
    for (uint64_t index : {uint64_t{1}, uint64_t{2}}) {
        auto size = source.value().ChunkBytes(index);
        RNET_CHECK_OK(size);
        std::vector<uint8_t> expected(static_cast<size_t>(size.value()));
        RNET_CHECK_OK(source.value().ReadChunkRange(index, 0, expected));
        RNET_CHECK(verified[index] == expected);
    }

    std::filesystem::remove(corpus.value().path);
}

// A node that keeps no corpus says so rather than leaving the asker to time out.
RNET_TEST(Node, ANodeWithoutACorpusSaysSo) {
    Node seed(TestNodeConfig(19819), crypto::Sha3_256(std::string_view("a")), 1111);
    Node worker(TestNodeConfig(0), crypto::Sha3_256(std::string_view("b")), 2222);
    RNET_CHECK_OK(seed.Start());
    RNET_CHECK_OK(worker.Start());

    bool told = false;
    worker.SetHandler(cmd::kNotFound,
                      [&](Peer&, const ReceivedMessage&, int64_t) -> Status {
                          told = true;
                          return Status::Ok();
                      });

    auto target = NetAddress::FromString("127.0.0.1:19819", 19819);
    RNET_CHECK_OK(target);
    RNET_CHECK_OK(worker.ConnectTo(target.value(), 1000));
    Pump(seed, worker, [&] { return seed.ReadyCount() == 1 && worker.ReadyCount() == 1; });

    GetCorpusMessage request;
    request.first_chunk = 0;
    request.chunk_count = 1;
    RNET_CHECK_EQ(worker.Broadcast(cmd::kGetCorpus, request.Serialize()), size_t{1});

    const bool answered = Pump(seed, worker, [&] { return told; });
    if (!answered) RNET_FAIL("a node without a corpus left the request unanswered");
}

// The message vocabulary.
//
// Small, and deliberately so: a protocol surface is an attack surface. Every
// message is length-checked, bounds-checked and rejected on anything unexpected,
// because all of it arrives from peers nobody vouches for.
//
// Two channels of meaning:
//   * control — handshake, peer exchange, inventory. Small, frequent.
//   * bulk    — pseudo-gradients and checkpoints, gigabytes, always chunked.
//
// Objects themselves are canon containers; the network layer carries bytes and
// checks framing, and never reinterprets what consensus already defined.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "canon/canon.h"
#include "crypto/merkle.h"
#include "crypto/sha3.h"
#include "util/result.h"

namespace rnet::net {

// Bumped when the message vocabulary changes. Peers below the minimum are
// disconnected rather than half-supported.
inline constexpr uint32_t kProtocolVersion = 1;
inline constexpr uint32_t kMinProtocolVersion = 1;

namespace cmd {
inline constexpr const char* kVersion = "version";
inline constexpr const char* kVerack = "verack";
inline constexpr const char* kPing = "ping";
inline constexpr const char* kPong = "pong";
inline constexpr const char* kGetAddr = "getaddr";
inline constexpr const char* kAddr = "addr";
inline constexpr const char* kInv = "inv";
inline constexpr const char* kGetData = "getdata";
inline constexpr const char* kObject = "object";       // a whole small object
inline constexpr const char* kGetChunks = "getchunks";  // request a bulk transfer
inline constexpr const char* kChunk = "chunk";          // one slice of a bulk object
inline constexpr const char* kGetCorpus = "getcorpus";   // request corpus chunks by index
inline constexpr const char* kCorpus = "corpus";         // one part of one corpus chunk
inline constexpr const char* kNotFound = "notfound";
inline constexpr const char* kReject = "reject";
}  // namespace cmd

// Services a peer offers. A node that keeps no full checkpoints cannot answer a
// verification challenge, so peers advertise what they can actually do rather
// than being assumed capable.
enum ServiceFlags : uint64_t {
    kServiceNone = 0,
    kServiceRelay = 1 << 0,      // forwards objects
    kServiceSeed = 1 << 1,       // serves corpus chunks
    kServiceCheckpoints = 1 << 2,  // serves full checkpoint weights
    kServiceVerifier = 1 << 3,   // performs recomputation challenges
};

struct VersionMessage {
    uint32_t protocol_version{kProtocolVersion};
    uint64_t services{kServiceNone};
    int64_t timestamp{0};
    uint64_t nonce{0};              // detects self-connection
    std::string user_agent;
    uint64_t worker_id{0};
    uint16_t determinism_class{0};  // which arithmetic class this node can verify
    crypto::Hash256 genesis_hash{};  // both anchors: a peer on a different round or
    crypto::Hash256 policy_hash{};   // a different policy is not a peer at all

    std::vector<uint8_t> Serialize() const;
    static Result<VersionMessage> Deserialize(std::span<const uint8_t> raw);
    Status Validate() const;
};

// A peer address as gossiped. Held as 16 bytes so IPv6 is native and IPv4 is
// mapped, exactly as in BIP155-style encodings.
struct NetAddress {
    std::array<uint8_t, 16> ip{};
    uint16_t port{0};
    uint64_t services{kServiceNone};
    int64_t last_seen{0};

    bool IsIPv4() const;
    bool IsRoutable() const;   // rejects loopback, private ranges, unspecified
    std::string ToString() const;

    static Result<NetAddress> FromString(std::string_view text, uint16_t default_port);
    bool operator==(const NetAddress& other) const;
    bool operator<(const NetAddress& other) const;
};

// Peer exchange. Bounded so a peer cannot flood a node's address database.
inline constexpr size_t kMaxAddrPerMessage = 1000;

struct AddrMessage {
    std::vector<NetAddress> addresses;

    std::vector<uint8_t> Serialize() const;
    static Result<AddrMessage> Deserialize(std::span<const uint8_t> raw);
};

// What kind of object an inventory entry refers to. Mirrors canon::ObjectType so
// a node knows what it is being offered before downloading it.
enum class InvType : uint16_t {
    Contribution = 1,
    Checkpoint = 2,
    Challenge = 3,
    Verdict = 4,
    SlashEvidence = 5,
    CorpusChunk = 6,
};

struct InvEntry {
    InvType type{InvType::Contribution};
    crypto::Hash256 hash{};

    bool operator==(const InvEntry& other) const;
};

inline constexpr size_t kMaxInvPerMessage = 5000;

struct InvMessage {
    std::vector<InvEntry> entries;

    std::vector<uint8_t> Serialize() const;
    static Result<InvMessage> Deserialize(std::span<const uint8_t> raw);
};

// Requests a bulk transfer. The responder answers with `chunk` messages; the
// requester already knows how many to expect from the object's declared size.
struct GetChunksMessage {
    InvType type{InvType::Checkpoint};
    crypto::Hash256 hash{};
    uint64_t first_chunk{0};
    uint64_t chunk_count{0};

    std::vector<uint8_t> Serialize() const;
    static Result<GetChunksMessage> Deserialize(std::span<const uint8_t> raw);
    Status Validate() const;
};

// One slice of a bulk object. `total_chunks` lets a receiver size its buffer up
// front, and the whole object is verified against its hash once complete — a
// chunk is never trusted on its own.
struct ChunkMessage {
    InvType type{InvType::Checkpoint};
    crypto::Hash256 hash{};
    uint64_t index{0};
    uint64_t total_chunks{0};
    std::vector<uint8_t> data;

    std::vector<uint8_t> Serialize() const;
    static Result<ChunkMessage> Deserialize(std::span<const uint8_t> raw);
    Status Validate() const;
};

// Why a message was refused. Sent so an honest peer can correct itself instead of
// guessing; never carries more than a bounded reason string.
enum class RejectCode : uint16_t {
    Malformed = 1,
    Invalid = 2,
    Obsolete = 3,
    Duplicate = 4,
    RateLimited = 5,
    NotServed = 6,
};

struct RejectMessage {
    std::string command;
    RejectCode code{RejectCode::Invalid};
    std::string reason;

    std::vector<uint8_t> Serialize() const;
    static Result<RejectMessage> Deserialize(std::span<const uint8_t> raw);
};

// A request for corpus chunks, addressed BY INDEX. A worker knows which chunk it
// needs — it derived the offset itself — but not what that chunk hashes to, so
// asking by hash would require it to already hold the thing it is asking for.
struct GetCorpusMessage {
    uint64_t first_chunk{0};
    uint32_t chunk_count{0};

    std::vector<uint8_t> Serialize() const;
    static Result<GetCorpusMessage> Deserialize(std::span<const uint8_t> raw);
    Status Validate() const;
};

// One part of one corpus chunk. The inclusion proof rides on part zero: repeating
// it on every part would cost nothing in correctness and a little in bytes, but
// sending it once makes it unambiguous which proof the assembled chunk was
// checked against.
struct CorpusMessage {
    uint64_t chunk_index{0};
    uint64_t chunk_bytes{0};      // total size of this chunk, so a receiver can size its buffer
    uint32_t part{0};
    crypto::MerkleProof proof{};  // meaningful only when part == 0
    std::vector<uint8_t> data;

    std::vector<uint8_t> Serialize() const;
    static Result<CorpusMessage> Deserialize(std::span<const uint8_t> raw);
    Status Validate() const;
};

// Reassembles a chunked object and refuses anything that does not add up.
class ChunkAssembler {
public:
    ChunkAssembler(InvType type, crypto::Hash256 hash, uint64_t total_bytes);

    Status Accept(const ChunkMessage& chunk);
    bool Complete() const;

    // Hands over the assembled bytes, and only if they hash to the expected value.
    // A partially-received or tampered object is never returned.
    //
    // Moves rather than copies: returning a gigabyte by value would double the
    // peak at the exact moment the transfer is already at its peak. The assembler
    // is spent afterwards.
    Result<std::vector<uint8_t>> Finish();

    uint64_t received_chunks() const { return received_; }
    uint64_t total_chunks() const { return total_chunks_; }

private:
    InvType type_;
    crypto::Hash256 hash_;
    uint64_t total_bytes_;
    uint64_t total_chunks_;
    uint64_t received_{0};
    std::vector<uint8_t> buffer_;
    std::vector<bool> have_;
};

}  // namespace rnet::net

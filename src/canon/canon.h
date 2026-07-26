// rnet-canon — the ONE canonical byte layout for every hashed or signed object.
//
// Format, not values: this layer says how bytes are laid out; consensus values
// live in src/consensus. Every node in any language must produce identical bytes,
// so the encoding is explicit: fixed-width big-endian integers, no padding, no
// implementation-defined types, length-prefixed variable data.
//
// Container:
//   magic[4]="RNET" | version:u32 | obj_type:u16 | content_len:u32 |
//   content[content_len] | content_hash:SHA3-256[32] | crc32c:u32
//
// content_hash gives content-addressed identity; crc32c (Castagnoli) is a cheap
// integrity check over everything before it, so a corrupted artifact is rejected
// before any structural parsing happens.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "crypto/sha3.h"
#include "util/result.h"

namespace rnet::canon {

inline constexpr uint8_t kMagic[4] = {'R', 'N', 'E', 'T'};

// major << 16 | minor. A change to any serialized layout REQUIRES bumping this;
// nodes reject containers whose version they do not implement.
// 1.1 — CheckpointHeader gained optimizer_state_hash.
//
// The outer optimizer's momentum is state the whole network shares, and until this
// version nothing committed to it: the update was accepted on the producer's word.
// Because the optimizer used to step only on the producing node, a follower
// arrived at its own turn with momentum nobody else had and computed a different
// update from identical contributions — 56 against 33 from the same aggregate,
// compounding after that because Nesterov feeds momentum through the lookahead
// point. Every test stayed green while the only product of the protocol diverged
// on every node.
//
// Bumping this re-pins all three networks' anchors. That is free before launch and
// a hard fork after it, which is why every format-breaking change was gathered
// into this one version.
inline constexpr uint32_t kProtocolVersion = 0x0001'0001;  // 1.1

enum class ObjectType : uint16_t {
    RoundDescriptor = 1,
    DatasetManifest = 2,
    BatchSeed = 3,
    Contribution = 4,   // a worker's quantised pseudo-gradient for one outer step
    Checkpoint = 5,     // one link in the canonical chain of published models
    Challenge = 6,      // an order to recompute one contribution
    Verdict = 7,        // the result of that recomputation
    SlashEvidence = 8,  // a quorum of verdicts proving a contribution was forged
    PolicyDescriptor = 9,  // operational policy: schedule, verification, economics
};

// Serializer. Big-endian, fixed width, append-only.
class Writer {
public:
    void U8(uint8_t v);
    void U16(uint16_t v);
    void U32(uint32_t v);
    void U64(uint64_t v);
    void Bytes(std::span<const uint8_t> data);          // raw, caller knows the width
    void Hash(const crypto::Hash256& h);
    void VarBytes(std::span<const uint8_t> data);       // u32 length prefix + data
    void String(std::string_view s);                    // u32 length prefix + utf-8

    const std::vector<uint8_t>& data() const { return buf_; }
    std::vector<uint8_t> take() { return std::move(buf_); }
    size_t size() const { return buf_.size(); }

private:
    std::vector<uint8_t> buf_;
};

// Bounds-checked deserializer: every read can fail and says why. Nothing here can
// read past the end or silently return zeroes.
class Reader {
public:
    explicit Reader(std::span<const uint8_t> data) : data_(data) {}

    Result<uint8_t> U8();
    Result<uint16_t> U16();
    Result<uint32_t> U32();
    Result<uint64_t> U64();
    Result<std::vector<uint8_t>> Bytes(size_t len);
    Result<crypto::Hash256> Hash();
    Result<std::vector<uint8_t>> VarBytes();
    Result<std::string> String();

    size_t remaining() const { return data_.size() - pos_; }
    bool exhausted() const { return pos_ == data_.size(); }
    // Trailing bytes are an error for consensus objects: an artifact must decode
    // exactly, or an attacker could append meaning that some parsers ignore.
    Status ExpectExhausted() const;

private:
    std::span<const uint8_t> data_;
    size_t pos_{0};
};

// CRC32C (Castagnoli, reflected, init/xorout 0xFFFFFFFF).
uint32_t Crc32c(std::span<const uint8_t> data);

struct Container {
    ObjectType type{};
    uint32_t version{};
    std::vector<uint8_t> content;
    crypto::Hash256 content_hash{};
};

std::vector<uint8_t> WrapContainer(ObjectType type, std::span<const uint8_t> content);

// Verifies magic, version, declared length, content hash and CRC before
// returning. Any mismatch is an error — never a partially trusted object.
Result<Container> ParseContainer(std::span<const uint8_t> raw);

// SHA3-256 over the whole container: the artifact's identity (its "hash" as
// pinned in genesis and referenced by other objects).
crypto::Hash256 ContainerId(std::span<const uint8_t> raw);

std::string ObjectTypeName(ObjectType type);

}  // namespace rnet::canon

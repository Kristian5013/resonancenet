// Wire framing for the peer-to-peer protocol.
//
//   magic[4] | command[12] | payload_len:u32 | checksum[4] | payload
//
// The magic is the network's, so a peer on the wrong network is rejected on the
// first four bytes instead of half-parsing foreign traffic. The checksum is the
// first four bytes of SHA3-256 over the payload — one hash primitive everywhere.
//
// THE SIZE RULE, and why it is stated as an invariant rather than a constant:
//
// The Lattica audit found a consensus-valid 32 MB block that could never cross a
// 4 MB message cap — an object the rules permitted and the transport forbade,
// discovered only when demand grew. The fix is not a bigger number; a bigger
// number just moves the cliff. The fix is that NOTHING large is ever sent as a
// single message. Bulk payloads (pseudo-gradients, checkpoints — gigabytes) are
// transferred as explicit chunk messages, each far below the cap, and the cap is
// checked against the chunk size at build time. There is no size a valid object
// can reach that the transport cannot carry.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "crypto/sha3.h"
#include "util/result.h"

namespace rnet::net {

inline constexpr size_t kCommandSize = 12;
inline constexpr size_t kChecksumSize = 4;
inline constexpr size_t kHeaderSize = 4 + kCommandSize + 4 + kChecksumSize;   // 24 bytes

// Ceiling for ONE message. Control messages are far smaller; bulk data is chunked
// below this, never sent whole. Deliberately generous so no legitimate control
// message can approach it, and small enough that a hostile peer cannot make a
// node allocate meaningfully before authentication.
inline constexpr uint32_t kMaxMessageSize = 1u << 20;   // 1 MiB

// Payload bytes per bulk chunk. Must leave room for the chunk message's own
// framing and metadata inside kMaxMessageSize — the static_assert below is what
// guarantees a chunk can always be transmitted.
inline constexpr uint32_t kBulkChunkSize = 512u * 1024;   // 512 KiB
inline constexpr uint32_t kChunkMetadataAllowance = 4096;

static_assert(kBulkChunkSize + kChunkMetadataAllowance < kMaxMessageSize,
              "a bulk chunk plus its metadata must fit in one message — otherwise the protocol "
              "could describe transfers it cannot perform");

struct MessageHeader {
    std::array<uint8_t, 4> magic{};
    std::string command;          // ASCII, NUL-padded on the wire
    uint32_t payload_size{0};
    std::array<uint8_t, 4> checksum{};
};

// Serialises header + payload. The caller has already ensured the payload is
// within kMaxMessageSize; SerializeMessage enforces it again rather than trusting.
Result<std::vector<uint8_t>> SerializeMessage(const std::array<uint8_t, 4>& magic,
                                              std::string_view command,
                                              std::span<const uint8_t> payload);

// Parses a header. Rejects foreign magic, an oversized declared length, and a
// command containing anything but printable ASCII before the NUL padding — a
// peer must not be able to inject control characters into logs.
Result<MessageHeader> ParseHeader(std::span<const uint8_t> raw,
                                  const std::array<uint8_t, 4>& expected_magic);

// Confirms the payload matches the header's checksum.
Status VerifyPayload(const MessageHeader& header, std::span<const uint8_t> payload);

std::array<uint8_t, 4> PayloadChecksum(std::span<const uint8_t> payload);

// Number of chunks a bulk object of `total_bytes` requires.
uint64_t ChunkCount(uint64_t total_bytes);

}  // namespace rnet::net

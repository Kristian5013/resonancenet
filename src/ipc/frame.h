// One IPC message on the wire.
//
// The transport is SOCK_SEQPACKET, which already preserves record boundaries — so
// why a length prefix at all? Because a boundary the kernel maintains is not a
// boundary the parser checked. The declared length is compared against the record
// the kernel delivered, and a disagreement means the sender and the transport
// describe different messages. That comparison is only possible if the length is
// written down.
//
// The header is fixed-width and big-endian, like every other framed thing in this
// project, so one reader can be written once and read the same bytes on any
// machine.
//
// Frames are SMALL BY CONSTRUCTION. The largest legitimate message is a handshake
// reply of a few hundred bytes. Bulk data — a contribution payload is 983,635,968
// bytes — never travels as a frame; it crosses as a sealed memfd whose descriptor
// rides alongside the frame. A cap far below what the transport can carry is
// therefore not a limitation, it is the statement that this channel is for
// instructions rather than for data.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "util/result.h"

namespace rnet::ipc {

// Distinct from the P2P wire magic. A frame fed to the network port, or a network
// message fed to the socket, must fail on the first four bytes rather than
// half-parse into something that looks plausible.
inline constexpr std::array<uint8_t, 4> kFrameMagic = {'R', 'N', 'I', 'P'};

inline constexpr uint16_t kIpcVersion = 1;

// Header: magic(4) | version(2) | type(2) | request_id(8) | payload_len(4)
inline constexpr size_t kFrameHeaderSize = 20;

// A SOCK_SEQPACKET record cannot exceed the socket's send buffer, and no
// legitimate message here comes close. 128 KiB leaves three orders of magnitude of
// headroom over the largest real message while staying comfortably inside what a
// default kernel will carry in one record — a limit that describes transfers the
// transport cannot perform is the defect this project already forbids elsewhere.
inline constexpr uint32_t kMaxFramePayload = 128u * 1024;
inline constexpr size_t kMaxFrameSize = kFrameHeaderSize + kMaxFramePayload;

enum class MessageType : uint16_t {
    Hello = 0x0001,
    HelloOk = 0x0002,
    Error = 0x0003,

    GetAssignment = 0x0010,
    AssignTrain = 0x0011,
    AssignVerify = 0x0012,
    AssignNone = 0x0013,
    // A third kind of assignment: apply an aggregated update and report what the
    // weights hash to. It arrives as an ANSWER to GetAssignment rather than as an
    // unsolicited request, so the worker remains the only initiator and neither
    // side needs a second state machine.
    AssignApply = 0x0014,

    SubmitContribution = 0x0030,
    SubmitAccepted = 0x0031,

    SubmitVerdict = 0x0040,
    VerdictAccepted = 0x0041,
    SubmitWeights = 0x0044,
    WeightsAccepted = 0x0045,

    GetWeights = 0x0020,
    Weights = 0x0021,

    // A worker asking its node for one chunk of the corpus.
    //
    // Without this a worker can only train on a corpus its own machine holds,
    // which makes a seed pointless: the reason chunks travel with inclusion
    // proofs is so they can safely come from a stranger. The node fetches from a
    // peer if it does not hold the corpus itself, checks the bytes against
    // dataset_root, and only then answers.
    GetCorpusChunk = 0x0060,
    CorpusChunk = 0x0061,

    GetStatus = 0x0050,
    Status = 0x0051,
};

std::string MessageTypeName(MessageType type);

// Which direction a message is allowed to travel. A type that is legal in one
// direction only cannot be replayed back at its sender, and a daemon that would
// act on its own reply is a daemon with two state machines.
bool IsWorkerToDaemon(MessageType type);
bool IsDaemonToWorker(MessageType type);

// How many file descriptors a message must carry. Not "may": a message that
// arrives with the wrong number of descriptors is malformed, and any descriptors
// it did bring are closed before the connection is dropped — leaking them would
// let a peer exhaust the daemon's fd table by being wrong repeatedly.
uint32_t RequiredDescriptors(MessageType type);

struct Frame {
    MessageType type{MessageType::Error};
    // Echoed on the reply, so a worker can tell an answer from a stale one. The
    // daemon never invents a request id.
    uint64_t request_id{0};
    std::vector<uint8_t> payload;   // canonical CBOR
};

// Serialises a frame. Fails rather than truncating when the payload is too large:
// a truncated frame would be a shorter, valid-looking message.
Result<std::vector<uint8_t>> EncodeFrame(const Frame& frame);

// Parses one complete record. `record` must be exactly what the kernel delivered,
// because the length check compares the declared payload against it.
Result<Frame> DecodeFrame(std::span<const uint8_t> record);

}  // namespace rnet::ipc

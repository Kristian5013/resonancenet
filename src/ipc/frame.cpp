#include "ipc/frame.h"

#include <cstring>
#include <format>

namespace rnet::ipc {

std::string MessageTypeName(MessageType type) {
    switch (type) {
        case MessageType::Hello:              return "hello";
        case MessageType::HelloOk:            return "hello-ok";
        case MessageType::Error:              return "error";
        case MessageType::GetAssignment:      return "get-assignment";
        case MessageType::AssignTrain:        return "assign-train";
        case MessageType::AssignVerify:       return "assign-verify";
        case MessageType::AssignNone:         return "assign-none";
        case MessageType::AssignApply:        return "assign-apply";
        case MessageType::SubmitContribution: return "submit-contribution";
        case MessageType::SubmitAccepted:     return "submit-accepted";
        case MessageType::SubmitVerdict:      return "submit-verdict";
        case MessageType::VerdictAccepted:    return "verdict-accepted";
        case MessageType::SubmitWeights:      return "submit-weights";
        case MessageType::WeightsAccepted:    return "weights-accepted";
        case MessageType::GetWeights:         return "get-weights";
        case MessageType::Weights:            return "weights";
        case MessageType::GetStatus:          return "get-status";
        case MessageType::Status:             return "status";
        case MessageType::GetCorpusChunk:     return "get-corpus-chunk";
        case MessageType::CorpusChunk:        return "corpus-chunk";
    }
    return "unknown";
}

bool IsWorkerToDaemon(MessageType type) {
    switch (type) {
        case MessageType::Hello:
        case MessageType::GetAssignment:
        case MessageType::SubmitContribution:
        case MessageType::SubmitVerdict:
        case MessageType::SubmitWeights:
        case MessageType::GetWeights:
        case MessageType::GetStatus:
        case MessageType::GetCorpusChunk:
            return true;
        default:
            return false;
    }
}

bool IsDaemonToWorker(MessageType type) {
    switch (type) {
        case MessageType::HelloOk:
        case MessageType::Error:
        case MessageType::AssignTrain:
        case MessageType::AssignVerify:
        case MessageType::AssignNone:
        case MessageType::AssignApply:
        case MessageType::SubmitAccepted:
        case MessageType::WeightsAccepted:
        case MessageType::Weights:
        case MessageType::VerdictAccepted:
        case MessageType::Status:
        case MessageType::CorpusChunk:
            return true;
        default:
            return false;
    }
}

uint32_t RequiredDescriptors(MessageType type) {
    // Both directions move bulk data the same way: a contribution's payload is one
    // byte per parameter and an aggregated update is four, so neither fits a frame
    // and both cross as a sealed memfd. Exactly one descriptor — zero means the
    // data is missing, and more than one means the sender is trying to make the
    // receiver choose.
    switch (type) {
        // A contribution payload is one byte per parameter, an aggregated update is
        // four, and a checkpoint's weights are two — none fits a frame, so all
        // cross as a descriptor. Exactly one: zero means the data is missing, more
        // than one means the sender is trying to make the receiver choose.
        case MessageType::SubmitContribution:
        case MessageType::AssignApply:
        case MessageType::SubmitWeights:
        case MessageType::Weights:
            return 1u;
        default:
            return 0u;
    }
}

namespace {

void PutU16(std::vector<uint8_t>& out, uint16_t v) {
    out.push_back(static_cast<uint8_t>(v >> 8));
    out.push_back(static_cast<uint8_t>(v & 0xFF));
}

void PutU32(std::vector<uint8_t>& out, uint32_t v) {
    for (int shift = 24; shift >= 0; shift -= 8) {
        out.push_back(static_cast<uint8_t>((v >> shift) & 0xFF));
    }
}

void PutU64(std::vector<uint8_t>& out, uint64_t v) {
    for (int shift = 56; shift >= 0; shift -= 8) {
        out.push_back(static_cast<uint8_t>((v >> shift) & 0xFF));
    }
}

uint16_t ReadU16(const uint8_t* p) {
    return static_cast<uint16_t>((static_cast<uint16_t>(p[0]) << 8) | p[1]);
}

uint32_t ReadU32(const uint8_t* p) {
    return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

uint64_t ReadU64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v = (v << 8) | p[i];
    return v;
}

bool IsKnownType(uint16_t raw) {
    const auto type = static_cast<MessageType>(raw);
    return IsWorkerToDaemon(type) || IsDaemonToWorker(type);
}

}  // namespace

Result<std::vector<uint8_t>> EncodeFrame(const Frame& frame) {
    if (frame.payload.size() > kMaxFramePayload) {
        // Truncating would produce a shorter message that still parses — the worst
        // possible failure, because it looks like success.
        return Err(std::format("ipc: payload of {} bytes exceeds the {} byte frame limit",
                               frame.payload.size(), kMaxFramePayload));
    }
    if (!IsKnownType(static_cast<uint16_t>(frame.type))) {
        return Err("ipc: refusing to encode an unknown message type");
    }

    std::vector<uint8_t> out;
    out.reserve(kFrameHeaderSize + frame.payload.size());
    out.insert(out.end(), kFrameMagic.begin(), kFrameMagic.end());
    PutU16(out, kIpcVersion);
    PutU16(out, static_cast<uint16_t>(frame.type));
    PutU64(out, frame.request_id);
    PutU32(out, static_cast<uint32_t>(frame.payload.size()));
    out.insert(out.end(), frame.payload.begin(), frame.payload.end());
    return out;
}

Result<Frame> DecodeFrame(std::span<const uint8_t> record) {
    if (record.size() < kFrameHeaderSize) {
        return Err(std::format("ipc: record of {} bytes is shorter than a header", record.size()));
    }
    if (std::memcmp(record.data(), kFrameMagic.data(), kFrameMagic.size()) != 0) {
        // Not a frame from this protocol. Failing on the first four bytes is what
        // stops a network message, or a file, from being half-parsed into
        // something plausible.
        return Err("ipc: bad frame magic");
    }

    const uint16_t version = ReadU16(record.data() + 4);
    if (version != kIpcVersion) {
        // No negotiation and no downgrade: a version this daemon does not
        // implement is refused rather than approximated.
        return Err(std::format("ipc: unsupported frame version {}", version));
    }

    const uint16_t raw_type = ReadU16(record.data() + 6);
    if (!IsKnownType(raw_type)) {
        return Err(std::format("ipc: unknown message type 0x{:04x}", raw_type));
    }

    Frame frame;
    frame.type = static_cast<MessageType>(raw_type);
    frame.request_id = ReadU64(record.data() + 8);
    const uint32_t declared = ReadU32(record.data() + 16);

    if (declared > kMaxFramePayload) {
        return Err(std::format("ipc: declared payload of {} bytes exceeds the limit", declared));
    }
    // The kernel preserved a record boundary; this checks that the sender's own
    // description of the message agrees with it. A mismatch means the two disagree
    // about where the message ends, and there is no safe way to pick a winner.
    if (kFrameHeaderSize + declared != record.size()) {
        return Err(std::format("ipc: frame declares {} payload bytes but the record holds {}",
                               declared, record.size() - kFrameHeaderSize));
    }

    frame.payload.assign(record.begin() + kFrameHeaderSize, record.end());
    return frame;
}

}  // namespace rnet::ipc

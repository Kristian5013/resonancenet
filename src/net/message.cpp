#include "net/message.h"

#include <cstring>
#include <format>

#include "canon/canon.h"

namespace rnet::net {

std::array<uint8_t, 4> PayloadChecksum(std::span<const uint8_t> payload) {
    const crypto::Hash256 digest = crypto::Sha3_256(payload);
    return {digest[0], digest[1], digest[2], digest[3]};
}

Result<std::vector<uint8_t>> SerializeMessage(const std::array<uint8_t, 4>& magic,
                                              std::string_view command,
                                              std::span<const uint8_t> payload) {
    if (command.empty() || command.size() > kCommandSize) {
        return Err(std::format("message: command '{}' must be 1..{} characters",
                               std::string(command), kCommandSize));
    }
    for (char c : command) {
        if (c < 0x20 || c > 0x7E) return Err("message: command must be printable ASCII");
    }
    if (payload.size() > kMaxMessageSize) {
        return Err(std::format("message: payload {} exceeds the {} byte limit — bulk data must be "
                               "chunked, never sent whole",
                               payload.size(), kMaxMessageSize));
    }

    canon::Writer w;
    w.Bytes(std::span<const uint8_t>(magic.data(), magic.size()));

    std::array<uint8_t, kCommandSize> padded{};
    std::memcpy(padded.data(), command.data(), command.size());
    w.Bytes(std::span<const uint8_t>(padded.data(), padded.size()));

    w.U32(static_cast<uint32_t>(payload.size()));
    const auto checksum = PayloadChecksum(payload);
    w.Bytes(std::span<const uint8_t>(checksum.data(), checksum.size()));
    w.Bytes(payload);
    return w.take();
}

Result<MessageHeader> ParseHeader(std::span<const uint8_t> raw,
                                  const std::array<uint8_t, 4>& expected_magic) {
    if (raw.size() < kHeaderSize) return Err("message: header truncated");

    MessageHeader header;
    std::copy(raw.begin(), raw.begin() + 4, header.magic.begin());
    if (header.magic != expected_magic) {
        // Wrong network: disconnect rather than interpret anything that follows.
        return Err("message: wrong network magic");
    }

    // Command is NUL-padded. Everything before the first NUL must be printable,
    // and everything after it must be NUL — otherwise a peer could smuggle bytes
    // through the padding.
    size_t command_len = 0;
    while (command_len < kCommandSize && raw[4 + command_len] != 0) ++command_len;
    for (size_t i = 0; i < command_len; ++i) {
        const uint8_t c = raw[4 + i];
        if (c < 0x20 || c > 0x7E) return Err("message: command contains non-printable bytes");
    }
    for (size_t i = command_len; i < kCommandSize; ++i) {
        if (raw[4 + i] != 0) return Err("message: command padding is not NUL");
    }
    if (command_len == 0) return Err("message: empty command");
    header.command.assign(reinterpret_cast<const char*>(raw.data() + 4), command_len);

    uint32_t size = 0;
    for (size_t i = 0; i < 4; ++i) size = (size << 8) | raw[4 + kCommandSize + i];
    if (size > kMaxMessageSize) {
        // Refused before a single byte is allocated for it.
        return Err(std::format("message: declared payload {} exceeds the {} byte limit", size,
                               kMaxMessageSize));
    }
    header.payload_size = size;

    const size_t checksum_offset = 4 + kCommandSize + 4;
    std::copy(raw.begin() + static_cast<ptrdiff_t>(checksum_offset),
              raw.begin() + static_cast<ptrdiff_t>(checksum_offset + kChecksumSize),
              header.checksum.begin());
    return header;
}

Status VerifyPayload(const MessageHeader& header, std::span<const uint8_t> payload) {
    if (payload.size() != header.payload_size) {
        return Err(std::format("message: payload is {} bytes, header declared {}", payload.size(),
                               header.payload_size));
    }
    if (PayloadChecksum(payload) != header.checksum) return Err("message: checksum mismatch");
    return Status::Ok();
}

uint64_t ChunkCount(uint64_t total_bytes) {
    if (total_bytes == 0) return 0;
    return (total_bytes + kBulkChunkSize - 1) / kBulkChunkSize;
}

}  // namespace rnet::net

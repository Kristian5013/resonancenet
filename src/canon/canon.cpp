#include "canon/canon.h"

#include <cstring>

#include "util/hex.h"

namespace rnet::canon {

namespace {
constexpr size_t kHeaderSize = 4 + 4 + 2 + 4;   // magic + version + type + len
constexpr size_t kTrailerSize = 32 + 4;         // content_hash + crc32c
constexpr uint32_t kCrc32cPoly = 0x82F63B78u;   // Castagnoli, reflected
}  // namespace

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------
void Writer::U8(uint8_t v) { buf_.push_back(v); }

void Writer::U16(uint16_t v) {
    buf_.push_back(static_cast<uint8_t>(v >> 8));
    buf_.push_back(static_cast<uint8_t>(v));
}

void Writer::U32(uint32_t v) {
    buf_.push_back(static_cast<uint8_t>(v >> 24));
    buf_.push_back(static_cast<uint8_t>(v >> 16));
    buf_.push_back(static_cast<uint8_t>(v >> 8));
    buf_.push_back(static_cast<uint8_t>(v));
}

void Writer::U64(uint64_t v) {
    for (int shift = 56; shift >= 0; shift -= 8) {
        buf_.push_back(static_cast<uint8_t>(v >> shift));
    }
}

void Writer::Bytes(std::span<const uint8_t> data) {
    buf_.insert(buf_.end(), data.begin(), data.end());
}

void Writer::Hash(const crypto::Hash256& h) {
    buf_.insert(buf_.end(), h.begin(), h.end());
}

void Writer::VarBytes(std::span<const uint8_t> data) {
    U32(static_cast<uint32_t>(data.size()));
    Bytes(data);
}

void Writer::String(std::string_view s) {
    U32(static_cast<uint32_t>(s.size()));
    buf_.insert(buf_.end(), s.begin(), s.end());
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------
Result<uint8_t> Reader::U8() {
    if (remaining() < 1) return Err("canon: truncated (u8)");
    return data_[pos_++];
}

Result<uint16_t> Reader::U16() {
    if (remaining() < 2) return Err("canon: truncated (u16)");
    const uint16_t v = static_cast<uint16_t>(static_cast<uint16_t>(data_[pos_]) << 8 |
                                             static_cast<uint16_t>(data_[pos_ + 1]));
    pos_ += 2;
    return v;
}

Result<uint32_t> Reader::U32() {
    if (remaining() < 4) return Err("canon: truncated (u32)");
    uint32_t v = 0;
    for (size_t i = 0; i < 4; ++i) v = (v << 8) | data_[pos_ + i];
    pos_ += 4;
    return v;
}

Result<uint64_t> Reader::U64() {
    if (remaining() < 8) return Err("canon: truncated (u64)");
    uint64_t v = 0;
    for (size_t i = 0; i < 8; ++i) v = (v << 8) | data_[pos_ + i];
    pos_ += 8;
    return v;
}

Result<std::vector<uint8_t>> Reader::Bytes(size_t len) {
    if (remaining() < len) return Err("canon: truncated (" + std::to_string(len) + " bytes)");
    std::vector<uint8_t> out(data_.begin() + static_cast<ptrdiff_t>(pos_),
                             data_.begin() + static_cast<ptrdiff_t>(pos_ + len));
    pos_ += len;
    return out;
}

Result<crypto::Hash256> Reader::Hash() {
    auto bytes = Bytes(32);
    if (!bytes) return Err(bytes.error());
    crypto::Hash256 h{};
    std::copy(bytes.value().begin(), bytes.value().end(), h.begin());
    return h;
}

Result<std::vector<uint8_t>> Reader::VarBytes() {
    auto len = U32();
    if (!len) return Err(len.error());
    // Length is checked against what is actually present, so a hostile length
    // prefix cannot cause a huge allocation.
    if (len.value() > remaining()) return Err("canon: varbytes length exceeds input");
    return Bytes(len.value());
}

Result<std::string> Reader::String() {
    auto bytes = VarBytes();
    if (!bytes) return Err(bytes.error());
    return std::string(bytes.value().begin(), bytes.value().end());
}

Status Reader::ExpectExhausted() const {
    if (pos_ != data_.size()) {
        return Err("canon: " + std::to_string(data_.size() - pos_) + " trailing byte(s)");
    }
    return Status::Ok();
}

// ---------------------------------------------------------------------------
// CRC32C
// ---------------------------------------------------------------------------
uint32_t Crc32c(std::span<const uint8_t> data) {
    uint32_t crc = 0xFFFFFFFFu;
    for (uint8_t b : data) {
        crc ^= b;
        for (int i = 0; i < 8; ++i) {
            crc = (crc & 1u) ? (crc >> 1) ^ kCrc32cPoly : (crc >> 1);
        }
    }
    return crc ^ 0xFFFFFFFFu;
}

// ---------------------------------------------------------------------------
// Container
// ---------------------------------------------------------------------------
std::vector<uint8_t> WrapContainer(ObjectType type, std::span<const uint8_t> content) {
    Writer w;
    w.Bytes(std::span<const uint8_t>(kMagic, 4));
    w.U32(kProtocolVersion);
    w.U16(static_cast<uint16_t>(type));
    w.U32(static_cast<uint32_t>(content.size()));
    w.Bytes(content);
    w.Hash(crypto::Sha3_256(content));
    std::vector<uint8_t> out = w.take();
    const uint32_t crc = Crc32c(out);
    Writer tail;
    tail.U32(crc);
    out.insert(out.end(), tail.data().begin(), tail.data().end());
    return out;
}

Result<Container> ParseContainer(std::span<const uint8_t> raw) {
    if (raw.size() < kHeaderSize + kTrailerSize) return Err("canon: container too short");
    if (std::memcmp(raw.data(), kMagic, 4) != 0) return Err("canon: bad magic");

    Reader r(raw);
    (void)r.Bytes(4);  // magic, already checked

    auto version = r.U32();
    if (!version) return Err(version.error());
    if (version.value() != kProtocolVersion) {
        return Err("canon: unsupported protocol version " + util::ToHex(std::span<const uint8_t>(
                       raw.data() + 4, 4)));
    }

    auto type = r.U16();
    if (!type) return Err(type.error());

    auto len = r.U32();
    if (!len) return Err(len.error());
    if (static_cast<size_t>(len.value()) + kHeaderSize + kTrailerSize != raw.size()) {
        return Err("canon: declared content length does not match container size");
    }

    auto content = r.Bytes(len.value());
    if (!content) return Err(content.error());

    auto content_hash = r.Hash();
    if (!content_hash) return Err(content_hash.error());

    auto crc = r.U32();
    if (!crc) return Err(crc.error());
    if (auto st = r.ExpectExhausted(); !st) return Err(st.error());

    if (Crc32c(std::span<const uint8_t>(raw.data(), raw.size() - 4)) != crc.value()) {
        return Err("canon: crc32c mismatch (corrupt artifact)");
    }
    if (crypto::Sha3_256(content.value()) != content_hash.value()) {
        return Err("canon: content hash mismatch");
    }

    Container c;
    c.type = static_cast<ObjectType>(type.value());
    c.version = version.value();
    c.content = std::move(content.value());
    c.content_hash = content_hash.value();
    return c;
}

crypto::Hash256 ContainerId(std::span<const uint8_t> raw) { return crypto::Sha3_256(raw); }

std::string ObjectTypeName(ObjectType type) {
    switch (type) {
        case ObjectType::RoundDescriptor: return "RoundDescriptor";
        case ObjectType::DatasetManifest: return "DatasetManifest";
        case ObjectType::BatchSeed: return "BatchSeed";
        case ObjectType::Contribution: return "Contribution";
        case ObjectType::Checkpoint: return "Checkpoint";
        case ObjectType::Challenge: return "Challenge";
        case ObjectType::Verdict: return "Verdict";
        case ObjectType::SlashEvidence: return "SlashEvidence";
        case ObjectType::PolicyDescriptor: return "PolicyDescriptor";
    }
    return "Unknown(" + std::to_string(static_cast<uint16_t>(type)) + ")";
}

}  // namespace rnet::canon

#include "net/protocol.h"

#include "net/corpus.h"

#include <algorithm>
#include <charconv>
#include <cstring>
#include <format>

#include "net/message.h"
#include "util/hex.h"

namespace rnet::net {

namespace {

constexpr size_t kMaxUserAgent = 64;
constexpr size_t kMaxRejectReason = 128;

// IPv4-mapped IPv6 prefix: ::ffff:0:0/96
constexpr uint8_t kV4MappedPrefix[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF};

Status ReadBounded(canon::Reader& r, std::string& out, size_t limit, const char* what) {
    auto s = r.String();
    if (!s) return Err(s.error());
    if (s.value().size() > limit) {
        return Err(std::format("{}: {} bytes exceeds the {} byte limit", what, s.value().size(),
                               limit));
    }
    for (char c : s.value()) {
        // Peer-supplied text ends up in logs; control characters must not.
        if (static_cast<unsigned char>(c) < 0x20 || static_cast<unsigned char>(c) == 0x7F) {
            return Err(std::format("{}: contains control characters", what));
        }
    }
    out = s.value();
    return Status::Ok();
}

}  // namespace

// ---------------------------------------------------------------------------
// VersionMessage
// ---------------------------------------------------------------------------
std::vector<uint8_t> VersionMessage::Serialize() const {
    canon::Writer w;
    w.U32(protocol_version);
    w.U64(services);
    w.U64(static_cast<uint64_t>(timestamp));
    w.U64(nonce);
    w.String(user_agent);
    w.U64(worker_id);
    w.U16(determinism_class);
    w.Hash(genesis_hash);
    w.Hash(policy_hash);
    return w.take();
}

Result<VersionMessage> VersionMessage::Deserialize(std::span<const uint8_t> raw) {
    canon::Reader r(raw);
    VersionMessage m;

    auto version = r.U32();
    if (!version) return Err(version.error());
    m.protocol_version = version.value();

    auto services = r.U64();
    if (!services) return Err(services.error());
    m.services = services.value();

    auto timestamp = r.U64();
    if (!timestamp) return Err(timestamp.error());
    m.timestamp = static_cast<int64_t>(timestamp.value());

    auto nonce = r.U64();
    if (!nonce) return Err(nonce.error());
    m.nonce = nonce.value();

    if (auto st = ReadBounded(r, m.user_agent, kMaxUserAgent, "version.user_agent"); !st) {
        return Err(st.error());
    }

    auto worker_id = r.U64();
    if (!worker_id) return Err(worker_id.error());
    m.worker_id = worker_id.value();

    auto det = r.U16();
    if (!det) return Err(det.error());
    m.determinism_class = det.value();

    auto genesis = r.Hash();
    if (!genesis) return Err(genesis.error());
    m.genesis_hash = genesis.value();

    auto policy = r.Hash();
    if (!policy) return Err(policy.error());
    m.policy_hash = policy.value();

    if (auto st = r.ExpectExhausted(); !st) return Err(st.error());
    if (auto st = m.Validate(); !st) return Err(st.error());
    return m;
}

Status VersionMessage::Validate() const {
    if (protocol_version < kMinProtocolVersion) {
        return Err(std::format("version: protocol {} is below the minimum {}", protocol_version,
                               kMinProtocolVersion));
    }
    if (genesis_hash == crypto::Hash256{}) return Err("version: genesis hash must be set");
    if (policy_hash == crypto::Hash256{}) return Err("version: policy hash must be set");
    return Status::Ok();
}

// ---------------------------------------------------------------------------
// NetAddress
// ---------------------------------------------------------------------------
bool NetAddress::IsIPv4() const { return std::memcmp(ip.data(), kV4MappedPrefix, 12) == 0; }

bool NetAddress::IsRoutable() const {
    if (port == 0) return false;

    bool all_zero = true;
    for (uint8_t b : ip) {
        if (b != 0) { all_zero = false; break; }
    }
    if (all_zero) return false;   // unspecified

    if (IsIPv4()) {
        const uint8_t a = ip[12], b = ip[13];
        if (a == 0 || a == 127) return false;              // this-network, loopback
        if (a == 10) return false;                         // private
        if (a == 172 && b >= 16 && b <= 31) return false;  // private
        if (a == 192 && b == 168) return false;            // private
        if (a == 169 && b == 254) return false;            // link-local
        if (a >= 224) return false;                        // multicast / reserved
        return true;
    }
    if (ip[0] == 0xFE && (ip[1] & 0xC0) == 0x80) return false;   // link-local
    if ((ip[0] & 0xFE) == 0xFC) return false;                    // unique-local
    if (ip[0] == 0xFF) return false;                             // multicast
    // ::1 loopback
    bool loopback = ip[15] == 1;
    for (size_t i = 0; i < 15 && loopback; ++i) loopback = ip[i] == 0;
    return !loopback;
}

std::string NetAddress::ToString() const {
    if (IsIPv4()) {
        return std::format("{}.{}.{}.{}:{}", ip[12], ip[13], ip[14], ip[15], port);
    }
    std::string out = "[";
    for (size_t i = 0; i < 16; i += 2) {
        if (i) out += ":";
        out += std::format("{:02x}{:02x}", ip[i], ip[i + 1]);
    }
    return out + std::format("]:{}", port);
}

Result<NetAddress> NetAddress::FromString(std::string_view text, uint16_t default_port) {
    NetAddress addr;
    addr.port = default_port;

    std::string_view host = text;
    if (!text.empty() && text.front() == '[') {   // [v6]:port
        const size_t close = text.find(']');
        if (close == std::string_view::npos) return Err("address: unterminated IPv6 literal");
        host = text.substr(1, close - 1);
        if (close + 1 < text.size() && text[close + 1] == ':') {
            const std::string_view port_text = text.substr(close + 2);
            uint16_t port = 0;
            if (std::from_chars(port_text.data(), port_text.data() + port_text.size(), port).ec !=
                std::errc()) {
                return Err("address: bad port");
            }
            addr.port = port;
        }
        return Err("address: IPv6 literal parsing is not implemented");
    }

    const size_t colon = text.rfind(':');
    if (colon != std::string_view::npos) {
        host = text.substr(0, colon);
        const std::string_view port_text = text.substr(colon + 1);
        uint16_t port = 0;
        if (std::from_chars(port_text.data(), port_text.data() + port_text.size(), port).ec !=
            std::errc()) {
            return Err("address: bad port");
        }
        addr.port = port;
    }

    std::memcpy(addr.ip.data(), kV4MappedPrefix, 12);
    size_t octet_index = 0;
    size_t start = 0;
    while (octet_index < 4) {
        const size_t dot = host.find('.', start);
        const std::string_view octet =
            host.substr(start, dot == std::string_view::npos ? std::string_view::npos : dot - start);
        if (octet.empty()) return Err("address: malformed IPv4 literal");
        unsigned value = 0;
        if (std::from_chars(octet.data(), octet.data() + octet.size(), value).ec != std::errc() ||
            value > 255) {
            return Err("address: octet out of range");
        }
        addr.ip[12 + octet_index] = static_cast<uint8_t>(value);
        ++octet_index;
        if (dot == std::string_view::npos) break;
        start = dot + 1;
    }
    if (octet_index != 4) return Err("address: IPv4 literal needs four octets");
    return addr;
}

bool NetAddress::operator==(const NetAddress& other) const {
    return ip == other.ip && port == other.port;
}

bool NetAddress::operator<(const NetAddress& other) const {
    if (ip != other.ip) return ip < other.ip;
    return port < other.port;
}

// ---------------------------------------------------------------------------
// AddrMessage
// ---------------------------------------------------------------------------
std::vector<uint8_t> AddrMessage::Serialize() const {
    canon::Writer w;
    w.U32(static_cast<uint32_t>(addresses.size()));
    for (const auto& a : addresses) {
        w.Bytes(std::span<const uint8_t>(a.ip.data(), a.ip.size()));
        w.U16(a.port);
        w.U64(a.services);
        w.U64(static_cast<uint64_t>(a.last_seen));
    }
    return w.take();
}

Result<AddrMessage> AddrMessage::Deserialize(std::span<const uint8_t> raw) {
    canon::Reader r(raw);
    auto count = r.U32();
    if (!count) return Err(count.error());
    if (count.value() > kMaxAddrPerMessage) {
        // Bounded so one peer cannot flood the address database.
        return Err(std::format("addr: {} addresses exceeds the {} limit", count.value(),
                               kMaxAddrPerMessage));
    }

    AddrMessage m;
    m.addresses.reserve(count.value());
    for (uint32_t i = 0; i < count.value(); ++i) {
        NetAddress a;
        auto ip = r.Bytes(16);
        if (!ip) return Err(ip.error());
        std::copy(ip.value().begin(), ip.value().end(), a.ip.begin());

        auto port = r.U16();
        if (!port) return Err(port.error());
        a.port = port.value();

        auto services = r.U64();
        if (!services) return Err(services.error());
        a.services = services.value();

        auto last_seen = r.U64();
        if (!last_seen) return Err(last_seen.error());
        a.last_seen = static_cast<int64_t>(last_seen.value());

        m.addresses.push_back(a);
    }
    if (auto st = r.ExpectExhausted(); !st) return Err(st.error());
    return m;
}

// ---------------------------------------------------------------------------
// InvMessage
// ---------------------------------------------------------------------------
bool InvEntry::operator==(const InvEntry& other) const {
    return type == other.type && hash == other.hash;
}

std::vector<uint8_t> InvMessage::Serialize() const {
    canon::Writer w;
    w.U32(static_cast<uint32_t>(entries.size()));
    for (const auto& e : entries) {
        w.U16(static_cast<uint16_t>(e.type));
        w.Hash(e.hash);
    }
    return w.take();
}

Result<InvMessage> InvMessage::Deserialize(std::span<const uint8_t> raw) {
    canon::Reader r(raw);
    auto count = r.U32();
    if (!count) return Err(count.error());
    if (count.value() > kMaxInvPerMessage) {
        return Err(std::format("inv: {} entries exceeds the {} limit", count.value(),
                               kMaxInvPerMessage));
    }

    InvMessage m;
    m.entries.reserve(count.value());
    for (uint32_t i = 0; i < count.value(); ++i) {
        auto type = r.U16();
        if (!type) return Err(type.error());
        if (type.value() < 1 || type.value() > 6) return Err("inv: unknown object type");

        auto hash = r.Hash();
        if (!hash) return Err(hash.error());
        m.entries.push_back(InvEntry{static_cast<InvType>(type.value()), hash.value()});
    }
    if (auto st = r.ExpectExhausted(); !st) return Err(st.error());
    return m;
}

// ---------------------------------------------------------------------------
// GetChunksMessage
// ---------------------------------------------------------------------------
std::vector<uint8_t> GetChunksMessage::Serialize() const {
    canon::Writer w;
    w.U16(static_cast<uint16_t>(type));
    w.Hash(hash);
    w.U64(first_chunk);
    w.U64(chunk_count);
    return w.take();
}

Result<GetChunksMessage> GetChunksMessage::Deserialize(std::span<const uint8_t> raw) {
    canon::Reader r(raw);
    GetChunksMessage m;

    auto type = r.U16();
    if (!type) return Err(type.error());
    if (type.value() < 1 || type.value() > 6) return Err("getchunks: unknown object type");
    m.type = static_cast<InvType>(type.value());

    auto hash = r.Hash();
    if (!hash) return Err(hash.error());
    m.hash = hash.value();

    auto first = r.U64();
    if (!first) return Err(first.error());
    m.first_chunk = first.value();

    auto count = r.U64();
    if (!count) return Err(count.error());
    m.chunk_count = count.value();

    if (auto st = r.ExpectExhausted(); !st) return Err(st.error());
    if (auto st = m.Validate(); !st) return Err(st.error());
    return m;
}

Status GetChunksMessage::Validate() const {
    if (chunk_count == 0) return Err("getchunks: must request at least one chunk");
    // Bounded so a single request cannot ask a node to serve an unbounded stream.
    constexpr uint64_t kMaxChunksPerRequest = 4096;
    if (chunk_count > kMaxChunksPerRequest) {
        return Err(std::format("getchunks: {} chunks exceeds the {} limit per request", chunk_count,
                               kMaxChunksPerRequest));
    }
    if (first_chunk + chunk_count < first_chunk) return Err("getchunks: range overflows");
    return Status::Ok();
}

// ---------------------------------------------------------------------------
// ChunkMessage
// ---------------------------------------------------------------------------
std::vector<uint8_t> ChunkMessage::Serialize() const {
    canon::Writer w;
    w.U16(static_cast<uint16_t>(type));
    w.Hash(hash);
    w.U64(index);
    w.U64(total_chunks);
    w.VarBytes(data);
    return w.take();
}

Result<ChunkMessage> ChunkMessage::Deserialize(std::span<const uint8_t> raw) {
    canon::Reader r(raw);
    ChunkMessage m;

    auto type = r.U16();
    if (!type) return Err(type.error());
    if (type.value() < 1 || type.value() > 6) return Err("chunk: unknown object type");
    m.type = static_cast<InvType>(type.value());

    auto hash = r.Hash();
    if (!hash) return Err(hash.error());
    m.hash = hash.value();

    auto index = r.U64();
    if (!index) return Err(index.error());
    m.index = index.value();

    auto total = r.U64();
    if (!total) return Err(total.error());
    m.total_chunks = total.value();

    auto data = r.VarBytes();
    if (!data) return Err(data.error());
    m.data = std::move(data.value());

    if (auto st = r.ExpectExhausted(); !st) return Err(st.error());
    if (auto st = m.Validate(); !st) return Err(st.error());
    return m;
}

Status ChunkMessage::Validate() const {
    if (total_chunks == 0) return Err("chunk: total_chunks must be non-zero");
    if (index >= total_chunks) return Err("chunk: index past the end of the object");
    if (data.empty()) return Err("chunk: empty payload");
    if (data.size() > kBulkChunkSize) {
        return Err(std::format("chunk: {} bytes exceeds the {} byte chunk size", data.size(),
                               kBulkChunkSize));
    }
    // Only the final chunk may be short; a short interior chunk would leave a hole
    // that the assembler could not detect from length alone.
    if (index + 1 < total_chunks && data.size() != kBulkChunkSize) {
        return Err("chunk: only the last chunk may be partial");
    }
    return Status::Ok();
}

// ---------------------------------------------------------------------------
// RejectMessage
// ---------------------------------------------------------------------------
std::vector<uint8_t> RejectMessage::Serialize() const {
    canon::Writer w;
    w.String(command);
    w.U16(static_cast<uint16_t>(code));
    w.String(reason);
    return w.take();
}

Result<RejectMessage> RejectMessage::Deserialize(std::span<const uint8_t> raw) {
    canon::Reader r(raw);
    RejectMessage m;

    if (auto st = ReadBounded(r, m.command, kCommandSize, "reject.command"); !st) {
        return Err(st.error());
    }

    auto code = r.U16();
    if (!code) return Err(code.error());
    if (code.value() < 1 || code.value() > 6) return Err("reject: unknown code");
    m.code = static_cast<RejectCode>(code.value());

    if (auto st = ReadBounded(r, m.reason, kMaxRejectReason, "reject.reason"); !st) {
        return Err(st.error());
    }
    if (auto st = r.ExpectExhausted(); !st) return Err(st.error());
    return m;
}

// ---------------------------------------------------------------------------
// ChunkAssembler
// ---------------------------------------------------------------------------
ChunkAssembler::ChunkAssembler(InvType type, crypto::Hash256 hash, uint64_t total_bytes)
    : type_(type), hash_(hash), total_bytes_(total_bytes), total_chunks_(ChunkCount(total_bytes)) {
    buffer_.resize(static_cast<size_t>(total_bytes));
    have_.assign(static_cast<size_t>(total_chunks_), false);
}

Status ChunkAssembler::Accept(const ChunkMessage& chunk) {
    if (auto st = chunk.Validate(); !st) return st;
    if (chunk.type != type_) return Err("assembler: chunk is for a different object type");
    if (chunk.hash != hash_) return Err("assembler: chunk is for a different object");
    if (chunk.total_chunks != total_chunks_) {
        return Err(std::format("assembler: peer claims {} chunks, expected {}", chunk.total_chunks,
                               total_chunks_));
    }
    if (chunk.index >= total_chunks_) return Err("assembler: chunk index out of range");

    const size_t offset = static_cast<size_t>(chunk.index) * kBulkChunkSize;
    const size_t expected = std::min<size_t>(kBulkChunkSize, buffer_.size() - offset);
    if (chunk.data.size() != expected) {
        return Err(std::format("assembler: chunk {} is {} bytes, expected {}", chunk.index,
                               chunk.data.size(), expected));
    }
    if (have_[static_cast<size_t>(chunk.index)]) return Status::Ok();   // duplicate, harmless

    std::copy(chunk.data.begin(), chunk.data.end(), buffer_.begin() + static_cast<ptrdiff_t>(offset));
    have_[static_cast<size_t>(chunk.index)] = true;
    ++received_;
    return Status::Ok();
}

bool ChunkAssembler::Complete() const { return received_ == total_chunks_; }

Result<std::vector<uint8_t>> ChunkAssembler::Finish() {
    if (!Complete()) {
        return Err(std::format("assembler: {} of {} chunks received", received_, total_chunks_));
    }
    // The object is only trustworthy as a whole: individual chunks carry no proof,
    // so nothing is handed out until the full hash matches.
    if (crypto::Sha3_256(buffer_) != hash_) {
        return Err("assembler: assembled object does not match its hash");
    }
    return std::move(buffer_);
}

}  // namespace rnet::net

namespace rnet::net {

// --- corpus requests --------------------------------------------------------

Status GetCorpusMessage::Validate() const {
    if (chunk_count == 0) return Err("getcorpus: must ask for at least one chunk");
    // A request for a million chunks is four terabytes of answer. Bounded so a
    // peer cannot make a seed node commit to work it never agreed to.
    if (chunk_count > 64) return Err("getcorpus: at most 64 chunks per request");
    return Status::Ok();
}

std::vector<uint8_t> GetCorpusMessage::Serialize() const {
    canon::Writer w;
    w.U64(first_chunk);
    w.U32(chunk_count);
    return w.take();
}

Result<GetCorpusMessage> GetCorpusMessage::Deserialize(std::span<const uint8_t> raw) {
    canon::Reader r(raw);
    GetCorpusMessage m;
    auto first = r.U64();
    if (!first) return Err(first.error());
    m.first_chunk = first.value();
    auto count = r.U32();
    if (!count) return Err(count.error());
    m.chunk_count = count.value();
    if (auto st = r.ExpectExhausted(); !st) return Err(st.error());
    if (auto st = m.Validate(); !st) return Err(st.error());
    return m;
}

Status CorpusMessage::Validate() const {
    if (chunk_bytes == 0) return Err("corpus: an empty chunk is not a chunk");
    if (data.size() > kCorpusPartBytes) return Err("corpus: part exceeds the part size");
    if (data.empty()) return Err("corpus: a part with no data");
    // A proof deeper than this describes a tree with more leaves than there are
    // atoms in the corpus; refusing it bounds the work before any hashing happens.
    if (proof.path.size() > 64) return Err("corpus: implausible proof depth");
    return Status::Ok();
}

std::vector<uint8_t> CorpusMessage::Serialize() const {
    canon::Writer w;
    w.U64(chunk_index);
    w.U64(chunk_bytes);
    w.U32(part);
    w.U64(proof.index);
    w.U64(proof.leaf_count);
    w.U32(static_cast<uint32_t>(proof.path.size()));
    for (const auto& node : proof.path) w.Hash(node);
    w.VarBytes(data);
    return w.take();
}

Result<CorpusMessage> CorpusMessage::Deserialize(std::span<const uint8_t> raw) {
    canon::Reader r(raw);
    CorpusMessage m;

    auto chunk_index = r.U64();
    if (!chunk_index) return Err(chunk_index.error());
    m.chunk_index = chunk_index.value();

    auto chunk_bytes = r.U64();
    if (!chunk_bytes) return Err(chunk_bytes.error());
    m.chunk_bytes = chunk_bytes.value();

    auto part = r.U32();
    if (!part) return Err(part.error());
    m.part = part.value();

    auto proof_index = r.U64();
    if (!proof_index) return Err(proof_index.error());
    m.proof.index = proof_index.value();

    auto leaf_count = r.U64();
    if (!leaf_count) return Err(leaf_count.error());
    m.proof.leaf_count = leaf_count.value();

    auto depth = r.U32();
    if (!depth) return Err(depth.error());
    // Bounded before allocating: the depth is a claim by the sender, and a claim
    // of four billion siblings must cost nothing to refuse.
    if (depth.value() > 64) return Err("corpus: implausible proof depth");
    for (uint32_t i = 0; i < depth.value(); ++i) {
        auto node = r.Hash();
        if (!node) return Err(node.error());
        m.proof.path.push_back(node.value());
    }

    auto data = r.VarBytes();
    if (!data) return Err(data.error());
    m.data = std::move(data.value());

    if (auto st = r.ExpectExhausted(); !st) return Err(st.error());
    if (auto st = m.Validate(); !st) return Err(st.error());
    return m;
}

}  // namespace rnet::net

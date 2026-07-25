#include "net/corpus.h"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <format>

#include "util/hex.h"

namespace rnet::net {

namespace {

size_t WidthOf(canon::TokenDtype dtype) { return dataset::TokenWidth(dtype); }

Result<size_t> ReadAt(const std::filesystem::path& path, uint64_t offset,
                      std::span<uint8_t> out) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return Err(std::format("corpus: cannot open {}: {}", path.string(), std::strerror(errno)));
    }
    size_t filled = 0;
    while (filled < out.size()) {
        const ssize_t got = ::pread(fd, out.data() + filled, out.size() - filled,
                                    static_cast<off_t>(offset + filled));
        if (got < 0) {
            if (errno == EINTR) continue;
            const auto message = std::string("corpus: pread: ") + std::strerror(errno);
            ::close(fd);
            return Err(message);
        }
        if (got == 0) break;
        filled += static_cast<size_t>(got);
    }
    ::close(fd);
    return filled;
}

}  // namespace

Result<CorpusSource> CorpusSource::Open(const std::filesystem::path& token_file,
                                        const canon::DatasetManifest& manifest) {
    if (auto st = manifest.Validate(); !st) return Err(st.error());

    auto index = dataset::DatasetIndex::Build(token_file, manifest.chunk_tokens, manifest.dtype);
    if (!index) return Err(index.error());

    // Checked here rather than left for the first worker to discover. A node
    // serving a file that is not the corpus would hand out tokens that fail every
    // proof, and the failure would look like the network accusing an honest seed.
    if (index.value().root() != manifest.dataset_root) {
        return Err(std::format(
            "corpus: {} has root {}, the manifest pins {} — this is not that corpus",
            token_file.string(), util::ToHex(index.value().root()),
            util::ToHex(manifest.dataset_root)));
    }
    if (index.value().n_tokens() != manifest.n_tokens) {
        return Err(std::format("corpus: file holds {} tokens, the manifest says {}",
                               index.value().n_tokens(), manifest.n_tokens));
    }

    CorpusSource source;
    source.token_file_ = token_file;
    source.manifest_ = manifest;
    source.index_ = std::move(index.value());
    return source;
}

Result<uint64_t> CorpusSource::ChunkBytes(uint64_t chunk_index) const {
    if (chunk_index >= manifest_.n_chunks) {
        return Err(std::format("corpus: chunk {} of {}", chunk_index, manifest_.n_chunks));
    }
    const uint64_t width = WidthOf(manifest_.dtype);
    const uint64_t first_token = chunk_index * manifest_.chunk_tokens;
    // The last chunk is short whenever the corpus does not divide evenly, which is
    // the normal case rather than an edge case.
    const uint64_t tokens =
        std::min<uint64_t>(manifest_.chunk_tokens, manifest_.n_tokens - first_token);
    return tokens * width;
}

Result<size_t> CorpusSource::ReadChunkRange(uint64_t chunk_index, uint64_t offset,
                                            std::span<uint8_t> out) const {
    auto size = ChunkBytes(chunk_index);
    if (!size) return Err(size.error());
    if (offset >= size.value()) return size_t{0};

    const uint64_t width = WidthOf(manifest_.dtype);
    const uint64_t chunk_start = chunk_index * manifest_.chunk_tokens * width;
    const size_t length =
        static_cast<size_t>(std::min<uint64_t>(out.size(), size.value() - offset));
    return ReadAt(token_file_, chunk_start + offset, out.subspan(0, length));
}

Result<crypto::MerkleProof> CorpusSource::ProofForChunk(uint64_t chunk_index) const {
    return index_.ProofForChunk(chunk_index);
}

// ---------------------------------------------------------------------------
// CorpusAssembler
// ---------------------------------------------------------------------------

CorpusAssembler::CorpusAssembler(uint64_t chunk_index, uint64_t total_bytes,
                                 crypto::MerkleProof proof)
    : chunk_index_(chunk_index),
      total_bytes_(total_bytes),
      total_parts_(static_cast<uint32_t>((total_bytes + kCorpusPartBytes - 1) / kCorpusPartBytes)),
      proof_(std::move(proof)),
      buffer_(static_cast<size_t>(total_bytes)),
      have_(total_parts_, false) {}

Status CorpusAssembler::Accept(uint32_t part, std::span<const uint8_t> data) {
    if (part >= total_parts_) {
        return Err(std::format("corpus: part {} of {}", part, total_parts_));
    }
    const uint64_t offset = static_cast<uint64_t>(part) * kCorpusPartBytes;
    const size_t expected =
        static_cast<size_t>(std::min<uint64_t>(kCorpusPartBytes, total_bytes_ - offset));
    // A part of the wrong length would leave a hole or overrun the buffer; either
    // way the assembled chunk would not be what the sender claimed to send.
    if (data.size() != expected) {
        return Err(std::format("corpus: part {} is {} bytes, expected {}", part, data.size(),
                               expected));
    }
    if (have_[part]) return Status::Ok();   // duplicate; relays overlap

    std::copy(data.begin(), data.end(), buffer_.begin() + static_cast<ptrdiff_t>(offset));
    have_[part] = true;
    ++received_;
    return Status::Ok();
}

bool CorpusAssembler::Complete() const { return received_ == total_parts_; }

Result<std::vector<uint8_t>> CorpusAssembler::Finish(const crypto::Hash256& dataset_root) {
    if (!Complete()) {
        return Err(std::format("corpus: {} of {} parts received", received_, total_parts_));
    }
    // Checked against consensus, not against whoever sent it. This is what makes it
    // safe to take the corpus from a stranger: an altered chunk fails the proof, and
    // the root it fails against is pinned in the round descriptor.
    if (!dataset::VerifyChunk(buffer_, proof_, dataset_root)) {
        return Err(std::format("corpus: chunk {} does not belong to the pinned dataset root",
                               chunk_index_));
    }
    return std::move(buffer_);
}

}  // namespace rnet::net

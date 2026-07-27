#include "net/corpus.h"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <format>

#include "util/hex.h"
#include "util/logging.h"

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

Result<CorpusSource> CorpusSource::Open(const std::filesystem::path& corpus_file,
                                        const canon::DatasetManifest& manifest) {
    if (auto st = manifest.Validate(); !st) return Err(st.error());

    // The cache first: building the index reads the whole corpus, which is two and
    // a half hours for 7.36 TB, and a node that spends that before it can complete
    // a handshake is indistinguishable from a node that is down.
    //
    // The cache is trusted only insofar as its leaves reproduce dataset_root — a
    // stale or tampered one fails that and the scan happens anyway. What it does
    // not re-check is that the file still matches those leaves, because that IS
    // the scan; a seed serving bytes that do not match is caught by the receiver's
    // inclusion proof, which is where the guarantee actually lives.
    const auto cache_path = std::filesystem::path(corpus_file.string() + ".rnidx");
    auto index = dataset::DatasetIndex::LoadCache(cache_path, manifest.dataset_root);
    if (index) {
        RNET_LOG_INFO("corpus: index loaded from {} ({} chunks), corpus not re-read",
                      cache_path.string(), index.value().n_chunks());
    } else {
        RNET_LOG_INFO("corpus: no usable index cache ({}); reading {} to build one",
                      index.error(), corpus_file.string());
        index = dataset::DatasetIndex::Build(corpus_file, manifest.target_chunk_bytes);
        if (!index) return Err(index.error());
        if (auto st = index.value().SaveCache(cache_path); !st) {
            // Not fatal: the node works, it will just pay the scan again next time.
            RNET_LOG_WARN("corpus: cannot write the index cache: {}", st.error());
        } else {
            RNET_LOG_INFO("corpus: index cached at {}", cache_path.string());
        }
    }

    // Checked here rather than left for the first worker to discover. A node
    // serving a file that is not the corpus would hand out tokens that fail every
    // proof, and the failure would look like the network accusing an honest seed.
    if (index.value().root() != manifest.dataset_root) {
        return Err(std::format(
            "corpus: {} has root {}, the manifest pins {} — this is not that corpus",
            corpus_file.string(), util::ToHex(index.value().root()),
            util::ToHex(manifest.dataset_root)));
    }
    if (index.value().n_bytes() != manifest.n_bytes) {
        return Err(std::format("corpus: file holds {} tokens, the manifest says {}",
                               index.value().n_bytes(), manifest.n_bytes));
    }

    CorpusSource source;
    source.token_file_ = corpus_file;
    source.manifest_ = manifest;
    source.index_ = std::move(index.value());
    return source;
}

Result<uint64_t> CorpusSource::ChunkBytes(uint64_t chunk_index) const {
    // From the index, not from arithmetic. Chunks end at document boundaries, so
    // their sizes are a property of the corpus rather than a formula, and the
    // index is where those boundaries were derived.
    auto extent = index_.ChunkExtent(chunk_index);
    if (!extent) return Err(extent.error());
    return extent.value().second - extent.value().first;
}

Result<size_t> CorpusSource::ReadChunkRange(uint64_t chunk_index, uint64_t offset,
                                            std::span<uint8_t> out) const {
    auto extent = index_.ChunkExtent(chunk_index);
    if (!extent) return Err(extent.error());
    const uint64_t size = extent.value().second - extent.value().first;
    if (offset >= size) return size_t{0};

    const size_t length = static_cast<size_t>(std::min<uint64_t>(out.size(), size - offset));
    return ReadAt(token_file_, extent.value().first + offset, out.subspan(0, length));
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
